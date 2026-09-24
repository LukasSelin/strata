package cog

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"sync"
	"sync/atomic"

	"github.com/klauspost/compress/lzw"
	"github.com/klauspost/compress/zlib"
	"github.com/klauspost/compress/zstd"

	"github.com/LukasSelin/strata/cog/internal/kern"
	"github.com/LukasSelin/strata/raster"
)

// Decoding one block: fetch its bytes, decompress them, undo the
// predictor, and convert one band's samples to float32 with validity.
// The steps follow libtiff, since libtiff is what writes these files
// under GDAL: the predictors work a block row at a time, horizontal
// differencing on integers in native byte order and the floating-point
// predictor on byte planes, most significant first.

// block is one decoded tile or strip of one band: w×rows cells, row-major,
// with their validity bits, or nil validity when every cell is valid.
//
// Its cells are in a buffer that is reused for another block once nobody
// holds this one: the source's cache while it keeps the block, and each
// reader while it copies from it (see cache). A block is born held once,
// by whoever decoded it.
type block struct {
	vals  []float32
	valid []uint64
	w     int
	rows  int
	refs  atomic.Int32
}

// newBlock returns a block of w×rows cells, held once, whose values are
// not cleared: the decoder writes every one.
func newBlock(w, rows int) *block {
	b := &block{w: w, rows: rows, vals: getVals(w * rows)}
	b.refs.Store(1)
	return b
}

func (b *block) hold() { b.refs.Add(1) }

// release gives back one reference, and the cells' buffer with the last.
func (b *block) release() {
	switch n := b.refs.Add(-1); {
	case n == 0:
		putVals(b.vals)
		b.vals = nil
	case n < 0:
		panic("cog: a block released more often than it was held")
	}
}

// valsPool holds the cells' buffers of released blocks. The blocks of a
// level are all one size but for a strip file's last strip, so a buffer
// that is too small is simply left for the collector.
var valsPool sync.Pool // of *[]float32

func getVals(n int) []float32 {
	var v []float32
	if p, ok := valsPool.Get().(*[]float32); ok && cap(*p) >= n {
		v = (*p)[:n]
	} else {
		v = make([]float32, n)
	}
	if poisonVals {
		for i := range v {
			v[i] = poison
		}
	}
	return v
}

func putVals(v []float32) {
	if cap(v) > 0 {
		valsPool.Put(&v)
	}
}

// poisonVals, which the tests set, fills every buffer a decoder is given
// with poison, new or reused, so that a decoder that leaves a cell
// unwritten reads as wrong rather than as the zero or stale value it
// happens to find.
var poisonVals bool

var poison = math.Float32frombits(0x7fc0dead) // a NaN no file holds

// size is the memory a block holds, for the cache's accounting.
func (b *block) size() int64 {
	return int64(4*len(b.vals) + 8*len(b.valid) + 64)
}

// noData is a NoData value prepared as GDAL's NoData mask band compares
// it with samples (gcore/gdalnodatamaskband.cpp):
//
//   - An integer type has a NoData value only if the value is within the
//     type's range. It is then truncated toward zero, as C++'s cast does,
//     and compared exactly: NoData 12.5 on bytes marks the cells holding
//     12.
//   - Floats compare with ARE_REAL_EQUAL: equal, or closer than
//     2·FLT_EPSILON·|a+b|. That is about two float32 ulps for float32
//     samples, and the same relative width, far more ulps, for float64.
//     A NaN NoData matches every NaN.
//   - A float32 file's NoData is first rounded to float32 by the GTiff
//     driver, after moving a value within 1e-10 of ±MaxFloat32 onto it
//     (GDALAdjustNoDataCloseToFloatMax). A value beyond float32's range
//     rounds to ±Inf and matches infinite cells.
type noData struct {
	set   bool
	nan   bool    // NoData is NaN: any NaN matches
	cmp   float64 // the value: integral for integers, float32-exact for float32
	cmp32 float32
}

// prepareNoData prepares GDAL's NoData value v for samples of the given
// format and bit depth.
func prepareNoData(v float64, has bool, format, bits int) noData {
	if !has {
		return noData{}
	}
	if format != sampleFloat {
		lo, hi := 0.0, math.Exp2(float64(bits))-1
		if format == sampleInt {
			lo, hi = -math.Exp2(float64(bits-1)), math.Exp2(float64(bits-1))-1
		}
		if !(v >= lo && v <= hi) { // NaN included
			return noData{}
		}
		return noData{set: true, cmp: math.Trunc(v)}
	}
	if math.IsNaN(v) {
		return noData{set: true, nan: true}
	}
	if bits == 32 {
		const maxF = math.MaxFloat32
		switch {
		case math.Abs(v-maxF) < 1e-10*maxF:
			v = maxF
		case math.Abs(v+maxF) < 1e-10*maxF:
			v = -maxF
		}
		f := toFloat32(v)
		return noData{set: true, cmp: float64(f), cmp32: f}
	}
	return noData{set: true, cmp: v}
}

// flt32Epsilon is C's FLT_EPSILON, which ARE_REAL_EQUAL uses for float32
// and float64 samples alike.
const flt32Epsilon = 0x1p-23

// realEqual32 is GDAL's ARE_REAL_EQUAL on floats, each operation rounded
// to float32 as the C++ is.
func realEqual32(a, b float32) bool {
	if a == b {
		return true
	}
	d := float32(math.Abs(float64(float32(a - b))))
	s := float32(math.Abs(float64(float32(a + b))))
	return d < float32(float32(flt32Epsilon*s)*2)
}

// realEqual64 is GDAL's ARE_REAL_EQUAL on doubles.
func realEqual64(a, b float64) bool {
	return a == b || math.Abs(a-b) < float64(flt32Epsilon*math.Abs(a+b))*2
}

// matches reports whether sample v, exact as a float64, is NoData.
func (nd noData) matches(v float64, format, bits int) bool {
	switch {
	case !nd.set:
		return false
	case nd.nan:
		return math.IsNaN(v)
	case format != sampleFloat:
		return v == nd.cmp
	case bits == 32:
		return realEqual32(float32(v), nd.cmp32)
	}
	return realEqual64(v, nd.cmp)
}

// decodeBlock reads and decodes block idx of im, at block row by, for
// band. nd is the prepared NoData value. read, if not nil, returns the
// block's stored bytes, which other sources share, so they are not
// modified; if nil, the block is read from the file into a scratch
// buffer.
func (c *container) decodeBlock(im *image, idx, by, band int, nd noData, read func(off, n uint64) ([]byte, error)) (*block, error) {
	rows := im.blockRows(by)
	n := im.blockW * rows
	off, count := im.offsets[idx], im.byteCounts[idx]
	if count == 0 {
		// An absent block, which GDAL writes as a sparse one (offset
		// and count 0) for all-NoData tiles, and reads, as it reads
		// any block of 0 bytes, as NoData, or as 0 without one.
		b := newBlock(im.blockW, rows)
		clear(b.vals)
		if nd.set {
			b.valid = make([]uint64, raster.MaskWords(n)) // all invalid
		}
		return b, nil
	}
	if count > 2*maxBlockBytes {
		return nil, fmt.Errorf("%d compressed bytes for a block, more than %d MiB", count, 2*maxBlockBytes>>20)
	}
	var raw []byte
	switch {
	case read != nil:
		var err error
		if raw, err = read(off, count); err != nil {
			return nil, fmt.Errorf("reading %d bytes at offset %d: %w", count, off, err)
		}
	case count <= readChunk && off <= math.MaxInt64-count:
		// The usual case, read into a scratch buffer; readFull reads in
		// chunks, for counts a short file may not back.
		rp := getScratch(int(count)) // #nosec G115 -- at most readChunk
		defer putScratch(rp)
		raw = *rp
		if err := readAtFull(c.r, raw, int64(off)); err != nil { // #nosec G115 -- checked above
			return nil, fmt.Errorf("reading %d bytes at offset %d: %w", count, off, err)
		}
	default:
		var err error
		if raw, err = c.readFull(off, count); err != nil {
			return nil, fmt.Errorf("reading %d bytes at offset %d: %w", count, off, err)
		}
	}
	if im.lsbFirst {
		for i, b := range raw {
			raw[i] = bits.Reverse8(b)
		}
	}
	spb := im.blockSamples()
	rowBytes := im.rowBytes()
	want := rowBytes * rows
	if read != nil && im.compression == compressionNone &&
		(im.predictor != predictorNone || c.order == binary.BigEndian && im.bytes > 1) {
		// decompress returns raw itself, and undoing the predictor or
		// the byte order writes to it, so shared bytes are copied first.
		rp := getScratch(len(raw))
		defer putScratch(rp)
		copy(*rp, raw)
		raw = *rp
	}
	var dst []byte // where a stream format decompresses to
	if im.compression != compressionNone {
		dp := getScratch(want)
		defer putScratch(dp)
		dst = *dp
	}
	data, err := decompress(im.compression, raw, dst, want)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", compressionName(im.compression), err)
	}
	// A tile that the image's last row cuts may stop there: GDAL
	// accepts that, and the rows below are outside the image anyway.
	need := want
	if im.tiled {
		need = rowBytes * min(rows, im.height-by*im.blockH)
	}
	if len(data) < need {
		return nil, fmt.Errorf("%s data decodes to %d bytes, want %d",
			compressionName(im.compression), len(data), need)
	}
	if len(data) < want {
		data = append(data, make([]byte, want-len(data))...)
	}
	data = data[:want]
	b := newBlock(im.blockW, rows)
	isFloat32 := im.format == sampleFloat && im.bytes == 4
	// With SIMD kernels, rows are converted and tested one at a time, the
	// validity while the row is in cache. Without, the block is converted
	// whole and tested after, as before the kernels: in scalar builds that
	// measured 7-10% faster on float32 end to end, though not in
	// isolation (benchmarks/cog), so it is kept for them.
	rowWise := kern.Backend() != "scalar"
	if isFloat32 && spb == 1 && im.predictor == predictorFloat {
		// The common float COG: the predictor's byte planes go straight
		// into the samples, without being put back into data first.
		v := newValidator(float32Test(nd), n)
		for r := range rows {
			kern.PlanesRow(b.vals[r*im.blockW:(r+1)*im.blockW], data[r*rowBytes:(r+1)*rowBytes])
			if rowWise {
				v.upto(b.vals, (r+1)*im.blockW)
			}
		}
		b.valid = v.finish(b.vals)
		return b, nil
	}
	if rowWise && isFloat32 && spb == 1 && im.predictor == predictorNone && c.order == binary.LittleEndian {
		// The samples are the data's bytes: a copy per row.
		v := newValidator(float32Test(nd), n)
		for r := range rows {
			kern.CopyRow(b.vals[r*im.blockW:(r+1)*im.blockW], data[r*rowBytes:(r+1)*rowBytes])
			v.upto(b.vals, (r+1)*im.blockW)
		}
		b.valid = v.finish(b.vals)
		return b, nil
	}
	first := 0
	if im.planar == planarChunky {
		first = band
	}
	if im.format != sampleFloat && (im.bits == 8 || im.bits == 16) {
		// 8- and 16-bit integers are exact in float32, and their NoData
		// is an integer, so neither needs convert's float64 detour.
		signed := im.format == sampleInt
		if spb == 1 && (im.bits == 8 || c.order == binary.LittleEndian) {
			// One band in native order: the horizontal predictor is
			// undone as the samples are written, in one pass per row.
			pred := im.predictor == predictorHorizontal
			v := newValidator(intTest(nd), n)
			for r := range rows {
				vals, row := b.vals[r*im.blockW:(r+1)*im.blockW], data[r*rowBytes:(r+1)*rowBytes]
				if im.bits == 16 && rowWise {
					kern.Uint16Row(vals, row, signed, pred)
				} else {
					intRow(vals, row, im.bits, signed, pred)
				}
				if rowWise {
					v.upto(b.vals, (r+1)*im.blockW)
				}
			}
			b.valid = v.finish(b.vals)
			return b, nil
		}
		toLittleEndian(data, im, c.order, rowBytes)
		convertInts(b.vals, data, im.bits, signed, spb, first)
		b.valid = exactValidity(b.vals, nd)
		return b, nil
	}
	toLittleEndian(data, im, c.order, rowBytes)
	if im.bits == 1 {
		expandBits(b.vals, data, im.blockW, rowBytes, spb, first)
		if nd.set && anyEqual(b.vals, float32(nd.cmp)) {
			b.valid = make([]uint64, raster.MaskWords(n))
			for i, v := range b.vals {
				if v != float32(nd.cmp) {
					b.valid[i>>6] |= 1 << uint(i&63)
				}
			}
		}
		return b, nil
	}
	if isFloat32 {
		// float32 needs no conversion, and its NoData comparison can be
		// made on the copied values, so a masked block costs one extra
		// pass over them rather than a second conversion.
		copyFloat32(b.vals, data, spb, first)
		b.valid = float32Validity(b.vals, nd)
		return b, nil
	}
	anyInvalid := convert(b.vals, data, im.format, im.bits, spb, first, nd, nil)
	if anyInvalid {
		b.valid = make([]uint64, raster.MaskWords(n))
		convert(b.vals, data, im.format, im.bits, spb, first, nd, b.valid)
	}
	return b, nil
}

// expandBits writes the 1-bit samples of data, every stride-th from the
// first-th in each row of rowBytes bytes, most significant bit first,
// into vals as 0 or 255: GDAL promotes a 1-bit mask to those values.
func expandBits(vals []float32, data []byte, w, rowBytes, stride, first int) {
	for i := range vals {
		r, c := i/w, i%w
		bit := c*stride + first
		vals[i] = 0 // the buffer is reused, so every cell is written
		if data[r*rowBytes+bit>>3]&(0x80>>uint(bit&7)) != 0 {
			vals[i] = 255
		}
	}
}

func anyEqual(vals []float32, v float32) bool {
	for _, x := range vals {
		if x == v {
			return true
		}
	}
	return false
}

// decompress returns data decoded, at most want bytes of it, so a
// corrupt stream cannot grow without bound; the result may be shorter,
// for the caller's length check. Deflate and LZW decode into dst, which
// holds want bytes; ZSTD decodes into it if the frame fits; PackBits
// allocates, and None returns data itself.
func decompress(scheme uint64, data, dst []byte, want int) ([]byte, error) {
	switch scheme {
	case compressionNone:
		return data, nil
	case compressionPackBits:
		return unpackBits(data, want)
	case compressionDeflate, compressionDeflate2:
		// A zlib stream: the header is checked here and the Deflate
		// data inflated raw, whole, into dst (see inflate). Decoding
		// stops at the block's size, so it never reaches the Adler-32
		// trailer, and a zlib reader would only compute a checksum it
		// never compares.
		if err := zlibHeader(data); err != nil {
			return nil, err
		}
		return inflate(dst, data[2:])
	case compressionLZW:
		if len(data) >= 2 && data[0] == 0 && data[1]&1 != 0 {
			// Old-style LZW, from libtiff before 5.0: codes least
			// significant bit first, widening a code later than the
			// TIFF 6 variant does. libtiff recognises it by its first
			// code, a Clear code (256) written LSB-first, and so do we.
			// Rare enough not to pool.
			r := lzw.NewReader(bytes.NewReader(data), lzw.LSB, 8)
			out, err := readInto(r, dst)
			_ = r.Close()
			return out, err
		}
		d := unlzws.Get().(*unlzw)
		defer unlzws.Put(d)
		d.src.Reset(data)
		d.lr.Reset(&d.src, lzw.MSB, 8)
		return readInto(&d.lr, dst)
	case compressionZSTD:
		// A frame declares its size up front; refuse one bigger than the
		// block before decoding it, as the stream formats stop at want.
		var h zstd.Header
		if err := h.Decode(data); err != nil {
			return nil, err
		}
		if h.HasFCS && h.FrameContentSize > uint64(want) { // #nosec G115 -- want is positive
			return nil, fmt.Errorf("a %d-byte frame for a %d-byte block", h.FrameContentSize, want)
		}
		d, err := zstdDecoder()
		if err != nil {
			return nil, err
		}
		return d.DecodeAll(data, dst[:0])
	}
	return nil, errors.New("unsupported")
}

// zlibHeader checks the two-byte header of a zlib stream (RFC 1950
// §2.2) as a zlib reader does: Deflate, a window of at most 32 KiB, a
// valid check, and no preset dictionary, which TIFF never uses.
func zlibHeader(data []byte) error {
	if len(data) < 2 {
		return io.ErrUnexpectedEOF
	}
	if data[0]&0x0f != 8 || data[0]>>4 > 7 || binary.BigEndian.Uint16(data)%31 != 0 {
		return zlib.ErrHeader
	}
	if data[1]&0x20 != 0 {
		return zlib.ErrDictionary
	}
	return nil
}

// readInto reads up to len(dst) bytes from r into dst. A stream that
// ends early is left for the caller's length check; one that errors is
// an error.
func readInto(r io.Reader, dst []byte) ([]byte, error) {
	n, err := io.ReadFull(r, dst)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return dst[:n], nil
}

// The LZW decoder keeps its tables between blocks, so it is pooled
// rather than made per block, with the reader it reads from.
type unlzw struct {
	src bytes.Reader
	lr  lzw.Reader
}

var (
	unlzws = sync.Pool{New: func() any {
		d := new(unlzw)
		d.lr.SetAldusCompatible(true) // libtiff's LZW; it survives Reset
		return d
	}}
	// scratch holds the byte buffers a decode needs only until its
	// samples are copied out: the block as read, and as decompressed.
	scratch sync.Pool // of *[]byte
)

// getScratch returns a pooled buffer of length n.
func getScratch(n int) *[]byte {
	if p, ok := scratch.Get().(*[]byte); ok && cap(*p) >= n {
		*p = (*p)[:n]
		return p
	}
	b := make([]byte, n)
	return &b
}

func putScratch(p *[]byte) { scratch.Put(p) }

// zstdDecoder is shared by every source: DecodeAll is safe for concurrent
// use, and one decoder's buffers serve them all.
var zstdDecoder = sync.OnceValues(func() (*zstd.Decoder, error) {
	return zstd.NewReader(nil, zstd.WithDecoderConcurrency(0), zstd.WithDecoderMaxMemory(2*maxBlockBytes))
})

// unpackBits decodes PackBits: a header byte n, then n+1 literal bytes
// for n in [0, 127], one byte repeated 1-n times for n in [-127, -1], and
// nothing for -128. It stops at want bytes.
func unpackBits(src []byte, want int) ([]byte, error) {
	out := make([]byte, 0, want)
	for i := 0; i < len(src) && len(out) < want; {
		n := int(int8(src[i])) // #nosec G115 -- PackBits headers are signed bytes
		i++
		switch {
		case n >= 0:
			if i+n+1 > len(src) {
				return nil, errors.New("PackBits literal run past the end of the data")
			}
			out = append(out, src[i:i+n+1]...)
			i += n + 1
		case n != -128:
			if i >= len(src) {
				return nil, errors.New("PackBits repeat run past the end of the data")
			}
			for range 1 - n {
				out = append(out, src[i])
			}
			i++
		}
	}
	return out, nil
}

// toLittleEndian undoes the predictor and leaves every sample in data
// little-endian, whatever the file's byte order.
func toLittleEndian(data []byte, im *image, order binary.ByteOrder, rowBytes int) {
	size, stride := im.bytes, im.blockSamples()
	if im.bits == 1 {
		return // bytes of bits, no predictor
	}
	if im.predictor == predictorFloat {
		tmp := make([]byte, rowBytes)
		for r := 0; r < len(data); r += rowBytes {
			floatPredictorRow(data[r:r+rowBytes], tmp, size, stride)
		}
		return
	}
	if order == binary.BigEndian && size > 1 {
		for i := 0; i+size <= len(data); i += size {
			s := data[i : i+size]
			for a, z := 0, size-1; a < z; a, z = a+1, z-1 {
				s[a], s[z] = s[z], s[a]
			}
		}
	}
	if im.predictor == predictorHorizontal {
		for r := 0; r < len(data); r += rowBytes {
			horizontalRow(data[r:r+rowBytes], size, stride)
		}
	}
}

// horizontalRow undoes horizontal differencing on one row of
// little-endian samples: each sample is added, wrapping, to the one
// stride samples before it.
func horizontalRow(row []byte, size, stride int) {
	le := binary.LittleEndian
	step := size * stride
	switch size {
	case 1:
		for i := step; i < len(row); i++ {
			row[i] += row[i-step]
		}
	case 2:
		for i := step; i+2 <= len(row); i += 2 {
			le.PutUint16(row[i:], le.Uint16(row[i:])+le.Uint16(row[i-step:]))
		}
	case 4:
		for i := step; i+4 <= len(row); i += 4 {
			le.PutUint32(row[i:], le.Uint32(row[i:])+le.Uint32(row[i-step:]))
		}
	case 8:
		for i := step; i+8 <= len(row); i += 8 {
			le.PutUint64(row[i:], le.Uint64(row[i:])+le.Uint64(row[i-step:]))
		}
	}
}

// floatPredictorRow undoes libtiff's floating-point predictor on one row:
// the bytes are differenced stride bytes apart, and the samples' bytes
// are stored as planes, most significant first. It leaves the samples
// little-endian.
//
// The usual float COG, one band of float32, never comes here: decodeBlock
// hands its rows to kern.PlanesRow. For the rest, the differencing is
// undone into tmp, by kern.SumBytes when stride is 1, and four-byte
// samples are reassembled from their planes a whole sample at a time.
func floatPredictorRow(row, tmp []byte, size, stride int) {
	tmp = tmp[:len(row)]
	if stride == 1 {
		copy(tmp, row)
		kern.SumBytes(tmp)
	} else {
		copy(tmp, row[:min(stride, len(row))])
		for i := stride; i < len(row); i++ {
			tmp[i] = row[i] + tmp[i-stride]
		}
	}
	n := len(row) / size
	if size == 4 {
		p0, p1, p2, p3 := tmp[:n], tmp[n:2*n], tmp[2*n:3*n], tmp[3*n:4*n]
		out := row[:4*n]
		for i := range p3 {
			binary.LittleEndian.PutUint32(out[4*i:],
				uint32(p3[i])|uint32(p2[i])<<8|uint32(p1[i])<<16|uint32(p0[i])<<24)
		}
		return
	}
	for i := range n {
		for k := range size {
			row[i*size+k] = tmp[(size-1-k)*n+i]
		}
	}
}

// overflow32 is the smallest float64 that rounds to +Inf as a float32:
// MaxFloat32 plus half its ulp, a tie that rounds to even, which is Inf.
const overflow32 = math.MaxFloat32 + 0x1p103

// toFloat32 rounds v to float32 as IEEE 754 does. Go leaves a conversion
// of an out-of-range value implementation-dependent, so the overflow is
// spelled out rather than left to the platform.
func toFloat32(v float64) float32 {
	switch {
	case v >= overflow32:
		return float32(math.Inf(1))
	case v <= -overflow32:
		return float32(math.Inf(-1))
	}
	return float32(v)
}

// copyFloat32 writes every stride-th little-endian float32 sample of
// data, from the first-th, into vals, bit for bit, as GDAL copies a
// Float32 band into a Float32 buffer.
func copyFloat32(vals []float32, data []byte, stride, first int) {
	le := binary.LittleEndian
	if stride == 1 {
		d := data[:4*len(vals)]
		for i := range vals {
			vals[i] = math.Float32frombits(le.Uint32(d[4*i:]))
		}
		return
	}
	step, p := 4*stride, 4*first
	for i := range vals {
		vals[i] = math.Float32frombits(le.Uint32(data[p:]))
		p += step
	}
}

// float32Validity is convert's NoData test for float32 samples, made on
// the values copyFloat32 wrote, in one pass: it returns the validity
// bits of vals, or nil if every cell is valid. It gives convert's
// answers, ARE_REAL_EQUAL included: the float32 values within its
// tolerance of a finite, non-zero NoData are a run of adjacent bit
// patterns, found once here, so each cell's test is a range check.
func float32Validity(vals []float32, nd noData) []uint64 {
	v := newValidator(float32Test(nd), len(vals))
	return v.finish(vals)
}

// realEqualRun32 returns the bit patterns of the float32 values that
// realEqual32 finds equal to b, as the first and the count past it, for
// a finite, non-zero b; ok is false for any other b, which only values
// with its own bits (either sign, for zero) equal. The tolerance is a
// few ulps, so the walk out from b is short.
func realEqualRun32(b float32) (lo uint32, span uint64, ok bool) {
	if b == 0 || math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) {
		return 0, 0, false
	}
	edge := func(toward float32) float32 {
		e := b
		for {
			n := math.Nextafter32(e, toward)
			if n == e || !realEqual32(n, b) {
				return e
			}
			e = n
		}
	}
	x, y := math.Float32bits(edge(float32(math.Inf(-1)))), math.Float32bits(edge(float32(math.Inf(1))))
	if x > y { // negative b: bits grow as values fall
		x, y = y, x
	}
	return x, uint64(y - x), true
}

// intRow writes one row of single-band little-endian integer samples of
// the given width and signedness to vals, undoing horizontal differencing
// first if pred: each sample is then the wrapping sum of those before it.
// It may use row as scratch.
func intRow(vals []float32, row []byte, bits int, signed, pred bool) {
	le := binary.LittleEndian
	switch {
	case bits == 8:
		kern.Uint8Row(vals, row, signed, pred)
	case pred:
		row = row[:2*len(vals)]
		var acc uint16
		if signed {
			for i := range vals {
				acc += le.Uint16(row[2*i:])
				vals[i] = float32(int16(acc)) // #nosec G115 -- as above
			}
			return
		}
		for i := range vals {
			acc += le.Uint16(row[2*i:])
			vals[i] = float32(acc)
		}
	default:
		row = row[:2*len(vals)]
		if signed {
			for i := range vals {
				vals[i] = float32(int16(le.Uint16(row[2*i:]))) // #nosec G115 -- as above
			}
			return
		}
		for i := range vals {
			vals[i] = float32(le.Uint16(row[2*i:]))
		}
	}
}

// convertInts writes every stride-th little-endian integer sample of
// data, from the first-th, into vals: the general case of intRow, for
// several bands and any byte order, after toLittleEndian.
func convertInts(vals []float32, data []byte, bits int, signed bool, stride, first int) {
	le := binary.LittleEndian
	size := bits / 8
	step, p := size*stride, first*size
	for i := range vals {
		s := data[p:]
		p += step
		switch {
		case size == 1 && signed:
			vals[i] = float32(int8(s[0])) // #nosec G115 -- reinterpreting the bits is the point
		case size == 1:
			vals[i] = float32(s[0])
		case signed:
			vals[i] = float32(int16(le.Uint16(s))) // #nosec G115 -- as above
		default:
			vals[i] = float32(le.Uint16(s))
		}
	}
}

// exactValidity is the NoData test for integer samples of at most 16
// bits, made on their float32 values, which hold them exactly: NoData is
// an integer in the type's range (prepareNoData), so it is exact too, and
// a cell is NoData if and only if its value equals it. It returns nil if
// every cell is valid.
func exactValidity(vals []float32, nd noData) []uint64 {
	v := newValidator(intTest(nd), len(vals))
	return v.finish(vals)
}

// convert writes every stride-th little-endian sample of data, from the
// first-th, into vals as float32. With valid nil it reports whether any
// sample matches nd, and writes no bits; otherwise it sets each cell's
// bit in valid. Samples convert as float32(v) does under IEEE 754, which
// is how GDAL converts them: rounded to nearest, and ±Inf past float32's
// range.
func convert(vals []float32, data []byte, format, bits, stride, first int, nd noData, valid []uint64) bool {
	le := binary.LittleEndian
	size := bits / 8
	step := size * stride
	p := first * size
	found := false
	var word uint64
	for i := range vals {
		s := data[p:]
		p += step
		var v float64 // the sample, exactly
		switch {
		case format == sampleFloat && size == 4:
			v = float64(math.Float32frombits(le.Uint32(s)))
		case format == sampleFloat:
			v = math.Float64frombits(le.Uint64(s))
		case format == sampleInt && size == 1:
			v = float64(int8(s[0])) // #nosec G115 -- reinterpreting the bits is the point
		case format == sampleInt && size == 2:
			v = float64(int16(le.Uint16(s))) // #nosec G115 -- as above
		case format == sampleInt:
			v = float64(int32(le.Uint32(s))) // #nosec G115 -- as above
		case size == 1:
			v = float64(s[0])
		case size == 2:
			v = float64(le.Uint16(s))
		default:
			v = float64(le.Uint32(s))
		}
		vals[i] = toFloat32(v)
		ok := !nd.matches(v, format, bits)
		if !ok {
			found = true
		}
		if valid != nil {
			if ok {
				word |= 1 << uint(i&63)
			}
			if i&63 == 63 || i == len(vals)-1 {
				valid[i>>6] = word
				word = 0
			}
		}
	}
	return found
}
