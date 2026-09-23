package cog

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/klauspost/compress/flate"
	"github.com/klauspost/compress/lzw"
	"github.com/klauspost/compress/zlib"
	"github.com/klauspost/compress/zstd"

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
type block struct {
	vals  []float32
	valid []uint64
	w     int
	rows  int
}

// size is the memory a block holds, for the cache's accounting.
func (b *block) size() int64 {
	return int64(4*len(b.vals) + 8*len(b.valid) + 64)
}

// noData is a NoData value prepared for comparison in the samples'
// native type: GDAL_NODATA is a decimal string, and a value that the
// type cannot hold exactly matches no cell.
type noData struct {
	set bool
	nan bool    // NoData is NaN: any NaN matches
	cmp float64 // the native value, as a float64; unmatched if !set
}

// prepareNoData converts GDAL's NoData value to the sample type, as
// GDAL's mask band does: for float32 samples it is rounded to float32,
// and an integer NoData is only ever equal to an integer sample.
func prepareNoData(v float64, has bool, format, size int) noData {
	if !has {
		return noData{}
	}
	if math.IsNaN(v) {
		return noData{set: format == sampleFloat, nan: true}
	}
	if format == sampleFloat && size == 4 {
		if !math.IsInf(v, 0) && math.Abs(v) > math.MaxFloat32 {
			return noData{} // not representable: nothing matches
		}
		return noData{set: true, cmp: float64(float32(v))}
	}
	return noData{set: true, cmp: v}
}

// decodeBlock reads and decodes block idx of im, at block row by, for
// band. nd is the prepared NoData value. read, if not nil, returns the
// block's stored bytes, which other sources share, so they are not
// modified; if nil, the block is read from the file into a scratch
// buffer.
func (c *container) decodeBlock(im *image, idx, by, band int, nd noData, read func(off, n uint64) ([]byte, error)) (*block, error) {
	rows := im.blockRows(by)
	n := im.blockW * rows
	b := &block{w: im.blockW, rows: rows}
	off, count := im.offsets[idx], im.byteCounts[idx]
	if off == 0 && count == 0 {
		// A sparse block, which GDAL writes for all-NoData tiles and
		// reads as NoData, or as 0 without one.
		b.vals = make([]float32, n)
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
	spb := im.blockSamples()
	rowBytes := im.blockW * spb * im.bytes
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
	if len(data) < want {
		return nil, fmt.Errorf("%s data decodes to %d bytes, want %d",
			compressionName(im.compression), len(data), want)
	}
	data = data[:want]
	b.vals = make([]float32, n)
	isFloat32 := im.format == sampleFloat && im.bytes == 4
	if isFloat32 && spb == 1 && im.predictor == predictorFloat {
		// The common float COG: the predictor's byte planes go straight
		// into the samples, without being put back into data first.
		for r := range rows {
			floatPredictorRow32(b.vals[r*im.blockW:(r+1)*im.blockW], data[r*rowBytes:(r+1)*rowBytes])
		}
		b.valid = float32Validity(b.vals, nd)
		return b, nil
	}
	toLittleEndian(data, im, c.order, rowBytes)
	first := 0
	if im.planar == planarChunky {
		first = band
	}
	if isFloat32 {
		// float32 needs no conversion, and its NoData comparison can be
		// made on the copied values, so a masked block costs one extra
		// pass over them rather than a second conversion.
		copyFloat32(b.vals, data, spb, first)
		b.valid = float32Validity(b.vals, nd)
		return b, nil
	}
	anyInvalid := convert(b.vals, data, im.format, im.bytes, spb, first, nd, nil)
	if anyInvalid {
		b.valid = make([]uint64, raster.MaskWords(n))
		convert(b.vals, data, im.format, im.bytes, spb, first, nd, b.valid)
	}
	return b, nil
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
		// data inflated raw. Decoding stops at the block's size, so it
		// never reaches the Adler-32 trailer, and a zlib reader would
		// only compute a checksum it never compares.
		if err := zlibHeader(data); err != nil {
			return nil, err
		}
		d := inflaters.Get().(*inflater)
		defer inflaters.Put(d)
		d.src.Reset(data[2:])
		if d.fr == nil {
			d.fr = flate.NewReader(&d.src)
		} else if err := d.fr.(flate.Resetter).Reset(&d.src, nil); err != nil {
			return nil, err
		}
		return readInto(d.fr, dst)
	case compressionLZW:
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

// The decoders keep tables and windows between blocks, so they are
// pooled rather than made per block, with the reader they read from.
type inflater struct {
	src bytes.Reader
	fr  io.ReadCloser // nil until first used
}

type unlzw struct {
	src bytes.Reader
	lr  lzw.Reader
}

var (
	inflaters = sync.Pool{New: func() any { return new(inflater) }}
	unlzws    = sync.Pool{New: func() any {
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
// It is the hottest loop in reading a float COG, so the differencing is
// undone straight into tmp, one byte a step with the running sum in a
// register when stride is 1, and four-byte samples are reassembled from
// their planes a whole sample at a time.
func floatPredictorRow(row, tmp []byte, size, stride int) {
	tmp = tmp[:len(row)]
	if stride == 1 {
		var acc byte
		for i, b := range row {
			acc += b
			tmp[i] = acc
		}
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

// floatPredictorRow32 is floatPredictorRow for one row of single-band
// float32 samples, writing them to vals rather than back into row, which
// it uses as scratch.
func floatPredictorRow32(vals []float32, row []byte) {
	var acc byte
	for i, b := range row {
		acc += b
		row[i] = acc
	}
	n := len(vals)
	p0, p1, p2, p3 := row[:n], row[n:2*n], row[2*n:3*n], row[3*n:4*n]
	for i := range vals {
		vals[i] = math.Float32frombits(uint32(p3[i]) | uint32(p2[i])<<8 | uint32(p1[i])<<16 | uint32(p0[i])<<24)
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
// answers: nd.cmp holds a float32 value exactly, so comparing in float32
// is comparing in float64.
func float32Validity(vals []float32, nd noData) []uint64 {
	if !nd.set {
		return nil
	}
	valid := make([]uint64, raster.MaskWords(len(vals)))
	// The test is on the bits, as integers, so that it has no branch:
	// a cell is valid where (bits ^ want) & care is not zero. Only ±0
	// compare equal with different bits, so NoData 0 ignores the sign;
	// and NaN, which never equals NoData, has bits that never do.
	want, care := math.Float32bits(float32(nd.cmp)), ^uint32(0)
	if nd.cmp == 0 {
		want, care = 0, 0x7fffffff
	}
	all := true
	for k := range valid {
		chunk := vals[k*64 : min(k*64+64, len(vals))]
		var word uint64
		if nd.nan {
			word = notNaNWord(chunk)
		} else {
			word = notEqualWord(chunk, want, care)
		}
		valid[k] = word
		all = all && word == ^uint64(0)>>(64-uint(len(chunk)))
	}
	if all {
		return nil
	}
	return valid
}

// notEqualWord returns bit j set where (bits of chunk[j] ^ want) & care
// is not zero, for at most 64 cells.
func notEqualWord(chunk []float32, want, care uint32) uint64 {
	var word uint64
	for j, v := range chunk {
		d := (math.Float32bits(v) ^ want) & care
		word |= uint64((d|-d)>>31) << (uint(j) & 63)
	}
	return word
}

// notNaNWord returns bit j set where chunk[j] is not NaN: its magnitude
// bits are at most +Inf's. For at most 64 cells.
func notNaNWord(chunk []float32) uint64 {
	var word uint64
	for j, v := range chunk {
		m := math.Float32bits(v) & 0x7fffffff
		word |= uint64((0x7f800000-m)>>31^1) << (uint(j) & 63)
	}
	return word
}

// convert writes every stride-th little-endian sample of data, from the
// first-th, into vals as float32. With valid nil it reports whether any
// sample matches nd, and writes no bits; otherwise it sets each cell's
// bit in valid. Samples convert as float32(v) does under IEEE 754, which
// is how GDAL converts them: rounded to nearest, and ±Inf past float32's
// range.
func convert(vals []float32, data []byte, format, size, stride, first int, nd noData, valid []uint64) bool {
	le := binary.LittleEndian
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
		ok := true
		switch {
		case !nd.set:
		case nd.nan:
			ok = v == v
		default:
			ok = v != nd.cmp
		}
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
