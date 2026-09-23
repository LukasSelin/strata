package cog

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"

	"github.com/klauspost/compress/zstd"
	"golang.org/x/image/tiff/lzw"

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
// band. nd is the prepared NoData value. read returns the block's stored
// bytes, which may be shared with other sources, so they are not
// modified.
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
	raw, err := read(off, count)
	if err != nil {
		return nil, fmt.Errorf("reading %d bytes at offset %d: %w", count, off, err)
	}
	spb := im.blockSamples()
	rowBytes := im.blockW * spb * im.bytes
	want := rowBytes * rows
	if im.compression == compressionNone && (im.predictor != predictorNone || c.order == binary.BigEndian && im.bytes > 1) {
		raw = slices.Clone(raw) // decompress returns raw itself, and toLittleEndian will write to it
	}
	data, err := decompress(im.compression, raw, want)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", compressionName(im.compression), err)
	}
	if len(data) < want {
		return nil, fmt.Errorf("%s data decodes to %d bytes, want %d",
			compressionName(im.compression), len(data), want)
	}
	data = data[:want]
	toLittleEndian(data, im, c.order, rowBytes)

	b.vals = make([]float32, n)
	first := 0
	if im.planar == planarChunky {
		first = band
	}
	anyInvalid := convert(b.vals, data, im.format, im.bytes, spb, first, nd, nil)
	if anyInvalid {
		b.valid = make([]uint64, raster.MaskWords(n))
		convert(b.vals, data, im.format, im.bytes, spb, first, nd, b.valid)
	}
	return b, nil
}

// decompress returns data decoded, reading at most want bytes of output
// for the stream formats, so a corrupt stream cannot grow without bound.
func decompress(scheme uint64, data []byte, want int) ([]byte, error) {
	switch scheme {
	case compressionNone:
		return data, nil
	case compressionPackBits:
		return unpackBits(data, want)
	case compressionDeflate, compressionDeflate2:
		r, err := zlib.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		return readUpTo(r, want)
	case compressionLZW:
		r := lzw.NewReader(bytes.NewReader(data), lzw.MSB, 8)
		out, err := readUpTo(r, want)
		_ = r.Close()
		return out, err
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
		var dst []byte
		if h.HasFCS {
			dst = make([]byte, 0, h.FrameContentSize) // at most want, checked above
		}
		return d.DecodeAll(data, dst)
	}
	return nil, errors.New("unsupported")
}

// readUpTo reads up to want bytes from r. A stream that ends early is
// left for the caller's length check; one that errors is an error.
func readUpTo(r io.Reader, want int) ([]byte, error) {
	var buf bytes.Buffer
	_, err := io.Copy(&buf, io.LimitReader(r, int64(want)))
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return nil, err
	}
	return buf.Bytes(), nil
}

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
func floatPredictorRow(row, tmp []byte, size, stride int) {
	for i := stride; i < len(row); i++ {
		row[i] += row[i-stride]
	}
	copy(tmp, row)
	n := len(row) / size
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
