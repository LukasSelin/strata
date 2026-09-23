package cog

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"slices"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// A minimal TIFF writer, for tests only: it builds the files the reader
// is tested on, across every layout, byte order, compression, predictor
// and sample type the reader claims. It follows the TIFF 6.0 and BigTIFF
// specifications and libtiff's predictors independently of the reader's
// code, so a shared misreading shows up as a disagreement with GDAL in
// acceptance/cogcheck.sh rather than here.

// imageSpec describes one image: its layout and its cells, one slice of
// native values per band, row-major.
type imageSpec struct {
	w, h, bands    int
	format, size   int // sampleUint/Int/Float; bytes per sample
	tiled          bool
	blockW, blockH int // tile size, or blockH rows per strip
	planar         int
	compression    uint64
	predictor      int
	vals           [][]float64
	sparse         func(block int) bool // blocks written as offset 0, count 0
	subfile        uint64
	extra          []tagValue // georeferencing, NoData, anything else
}

// tagValue is one entry to write: uints for integer types, floats for
// DOUBLE, str for ASCII.
type tagValue struct {
	tag    uint16
	typ    uint16
	uints  []uint64
	floats []float64
	str    string
}

// order is a byte order that can also append.
type order interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

// fileSpec is a whole file.
type fileSpec struct {
	order  order
	big    bool
	images []imageSpec
}

// blocks encodes an image's blocks, in TIFF order: row-major within a
// plane, planes in band order.
func (sp imageSpec) blocks(order binary.ByteOrder) [][]byte {
	bw, bh := sp.blockW, sp.blockH
	if !sp.tiled {
		bw = sp.w
	}
	across, down := (sp.w+bw-1)/bw, (sp.h+bh-1)/bh
	planes, spb := 1, sp.bands
	if sp.planar == planarSeparate {
		planes, spb = sp.bands, 1
	}
	var out [][]byte
	for p := range planes {
		for by := range down {
			for bx := range across {
				rows := bh
				if !sp.tiled {
					rows = min(bh, sp.h-by*bh)
				}
				idx := len(out)
				if sp.sparse != nil && sp.sparse(idx) {
					out = append(out, nil)
					continue
				}
				rowBytes := bw * spb * sp.size
				raw := make([]byte, rowBytes*rows)
				for r := range rows {
					row := raw[r*rowBytes : (r+1)*rowBytes]
					for c := range bw {
						for s := range spb {
							band := s
							if planes > 1 {
								band = p
							}
							x, y := bx*bw+c, by*bh+r
							v := 0.0
							if x < sp.w && y < sp.h {
								v = sp.vals[band][y*sp.w+x]
							}
							putSample(row[(c*spb+s)*sp.size:], v, sp.format, sp.size)
						}
					}
					encodeRow(row, sp, spb, order)
				}
				out = append(out, compress(raw, sp.compression))
			}
		}
	}
	return out
}

// putSample writes v little-endian as the sample type.
func putSample(b []byte, v float64, format, size int) {
	le := binary.LittleEndian
	switch {
	case format == sampleFloat && size == 4:
		le.PutUint32(b, math.Float32bits(float32(v)))
	case format == sampleFloat && size == 8:
		le.PutUint64(b, math.Float64bits(v))
	case size == 1:
		b[0] = byte(int64(v))
	case size == 2:
		le.PutUint16(b, uint16(int64(v)))
	default:
		le.PutUint32(b, uint32(int64(v)))
	}
}

// encodeRow applies the predictor to one row of little-endian samples
// and leaves it in the file's byte order, as libtiff writes it.
func encodeRow(row []byte, sp imageSpec, stride int, order binary.ByteOrder) {
	size := sp.size
	switch sp.predictor {
	case predictorHorizontal:
		// Difference from the end, so each sample subtracts the
		// original of its predecessor.
		le := binary.LittleEndian
		step := size * stride
		for i := len(row) - size; i >= step; i -= size {
			switch size {
			case 1:
				row[i] -= row[i-step]
			case 2:
				le.PutUint16(row[i:], le.Uint16(row[i:])-le.Uint16(row[i-step:]))
			case 4:
				le.PutUint32(row[i:], le.Uint32(row[i:])-le.Uint32(row[i-step:]))
			case 8:
				le.PutUint64(row[i:], le.Uint64(row[i:])-le.Uint64(row[i-step:]))
			}
		}
	case predictorFloat:
		// Byte planes, most significant first, then bytes differenced
		// stride apart. Byte order does not enter into it.
		n := len(row) / size
		planes := make([]byte, len(row))
		for i := range n {
			for k := range size {
				planes[(size-1-k)*n+i] = row[i*size+k]
			}
		}
		for i := len(planes) - 1; i >= stride; i-- {
			planes[i] -= planes[i-stride]
		}
		copy(row, planes)
		return
	}
	if order == binary.BigEndian {
		for i := 0; i < len(row); i += size {
			slices.Reverse(row[i : i+size])
		}
	}
}

func compress(raw []byte, scheme uint64) []byte {
	switch scheme {
	case compressionDeflate, compressionDeflate2:
		var b bytes.Buffer
		w := zlib.NewWriter(&b)
		_, _ = w.Write(raw)
		_ = w.Close()
		return b.Bytes()
	case compressionPackBits:
		return packBits(raw)
	case compressionZSTD:
		return zstdEncoder().EncodeAll(raw, nil)
	}
	return raw
}

// zstdEncoder is shared: EncodeAll is safe for concurrent use, and
// building an encoder per block would dominate the tests.
var zstdEncoder = sync.OnceValue(func() *zstd.Encoder {
	e, err := zstd.NewWriter(nil, zstd.WithEncoderConcurrency(1))
	if err != nil {
		panic(err)
	}
	return e
})

// packBits encodes runs of three or more equal bytes as repeats and
// everything else as literals.
func packBits(src []byte) []byte {
	var out []byte
	for i := 0; i < len(src); {
		run := 1
		for i+run < len(src) && run < 128 && src[i+run] == src[i] {
			run++
		}
		if run >= 3 {
			out = append(out, byte(int8(1-run)), src[i])
			i += run
			continue
		}
		j := i
		for j < len(src) && j-i < 128 {
			if j+2 < len(src) && src[j] == src[j+1] && src[j] == src[j+2] {
				break
			}
			j++
		}
		out = append(out, byte(j-i-1))
		out = append(out, src[i:j]...)
		i = j
	}
	return out
}

// write lays the file out: header, every image's blocks, then the IFDs,
// each followed by the values that do not fit in its entries.
func (fs fileSpec) write() []byte {
	o := fs.order
	var buf []byte
	if o == binary.BigEndian {
		buf = append(buf, 'M', 'M')
	} else {
		buf = append(buf, 'I', 'I')
	}
	if fs.big {
		buf = o.AppendUint16(buf, 43)
		buf = o.AppendUint16(buf, 8)
		buf = o.AppendUint16(buf, 0)
		buf = o.AppendUint64(buf, 0) // first IFD, patched below
	} else {
		buf = o.AppendUint16(buf, 42)
		buf = o.AppendUint32(buf, 0)
	}
	offType := uint16(typeLong)
	if fs.big {
		offType = typeLong8
	}

	tags := make([][]tagValue, len(fs.images))
	for i, sp := range fs.images {
		var offs, counts []uint64
		for _, b := range sp.blocks(o) {
			if b == nil {
				offs, counts = append(offs, 0), append(counts, 0)
				continue
			}
			offs, counts = append(offs, uint64(len(buf))), append(counts, uint64(len(b)))
			buf = append(buf, b...)
		}
		perSample := func(v int) []uint64 {
			s := make([]uint64, sp.bands)
			for k := range s {
				s[k] = uint64(v)
			}
			return s
		}
		t := []tagValue{
			{tag: tagImageWidth, typ: typeLong, uints: []uint64{uint64(sp.w)}},
			{tag: tagImageLength, typ: typeLong, uints: []uint64{uint64(sp.h)}},
			{tag: tagBitsPerSample, typ: typeShort, uints: perSample(8 * sp.size)},
			{tag: tagCompression, typ: typeShort, uints: []uint64{sp.compression}},
			{tag: 262, typ: typeShort, uints: []uint64{1}}, // BlackIsZero
			{tag: tagSamplesPerPixel, typ: typeShort, uints: []uint64{uint64(sp.bands)}},
			{tag: tagPlanarConfig, typ: typeShort, uints: []uint64{uint64(sp.planar)}},
			{tag: tagSampleFormat, typ: typeShort, uints: perSample(sp.format)},
		}
		if sp.subfile != 0 {
			t = append(t, tagValue{tag: tagNewSubfileType, typ: typeLong, uints: []uint64{sp.subfile}})
		}
		if sp.predictor != 0 {
			t = append(t, tagValue{tag: tagPredictor, typ: typeShort, uints: []uint64{uint64(sp.predictor)}})
		}
		if sp.tiled {
			t = append(t,
				tagValue{tag: tagTileWidth, typ: typeShort, uints: []uint64{uint64(sp.blockW)}},
				tagValue{tag: tagTileLength, typ: typeShort, uints: []uint64{uint64(sp.blockH)}},
				tagValue{tag: tagTileOffsets, typ: offType, uints: offs},
				tagValue{tag: tagTileByteCounts, typ: offType, uints: counts})
		} else {
			t = append(t,
				tagValue{tag: tagRowsPerStrip, typ: typeLong, uints: []uint64{uint64(sp.blockH)}},
				tagValue{tag: tagStripOffsets, typ: offType, uints: offs},
				tagValue{tag: tagStripByteCounts, typ: offType, uints: counts})
		}
		t = append(t, sp.extra...)
		slices.SortFunc(t, func(a, b tagValue) int { return int(a.tag) - int(b.tag) })
		tags[i] = t
	}

	patch := 4 // where the previous IFD's next pointer goes
	if fs.big {
		patch = 8
	}
	for _, t := range tags {
		if len(buf)%2 == 1 {
			buf = append(buf, 0) // IFDs start on a word boundary
		}
		start := uint64(len(buf))
		if fs.big {
			o.PutUint64(buf[patch:], start)
		} else {
			o.PutUint32(buf[patch:], uint32(start))
		}
		buf, patch = fs.writeIFD(buf, t)
	}
	return buf
}

// writeIFD appends an IFD with a zero next pointer, followed by the
// values that do not fit in its entries, and returns the buffer and
// where the next pointer is.
func (fs fileSpec) writeIFD(buf []byte, tags []tagValue) ([]byte, int) {
	o := fs.order
	countSize, entrySize, field := 2, 12, 4
	if fs.big {
		countSize, entrySize, field = 8, 20, 8
	}
	start := len(buf)
	valuesAt := start + countSize + len(tags)*entrySize + field
	var values []byte
	if fs.big {
		buf = o.AppendUint64(buf, uint64(len(tags)))
	} else {
		buf = o.AppendUint16(buf, uint16(len(tags)))
	}
	for _, t := range tags {
		v, count := encodeValue(o, t)
		buf = o.AppendUint16(buf, t.tag)
		buf = o.AppendUint16(buf, t.typ)
		if fs.big {
			buf = o.AppendUint64(buf, uint64(count))
		} else {
			buf = o.AppendUint32(buf, uint32(count))
		}
		if len(v) <= field {
			f := make([]byte, field)
			copy(f, v)
			buf = append(buf, f...)
			continue
		}
		off := uint64(valuesAt + len(values))
		if fs.big {
			buf = o.AppendUint64(buf, off)
		} else {
			buf = o.AppendUint32(buf, uint32(off))
		}
		values = append(values, v...)
		if len(values)%2 == 1 {
			values = append(values, 0)
		}
	}
	next := len(buf)
	buf = append(buf, make([]byte, field)...) // next IFD: none yet
	return append(buf, values...), next
}

func encodeValue(o order, t tagValue) ([]byte, int) {
	var b []byte
	switch t.typ {
	case typeASCII:
		b = append([]byte(t.str), 0)
		return b, len(b)
	case typeDouble:
		for _, f := range t.floats {
			b = o.AppendUint64(b, math.Float64bits(f))
		}
		return b, len(t.floats)
	}
	for _, u := range t.uints {
		switch t.typ {
		case typeByte:
			b = append(b, byte(u))
		case typeShort:
			b = o.AppendUint16(b, uint16(u))
		case typeLong:
			b = o.AppendUint32(b, uint32(u))
		case typeLong8:
			b = o.AppendUint64(b, u)
		}
	}
	return b, len(t.uints)
}
