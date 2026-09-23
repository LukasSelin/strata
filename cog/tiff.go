package cog

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// The TIFF container: a header, then a chain of image file directories
// (IFDs), each a sorted list of tagged entries whose values sit inline
// when they fit and at an offset when they do not. Classic TIFF uses
// 32-bit offsets and counts, BigTIFF 64-bit ones; both come in either
// byte order. Nothing here knows what a tag means; ifd.go does.

// Limits that keep a hostile or corrupt file from making Open allocate
// or loop without bound. Real files are nowhere near them: a GDAL COG
// has a handful of IFDs of a few dozen entries each.
const (
	maxIFDs    = 1024
	maxEntries = 4096
	// maxTagValues bounds one tag's value count. A 1 000 000² raster in
	// 256² tiles has 15.3M tiles; this admits 64M.
	maxTagValues = 1 << 26
	// readChunk is the most readFull allocates ahead of the bytes it
	// has actually read, so a count claimed by a short file costs at
	// most this much memory before the read fails.
	readChunk = 1 << 20
)

// TIFF field types.
const (
	typeByte      = 1
	typeASCII     = 2
	typeShort     = 3
	typeLong      = 4
	typeRational  = 5
	typeSByte     = 6
	typeUndefined = 7
	typeSShort    = 8
	typeSLong     = 9
	typeSRational = 10
	typeFloat     = 11
	typeDouble    = 12
	typeIFD       = 13
	typeLong8     = 16
	typeSLong8    = 17
	typeIFD8      = 18
)

// typeSize is the size in bytes of one value of each field type, or 0
// for a type this reader does not know.
func typeSize(t uint16) int {
	switch t {
	case typeByte, typeASCII, typeSByte, typeUndefined:
		return 1
	case typeShort, typeSShort:
		return 2
	case typeLong, typeSLong, typeFloat, typeIFD:
		return 4
	case typeRational, typeSRational, typeDouble, typeLong8, typeSLong8, typeIFD8:
		return 8
	}
	return 0
}

// container is an open TIFF file: its reader, byte order and flavour.
type container struct {
	r     io.ReaderAt
	order binary.ByteOrder
	big   bool
}

// entry is one IFD entry, its value not yet fetched. inline holds the
// value field's bytes when the value fits there; otherwise offset points
// at the value.
type entry struct {
	tag, typ uint16
	count    uint64
	inline   []byte
	offset   uint64
}

// rawIFD is one directory's entries by tag.
type rawIFD struct {
	offset  uint64
	entries map[uint16]entry
}

// readHeader reads the header and returns the container and the offset
// of the first IFD.
func readHeader(r io.ReaderAt) (*container, uint64, error) {
	var h [16]byte
	if err := readAtFull(r, h[:8], 0); err != nil {
		return nil, 0, fmt.Errorf("reading the header: %w", err)
	}
	c := &container{r: r}
	switch string(h[:2]) {
	case "II":
		c.order = binary.LittleEndian
	case "MM":
		c.order = binary.BigEndian
	default:
		return nil, 0, errors.New("not a TIFF file: no II or MM byte-order mark")
	}
	switch v := c.order.Uint16(h[2:]); v {
	case 42:
		return c, uint64(c.order.Uint32(h[4:])), nil
	case 43:
		c.big = true
		if err := readAtFull(r, h[8:16], 8); err != nil {
			return nil, 0, fmt.Errorf("reading the BigTIFF header: %w", err)
		}
		if c.order.Uint16(h[4:]) != 8 || c.order.Uint16(h[6:]) != 0 {
			return nil, 0, errors.New("BigTIFF header: offsets are not 8 bytes")
		}
		return c, c.order.Uint64(h[8:]), nil
	default:
		return nil, 0, fmt.Errorf("not a TIFF file: version %d, want 42 or 43 (BigTIFF)", v)
	}
}

// readIFDs reads the chain of IFDs starting at off. Only the first IFD
// must be readable. The chain stops quietly where it loops, grows past
// maxIFDs or reaches an IFD that cannot be read, as libtiff's does: the
// first image is still there, and what follows is at most overviews,
// which GDAL does not see either.
func (c *container) readIFDs(off uint64) ([]rawIFD, error) {
	var ifds []rawIFD
	seen := map[uint64]bool{}
	for off != 0 && !seen[off] && len(ifds) < maxIFDs {
		seen[off] = true
		ifd, next, err := c.readIFD(off)
		if err != nil {
			if len(ifds) == 0 {
				return nil, fmt.Errorf("IFD 0 at offset %d: %w", off, err)
			}
			break
		}
		ifds = append(ifds, ifd)
		off = next
	}
	if len(ifds) == 0 {
		return nil, errors.New("the file has no IFD")
	}
	return ifds, nil
}

// readIFD reads the IFD at off and returns it and the next IFD's offset.
func (c *container) readIFD(off uint64) (rawIFD, uint64, error) {
	countSize, entrySize, valueSize := 2, 12, 4
	if c.big {
		countSize, entrySize, valueSize = 8, 20, 8
	}
	var cb [8]byte
	if err := c.readAt(cb[:countSize], off); err != nil {
		return rawIFD{}, 0, fmt.Errorf("reading the entry count: %w", err)
	}
	var n uint64
	if c.big {
		n = c.order.Uint64(cb[:])
	} else {
		n = uint64(c.order.Uint16(cb[:]))
	}
	if n > maxEntries {
		return rawIFD{}, 0, fmt.Errorf("%d entries, more than %d", n, maxEntries)
	}
	buf := make([]byte, int(n)*entrySize+valueSize)
	if err := c.readAt(buf, off+uint64(countSize)); err != nil {
		return rawIFD{}, 0, fmt.Errorf("reading %d entries: %w", n, err)
	}
	ifd := rawIFD{offset: off, entries: make(map[uint16]entry, n)}
	for i := range int(n) {
		b := buf[i*entrySize : (i+1)*entrySize]
		e := entry{tag: c.order.Uint16(b), typ: c.order.Uint16(b[2:])}
		var field []byte
		if c.big {
			e.count = c.order.Uint64(b[4:])
			field = b[12:20]
		} else {
			e.count = uint64(c.order.Uint32(b[4:]))
			field = b[8:12]
		}
		size := typeSize(e.typ)
		if size == 0 {
			continue // an unknown type: TIFF readers must skip it
		}
		if e.count > maxTagValues {
			return rawIFD{}, 0, fmt.Errorf("tag %d has %d values, more than %d", e.tag, e.count, maxTagValues)
		}
		if e.count*uint64(size) <= uint64(valueSize) { // #nosec G115 -- size and valueSize are at most 8
			e.inline = field
		} else if c.big {
			e.offset = c.order.Uint64(field)
		} else {
			e.offset = uint64(c.order.Uint32(field))
		}
		ifd.entries[e.tag] = e
	}
	tail := buf[int(n)*entrySize:]
	if c.big {
		return ifd, c.order.Uint64(tail), nil
	}
	return ifd, uint64(c.order.Uint32(tail)), nil
}

// bytes returns an entry's value bytes, in the file's byte order.
func (c *container) bytes(e entry) ([]byte, error) {
	if e.inline != nil {
		return e.inline[:int(e.count)*typeSize(e.typ)], nil // #nosec G115 -- inline: at most 8 bytes
	}
	b, err := c.readFull(e.offset, e.count*uint64(typeSize(e.typ))) // #nosec G115 -- a type size, at most 8
	if err != nil {
		return nil, fmt.Errorf("tag %d: reading %d values at offset %d: %w", e.tag, e.count, e.offset, err)
	}
	return b, nil
}

// uints returns an entry's values as unsigned integers. It accepts the
// integer types only, the signed ones when no value is negative: some
// writers store offsets as SLONG8, which libtiff accepts.
func (c *container) uints(e entry) ([]uint64, error) {
	b, err := c.bytes(e)
	if err != nil {
		return nil, err
	}
	v := make([]uint64, e.count)
	for i := range v {
		switch e.typ {
		case typeByte, typeUndefined:
			v[i] = uint64(b[i])
		case typeShort:
			v[i] = uint64(c.order.Uint16(b[2*i:]))
		case typeLong, typeIFD:
			v[i] = uint64(c.order.Uint32(b[4*i:]))
		case typeLong8, typeIFD8:
			v[i] = c.order.Uint64(b[8*i:])
		case typeSByte, typeSShort, typeSLong, typeSLong8:
			var x int64
			switch e.typ {
			case typeSByte:
				x = int64(int8(b[i])) // #nosec G115 -- reinterpreting the bits is the point
			case typeSShort:
				x = int64(int16(c.order.Uint16(b[2*i:]))) // #nosec G115 -- as above
			case typeSLong:
				x = int64(int32(c.order.Uint32(b[4*i:]))) // #nosec G115 -- as above
			default:
				x = int64(c.order.Uint64(b[8*i:])) // #nosec G115 -- as above
			}
			if x < 0 {
				return nil, fmt.Errorf("tag %d: a negative value %d", e.tag, x)
			}
			v[i] = uint64(x)
		default:
			return nil, fmt.Errorf("tag %d: type %d is not an unsigned integer type", e.tag, e.typ)
		}
	}
	return v, nil
}

// floats returns an entry's values as float64s. It accepts DOUBLE and
// FLOAT, which is what the GeoTIFF tags hold.
func (c *container) floats(e entry) ([]float64, error) {
	b, err := c.bytes(e)
	if err != nil {
		return nil, err
	}
	v := make([]float64, e.count)
	for i := range v {
		switch e.typ {
		case typeDouble:
			v[i] = math.Float64frombits(c.order.Uint64(b[8*i:]))
		case typeFloat:
			v[i] = float64(math.Float32frombits(c.order.Uint32(b[4*i:])))
		default:
			return nil, fmt.Errorf("tag %d: type %d is not a floating-point type", e.tag, e.typ)
		}
	}
	return v, nil
}

// ascii returns an entry's value as a string, up to its first NUL.
func (c *container) ascii(e entry) (string, error) {
	if e.typ != typeASCII {
		return "", fmt.Errorf("tag %d: type %d is not ASCII", e.tag, e.typ)
	}
	b, err := c.bytes(e)
	if err != nil {
		return "", err
	}
	for i, ch := range b {
		if ch == 0 {
			return string(b[:i]), nil
		}
	}
	return string(b), nil
}

// readAt reads len(p) bytes at off.
func (c *container) readAt(p []byte, off uint64) error {
	if off > math.MaxInt64 {
		return fmt.Errorf("offset %d is past the end of any file", off)
	}
	return readAtFull(c.r, p, int64(off))
}

// readFull reads n bytes at off. It allocates as it reads, at most
// readChunk ahead of the data, so a count that a short file cannot back
// fails without first allocating all of it.
func (c *container) readFull(off, n uint64) ([]byte, error) {
	if off > math.MaxInt64 || n > math.MaxInt64-off {
		return nil, fmt.Errorf("%d bytes at offset %d are past the end of any file", n, off)
	}
	if n <= readChunk {
		b := make([]byte, n)
		return b, readAtFull(c.r, b, int64(off))
	}
	b := make([]byte, 0, readChunk)
	for uint64(len(b)) < n {
		k := min(n-uint64(len(b)), readChunk)
		start := len(b)
		b = append(b, make([]byte, k)...)
		if err := readAtFull(c.r, b[start:], int64(off)+int64(start)); err != nil {
			return nil, err
		}
	}
	return b, nil
}

// readAtFull reads len(p) bytes at off, turning a short read into
// io.ErrUnexpectedEOF.
func readAtFull(r io.ReaderAt, p []byte, off int64) error {
	n, err := r.ReadAt(p, off)
	if n < len(p) {
		if err == nil || errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	return nil
}
