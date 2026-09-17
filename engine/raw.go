package engine

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/bits"
	"unsafe"

	"strata/raster"
)

// A raw float32 file is a raster stored as little-endian IEEE 754 float32
// cells in row-major order, with no header: cell (x, y) is the 4 bytes at
// offset 4·(y·width + x). It is the minimum needed to process rasters
// larger than memory (DESIGN.md §24, §43), not a format: the width and
// height are the caller's to know. For a file with a header, read through
// io.NewSectionReader and write through io.NewOffsetWriter.
//
// The file has no validity of its own. By default every cell is valid and
// the source and sink are not Masked. RawOptions.Fill declares a NoData
// value instead, as an IO adapter does (DESIGN.md §31): reading marks
// cells holding it invalid, and writing puts it under invalid cells. A
// valid cell that holds the fill value does not survive a round trip, the
// hazard of any fill value, so choose one the data cannot hold.

// RawOptions configures a raw float32 file source or sink.
type RawOptions struct {
	// Fill is the file's NoData value, used if HasFill is set. A
	// RawSource marks a cell invalid when its value equals Fill (by ==,
	// so -0 matches 0), or when Fill is a NaN and the value is any NaN.
	// A RawSink writes Fill under invalid cells. Either is then Masked.
	Fill    float32
	HasFill bool
}

// RawSource is a RasterSource reading a raw float32 file (see RawOptions)
// through an io.ReaderAt, such as a RawFile or an *os.File. It reads
// straight into the destination's memory: a window as wide as the raster
// into a destination without row padding (Stride == Width) with one
// ReadAt call per rawCallBytes, about 1 MiB, of consecutive rows, and any
// other window with one call per row. It allocates nothing.
type RawSource struct {
	r    io.ReaderAt
	w, h int
	opts RawOptions
}

// NewRawSource returns a source reading a width×height raster from r. The
// file is not checked here; a read past its end fails with
// io.ErrUnexpectedEOF. It panics if a dimension is not positive or the
// file would exceed 2⁶³ bytes.
func NewRawSource(r io.ReaderAt, width, height int, opts RawOptions) *RawSource {
	requireRawSize("NewRawSource", r == nil, width, height)
	return &RawSource{r: r, w: width, h: height, opts: opts}
}

// Size returns the raster's width and height.
func (s *RawSource) Size() (width, height int) { return s.w, s.h }

// Masked reports whether the source has a fill value.
func (s *RawSource) Masked() bool { return s.opts.HasFill }

// ReadWindow reads the region at (x, y) into dst. See RasterSource. It
// returns ctx.Err() without reading if ctx is done, and errors from the
// io.ReaderAt wrapped with the row that failed.
func (s *RawSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	requireRegion("RawSource.ReadWindow", "dst", dst, x, y, s.w, s.h)
	if s.opts.HasFill && dst.Valid == nil {
		panic("engine: RawSource.ReadWindow: the source has a fill value and dst has no validity mask")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	per := rowsPerCall(dst, s.w)
	for row := 0; row < dst.Height; row += per {
		k := min(per, dst.Height-row)
		start := row * dst.Stride
		cells := dst.Data[start : start+(k-1)*dst.Stride+dst.Width]
		b := floatBytes(cells)
		off := 4 * (int64(y+row)*int64(s.w) + int64(x))
		n, err := s.r.ReadAt(b, off)
		if n < len(b) {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return fmt.Errorf("engine: raw source: reading %s: %w", rowsName(y+row, k), err)
		}
		if !littleEndian {
			swapBytes(cells)
		}
		if dst.Valid != nil {
			for i := row; i < row+k; i++ {
				s.rowValidity(dst.Valid, dst.ValidOffset+i*dst.Stride, dst.Row(i))
			}
		}
	}
	return nil
}

// rawCallBytes is the size of one ReadAt or WriteAt call for consecutive
// rows. Measured on the benchmarks/chunked machine, reading and writing a
// cached file in 1 MiB calls is 17–21% faster than 80 KiB rows, and 16 MiB
// calls are slower again. It is a variable so tests can use small calls.
var rawCallBytes = 1 << 20

// rowsPerCall is the number of rows of r that one call reads or writes.
// Rows are consecutive in the file only for windows as wide as the file,
// and consecutive in r's memory only without row padding; anything else
// takes a call per row. Joining rows across gaps does not pay: copying a
// row's worth of unwanted cells costs more than a call.
func rowsPerCall(r raster.Float32Raster, fileWidth int) int {
	if r.Width != fileWidth || r.Stride != r.Width {
		return 1
	}
	return max(1, rawCallBytes/(4*r.Width))
}

func rowsName(y, k int) string {
	if k == 1 {
		return fmt.Sprintf("row %d", y)
	}
	return fmt.Sprintf("rows %d to %d", y, y+k-1)
}

// rowValidity sets the validity bits at off of a row of cells from the
// fill value, or all valid without one.
func (s *RawSource) rowValidity(m []uint64, off int, cells []float32) {
	if !s.opts.HasFill {
		raster.MaskFillRange(m, off, len(cells), true)
		return
	}
	fill, nanFill := s.opts.Fill, s.opts.Fill != s.opts.Fill
	for i := 0; i < len(cells); i += 64 {
		chunk := cells[i:min(i+64, len(cells))]
		var word [1]uint64
		for k, c := range chunk {
			if nanFill {
				if c == c {
					word[0] |= 1 << uint(k)
				}
			} else if c != fill {
				word[0] |= 1 << uint(k)
			}
		}
		raster.MaskCopyRange(m, off+i, word[:], 0, len(chunk))
	}
}

// RawSink is a RasterSink writing a raw float32 file (see RawOptions)
// through an io.WriterAt, such as a RawFile or an *os.File. It writes
// straight from the source's memory, with WriteAt calls grouped as
// RawSource groups reads, and allocates nothing. Writes
// past the end of a file extend it, so a sink can fill a new, empty file
// in any order.
type RawSink struct {
	f    io.WriterAt
	w, h int
	opts RawOptions
}

// NewRawSink returns a sink writing a width×height raster to w. It panics
// if a dimension is not positive or the file would exceed 2⁶³ bytes.
func NewRawSink(w io.WriterAt, width, height int, opts RawOptions) *RawSink {
	requireRawSize("NewRawSink", w == nil, width, height)
	return &RawSink{f: w, w: width, h: height, opts: opts}
}

// Size returns the raster's width and height.
func (s *RawSink) Size() (width, height int) { return s.w, s.h }

// Masked reports whether the sink has a fill value.
func (s *RawSink) Masked() bool { return s.opts.HasFill }

// WriteWindow writes src into the region at (x, y). See RasterSink. With a
// fill value it first stores the fill value in the Data of src's invalid
// cells. It returns ctx.Err() without writing if ctx is done, and errors
// from the io.WriterAt wrapped with the row that failed.
func (s *RawSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	requireRegion("RawSink.WriteWindow", "src", src, x, y, s.w, s.h)
	if src.Valid != nil && !s.opts.HasFill {
		panic("engine: RawSink.WriteWindow: src has a validity mask and the sink has no fill value")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	per := rowsPerCall(src, s.w)
	for row := 0; row < src.Height; row += per {
		k := min(per, src.Height-row)
		if src.Valid != nil {
			for i := row; i < row+k; i++ {
				fillInvalid(src.Row(i), src.Valid, src.ValidOffset+i*src.Stride, s.opts.Fill)
			}
		}
		start := row * src.Stride
		cells := src.Data[start : start+(k-1)*src.Stride+src.Width]
		if !littleEndian {
			swapBytes(cells)
		}
		off := 4 * (int64(y+row)*int64(s.w) + int64(x))
		_, err := s.f.WriteAt(floatBytes(cells), off)
		if !littleEndian {
			swapBytes(cells)
		}
		if err != nil {
			return fmt.Errorf("engine: raw sink: writing %s: %w", rowsName(y+row, k), err)
		}
	}
	return nil
}

// fillInvalid stores fill in the cells whose validity bit, from off in m,
// is clear.
func fillInvalid(cells []float32, m []uint64, off int, fill float32) {
	for i := 0; i < len(cells); i += 64 {
		chunk := cells[i:min(i+64, len(cells))]
		var word [1]uint64
		raster.MaskCopyRange(word[:], 0, m, off+i, len(chunk))
		invalid := ^word[0]
		if len(chunk) < 64 {
			invalid &= 1<<uint(len(chunk)) - 1
		}
		for ; invalid != 0; invalid &= invalid - 1 {
			chunk[bits.TrailingZeros64(invalid)] = fill
		}
	}
}

func requireRawSize(fn string, nilIO bool, width, height int) {
	if nilIO {
		panic(fmt.Sprintf("engine: %s: nil file", fn))
	}
	if width <= 0 || height <= 0 {
		panic(fmt.Sprintf("engine: %s: dimensions must be positive, got %d×%d", fn, width, height))
	}
	if uint64(width) > math.MaxInt64/4/uint64(height) {
		panic(fmt.Sprintf("engine: %s: %d×%d cells do not fit in a file", fn, width, height))
	}
}

// littleEndian reports whether float32 memory is laid out as the file is,
// so rows can be read and written in place.
var littleEndian = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

// floatBytes returns the memory of cells as bytes.
func floatBytes(cells []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(cells))), 4*len(cells))
}

// swapBytes reverses the byte order of every cell, converting between the
// file's little-endian layout and a big-endian machine's.
func swapBytes(cells []float32) {
	// Through uint32s, so that no NaN payload passes through a float
	// register.
	words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(cells))), len(cells))
	for i, v := range words {
		words[i] = bits.ReverseBytes32(v)
	}
}
