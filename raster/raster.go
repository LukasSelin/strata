// Package raster holds strata's in-memory raster types: flat float32
// cell data with an optional validity mask (Float32Raster), zero-copy
// windows onto that data, and spatial metadata (Grid) kept separate from
// the numbers.
//
// # Layout
//
// Cells are stored row-major in Data. Cell (x, y) is Data[y*Stride+x];
// Stride may exceed Width, leaving padding between rows (for example a
// Stride rounded up to a multiple of 64 so mask rows are word-aligned).
// A raster's Data always ends at its last cell:
// len(Data) == (Height-1)*Stride + Width.
//
// # NoData
//
// NoData is a separate validity bitmask, never a value in Data (STRATA-3,
// benchmarks/nodata/RESULTS.md). Valid == nil means every cell is valid.
// Otherwise the validity of Data[i] is bit ValidOffset+i of Valid, where
// bit b lives in Valid[b>>6], LSB first, and a set bit means valid. Data
// under a cleared bit is unspecified. A NaN or -9999 with its bit set is
// an ordinary value. Fill values belong to IO adapters, which translate
// them to and from the mask.
//
// # Errors
//
// Like internal/vec, constructors and accessors panic on invalid
// arguments (non-positive dimensions, short data, out-of-bounds windows or
// cells). These are programming errors, not runtime conditions. Use
// Float32Raster.Validate to check a hand-assembled raster without
// panicking.
package raster

import (
	"fmt"
	"math"
)

// Float32Raster is a 2D float32 raster, or a view into one. Windows are
// also Float32Rasters, so anything that accepts a raster accepts a
// window, and windows can be windowed again.
type Float32Raster struct {
	Data []float32

	Width  int
	Height int
	Stride int

	// Valid is the validity mask shared by a raster and all its windows,
	// or nil when every cell is valid. See the package documentation.
	Valid []uint64
	// ValidOffset is the bit index in Valid of Data[0]. It is 0 for
	// rasters built by the constructors and non-zero for most windows.
	ValidOffset int
}

// NewFloat32 wraps data as a width×height raster with Stride == width and
// no validity mask (every cell valid). data is not copied. It panics if
// either dimension is not positive or data holds fewer than width*height
// cells. Extra elements past width*height are not part of the raster.
func NewFloat32(width, height int, data []float32) Float32Raster {
	return NewFloat32Stride(width, height, width, data)
}

// NewFloat32Stride is NewFloat32 with an explicit row stride, which must
// be at least width. data must hold at least (height-1)*stride + width
// cells; the padding after the last cell of the last row is not required.
func NewFloat32Stride(width, height, stride int, data []float32) Float32Raster {
	n := requireShape(width, height, stride)
	if len(data) < n {
		panic(fmt.Sprintf("raster: data has %d cells, need %d for %d×%d with stride %d",
			len(data), n, width, height, stride))
	}
	return Float32Raster{
		Data:   data[:n:n],
		Width:  width,
		Height: height,
		Stride: stride,
	}
}

// NewFloat32Like allocates a zeroed raster with r's width and height.
// The result is compact (Stride == Width) even when r is a window with a
// wider stride. If r has a validity mask, the result gets its own
// all-valid mask, so kernels writing into it can record NoData; if r has
// none, neither does the result.
func NewFloat32Like(r Float32Raster) Float32Raster {
	n := requireShape(r.Width, r.Height, r.Width)
	out := Float32Raster{
		Data:   make([]float32, n),
		Width:  r.Width,
		Height: r.Height,
		Stride: r.Width,
	}
	if r.Valid != nil {
		out.Valid = NewMask(n)
	}
	return out
}

// requireShape panics on a bad shape and returns the Data length it needs.
func requireShape(width, height, stride int) int {
	if width <= 0 || height <= 0 {
		panic(fmt.Sprintf("raster: dimensions must be positive, got %d×%d", width, height))
	}
	if stride < width {
		panic(fmt.Sprintf("raster: stride %d is less than width %d", stride, width))
	}
	n, ok := shapeLen(width, height, stride)
	if !ok {
		panic(fmt.Sprintf("raster: %d×%d with stride %d has more cells than an int holds",
			width, height, stride))
	}
	return n
}

// dataLen is the Data length of a shape known to fit in an int.
func dataLen(width, height, stride int) int {
	return (height-1)*stride + width
}

// shapeLen is dataLen for a shape with positive dimensions and stride >=
// width that may not fit: ok is false if (height-1)*stride + width
// overflows an int.
func shapeLen(width, height, stride int) (n int, ok bool) {
	if height-1 > (math.MaxInt-width)/stride {
		return 0, false
	}
	return dataLen(width, height, stride), true
}

// Row returns the Width cells of row y as a slice sharing r's memory.
// Its capacity is also Width, so appending to it never overwrites the
// row padding or the next row. It panics if y is out of range.
func (r Float32Raster) Row(y int) []float32 {
	if uint(y) >= uint(r.Height) {
		panic(fmt.Sprintf("raster: row %d out of range [0, %d)", y, r.Height))
	}
	start := y * r.Stride
	end := start + r.Width
	return r.Data[start:end:end]
}

// Index returns the Data index of cell (x, y). It panics if the cell is
// outside the raster.
func (r Float32Raster) Index(x, y int) int {
	if uint(x) >= uint(r.Width) || uint(y) >= uint(r.Height) {
		panic(fmt.Sprintf("raster: cell (%d, %d) outside %d×%d raster", x, y, r.Width, r.Height))
	}
	return y*r.Stride + x
}

// IsValid reports whether cell (x, y) holds data. It is true for every
// cell of a raster without a mask.
func (r Float32Raster) IsValid(x, y int) bool {
	i := r.Index(x, y)
	return r.Valid == nil || MaskGet(r.Valid, r.ValidOffset+i)
}

// SetValid marks cell (x, y) valid or invalid in the shared mask, so the
// change is visible to the parent raster and every overlapping window.
// It panics if r has no mask: a mask must be attached to the root raster
// (for example r.Valid = NewMask(len(r.Data))) before windowing, because a
// mask attached to a window would not be seen by its parent.
func (r Float32Raster) SetValid(x, y int, valid bool) {
	i := r.Index(x, y)
	if r.Valid == nil {
		panic("raster: SetValid on a raster without a validity mask")
	}
	MaskSet(r.Valid, r.ValidOffset+i, valid)
}

// Validate reports whether r's fields are consistent: positive
// dimensions, Stride >= Width, Data long enough for every cell, and, if
// Valid is set, a non-negative ValidOffset with Valid long enough to
// cover every cell. It does not inspect cell values or mask bits.
func (r Float32Raster) Validate() error {
	if r.Width <= 0 || r.Height <= 0 {
		return fmt.Errorf("raster: dimensions must be positive, got %d×%d", r.Width, r.Height)
	}
	if r.Stride < r.Width {
		return fmt.Errorf("raster: stride %d is less than width %d", r.Stride, r.Width)
	}
	n, ok := shapeLen(r.Width, r.Height, r.Stride)
	if !ok {
		return fmt.Errorf("raster: %d×%d with stride %d has more cells than an int holds",
			r.Width, r.Height, r.Stride)
	}
	if len(r.Data) < n {
		return fmt.Errorf("raster: data has %d cells, need %d", len(r.Data), n)
	}
	if r.Valid != nil {
		if r.ValidOffset < 0 {
			return fmt.Errorf("raster: negative ValidOffset %d", r.ValidOffset)
		}
		// len(r.Valid)*64 cannot overflow: a slice of that many words
		// could not be allocated.
		if bits := len(r.Valid) * 64; r.ValidOffset > bits-n {
			return fmt.Errorf("raster: mask has %d bits, need %d (offset %d + %d cells)",
				bits, r.ValidOffset+n, r.ValidOffset, n)
		}
	}
	return nil
}
