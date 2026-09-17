package engine

import (
	"context"
	"fmt"
	"sync"

	"strata/raster"
)

// RasterSource is a raster that the engine reads a window at a time
// (DESIGN.md §24), such as a raster in memory (MemorySource) or a raw
// float32 file (RawSource). The Chunked entry points, such as
// terrain.SlopeChunked, read each tile and its halo from their sources
// into buffers of their own.
//
// Implementations must be safe for concurrent ReadWindow calls, of any
// regions.
type RasterSource interface {
	// Size returns the raster's width and height in cells. Both are
	// positive and do not change.
	Size() (width, height int)
	// Masked reports whether the source has validity, that is whether
	// ReadWindow can report cells as invalid (DESIGN.md §31). A source
	// that is not Masked has every cell valid.
	Masked() bool
	// ReadWindow reads the dst.Width×dst.Height region of the raster
	// whose top-left cell is (x, y) into dst's cells: their Data, and,
	// if dst has a mask, their validity bits, which are all set when the
	// source is not Masked. It writes nothing else: not dst's row
	// padding, nor mask bits outside dst's cells.
	//
	// It returns an error if the read fails, for example on an IO error
	// or a file too short for the raster; dst's cells are then
	// unspecified. It may return ctx.Err() if ctx is done. It panics on
	// programming errors: dst fails Validate, the region is not inside
	// the raster, or the source is Masked and dst has no mask, which
	// would lose the validity.
	ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error
}

// RasterSink is a raster that the engine writes a window at a time
// (DESIGN.md §24), such as a raster in memory (MemorySink) or a raw
// float32 file (RawSink).
//
// Implementations must be safe for concurrent WriteWindow calls of
// disjoint regions. The engine never writes a region twice in one call.
type RasterSink interface {
	// Size returns the raster's width and height in cells. Both are
	// positive and do not change.
	Size() (width, height int)
	// Masked reports whether the sink stores validity. The engine
	// computes validity only for sinks that do, and a call whose sources
	// include a Masked one panics unless every sink is Masked, as the
	// Tiled entry points panic for a dst without a mask.
	Masked() bool
	// WriteWindow stores src's cells as the src.Width×src.Height region
	// of the raster whose top-left cell is (x, y): their Data and, if
	// the sink is Masked, their validity (all valid if src has no mask).
	//
	// It may overwrite the Data of src's invalid cells, which is
	// unspecified (DESIGN.md §31), for example with a fill value; it
	// changes nothing else in src.
	//
	// It returns an error if the write fails; the region's cells in the
	// sink are then unspecified. It may return ctx.Err() if ctx is done.
	// It panics on programming errors: src fails Validate, the region is
	// not inside the raster, or src has a mask and the sink is not
	// Masked, which would lose the validity.
	WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error
}

// MemorySource is a RasterSource over a raster in memory. It is Masked if
// the raster has a validity mask. It never blocks or fails, and ignores
// ctx.
type MemorySource struct {
	r raster.Float32Raster
}

// NewMemorySource returns a source reading r, which may be a window. r
// is not copied, so later changes to its cells are visible to reads. It
// panics if r fails Validate.
func NewMemorySource(r raster.Float32Raster) *MemorySource {
	requireRaster("NewMemorySource", r)
	return &MemorySource{r: r}
}

// Raster returns the raster s reads.
func (s *MemorySource) Raster() raster.Float32Raster { return s.r }

// Size returns the raster's width and height.
func (s *MemorySource) Size() (width, height int) { return s.r.Width, s.r.Height }

// Masked reports whether the raster has a validity mask.
func (s *MemorySource) Masked() bool { return s.r.Valid != nil }

// ReadWindow copies the region at (x, y) into dst. See RasterSource.
func (s *MemorySource) ReadWindow(_ context.Context, dst raster.Float32Raster, x, y int) error {
	requireRegion("MemorySource.ReadWindow", "dst", dst, x, y, s.r.Width, s.r.Height)
	if s.r.Valid != nil && dst.Valid == nil {
		panic("engine: MemorySource.ReadWindow: the source has a validity mask and dst has none")
	}
	copyCells(dst, s.r.Window(x, y, dst.Width, dst.Height), dst.Valid != nil)
	return nil
}

// MemorySink is a RasterSink over a raster in memory. It is Masked if the
// raster has a validity mask. It never blocks or fails, and ignores ctx.
type MemorySink struct {
	r raster.Float32Raster
	// maskMu serialises writes of validity bits: disjoint regions can
	// share mask words.
	maskMu sync.Mutex
}

// NewMemorySink returns a sink writing into r, which may be a window. It
// writes only r's cells (Data, and validity bits if r has a mask). It
// panics if r fails Validate.
func NewMemorySink(r raster.Float32Raster) *MemorySink {
	requireRaster("NewMemorySink", r)
	return &MemorySink{r: r}
}

// Raster returns the raster s writes.
func (s *MemorySink) Raster() raster.Float32Raster { return s.r }

// Size returns the raster's width and height.
func (s *MemorySink) Size() (width, height int) { return s.r.Width, s.r.Height }

// Masked reports whether the raster has a validity mask.
func (s *MemorySink) Masked() bool { return s.r.Valid != nil }

// WriteWindow copies src into the region at (x, y). See RasterSink. It
// never changes src.
func (s *MemorySink) WriteWindow(_ context.Context, src raster.Float32Raster, x, y int) error {
	requireRegion("MemorySink.WriteWindow", "src", src, x, y, s.r.Width, s.r.Height)
	if src.Valid != nil && s.r.Valid == nil {
		panic("engine: MemorySink.WriteWindow: src has a validity mask and the sink has none")
	}
	dst := s.r.Window(x, y, src.Width, src.Height)
	copyCells(dst, src, false)
	if dst.Valid != nil {
		s.maskMu.Lock()
		copyValidity(dst, src)
		s.maskMu.Unlock()
	}
	return nil
}

// copyCells copies src's Data into dst's cells, which have the same size,
// and with valid also its validity.
func copyCells(dst, src raster.Float32Raster, valid bool) {
	if dst.Stride == dst.Width && src.Stride == src.Width {
		copy(dst.Data, src.Data)
	} else {
		for y := range dst.Height {
			copy(dst.Row(y), src.Row(y))
		}
	}
	if valid {
		copyValidity(dst, src)
	}
}

// copyValidity copies src's validity bits into dst's, or sets dst's if
// src has no mask. dst has a mask.
func copyValidity(dst, src raster.Float32Raster) {
	whole := dst.Stride == dst.Width && src.Stride == src.Width
	switch {
	case src.Valid == nil && whole:
		raster.MaskFillRange(dst.Valid, dst.ValidOffset, dst.Width*dst.Height, true)
	case src.Valid == nil:
		for y := range dst.Height {
			raster.MaskFillRange(dst.Valid, dst.ValidOffset+y*dst.Stride, dst.Width, true)
		}
	case whole:
		raster.MaskCopyRange(dst.Valid, dst.ValidOffset, src.Valid, src.ValidOffset, dst.Width*dst.Height)
	default:
		for y := range dst.Height {
			raster.MaskCopyRange(dst.Valid, dst.ValidOffset+y*dst.Stride, src.Valid, src.ValidOffset+y*src.Stride, dst.Width)
		}
	}
}

func requireRaster(fn string, r raster.Float32Raster) {
	if err := r.Validate(); err != nil {
		panic(fmt.Sprintf("engine: %s: %v", fn, err))
	}
}

// requireRegion panics unless r is a valid raster whose size placed at
// (x, y) lies inside a w×h raster.
func requireRegion(fn, name string, r raster.Float32Raster, x, y, w, h int) {
	if err := r.Validate(); err != nil {
		panic(fmt.Sprintf("engine: %s: %s: %v", fn, name, err))
	}
	if x < 0 || y < 0 || x > w-r.Width || y > h-r.Height {
		panic(fmt.Sprintf("engine: %s: %d×%d region at (%d, %d) outside %d×%d raster",
			fn, r.Width, r.Height, x, y, w, h))
	}
}
