package raster

import "fmt"

// Window returns the w×h region of r whose top-left cell is (x, y), as a
// view: the result's Data and Valid share r's memory, so writes through
// either are visible in both. The window keeps r's Stride and points its
// ValidOffset at its own first cell, so it can be windowed again. Nothing
// is copied or allocated.
//
// It panics if w or h is not positive or the region extends outside r.
func (r Float32Raster) Window(x, y, w, h int) Float32Raster {
	if w <= 0 || h <= 0 {
		panic(fmt.Sprintf("raster: window dimensions must be positive, got %d×%d", w, h))
	}
	if x < 0 || y < 0 || x > r.Width-w || y > r.Height-h {
		panic(fmt.Sprintf("raster: window %d×%d at (%d, %d) outside %d×%d raster",
			w, h, x, y, r.Width, r.Height))
	}
	start := y*r.Stride + x
	end := start + dataLen(w, h, r.Stride)
	win := Float32Raster{
		Data:   r.Data[start:end:end],
		Width:  w,
		Height: h,
		Stride: r.Stride,
		Valid:  r.Valid,
	}
	if r.Valid != nil {
		win.ValidOffset = r.ValidOffset + start
	}
	return win
}
