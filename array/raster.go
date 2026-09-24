package array

import (
	"fmt"

	"github.com/LukasSelin/strata/raster"
)

// FromRaster returns r as a [Height, Width] array sharing its Data and
// Valid: strides [Stride, 1], and the same ValidOffset, so the mask bits
// line up without a copy. Row padding is not part of the array. It panics
// if r fails Validate.
func FromRaster(r raster.Float32Raster) Array[float32] {
	if err := r.Validate(); err != nil {
		panic("array.FromRaster: " + err.Error())
	}
	return Array[float32]{
		Data:        r.Data,
		Shape:       []int{r.Height, r.Width},
		Stride:      []int{r.Stride, 1},
		Valid:       r.Valid,
		ValidOffset: r.ValidOffset,
	}
}

// ToRaster returns a rank-2 float32 array as a raster sharing its Data
// and Valid, so every package that takes a raster — algebra, terrain,
// focal, reduce, the engine — takes a 2-D slice of an N-D array:
//
//	dem := array.ToRaster(stack.Select(0, t)) // time step t of [time, y, x]
//
// It panics unless a has rank 2, consecutive elements along its last
// axis (stride 1, or any stride on a length-1 axis), and rows that do not
// overlap (a first stride of at least the row length, or any on one row).
// A transposed or broadcast view must be copied into a New array first.
func ToRaster(a Array[float32]) raster.Float32Raster {
	a.must("array.ToRaster", "a")
	if len(a.Shape) != 2 {
		panic(fmt.Sprintf("array.ToRaster: rank %d, want 2", len(a.Shape)))
	}
	h, w := a.Shape[0], a.Shape[1]
	stride := a.Stride[0]
	if h == 1 {
		stride = w
	}
	if w > 1 && a.Stride[1] != 1 || stride < w {
		panic(fmt.Sprintf("array.ToRaster: strides %v for shape %v are not a raster layout", a.Stride, a.Shape))
	}
	n := (h-1)*stride + w
	return raster.Float32Raster{
		Data:        a.Data[:n:n],
		Width:       w,
		Height:      h,
		Stride:      stride,
		Valid:       a.Valid,
		ValidOffset: a.ValidOffset,
	}
}
