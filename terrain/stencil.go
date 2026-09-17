package terrain

import (
	"fmt"
	"math"
	"unsafe"

	"strata/internal/stencil"
	"strata/raster"
)

// cellSizes resolves and checks the shared cell-size and z-factor options.
func cellSizes(cellSize, cellSizeY, zFactor float64) (kx, ky float32) {
	if !(cellSize > 0) || math.IsInf(cellSize, 0) {
		panic(fmt.Sprintf("terrain: CellSize must be positive and finite, got %v", cellSize))
	}
	if cellSizeY == 0 {
		cellSizeY = cellSize
	}
	if !(cellSizeY > 0) || math.IsInf(cellSizeY, 0) {
		panic(fmt.Sprintf("terrain: CellSizeY must be positive and finite (or 0 for CellSize), got %v", cellSizeY))
	}
	if zFactor == 0 {
		zFactor = 1
	}
	if math.IsNaN(zFactor) || math.IsInf(zFactor, 0) {
		panic(fmt.Sprintf("terrain: ZFactor must be finite, got %v", zFactor))
	}
	return stencil.HornScales(cellSize, cellSizeY, zFactor)
}

// checkStencil panics unless dem and every output are valid rasters of
// the same size whose Data and mask bits do not overlap, and every output
// has a mask if dem does. names[0] is dem's parameter name.
func checkStencil(names []string, dem raster.Float32Raster, outs ...raster.Float32Raster) {
	all := append([]raster.Float32Raster{dem}, outs...)
	for i, r := range all {
		if err := r.Validate(); err != nil {
			panic(fmt.Sprintf("terrain: %s: %v", names[i], err))
		}
	}
	for i, out := range outs {
		name := names[i+1]
		if out.Width != dem.Width || out.Height != dem.Height {
			panic(fmt.Sprintf("terrain: %s is %d×%d, %s is %d×%d",
				name, out.Width, out.Height, names[0], dem.Width, dem.Height))
		}
		if dem.Valid != nil && out.Valid == nil {
			panic(fmt.Sprintf("terrain: %s has a validity mask but %s has none; allocate %s with raster.NewFloat32Like(%s)",
				names[0], name, name, names[0]))
		}
	}
	for i := range all {
		for j := i + 1; j < len(all); j++ {
			if dataOverlaps(all[i], all[j]) {
				panic(fmt.Sprintf("terrain: %s and %s share Data", names[i], names[j]))
			}
			if maskOverlaps(all[i], all[j]) {
				panic(fmt.Sprintf("terrain: %s and %s share validity bits", names[i], names[j]))
			}
		}
	}
}

func dataOverlaps(a, b raster.Float32Raster) bool {
	const size = unsafe.Sizeof(float32(0))
	a0 := uintptr(unsafe.Pointer(unsafe.SliceData(a.Data)))
	b0 := uintptr(unsafe.Pointer(unsafe.SliceData(b.Data)))
	return a0 < b0+uintptr(len(b.Data))*size && b0 < a0+uintptr(len(a.Data))*size
}

func maskOverlaps(a, b raster.Float32Raster) bool {
	if a.Valid == nil || b.Valid == nil {
		return false
	}
	// Absolute bit addresses of each raster's first and last-plus-one cell.
	a0 := uintptr(unsafe.Pointer(unsafe.SliceData(a.Valid)))*8 + uintptr(a.ValidOffset)
	b0 := uintptr(unsafe.Pointer(unsafe.SliceData(b.Valid)))*8 + uintptr(b.ValidOffset)
	return a0 < b0+uintptr(len(b.Data)) && b0 < a0+uintptr(len(a.Data))
}

// forInterior calls row for every interior row y of dem with the output
// cells 1..Width-2 of that row in each output and dem rows y-1, y and
// y+1. It does nothing for rasters narrower or shorter than 3.
func forInterior(dem raster.Float32Raster, outs []raster.Float32Raster, row func(dst [][]float32, r0, r1, r2 []float32)) {
	w, h := dem.Width, dem.Height
	if w < 3 || h < 3 {
		return
	}
	dst := make([][]float32, len(outs))
	for y := 1; y < h-1; y++ {
		for i, out := range outs {
			dst[i] = out.Row(y)[1 : w-1]
		}
		row(dst, dem.Row(y-1), dem.Row(y), dem.Row(y+1))
	}
}

// finishBorder applies the edge and validity policy to one output: NaN in
// the border, and either the eroded mask of dem or, for a dem without a
// mask, valid interior cells and cleared border bits (as in package
// algebra, stale bits from earlier use of out do not survive).
func finishBorder(out, dem raster.Float32Raster) {
	fillBorder(out, float32(math.NaN()))
	switch {
	case dem.Valid != nil:
		stencil.Erode3x3(out.Valid, out.ValidOffset, out.Stride,
			dem.Valid, dem.ValidOffset, dem.Stride, dem.Width, dem.Height)
	case out.Valid != nil:
		for y := range out.Height {
			raster.MaskFillRange(out.Valid, out.ValidOffset+y*out.Stride, out.Width, true)
		}
		stencil.ClearBorder(out.Valid, out.ValidOffset, out.Stride, out.Width, out.Height)
	}
}

func fillBorder(r raster.Float32Raster, v float32) {
	first, last := r.Row(0), r.Row(r.Height-1)
	for x := range first {
		first[x] = v
		last[x] = v
	}
	for y := 1; y < r.Height-1; y++ {
		row := r.Row(y)
		row[0] = v
		row[len(row)-1] = v
	}
}
