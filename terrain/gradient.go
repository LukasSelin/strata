package terrain

import (
	"strata/internal/stencil"
	"strata/raster"
)

// GradientOptions configures Gradient.
type GradientOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
}

// Gradient computes Horn's 3×3 estimate of the elevation gradient of dem:
//
//	z1 z2 z3
//	z4 z5 z6      dx = ((z3 + 2·z6 + z9) - (z1 + 2·z4 + z7)) · ZFactor / (8·CellSize)
//	z7 z8 z9      dy = ((z7 + 2·z8 + z9) - (z1 + 2·z2 + z3)) · ZFactor / (8·CellSizeY)
//
// dx is positive when elevation rises with the column index (east on a
// north-up grid) and dy when it rises with the row index (south), as in
// gdaldem's aspect kernel; see the package documentation for conventions,
// edges and validity. dx, dy and dem must have the same dimensions and
// must not overlap; their strides may differ.
func Gradient(dx, dy, dem raster.Float32Raster, opts GradientOptions) {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	checkStencil([]string{"dem", "dx", "dy"}, dem, dx, dy)
	forInterior(dem, []raster.Float32Raster{dx, dy}, func(dst [][]float32, r0, r1, r2 []float32) {
		stencil.HornGradientRow(dst[0], dst[1], r0, r1, r2, kx, ky)
	})
	finishBorder(dx, dem)
	finishBorder(dy, dem)
}
