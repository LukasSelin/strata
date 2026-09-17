package terrain

import (
	"strata/engine"
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
	run(GradientKernel(opts), dem, dx, dy)
}

// GradientKernel returns Gradient as an engine kernel with radius 1, one
// input (the DEM) and two outputs, dx then dy, for engine.ProcessN. It
// panics on invalid options, as Gradient does.
func GradientKernel(opts GradientOptions) engine.Kernel {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	return gradientKernel{horn{kx, ky}}
}

type gradientKernel struct{ horn }

func (gradientKernel) Arity() (inputs, outputs int) { return 1, 2 }

func (k gradientKernel) Process(dst engine.Span, src engine.Window) {
	dx, dy, dem := dst.Dst[0], dst.Dst[1], src.Src[0]
	for y := range dst.Height {
		stencil.HornGradientRow(dx.Row(y), dy.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kx, k.ky)
	}
}
