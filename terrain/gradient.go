package terrain

import (
	"context"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
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
	run(newGradientKernel(opts), dem, dx, dy)
}

// GradientTiled is Gradient run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func GradientTiled(ctx context.Context, dx, dy, dem raster.Float32Raster, opts GradientOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newGradientKernel(opts), dem, dx, dy)
}

// GradientChunked is Gradient run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Gradient would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func GradientChunked(ctx context.Context, dx, dy engine.RasterSink, dem engine.RasterSource, opts GradientOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newGradientKernel(opts), dem, dx, dy)
}

// newGradientKernel resolves and checks opts for Gradient's kernel.
func newGradientKernel(opts GradientOptions) gradientKernel {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	return gradientKernel{horn{kx: kx, ky: ky}}
}

type gradientKernel struct{ horn }

func (gradientKernel) Arity() (inputs, outputs int) { return 1, 2 }

func (k gradientKernel) Process(dst exec.Span, src exec.Window) {
	dx, dy, dem := dst.Dst[0], dst.Dst[1], src.Src[0]
	for y := range dst.Height {
		stencil.HornGradientRow(dx.Row(y), dy.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kx, k.ky)
	}
}
