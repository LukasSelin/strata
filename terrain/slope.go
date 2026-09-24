package terrain

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// SlopeUnits selects how Slope expresses steepness.
type SlopeUnits int

const (
	// SlopeDegrees is the slope angle in degrees, 0 (flat) to 90.
	SlopeDegrees SlopeUnits = iota
	// SlopePercent is 100 × rise / run, so 45° is 100.
	SlopePercent
	// SlopeRadians is the slope angle in radians, 0 to π/2.
	SlopeRadians
)

// SlopeOptions configures Slope.
type SlopeOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
	// FitRadius selects how the derivatives are estimated: 0 is Horn's
	// 3×3 kernel, as gdaldem uses; 1 to MaxRadius fits Wood's quadratic
	// by least squares to the (2·FitRadius+1)² window around each cell,
	// for the same measure at a coarser scale (see the package
	// documentation). At 1 the fit is not Horn's kernel.
	FitRadius int
	// Units of the result. The zero value is SlopeDegrees.
	Units SlopeUnits
}

// Slope computes the steepness of dem from the magnitude of its Horn
// gradient, m = sqrt(dx² + dy²) with dx and dy as in Gradient: atan(m) in
// degrees or radians, or 100·m in percent. See the package documentation
// for edges and validity. dst and dem must have the same dimensions and
// must not overlap; their strides may differ.
//
// Angles use a float32 arctangent (Cephes atanf) that the scalar and SIMD
// kernels share bit-for-bit. Checked over every float32 input, it is
// within 1.41e-7 radians (8.1e-6 degrees) of math.Atan.
func Slope(dst, dem raster.Float32Raster, opts SlopeOptions) {
	run(slopeOp(opts), dem, dst)
}

// SlopeTiled is Slope run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func SlopeTiled(ctx context.Context, dst, dem raster.Float32Raster, opts SlopeOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, slopeOp(opts), dem, dst)
}

// SlopeChunked is Slope run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Slope would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func SlopeChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts SlopeOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, slopeOp(opts), dem, dst)
}

// newSlopeKernel resolves and checks opts for Slope's kernel.
func newSlopeKernel(opts SlopeOptions) slopeKernel {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	scale, atan := slopeScale(opts.Units)
	return slopeKernel{horn{kx: kx, ky: ky}, scale, atan}
}

// slopeScale is what the slope kernels multiply by, and whether they
// take the arctangent first, for units u.
func slopeScale(u SlopeUnits) (scale float32, atan bool) {
	switch u {
	case SlopeDegrees:
		return float32(180 / math.Pi), true
	case SlopeRadians:
		return 1, true
	case SlopePercent:
		return 100, false
	}
	panic(fmt.Sprintf("terrain: unknown SlopeUnits %d", u))
}

type slopeKernel struct {
	horn
	scale float32
	atan  bool
}

func (k slopeKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		stencil.HornSlopeRow(out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kx, k.ky, k.scale, k.atan)
	}
}
