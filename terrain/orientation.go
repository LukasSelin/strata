package terrain

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// OrientationComponent selects which component of the aspect Orientation
// writes.
type OrientationComponent int

const (
	// Northness is the aspect's north component: +1 for a slope facing
	// north, -1 for one facing south.
	Northness OrientationComponent = iota
	// Eastness is the aspect's east component: +1 for a slope facing east,
	// -1 for one facing west.
	Eastness
)

// OrientationOptions configures Orientation.
type OrientationOptions struct {
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
	// Component is Northness or Eastness. The zero value is Northness.
	Component OrientationComponent
	// Unweighted writes the cosine or sine of the aspect alone, instead
	// of weighting it by the sine of the slope. Flat cells, which have no
	// aspect, get 0.
	Unweighted bool
}

// Orientation computes the northness or eastness of dem: by default
//
//	northness = sin(slope)·cos(aspect)    eastness = sin(slope)·sin(aspect)
//
// as Geomorpho90m defines them (Amatulli et al. 2020), with aspect the
// compass bearing of Aspect. They are the north and east components of the
// unit surface normal, in [-1, 1]: continuous where aspect jumps from 359°
// to 0°, near 0 on gentle slopes whatever their direction, and ±1 only on
// vertical ones. With Unweighted they are cos(aspect) and sin(aspect)
// alone, the components of the unit downslope direction, as many
// ecological models use them; flat cells, which have no aspect, get 0,
// and on nearly flat ground they follow the noise of the gradient.
//
// Both come from the Horn gradient (dx and dy as in Gradient), or the
// quadratic fit's with FitRadius, without angles: the downslope direction
// in (east, north) components is (-dx, dy), so northness is
// dy/sqrt(1 + dx² + dy²) and eastness -dx/sqrt(1 + dx² + dy²), or over
// sqrt(dx² + dy²) with Unweighted. They are evaluated in float64, so no
// gradient is large or small enough to overflow or underflow, and rounded
// to float32 once. NaN or infinite gradients give NaN. See the package
// documentation for edges and validity. dst and dem must have the same
// dimensions and must not overlap; their strides may differ.
func Orientation(dst, dem raster.Float32Raster, opts OrientationOptions) {
	run(orientationOp(opts), dem, dst)
}

// OrientationTiled is Orientation run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func OrientationTiled(ctx context.Context, dst, dem raster.Float32Raster, opts OrientationOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, orientationOp(opts), dem, dst)
}

// OrientationChunked is Orientation run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Orientation would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func OrientationChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts OrientationOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, orientationOp(opts), dem, dst)
}

// orientationEast resolves and checks the component: whether it is
// eastness.
func orientationEast(c OrientationComponent) bool {
	switch c {
	case Northness:
		return false
	case Eastness:
		return true
	}
	panic(fmt.Sprintf("terrain: unknown OrientationComponent %d", c))
}

// orientationOp is Orientation's kernel: Horn's gradient when FitRadius
// is 0, the quadratic fit's otherwise.
func orientationOp(o OrientationOptions) exec.Kernel {
	east := orientationEast(o.Component)
	if o.FitRadius == 0 {
		kx, ky := cellSizes(o.CellSize, o.CellSizeY, o.ZFactor)
		return orientationKernel{horn{kx: kx, ky: ky}, east, o.Unweighted}
	}
	return fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitOrientation,
		east: east, unweighted: o.Unweighted}
}

// orientationKernel computes Horn's gradient a block of cells at a time
// into stack buffers, then the component from it, so that it is Gradient
// followed by orientationFromGradient by construction.
type orientationKernel struct {
	horn
	east, unweighted bool
}

func (k orientationKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	var gx, gy [fitBlock]float32
	for y := range dst.Height {
		o, r0, r1, r2 := out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2)
		for x0 := 0; x0 < dst.Width; x0 += fitBlock {
			m := min(fitBlock, dst.Width-x0)
			stencil.HornGradientRow(gx[:m], gy[:m], r0[x0:], r1[x0:], r2[x0:], k.kx, k.ky)
			stencil.OrientationFromGradientRow(o[x0:x0+m], gx[:m], gy[:m], k.east, k.unweighted)
		}
	}
}

// orientationFromGradient is the pointwise stage: dx and dy in, the
// component out.
type orientationFromGradient struct{ east, unweighted bool }

func (orientationFromGradient) Radius() int                  { return 0 }
func (orientationFromGradient) Arity() (inputs, outputs int) { return 2, 1 }

func (k orientationFromGradient) Process(dst exec.Span, src exec.Window) {
	out, gx, gy := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		stencil.OrientationFromGradientRow(out.Row(y), gx.Row(y), gy.Row(y), k.east, k.unweighted)
	}
}
