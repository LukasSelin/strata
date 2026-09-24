package terrain

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// CurvatureType selects which curvature Curvature computes.
type CurvatureType int

const (
	// CurvatureProfile is the profile (vertical) curvature: the normal
	// curvature of the surface along the gradient, the direction water
	// flows. Positive where the slope steepens downhill (convex), which
	// accelerates flow.
	CurvatureProfile CurvatureType = iota
	// CurvaturePlan is the plan (contour) curvature: the curvature of the
	// contour line through the cell. Positive where contours bulge
	// outwards (convex, spurs and ridges), which disperses flow;
	// negative in hollows and valleys, which concentrate it.
	CurvaturePlan
	// CurvatureMean is the mean curvature of the surface, the average of
	// its two principal curvatures. Positive on domes, negative in bowls.
	CurvatureMean
)

// CurvatureOptions configures Curvature.
type CurvatureOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
	// FitRadius selects how the derivatives are estimated: 0 is the
	// Zevenbergen–Thorne quadratic through the 3×3 window; 1 to MaxRadius
	// fits Wood's quadratic by least squares to the (2·FitRadius+1)²
	// window around each cell, for the same measure at a coarser scale
	// (see the package documentation). At 1 the fit is not ZT's.
	FitRadius int
	// Type of curvature. The zero value is CurvatureProfile.
	Type CurvatureType
}

// Curvature computes a curvature of dem, in reciprocal units of the cell
// size (1/m for metres), positive where the surface is convex. See the
// package documentation for edges and validity. dst and dem must have the
// same dimensions and must not overlap; their strides may differ.
//
// It fits the Zevenbergen–Thorne (1987) quadratic to each 3×3 window
// (z1..z9 as in Gradient, z5 the centre, cx and cy the cell sizes and Z
// the ZFactor):
//
//	p = Z·(z6 - z4) / (2·cx)          ∂z/∂x
//	q = Z·(z8 - z2) / (2·cy)          ∂z/∂y
//	r = Z·(z4 - 2·z5 + z6) / cx²      ∂²z/∂x²
//	t = Z·(z2 - 2·z5 + z8) / cy²      ∂²z/∂y²
//	s = Z·(z1 - z3 - z7 + z9) / (4·cx·cy)
//
// and evaluates Florinsky's (2016) normal-section curvatures, with
// g = p² + q²:
//
//	profile  -(p²·r + 2·p·q·s + q²·t) / (g·(1 + g)^(3/2))
//	plan     -(q²·r - 2·p·q·s + p²·t) / g^(3/2)
//	mean     -((1 + q²)·r - 2·p·q·s + (1 + p²)·t) / (2·(1 + g)^(3/2))
//
// Unlike the Horn operations, Curvature reads the centre cell. The
// results do not depend on whether y runs north or south.
//
// Profile and plan curvature are undefined where the surface is flat
// (p = q = 0). Those cells get 0 and stay valid, like Aspect's flat
// cells, unless r, s or t is not finite (a NaN or infinite centre, for
// example), which gives NaN. A zero result is +0.
//
// Esri's Curvature tool differs: it omits the (1 + g) terms, multiplies
// by 100, and its profile curvature has the opposite sign. Its "total"
// curvature is -(r + t)·100, the Laplacian, which is twice the mean
// curvature only where the ground is flat; Wood's (1996) total curvature
// r² + 2s² + t² is another quantity again. CurvatureMean is the
// curvature of the surface itself.
//
// The kernels compute in float32, dividing num/g before the (1 + g)
// term so that steep ground does not overflow, and the scalar and SIMD
// kernels agree bit-for-bit.
func Curvature(dst, dem raster.Float32Raster, opts CurvatureOptions) {
	run(curvatureOp(opts), dem, dst)
}

// CurvatureTiled is Curvature run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func CurvatureTiled(ctx context.Context, dst, dem raster.Float32Raster, opts CurvatureOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, curvatureOp(opts), dem, dst)
}

// CurvatureChunked is Curvature run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Curvature would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func CurvatureChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts CurvatureOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, curvatureOp(opts), dem, dst)
}

// newCurvatureKernel resolves and checks opts for Curvature's kernel.
func newCurvatureKernel(opts CurvatureOptions) curvatureKernel {
	cx, cy, z := resolveCells(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	kind := curvatureKind(opts.Type)
	// As for the Horn factors: one that overflows turns a flat window
	// (0·Inf) into NaN, and one that underflows flattens every cell.
	kp, kq, kr, kt, ks := stencil.ZTScales(cx, cy, z)
	for _, k := range []float32{kp, kq, kr, kt, ks} {
		if !usableScale(k) {
			panic(fmt.Sprintf("terrain: ZFactor/(2·CellSize), ZFactor/CellSize² and ZFactor/(4·CellSize·CellSizeY), "+
				"and the same for CellSizeY, must be finite and non-zero as float32, got %v, %v, %v, %v and %v "+
				"for CellSize %v, CellSizeY %v and ZFactor %v", kp, kq, kr, kt, ks, opts.CellSize, opts.CellSizeY, opts.ZFactor))
		}
	}
	return curvatureKernel{kp: kp, kq: kq, kr: kr, kt: kt, ks: ks, kind: kind}
}

// curvatureKind checks t and returns the kernels' name for it.
func curvatureKind(t CurvatureType) stencil.CurvatureKind {
	switch t {
	case CurvatureProfile:
		return stencil.CurvProfile
	case CurvaturePlan:
		return stencil.CurvPlan
	case CurvatureMean:
		return stencil.CurvMean
	}
	panic(fmt.Sprintf("terrain: unknown CurvatureType %d", t))
}

type curvatureKernel struct {
	window3
	kp, kq, kr, kt, ks float32
	kind               stencil.CurvatureKind
}

func (k curvatureKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		stencil.ZTCurvatureRow(out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kp, k.kq, k.kr, k.kt, k.ks, k.kind)
	}
}
