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

// HeatLoadEquation selects one of McCune and Keon's (2002) three
// regressions (their Table 2).
type HeatLoadEquation int

const (
	// HeatLoadEquation1 is fitted to ln(radiation) over slopes of 0–90°
	// at latitudes 0–60°: the broadest, and the least precise
	// (adjusted R² 0.958).
	HeatLoadEquation1 HeatLoadEquation = iota
	// HeatLoadEquation2 is fitted to ln(radiation) over slopes of 0–60°
	// (adjusted R² 0.978).
	HeatLoadEquation2
	// HeatLoadEquation3 is fitted to radiation itself, not its
	// logarithm, over slopes of 0–60° at latitudes 30–60° (adjusted R²
	// 0.983). Outside that range it can go negative.
	HeatLoadEquation3
)

// HeatLoadOptions configures HeatLoad.
type HeatLoadOptions struct {
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
	// Latitude is the site's latitude in degrees, in [-90, 90], one value
	// for the whole raster. Negative is south, and selects McCune's
	// southern-hemisphere folds. 0 is the equator: there is no default.
	// The equations were fitted over 0–60°, and Equation 3 over 30–60°.
	Latitude float64
	// Equation is the regression. The zero value is HeatLoadEquation1.
	Equation HeatLoadEquation
	// Radiation folds the aspect about the north–south axis, for
	// potential annual direct incident radiation, instead of about the
	// northeast–southwest axis for the heat load index.
	Radiation bool
	// Linear writes radiation on its arithmetic scale, in
	// MJ·cm⁻²·yr⁻¹: exp of Equations 1 and 2, which are fitted to its
	// logarithm. Equation 3 is fitted to radiation itself and is
	// unchanged by it.
	Linear bool
}

// HeatLoad computes McCune and Keon's (2002) index of heat load, or with
// Radiation their estimate of potential annual direct incident
// radiation, from latitude and dem's slope and aspect:
//
//	v = k0 + k1·cos(L)·cos(S) + k2·cos(A')·sin(S)·sin(L) + k3·sin(L)·sin(S)
//	       + k4·sin(A')·sin(S) + k5·cos(A')·sin(S)
//
// with the coefficients of the chosen Equation (their Table 2), L the
// latitude, S the slope angle and A' the folded aspect: the angle between
// the cell's aspect and the coolest bearing, 0 to 180°. For heat load the
// coolest bearing is northeast (45°), so southwest slopes, which take the
// afternoon sun, are the warmest; for Radiation it is north (0°). South
// of the equator, as McCune (2004) prescribes, L is the latitude's
// absolute value and the coolest bearings are southeast (135°) and south
// (180°). Equations 1 and 2 give ln(radiation, MJ·cm⁻²·yr⁻¹), and 3
// radiation itself; Linear gives radiation for all three. For heat load
// the result is a unitless index on the same scale. Neither accounts for
// cloud, the atmosphere or shading by surrounding terrain.
//
// S and A' come from the Horn gradient (dx and dy as in Gradient), or
// the quadratic fit's with FitRadius, without angles: cos(S) is
// w = 1/sqrt(1 + dx² + dy²), and sin(S) times the aspect's unit vector
// is w·(-dx, dy) in (east, north) components, so the equation is a
// function of dx, dy and one square root. A flat cell, whose aspect is
// undefined, has sin(S) = 0 and gets k0 + k1·cos(L). The gradient's
// function is evaluated in float64 and rounded to float32 once, so a
// result is within an ulp or two of the exact equation at the float32
// gradient Gradient writes. NaN or infinite gradients give NaN. See the
// package documentation for edges and validity. dst and dem must have
// the same dimensions and must not overlap; their strides may differ.
//
// Linear's exp is Go's math.Exp, which on some architectures is not
// correctly rounded, so on two architectures a result can differ by an
// ulp.
func HeatLoad(dst, dem raster.Float32Raster, opts HeatLoadOptions) {
	run(heatLoadOp(opts), dem, dst)
}

// HeatLoadTiled is HeatLoad run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func HeatLoadTiled(ctx context.Context, dst, dem raster.Float32Raster, opts HeatLoadOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, heatLoadOp(opts), dem, dst)
}

// HeatLoadChunked is HeatLoad run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// HeatLoad would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func HeatLoadChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts HeatLoadOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, heatLoadOp(opts), dem, dst)
}

// heatLoadCoefficients are k0 to k5 of HeatLoad's equation, from McCune
// and Keon (2002), Table 2: the constant, then the coefficients of
// cos(L)·cos(S), cos(A')·sin(S)·sin(L), sin(L)·sin(S), sin(A')·sin(S)
// and cos(A')·sin(S).
var heatLoadCoefficients = [...][6]float64{
	HeatLoadEquation1: {-1.467, 1.582, -1.500, -0.262, 0.607, 0},
	HeatLoadEquation2: {-1.236, 1.350, -1.376, -0.331, 0.375, 0},
	HeatLoadEquation3: {0.339, 0.808, 0, -0.196, 0, -0.482},
}

// heatLoadTerms resolves and checks the options that do not concern the
// gradient, and rewrites the equation as a function of it.
//
// With the aspect's bearing θ and the coolest bearing c, A' = |θ − c|
// folded into [0, 180°], so cos(A') = cos(θ − c) and sin(A') =
// |sin(θ − c)|. sin(S)·(sin θ, cos θ) is w·(-dx, dy), so
//
//	sin(S)·cos(A') = w·(dy·cos c − dx·sin c)
//	sin(S)·sin(A') = w·|dx·cos c + dy·sin c|
//	sin(S)         = w·sqrt(dx² + dy²)
//	cos(S)         = w
func heatLoadTerms(o HeatLoadOptions) stencil.HeatLoadTerms {
	if o.Equation < HeatLoadEquation1 || o.Equation > HeatLoadEquation3 {
		panic(fmt.Sprintf("terrain: unknown HeatLoadEquation %d", o.Equation))
	}
	if !(o.Latitude >= -90 && o.Latitude <= 90) {
		panic(fmt.Sprintf("terrain: Latitude must be in [-90, 90], got %v", o.Latitude))
	}
	k := heatLoadCoefficients[o.Equation]
	lat := math.Abs(o.Latitude) * math.Pi / 180
	sinL, cosL := math.Sin(lat), math.Cos(lat)
	// cos c and sin c of the coolest bearing, exactly where they can be.
	h := math.Sqrt2 / 2
	cosC, sinC := h, h // northeast
	switch south := o.Latitude < 0; {
	case o.Radiation && !south:
		cosC, sinC = 1, 0 // north
	case o.Radiation && south:
		cosC, sinC = -1, 0 // south
	case south:
		cosC, sinC = -h, h // southeast
	}
	kc := k[2]*sinL + k[5]
	return stencil.HeatLoadTerms{
		K0: k[0],
		A:  k[1] * cosL,
		BX: -kc * sinC, BY: kc * cosC,
		E:  k[3] * sinL,
		F:  k[4],
		CX: cosC, CY: sinC,
		Exp: o.Linear && o.Equation != HeatLoadEquation3,
	}
}

// heatLoadOp is HeatLoad's kernel: Horn's gradient when FitRadius is 0,
// the quadratic fit's otherwise.
func heatLoadOp(o HeatLoadOptions) exec.Kernel {
	t := heatLoadTerms(o)
	if o.FitRadius == 0 {
		kx, ky := cellSizes(o.CellSize, o.CellSizeY, o.ZFactor)
		return heatLoadKernel{horn{kx: kx, ky: ky}, t}
	}
	return fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitHeatLoad, heat: t}
}

// heatLoadKernel computes Horn's gradient a block of cells at a time into
// stack buffers, then the equation from it, so that it is Gradient
// followed by heatLoadFromGradient by construction.
type heatLoadKernel struct {
	horn
	t stencil.HeatLoadTerms
}

func (k heatLoadKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	var gx, gy [fitBlock]float32
	for y := range dst.Height {
		o, r0, r1, r2 := out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2)
		for x0 := 0; x0 < dst.Width; x0 += fitBlock {
			m := min(fitBlock, dst.Width-x0)
			stencil.HornGradientRow(gx[:m], gy[:m], r0[x0:], r1[x0:], r2[x0:], k.kx, k.ky)
			stencil.HeatLoadFromGradientRow(o[x0:x0+m], gx[:m], gy[:m], k.t)
		}
	}
}

// heatLoadFromGradient is the pointwise stage: dx and dy in, the
// equation out.
type heatLoadFromGradient struct{ t stencil.HeatLoadTerms }

func (heatLoadFromGradient) Radius() int                  { return 0 }
func (heatLoadFromGradient) Arity() (inputs, outputs int) { return 2, 1 }

func (k heatLoadFromGradient) Process(dst exec.Span, src exec.Window) {
	out, gx, gy := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		stencil.HeatLoadFromGradientRow(out.Row(y), gx.Row(y), gy.Row(y), k.t)
	}
}
