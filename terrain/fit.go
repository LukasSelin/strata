package terrain

import (
	"fmt"
	"math"

	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/internal/stencil"
)

// woodFit is Wood's (1996) least-squares quadratic
//
//	z = a·x² + b·y² + c·x·y + d·x + e·y + f
//
// fitted to the (2r+1)×(2r+1) window around a cell, x = i·cx and
// y = j·cy for column and row offsets i, j in −r..r. On a square window
// the normal equations separate, and each coefficient is one weighted
// sum of the window with integer taps (see the package documentation):
//
//	p = d = Σ i·z / (K·S·cx)            ti  = i
//	q = e = Σ j·z / (K·S·cy)            t2  = 3i² − r(r+1)
//	r = 2a = 6·Σ t2_i·z / (K·Q·cx²)     K = 2r+1, S = Σ i² = r(r+1)(2r+1)/3,
//	t = 2b = 6·Σ t2_j·z / (K·Q·cy²)     Q = Σ t2_i²
//	s = c = Σ i·j·z / (S²·cx·cy)
//
// each times ZFactor. The divisors are folded into one float32 factor
// per derivative, computed in float64 and rounded once, like HornScales.
type woodFit struct {
	r                  int
	kp, kq, kr, kt, ks float32
	// ti, t2 and ones are the taps for offsets −r..r.
	ti, t2, ones []float32
}

// newWoodFit resolves and checks the fit's options.
func newWoodFit(fitRadius int, cellSize, cellSizeY, zFactor float64) woodFit {
	if fitRadius < 1 || fitRadius > MaxRadius {
		panic(fmt.Sprintf("terrain: FitRadius must be 0 to %d, got %d", MaxRadius, fitRadius))
	}
	cx, cy, z := resolveCells(cellSize, cellSizeY, zFactor)
	r := fitRadius
	f := woodFit{r: r}
	var q float64
	for i := -r; i <= r; i++ {
		w := 3*i*i - r*(r+1)
		f.ti = append(f.ti, float32(i))
		f.t2 = append(f.t2, float32(w))
		f.ones = append(f.ones, 1)
		q += float64(w * w)
	}
	k := float64(2*r + 1)
	s := float64(r*(r+1)*(2*r+1)) / 3
	f.kp = float32(z / (k * s * cx))
	f.kq = float32(z / (k * s * cy))
	f.kr = float32(6 * z / (k * q * cx * cx))
	f.kt = float32(6 * z / (k * q * cy * cy))
	f.ks = float32(z / (s * s * cx * cy))
	// As for the Horn factors: one that overflows turns a flat window
	// (0·Inf) into NaN, and one that underflows flattens every cell.
	for _, v := range []float32{f.kp, f.kq, f.kr, f.kt, f.ks} {
		if !usableScale(v) {
			panic(fmt.Sprintf("terrain: the quadratic fit's factors ZFactor/(K·S·CellSize), 6·ZFactor/(K·Q·CellSize²) "+
				"and ZFactor/(S²·CellSize·CellSizeY), and the same for CellSizeY, must be finite and non-zero as float32, "+
				"got %v, %v, %v, %v and %v for CellSize %v, CellSizeY %v, ZFactor %v and FitRadius %d",
				f.kp, f.kq, f.kr, f.kt, f.ks, cellSize, cellSizeY, zFactor, fitRadius))
		}
	}
	return f
}

// fitProduct is what a fitKernel writes from the fitted derivatives.
type fitProduct int

const (
	fitGradient fitProduct = iota
	fitSlope
	fitAspect
	fitHillshade
	fitHeatLoad
	fitCurvature
)

// fitKernel computes a product of the quadratic fit: the gradient
// (p, q) through the same from-gradient kernels Surface finishes Horn's
// gradient with, or a curvature through ZT's curvature formulas.
// HeatLoad is a from-gradient product too.
type fitKernel struct {
	fit     woodFit
	product fitProduct
	scale   float32 // slope
	atan    bool
	flat    float32 // aspect
	trig    bool
	c       float32 // hillshade
	bx, by  float32
	curv    stencil.CurvatureKind
	heat    stencil.HeatLoadTerms
}

func (k fitKernel) Radius() int { return k.fit.r }

func (k fitKernel) Arity() (inputs, outputs int) {
	if k.product == fitGradient {
		return 1, 2
	}
	return 1, 1
}

func (fitKernel) Edge() float32 { return float32(math.NaN()) }

// fitBlock is how many cells a fitKernel takes at a time, so that its
// column and derivative rows fit on the stack rather than in engine
// scratch, which a Pipeline stage cannot have (DESIGN.md §52).
const fitBlock = 256

// Process computes each row a block at a time. The column pass folds
// the 2r+1 rows under the block, top to bottom, into the plain and the
// j-weighted column sums (and for curvature the t2-weighted ones); the
// row pass folds each run of 2r+1 of those, left to right, with the
// taps above, and the result is multiplied by its factor. Every tap is
// applied, zeros included, as in focal (DESIGN.md §53).
func (k fitKernel) Process(dst exec.Span, src exec.Window) {
	dem := src.Src[0]
	f := &k.fit
	n := 2*f.r + 1
	var colSum, colJ, colT2 [fitBlock + 2*MaxRadius]float32
	var p, q, r, s, t [fitBlock]float32
	for y := range dst.Height {
		in := dem.Data[dem.Index(0, y):]
		for x0 := 0; x0 < dst.Width; x0 += fitBlock {
			m := min(fitBlock, dst.Width-x0)
			cols := m + n - 1
			win := in[x0:]
			cs, cj := colSum[:cols], colJ[:cols]
			focalrow.ColumnSum(cs, win, dem.Stride, n)
			focalrow.ColumnCorrelate(cj, win, dem.Stride, f.ti)
			pp, qq := p[:m], q[:m]
			focalrow.RowCorrelate(pp, cs, f.ti)
			focalrow.RowCorrelate(qq, cj, f.ones)
			scaleRow(pp, f.kp)
			scaleRow(qq, f.kq)
			out := dst.Dst[0].Row(y)[x0 : x0+m]
			switch k.product {
			case fitGradient:
				copy(out, pp)
				copy(dst.Dst[1].Row(y)[x0:x0+m], qq)
			case fitSlope:
				stencil.SlopeFromGradientRow(out, pp, qq, k.scale, k.atan)
			case fitAspect:
				stencil.AspectFromGradientRow(out, pp, qq, k.flat, k.trig)
			case fitHillshade:
				stencil.HillshadeFromGradientRow(out, pp, qq, k.c, k.bx, k.by)
			case fitHeatLoad:
				stencil.HeatLoadFromGradientRow(out, pp, qq, k.heat)
			default:
				ct := colT2[:cols]
				focalrow.ColumnCorrelate(ct, win, dem.Stride, f.t2)
				rr, ss, tt := r[:m], s[:m], t[:m]
				focalrow.RowCorrelate(rr, cs, f.t2)
				focalrow.RowCorrelate(tt, ct, f.ones)
				focalrow.RowCorrelate(ss, cj, f.ti)
				scaleRow(rr, f.kr)
				scaleRow(tt, f.kt)
				scaleRow(ss, f.ks)
				stencil.CurvatureFromDerivsRow(out, pp, qq, rr, ss, tt, k.curv)
			}
		}
	}
}

// scaleRow sets x[i] = x[i]·k, rounded to float32.
func scaleRow(x []float32, k float32) {
	for i := range x {
		x[i] = float32(x[i] * k)
	}
}

// The operations' kernels: the 3×3 method when FitRadius is 0, the
// quadratic fit otherwise.

func gradientOp(o GradientOptions) exec.Kernel {
	if o.FitRadius == 0 {
		return newGradientKernel(o)
	}
	return fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitGradient}
}

func slopeOp(o SlopeOptions) exec.Kernel {
	if o.FitRadius == 0 {
		return newSlopeKernel(o)
	}
	k := fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitSlope}
	k.scale, k.atan = slopeScale(o.Units)
	return k
}

func aspectOp(o AspectOptions) exec.Kernel {
	if o.FitRadius == 0 {
		return newAspectKernel(o)
	}
	return fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitAspect,
		flat: aspectFlat(o.ZeroForFlat), trig: o.Trigonometric}
}

func hillshadeOp(o HillshadeOptions) exec.Kernel {
	if o.FitRadius == 0 {
		return newHillshadeKernel(o)
	}
	k := fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitHillshade}
	k.c, k.bx, k.by = hillshadeLight(o.Azimuth, o.Altitude)
	return k
}

func curvatureOp(o CurvatureOptions) exec.Kernel {
	if o.FitRadius == 0 {
		return newCurvatureKernel(o)
	}
	return fitKernel{fit: newWoodFit(o.FitRadius, o.CellSize, o.CellSizeY, o.ZFactor), product: fitCurvature,
		curv: curvatureKind(o.Type)}
}
