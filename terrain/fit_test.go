package terrain

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// useScalar switches both kernel packages the fit runs on.
func useScalar(scalar bool) {
	stencil.UseScalar(scalar)
	focalrow.UseScalar(scalar)
}

// quadratic returns a w×h DEM of z = A·x² + B·y² + C·x·y + D·x + E·y + F
// at x = col·cx, y = row·cy, as float32, and the coefficients.
func quadratic(w, h int, cx, cy float64, c [6]float64) raster.Float32Raster {
	d := make([]float32, w*h)
	for row := range h {
		for col := range w {
			x, y := float64(col)*cx, float64(row)*cy
			d[row*w+col] = float32(c[0]*x*x + c[1]*y*y + c[2]*x*y + c[3]*x + c[4]*y + c[5])
		}
	}
	return raster.NewFloat32(w, h, d)
}

// TestFitPlane checks the fit at every radius on integer planes, where
// every column sum and weighted sum is an exact integer: the gradient is
// the plane's to a rounding of its factor, and every curvature is +0,
// since the second-derivative taps sum to zero against any plane.
func TestFitPlane(t *testing.T) {
	for _, ab := range [][2]float64{{0, 0}, {1, 0}, {0, -3}, {2, 5}, {-7, 4}} {
		for r := 1; r <= MaxRadius; r++ {
			dem := plane(2*r+5, 2*r+4, ab[0], ab[1], 1, 1)
			dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
			Gradient(dx, dy, dem, GradientOptions{CellSize: 1, FitRadius: r})
			curv := raster.NewFloat32Like(dem)
			for _, ct := range []CurvatureType{CurvatureProfile, CurvaturePlan, CurvatureMean} {
				Curvature(curv, dem, CurvatureOptions{CellSize: 1, Type: ct, FitRadius: r})
				for y := range dem.Height {
					for x := range dem.Width {
						border := x < r || y < r || x >= dem.Width-r || y >= dem.Height-r
						gx, gy, cv := dx.Data[dx.Index(x, y)], dy.Data[dy.Index(x, y)], curv.Data[curv.Index(x, y)]
						if border {
							if gx == gx || gy == gy || cv == cv {
								t.Fatalf("plane %v r=%d: border cell (%d, %d) not NaN", ab, r, x, y)
							}
							continue
						}
						if math.Abs(float64(gx)-ab[0]) > 1e-6*math.Abs(ab[0]) || math.Abs(float64(gy)-ab[1]) > 1e-6*math.Abs(ab[1]) {
							t.Fatalf("plane %v r=%d cell (%d, %d): gradient (%v, %v)", ab, r, x, y, gx, gy)
						}
						if math.Float32bits(cv) != 0 {
							t.Fatalf("plane %v r=%d type %d cell (%d, %d): curvature %v, want +0", ab, r, ct, x, y, cv)
						}
					}
				}
			}
		}
	}
}

// TestFitQuadratic checks that the fit reproduces a quadratic surface,
// which least squares fits exactly: at every cell the gradient is the
// surface's there and the curvatures are Florinsky's formulas of its
// exact derivatives, on rectangular cells and with a ZFactor. The
// tolerance is float32 rounding of sums of up to (2r+1)² elevations of
// magnitude zmax, over the fit's divisors.
func TestFitQuadratic(t *testing.T) {
	const w, h = 40, 36
	cx, cy, zf := 2.0, 3.0, 1.5
	coef := [6]float64{0.03, -0.02, 0.01, 0.4, -0.25, 50}
	dem := quadratic(w, h, cx, cy, coef)
	var zmax float64
	for _, v := range dem.Data {
		zmax = max(zmax, math.Abs(float64(v)))
	}
	for _, r := range []int{1, 2, 4, 8} {
		dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
		Gradient(dx, dy, dem, GradientOptions{CellSize: cx, CellSizeY: cy, ZFactor: zf, FitRadius: r})
		curv := map[CurvatureType]raster.Float32Raster{}
		for _, ct := range []CurvatureType{CurvatureProfile, CurvaturePlan, CurvatureMean} {
			curv[ct] = raster.NewFloat32Like(dem)
			Curvature(curv[ct], dem, CurvatureOptions{CellSize: cx, CellSizeY: cy, ZFactor: zf, Type: ct, FitRadius: r})
		}
		k := float64(2*r + 1)
		// The elevations are rounded to float32 before the fit sees them,
		// which is an error of up to zmax·2⁻²⁴ in every cell, and the
		// sums add (2r+1)² more roundings of that size.
		e := 4 * k * k * zmax * math.Pow(2, -24)
		s2 := float64(r*(r+1)*(2*r+1)) / 3
		gtolX := zf * e * float64(r) * k / (k * s2 * cx)
		gtolY := zf * e * float64(r) * k / (k * s2 * cy)
		for row := r; row < h-r; row++ {
			for col := r; col < w-r; col++ {
				x, y := float64(col)*cx, float64(row)*cy
				p := zf * (2*coef[0]*x + coef[2]*y + coef[3])
				q := zf * (2*coef[1]*y + coef[2]*x + coef[4])
				gx, gy := float64(dx.Data[dx.Index(col, row)]), float64(dy.Data[dy.Index(col, row)])
				if math.Abs(gx-p) > gtolX+1e-6*math.Abs(p) || math.Abs(gy-q) > gtolY+1e-6*math.Abs(q) {
					t.Fatalf("r=%d cell (%d, %d): gradient (%v, %v), want (%v, %v)", r, col, row, gx, gy, p, q)
				}
				rr, tt, ss := zf*2*coef[0], zf*2*coef[1], zf*coef[2]
				g := p*p + q*q
				want := map[CurvatureType]float64{
					CurvatureProfile: -(p*p*rr + 2*p*q*ss + q*q*tt) / (g * math.Pow(1+g, 1.5)),
					CurvaturePlan:    -(q*q*rr - 2*p*q*ss + p*p*tt) / math.Pow(g, 1.5),
					CurvatureMean:    -((1+q*q)*rr - 2*p*q*ss + (1+p*p)*tt) / (2 * math.Pow(1+g, 1.5)),
				}
				for ct, v := range want {
					got := float64(curv[ct].Data[curv[ct].Index(col, row)])
					// Second differences lose more to the elevations' rounding:
					// relative to the curvature itself, a few parts in 10⁴ here.
					if math.Abs(got-v) > 2e-3*math.Abs(v)+1e-6 {
						t.Fatalf("r=%d type %d cell (%d, %d): %v, want %v", r, ct, col, row, got, v)
					}
				}
			}
		}
	}
}

// TestFitSurfaceIsTheStandaloneProducts holds Surface with a FitRadius
// to Gradient, Slope, Aspect and Hillshade with the same FitRadius, bit
// for bit, Data and validity, on both backends and in every form.
func TestFitSurfaceIsTheStandaloneProducts(t *testing.T) {
	const w, h = 57, 31
	runs := []engine.Options{
		{Workers: 1},
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 5, TileHeight: 3, Workers: 2},
	}
	defer useScalar(false)
	for _, scalar := range []bool{false, true} {
		useScalar(scalar)
		for _, masked := range []bool{false, true} {
			for _, r := range []int{1, 3} {
				opts := SurfaceOptions{CellSize: 12.5, CellSizeY: 9, ZFactor: 1.5, Units: SlopePercent,
					ZeroForFlat: true, Azimuth: 200, Altitude: 30, FitRadius: r}
				rng := rand.New(rand.NewPCG(81, uint64(r)))
				dem := weightedOperand(rng, w, h, 800, true, masked)
				for y := 3; y < 12; y++ { // a flat patch
					for x := 10; x < 24; x++ {
						dem.Data[dem.Index(x, y)] = 812
					}
				}
				newOut := func() raster.Float32Raster {
					out := raster.NewFloat32(w, h, make([]float32, w*h))
					if masked {
						out.Valid = make([]uint64, raster.MaskWords(w*h))
					}
					return out
				}
				var want [surfaceProducts]raster.Float32Raster
				for k := range want {
					want[k] = newOut()
				}
				Gradient(want[surfaceDx], want[surfaceDy], dem, GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY,
					ZFactor: opts.ZFactor, FitRadius: r})
				Slope(want[surfaceSlope], dem, SlopeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
					Units: opts.Units, FitRadius: r})
				Aspect(want[surfaceAspect], dem, AspectOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
					ZeroForFlat: opts.ZeroForFlat, FitRadius: r})
				Hillshade(want[surfaceHillshade], dem, HillshadeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY,
					ZFactor: opts.ZFactor, Azimuth: opts.Azimuth, Altitude: opts.Altitude, FitRadius: r})
				for _, eo := range runs {
					var got [surfaceProducts]raster.Float32Raster
					for k := range got {
						got[k] = newOut()
					}
					id := fmt.Sprintf("scalar=%v masked=%v r=%d %+v", scalar, masked, r, eo)
					if err := SurfaceTiled(context.Background(), SurfaceOutputs{Dx: got[0], Dy: got[1], Slope: got[2],
						Aspect: got[3], Hillshade: got[4]}, dem, opts, eo); err != nil {
						t.Fatal(err)
					}
					for k := range got {
						sameCells(t, fmt.Sprintf("%s tiled product %d", id, k), got[k], want[k])
					}
					for k := range got {
						got[k] = newOut()
					}
					sink := func(k int) engine.RasterSink { return engine.NewMemorySink(got[k]) }
					if err := SurfaceChunked(context.Background(), SurfaceSinks{Dx: sink(0), Dy: sink(1), Slope: sink(2),
						Aspect: sink(3), Hillshade: sink(4)}, engine.NewMemorySource(dem), opts, eo); err != nil {
						t.Fatal(err)
					}
					for k := range got {
						sameCells(t, fmt.Sprintf("%s chunked product %d", id, k), got[k], want[k])
					}
				}
			}
		}
	}
}

// TestFitBackendsAgree holds the scalar kernels to the SIMD ones for
// every fitted product, over hazards.
func TestFitBackendsAgree(t *testing.T) {
	defer useScalar(false)
	rng := rand.New(rand.NewPCG(82, 1))
	dem := weightedOperand(rng, 300, 23, 500, true, true)
	for _, op := range []FeatureOp{
		SlopeOptions{CellSize: 5, FitRadius: 2},
		AspectOptions{CellSize: 5, FitRadius: 5, Trigonometric: true},
		HillshadeOptions{CellSize: 5, FitRadius: 3},
		CurvatureOptions{CellSize: 5, FitRadius: 4, Type: CurvatureMean},
		CurvatureOptions{CellSize: 5, FitRadius: 8, Type: CurvaturePlan},
	} {
		var out [2]raster.Float32Raster
		for i, scalar := range []bool{false, true} {
			useScalar(scalar)
			out[i] = raster.NewFloat32Like(dem)
			Features([]Feature{{Op: op, Dst: out[i]}}, dem)
		}
		sameCells(t, fmt.Sprintf("%+v", op), out[1], out[0])
	}
}

// TestFitValidity checks that a fitted product's border and validity
// are those of its whole window: the same as a ruggedness measure's of
// the same radius, which TestRuggednessRadiusValidity checks per cell.
func TestFitValidity(t *testing.T) {
	rng := rand.New(rand.NewPCG(83, 2))
	for _, r := range []int{1, 3, 8} {
		dem := weightedOperand(rng, 70, 2*r+11, 300, true, true)
		want := raster.NewFloat32Like(dem)
		Ruggedness(want, dem, RuggednessOptions{Type: RuggednessRoughness, Radius: r})
		for _, op := range []FeatureOp{SlopeOptions{CellSize: 1, FitRadius: r}, CurvatureOptions{CellSize: 1, FitRadius: r}} {
			got := raster.NewFloat32Like(dem)
			Features([]Feature{{Op: op, Dst: got}}, dem)
			for y := range dem.Height {
				for x := range dem.Width {
					if got.IsValid(x, y) != want.IsValid(x, y) {
						t.Fatalf("r=%d %T: cell (%d, %d) valid=%v, want %v", r, op, x, y, got.IsValid(x, y), want.IsValid(x, y))
					}
				}
			}
		}
	}
}

func TestFitPanics(t *testing.T) {
	r := raster.NewFloat32(30, 30, make([]float32, 900))
	o := func() raster.Float32Raster { return raster.NewFloat32Like(r) }
	for _, fr := range []int{-1, MaxRadius + 1} {
		mustPanic(t, fmt.Sprintf("slope FitRadius %d", fr), func() { Slope(o(), r, SlopeOptions{CellSize: 1, FitRadius: fr}) })
		mustPanic(t, fmt.Sprintf("surface FitRadius %d", fr), func() {
			Surface(SurfaceOutputs{Slope: o()}, r, SurfaceOptions{CellSize: 1, FitRadius: fr})
		})
	}
	mustPanic(t, "fit without a cell size", func() { Curvature(o(), r, CurvatureOptions{FitRadius: 2}) })
	mustPanic(t, "fit with bad units", func() { Slope(o(), r, SlopeOptions{CellSize: 1, FitRadius: 2, Units: 7}) })
	mustPanic(t, "fit with bad altitude", func() { Hillshade(o(), r, HillshadeOptions{CellSize: 1, FitRadius: 2, Altitude: 95}) })
	mustPanic(t, "fit with a factor that underflows", func() {
		Slope(o(), r, SlopeOptions{CellSize: 1e30, ZFactor: 1e-30, FitRadius: 2})
	})
}
