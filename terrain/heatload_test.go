package terrain

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// mcCuneKeonTest is one row of testrad.xls, the spreadsheet McCune and
// Keon publish with their paper (sites.science.oregonstate.edu/~mccuneb/
// testrad.xls), evaluated by Excel: latitude, slope and aspect in
// degrees, then the predicted direct incident radiation from Equations
// 1, 2 and 3 and the heat load from the same three, each as the equation
// gives it (ln for 1 and 2). The first rows are the paper's Table 2
// examples, with the erratum to Eq. 1's north-aspect value (-0.984).
var mcCuneKeonTest = []struct {
	lat, slope, aspect float64
	rad, heat          [3]float64
}{
	{40, 0, 0,
		[3]float64{-0.25511769098577686, -0.20184000178937955, 0.9579639100401343},
		[3]float64{-0.25511769098577686, -0.20184000178937955, 0.9579639100401343}},
	{40, 30, 0,
		[3]float64{-0.9837750181305804, -0.8890103948393941, 0.5710452843712215},
		[3]float64{-0.6279670110295916, -0.6268993985409437, 0.6416325501052635}},
	{40, 30, 180,
		[3]float64{-0.01959360360077163, -0.00453464391071629, 1.0530452843712215},
		[3]float64{0.053812205478473735, -0.001480597264211625, 0.9824580186371794}},
	{60, 0, 0,
		[3]float64{-0.6759999999999998, -0.5609999999999998, 0.7430000000000001},
		[3]float64{-0.6759999999999998, -0.5609999999999998, 0.7430000000000001}},
	{60, 30, 0,
		[3]float64{-1.544942286340599, -1.3905855345755218, 0.36300377355803837},
		[3]float64{-1.1400956521839989, -1.0834897710580573, 0.4335910392920804}},
	{60, 30, 180,
		[3]float64{-0.24590418066394135, -0.19893457896813463, 0.8450037735580384},
		[3]float64{-0.22153699864030738, -0.24086529954064415, 0.7744165078239963}},
}

// slopePlane is a 5×5 DEM on unit cells whose aspect is north (0) or
// south (180) at the given slope in degrees: z = ±tan(slope)·row, which
// falls towards row 0, north, for aspect 0. Elevations stay below 3, so
// their float32 rounding moves Horn's gradient by about 1e-7.
func slopePlane(slope, aspect float64) raster.Float32Raster {
	t := math.Tan(slope * math.Pi / 180)
	if aspect == 180 {
		t = -t
	}
	d := make([]float32, 25)
	for y := range 5 {
		for x := range 5 {
			d[y*5+x] = float32(t * float64(y))
		}
	}
	return raster.NewFloat32(5, 5, d)
}

// TestHeatLoadMcCuneKeon holds HeatLoad to the authors' own numbers: all
// 36 values of their test spreadsheet, and the arithmetic-scale column
// the spreadsheet derives from them. The planes' gradients are within
// about 1e-7 of tan(slope), which moves any equation by less than 1e-6;
// the result's rounding to float32 is 6e-8 at these magnitudes.
func TestHeatLoadMcCuneKeon(t *testing.T) {
	const tol = 2e-6
	for _, c := range mcCuneKeonTest {
		dem := slopePlane(c.slope, c.aspect)
		for eq := range 3 {
			for _, radiation := range []bool{true, false} {
				want := c.heat[eq]
				if radiation {
					want = c.rad[eq]
				}
				for _, linear := range []bool{false, true} {
					w := want
					if linear && eq < 2 {
						w = math.Exp(want)
					}
					o := HeatLoadOptions{CellSize: 1, Latitude: c.lat, Equation: HeatLoadEquation(eq),
						Radiation: radiation, Linear: linear}
					dst := raster.NewFloat32Like(dem)
					HeatLoad(dst, dem, o)
					checkBorderNaN(t, "heat load", dst)
					eachInterior(dst, func(x, y int, v float32) {
						if math.Abs(float64(v)-w) > tol {
							t.Fatalf("lat %v slope %v aspect %v %+v: cell (%d, %d) = %.9f, want %.9f",
								c.lat, c.slope, c.aspect, o, x, y, v, w)
						}
					})
				}
			}
		}
	}
}

// TestHeatLoadHemispheres checks McCune's (2004) southern-hemisphere
// rule as a symmetry: south of the equator the sun is to the north, so a
// DEM mirrored north to south at latitude -L has the heat load and
// radiation the original has at L, cell for cell. Mirroring the rows
// negates Horn's dy exactly, and the southern folds are the northern ones
// reflected, so with Horn's gradient the bits agree. The fit folds its
// column sums top to bottom, so mirroring reorders them and its dy is
// negated only to within their rounding: a few float32 ulps of the
// window's elevations over the factor, which moves the result by well
// under 1e-5 here.
func TestHeatLoadHemispheres(t *testing.T) {
	rng := rand.New(rand.NewPCG(91, 4))
	const w, h = 37, 29
	dem := weightedOperand(rng, w, h, 500, false, true)
	flip := raster.NewFloat32(w, h, make([]float32, w*h))
	flip.Valid = make([]uint64, raster.MaskWords(w*h))
	for y := range h {
		for x := range w {
			flip.Data[flip.Index(x, h-1-y)] = dem.Data[dem.Index(x, y)]
			raster.MaskSet(flip.Valid, flip.Index(x, h-1-y), dem.IsValid(x, y))
		}
	}
	for _, fit := range []int{0, 2} {
		for eq := range 3 {
			for _, radiation := range []bool{false, true} {
				north := HeatLoadOptions{CellSize: 20, CellSizeY: 15, Latitude: 38.5, Equation: HeatLoadEquation(eq),
					Radiation: radiation, FitRadius: fit}
				south := north
				south.Latitude = -north.Latitude
				a, b := raster.NewFloat32Like(dem), raster.NewFloat32Like(flip)
				HeatLoad(a, dem, north)
				HeatLoad(b, flip, south)
				for y := range h {
					for x := range w {
						g, want := b.Data[b.Index(x, h-1-y)], a.Data[a.Index(x, y)]
						tol := 0.0
						if fit > 0 {
							tol = 1e-5
						}
						if !(math.Abs(float64(g)-float64(want)) <= tol) && !(g != g && want != want) {
							t.Fatalf("%+v: mirrored cell (%d, %d) = %v, want %v", south, x, y, g, want)
						}
						if b.IsValid(x, h-1-y) != a.IsValid(x, y) {
							t.Fatalf("%+v: mirrored validity differs at (%d, %d)", south, x, y)
						}
					}
				}
			}
		}
	}
}

// heatLoadAngles is McCune and Keon's equation as they write it, in
// float64, from the slope and aspect of the gradient (gx, gy): aspect by
// atan2, folded by their formulas (and McCune 2004's for the south), and
// the terms by cos and sin.
func heatLoadAngles(o HeatLoadOptions, gx, gy float64) float64 {
	k := heatLoadCoefficients[o.Equation]
	L := math.Abs(o.Latitude) * math.Pi / 180
	S := math.Atan(math.Hypot(gx, gy))
	asp := math.Mod(math.Atan2(-gx, gy)*180/math.Pi+360, 360)
	var folded float64
	switch south := o.Latitude < 0; {
	case o.Radiation && !south:
		folded = 180 - math.Abs(asp-180)
	case o.Radiation:
		folded = math.Abs(asp - 180)
	case !south:
		folded = math.Abs(180 - math.Abs(asp-225))
	default:
		folded = math.Abs(180 - math.Abs(asp-315))
	}
	A := folded * math.Pi / 180
	v := k[0] + k[1]*math.Cos(L)*math.Cos(S) + k[2]*math.Cos(A)*math.Sin(S)*math.Sin(L) +
		k[3]*math.Sin(L)*math.Sin(S) + k[4]*math.Sin(A)*math.Sin(S) + k[5]*math.Cos(A)*math.Sin(S)
	if o.Linear && o.Equation != HeatLoadEquation3 {
		v = math.Exp(v)
	}
	return v
}

// TestHeatLoadIsTheAngleForm checks the rewrite of the equation as a
// function of the gradient: at every cell of a rough DEM, for every
// equation, fold, hemisphere and scale, HeatLoad is the angle form at
// the gradient Gradient writes, to within the float32 rounding of the
// result (half an ulp, plus the angle form's own float64 error).
func TestHeatLoadIsTheAngleForm(t *testing.T) {
	rng := rand.New(rand.NewPCG(92, 7))
	const w, h = 43, 31
	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range dem.Data {
		dem.Data[i] = float32(300 + 40*rng.NormFloat64())
	}
	for y := 10; y < 16; y++ { // a flat patch
		for x := 5; x < 12; x++ {
			dem.Data[dem.Index(x, y)] = 300
		}
	}
	for _, fit := range []int{0, 3} {
		g := GradientOptions{CellSize: 10, CellSizeY: 14, ZFactor: 0.8, FitRadius: fit}
		dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
		Gradient(dx, dy, dem, g)
		for _, lat := range []float64{0, 23, 47.25, 60, 90, -35, -60} {
			for eq := range 3 {
				for _, radiation := range []bool{false, true} {
					for _, linear := range []bool{false, true} {
						o := HeatLoadOptions{CellSize: g.CellSize, CellSizeY: g.CellSizeY, ZFactor: g.ZFactor, FitRadius: fit,
							Latitude: lat, Equation: HeatLoadEquation(eq), Radiation: radiation, Linear: linear}
						dst := raster.NewFloat32Like(dem)
						HeatLoad(dst, dem, o)
						for y := fit + 1; y < h-fit-1; y++ {
							for x := fit + 1; x < w-fit-1; x++ {
								want := heatLoadAngles(o, float64(dx.Data[dx.Index(x, y)]), float64(dy.Data[dy.Index(x, y)]))
								got := float64(dst.Data[dst.Index(x, y)])
								f := float32(math.Abs(want))
								ulp := float64(math.Nextafter32(f, float32(math.Inf(1))) - f)
								if math.Abs(got-want) > ulp+1e-12 {
									t.Fatalf("%+v: cell (%d, %d) = %v, angle form %v (ulp %v)", o, x, y, got, want, ulp)
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestHeatLoadIsGradientThenEquation checks that HeatLoad is Gradient
// followed by the from-gradient kernel bit for bit, Horn's and the
// fit's, on both backends, with the hazards of weightedOperand: the
// graph's shared-gradient plan runs exactly that.
func TestHeatLoadIsGradientThenEquation(t *testing.T) {
	defer stencil.UseScalar(false)
	for _, scalar := range []bool{false, true} {
		stencil.UseScalar(scalar)
		rng := rand.New(rand.NewPCG(93, 1))
		dem := weightedOperand(rng, 300, 13, 900, true, true)
		for _, fit := range []int{0, 1, 4} {
			o := HeatLoadOptions{CellSize: 5, CellSizeY: 7, FitRadius: fit, Latitude: 51, Equation: HeatLoadEquation2, Linear: true}
			dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
			Gradient(dx, dy, dem, GradientOptions{CellSize: o.CellSize, CellSizeY: o.CellSizeY, FitRadius: fit})
			want := raster.NewFloat32Like(dem)
			copy(want.Valid, dx.Valid)
			for y := range dem.Height {
				stencil.HeatLoadFromGradientRow(want.Data[want.Index(0, y):want.Index(0, y)+dem.Width],
					dx.Data[dx.Index(0, y):dx.Index(0, y)+dem.Width], dy.Data[dy.Index(0, y):dy.Index(0, y)+dem.Width],
					heatLoadTerms(o))
			}
			got := raster.NewFloat32Like(dem)
			HeatLoad(got, dem, o)
			sameCells(t, fmt.Sprintf("scalar=%v fit=%d", scalar, fit), got, want)
		}
	}
}

// TestHeatLoadForms checks that the Tiled and Chunked forms write the
// plain function's bits, and that validity is the gradient's: the same
// erosion as Slope's with the same FitRadius.
func TestHeatLoadForms(t *testing.T) {
	rng := rand.New(rand.NewPCG(94, 3))
	dem := weightedOperand(rng, 70, 41, 600, true, true)
	runs := []engine.Options{
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 7, TileHeight: 3, Workers: 2},
	}
	for _, fit := range []int{0, 2} {
		o := HeatLoadOptions{CellSize: 25, FitRadius: fit, Latitude: -12, Radiation: true}
		want := raster.NewFloat32Like(dem)
		HeatLoad(want, dem, o)
		slope := raster.NewFloat32Like(dem)
		Slope(slope, dem, SlopeOptions{CellSize: 25, FitRadius: fit})
		for y := range dem.Height {
			for x := range dem.Width {
				if want.IsValid(x, y) != slope.IsValid(x, y) {
					t.Fatalf("fit=%d: validity at (%d, %d) = %v, Slope's is %v", fit, x, y, want.IsValid(x, y), slope.IsValid(x, y))
				}
			}
		}
		for _, eo := range runs {
			got := raster.NewFloat32Like(dem)
			if err := HeatLoadTiled(context.Background(), got, dem, o, eo); err != nil {
				t.Fatal(err)
			}
			sameCells(t, fmt.Sprintf("fit=%d tiled %+v", fit, eo), got, want)
			got = raster.NewFloat32Like(dem)
			if err := HeatLoadChunked(context.Background(), engine.NewMemorySink(got), engine.NewMemorySource(dem), o, eo); err != nil {
				t.Fatal(err)
			}
			sameCells(t, fmt.Sprintf("fit=%d chunked %+v", fit, eo), got, want)
		}
	}
}

func TestHeatLoadPanics(t *testing.T) {
	r := raster.NewFloat32(8, 8, make([]float32, 64))
	o := func() raster.Float32Raster { return raster.NewFloat32Like(r) }
	for _, lat := range []float64{math.NaN(), 90.5, -91, math.Inf(1)} {
		mustPanic(t, fmt.Sprintf("latitude %v", lat), func() { HeatLoad(o(), r, HeatLoadOptions{CellSize: 1, Latitude: lat}) })
	}
	for _, eq := range []HeatLoadEquation{-1, 3} {
		mustPanic(t, fmt.Sprintf("equation %d", eq), func() { HeatLoad(o(), r, HeatLoadOptions{CellSize: 1, Equation: eq}) })
	}
	mustPanic(t, "cell size", func() { HeatLoad(o(), r, HeatLoadOptions{Latitude: 40}) })
	mustPanic(t, "fit radius", func() { HeatLoad(o(), r, HeatLoadOptions{CellSize: 1, FitRadius: MaxRadius + 1}) })
}
