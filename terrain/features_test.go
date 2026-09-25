package terrain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// TestRuggednessRadiusPlane checks every measure at every radius on
// integer planes z = a·x + b·y + 100, where every elevation, difference
// and sum is a small exact integer: TPI is 0 (the window averages to its
// centre), roughness is the window's span 2r(|a| + |b|), and Riley's sum
// of squares is (a² + b²)·(2r+1)·Σi² with Σi² = r(r+1)(2r+1)/3 over
// i = -r..r, since the cross terms cancel.
func TestRuggednessRadiusPlane(t *testing.T) {
	for _, ab := range [][2]int{{0, 0}, {1, 0}, {0, -3}, {2, 5}, {-7, 4}} {
		a, b := ab[0], ab[1]
		for r := 1; r <= MaxRadius; r++ {
			d := 2*r + 1
			dem := plane(2*r+4, 2*r+3, float64(a), float64(b), 1, 1)
			sumSq := (a*a + b*b) * d * r * (r + 1) * d / 3
			var sumAbs int
			for j := -r; j <= r; j++ {
				for i := -r; i <= r; i++ {
					sumAbs += abs(a*i + b*j)
				}
			}
			n := float32(d*d - 1)
			want := map[RuggednessType]float32{
				RuggednessTRI:       float32(math.Sqrt(float64(sumSq))),
				RuggednessTRIWilson: float32(sumAbs) / n,
				RuggednessTPI:       0,
				RuggednessRoughness: float32(2 * r * (abs(a) + abs(b))),
			}
			for _, rt := range ruggednessTypes {
				dst := raster.NewFloat32Like(dem)
				Ruggedness(dst, dem, RuggednessOptions{Type: rt, Radius: r})
				for y := range dem.Height {
					for x := range dem.Width {
						got := dst.Data[dst.Index(x, y)]
						border := x < r || y < r || x >= dem.Width-r || y >= dem.Height-r
						switch {
						case border && got == got:
							t.Fatalf("plane %v r=%d type %d: border cell (%d, %d) = %v, want NaN", ab, r, rt, x, y, got)
						case !border && math.Float32bits(got) != math.Float32bits(want[rt]):
							t.Fatalf("plane %v r=%d type %d: cell (%d, %d) = %v, want %v", ab, r, rt, x, y, got, want[rt])
						}
					}
				}
			}
		}
	}
}

func abs(x int) int { return max(x, -x) }

// TestRuggednessRadiusOneIsDefault checks that Radius 0 and 1 are the
// same 3×3 window, on both backends.
func TestRuggednessRadiusOneIsDefault(t *testing.T) {
	defer stencil.UseScalar(false)
	for _, scalar := range []bool{false, true} {
		stencil.UseScalar(scalar)
		rng := rand.New(rand.NewPCG(71, 2))
		dem := weightedOperand(rng, 41, 23, 700, true, true)
		for _, rt := range ruggednessTypes {
			want, got := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
			Ruggedness(want, dem, RuggednessOptions{Type: rt})
			Ruggedness(got, dem, RuggednessOptions{Type: rt, Radius: 1})
			sameCells(t, fmt.Sprintf("scalar=%v type %d", scalar, rt), got, want)
		}
	}
}

// TestRuggednessRadiusValidity checks the border and the validity of a
// radius-r measure against a per-cell reference: a cell is valid iff it
// is r or more from the edge and its whole (2r+1)² window is valid.
func TestRuggednessRadiusValidity(t *testing.T) {
	rng := rand.New(rand.NewPCG(72, 3))
	for _, r := range []int{2, 3, 8} {
		for _, windowed := range []bool{false, true} {
			dem := weightedOperand(rng, 70, 2*r+9, 300, windowed, true)
			dst := raster.NewFloat32(dem.Width, dem.Height, make([]float32, dem.Width*dem.Height))
			dst.Valid = make([]uint64, raster.MaskWords(len(dst.Data)))
			for k := range dst.Valid {
				dst.Valid[k] = rng.Uint64() // stale bits must not survive
			}
			Ruggedness(dst, dem, RuggednessOptions{Type: RuggednessRoughness, Radius: r})
			for y := range dem.Height {
				for x := range dem.Width {
					want := x >= r && y >= r && x < dem.Width-r && y < dem.Height-r
					for dy := -r; want && dy <= r; dy++ {
						for dx := -r; want && dx <= r; dx++ {
							want = dem.IsValid(x+dx, y+dy)
						}
					}
					if got := dst.IsValid(x, y); got != want {
						t.Fatalf("r=%d windowed=%v: cell (%d, %d) valid=%v, want %v", r, windowed, x, y, got, want)
					}
					if v := dst.Data[dst.Index(x, y)]; (x < r || y < r || x >= dem.Width-r || y >= dem.Height-r) && v == v {
						t.Fatalf("r=%d windowed=%v: border cell (%d, %d) = %v, want NaN", r, windowed, x, y, v)
					}
				}
			}
		}
	}
}

// TestRuggednessRadiusForms holds RuggednessTiled and RuggednessChunked
// to Ruggedness at radii above 1, for tilings whose tiles are narrower
// than the window.
func TestRuggednessRadiusForms(t *testing.T) {
	runs := []engine.Options{
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 5, TileHeight: 3, Workers: 2},
		{TileWidth: 64, TileHeight: 1, Workers: 4},
	}
	rng := rand.New(rand.NewPCG(73, 4))
	for _, r := range []int{2, 5, 8} {
		for _, masked := range []bool{false, true} {
			dem := weightedOperand(rng, 53, 37, 900, true, masked)
			for _, rt := range ruggednessTypes {
				o := RuggednessOptions{Type: rt, Radius: r}
				newOut := func() raster.Float32Raster {
					out := raster.NewFloat32(dem.Width, dem.Height, make([]float32, dem.Width*dem.Height))
					if masked {
						out.Valid = make([]uint64, raster.MaskWords(len(out.Data)))
					}
					return out
				}
				want := newOut()
				Ruggedness(want, dem, o)
				for _, eo := range runs {
					id := fmt.Sprintf("r=%d masked=%v type %d %+v", r, masked, rt, eo)
					got := newOut()
					if err := RuggednessTiled(context.Background(), got, dem, o, eo); err != nil {
						t.Fatal(err)
					}
					sameCells(t, id+" tiled", got, want)
					got = newOut()
					if err := RuggednessChunked(context.Background(), engine.NewMemorySink(got), engine.NewMemorySource(dem), o, eo); err != nil {
						t.Fatal(err)
					}
					sameCells(t, id+" chunked", got, want)
				}
			}
		}
	}
}

// featureCase is one output of a Features call and the standalone call
// that must write the same bits.
type featureCase struct {
	op    FeatureOp
	alone func(dst, dem raster.Float32Raster)
}

func featureCases() []featureCase {
	const cs, csy = 12.5, 9
	slope := SlopeOptions{CellSize: cs, CellSizeY: csy, ZFactor: 1.5, Units: SlopePercent}
	aspect := AspectOptions{CellSize: cs, CellSizeY: csy, ZeroForFlat: true}
	shade := HillshadeOptions{CellSize: cs, CellSizeY: csy, Azimuth: 200, Altitude: 30}
	curv := CurvatureOptions{CellSize: cs, CellSizeY: csy, Type: CurvaturePlan}
	cases := []featureCase{
		{slope, func(d, m raster.Float32Raster) { Slope(d, m, slope) }},
		{aspect, func(d, m raster.Float32Raster) { Aspect(d, m, aspect) }},
		{shade, func(d, m raster.Float32Raster) { Hillshade(d, m, shade) }},
		{curv, func(d, m raster.Float32Raster) { Curvature(d, m, curv) }},
	}
	for _, rt := range ruggednessTypes {
		for _, r := range []int{0, 2, 4} {
			o := RuggednessOptions{Type: rt, Radius: r}
			cases = append(cases, featureCase{o, func(d, m raster.Float32Raster) { Ruggedness(d, m, o) }})
		}
	}
	// The quadratic fit at several radii, after the others so that the
	// sets below keep their indices.
	fslope, fcurv := slope, curv
	fslope.FitRadius, fcurv.FitRadius = 2, 3
	faspect, fshade := aspect, shade
	faspect.FitRadius, fshade.FitRadius = 1, 4
	// Heat load, Horn's and fitted, north and south, after the fit.
	heat := HeatLoadOptions{CellSize: cs, CellSizeY: csy, Latitude: 47}
	fheat := HeatLoadOptions{CellSize: cs, CellSizeY: csy, FitRadius: 3, Latitude: -33,
		Equation: HeatLoadEquation3, Radiation: true, Linear: true}
	// Northness, Horn's, and unweighted eastness from the fit.
	north := OrientationOptions{CellSize: cs, CellSizeY: csy}
	feast := OrientationOptions{CellSize: cs, CellSizeY: csy, FitRadius: 2, Component: Eastness, Unweighted: true}
	return append(cases,
		featureCase{fslope, func(d, m raster.Float32Raster) { Slope(d, m, fslope) }},
		featureCase{fcurv, func(d, m raster.Float32Raster) { Curvature(d, m, fcurv) }},
		featureCase{faspect, func(d, m raster.Float32Raster) { Aspect(d, m, faspect) }},
		featureCase{fshade, func(d, m raster.Float32Raster) { Hillshade(d, m, fshade) }},
		featureCase{heat, func(d, m raster.Float32Raster) { HeatLoad(d, m, heat) }},
		featureCase{fheat, func(d, m raster.Float32Raster) { HeatLoad(d, m, fheat) }},
		featureCase{north, func(d, m raster.Float32Raster) { Orientation(d, m, north) }},
		featureCase{feast, func(d, m raster.Float32Raster) { Orientation(d, m, feast) }},
	)
}

// TestFeaturesAreTheStandaloneProducts checks Features' promise: each
// output is the bits its own function writes, Data and validity, for
// sets that mix radii, repeat an operation or ask for one output, on
// both backends, masked or not, windowed or not, for every entry point,
// tiling and worker count. Mixed radii are the case that needs the
// engine's per-output rings: a 3×3 output next to a radius-4 one keeps
// its values out to one cell from the edge.
func TestFeaturesAreTheStandaloneProducts(t *testing.T) {
	const w, h = 61, 27
	cases := featureCases()
	sets := [][]int{
		{0},                   // one 3×3 output
		{6},                   // one radius-4 output
		{0, 1, 2, 3},          // the 3×3 derivatives
		{4, 5, 6},             // TRI at three radii
		{12, 0, 13, 8, 15, 2}, // mixed radii, out of order
		{5, 5},                // one operation twice
		{16, 0, 17, 13},       // fitted slope and curvature next to Horn's and ruggedness
		{20, 0, 21, 6},        // heat load, Horn's and fitted, next to slope and TRI
		{22, 23, 1, 20},       // northness and fitted eastness next to aspect and heat load
		func() []int { // everything
			all := make([]int, len(cases))
			for i := range all {
				all[i] = i
			}
			return all
		}(),
	}
	runs := []engine.Options{
		{Workers: 1},
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 5, TileHeight: 3, Workers: 2},
	}
	defer stencil.UseScalar(false)
	for _, scalar := range []bool{false, true} {
		stencil.UseScalar(scalar)
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(74, 5))
				dem := weightedOperand(rng, w, h, 800, windowed, masked)
				for y := 3; y < 9; y++ { // a flat patch, for aspect and curvature
					for x := 10; x < 20; x++ {
						dem.Data[dem.Index(x, y)] = 812
					}
				}
				newOut := func() raster.Float32Raster {
					out := raster.NewFloat32(w, h, make([]float32, w*h))
					if masked {
						out.Valid = make([]uint64, raster.MaskWords(w*h))
						for k := range out.Valid {
							out.Valid[k] = rng.Uint64()
						}
					}
					return out
				}
				want := make([]raster.Float32Raster, len(cases))
				for i, c := range cases {
					want[i] = newOut()
					c.alone(want[i], dem)
				}
				for _, set := range sets {
					id := fmt.Sprintf("scalar=%v masked=%v windowed=%v set=%v", scalar, masked, windowed, set)
					outputs := func() []Feature {
						out := make([]Feature, len(set))
						for k, i := range set {
							out[k] = Feature{Op: cases[i].op, Dst: newOut()}
						}
						return out
					}
					check := func(how string, out []Feature) {
						t.Helper()
						for k, i := range set {
							sameCells(t, fmt.Sprintf("%s %s output %d (case %d)", id, how, k, i), out[k].Dst, want[i])
						}
					}
					out := outputs()
					Features(out, dem)
					check("plain", out)
					for _, eo := range runs {
						out := outputs()
						if err := FeaturesTiled(context.Background(), out, dem, eo); err != nil {
							t.Fatal(err)
						}
						check(fmt.Sprintf("tiled %+v", eo), out)
						out = outputs()
						sinks := make([]FeatureSink, len(out))
						for k, f := range out {
							sinks[k] = FeatureSink{Op: f.Op, Dst: engine.NewMemorySink(f.Dst)}
						}
						if err := FeaturesChunked(context.Background(), sinks, engine.NewMemorySource(dem), eo); err != nil {
							t.Fatal(err)
						}
						check(fmt.Sprintf("chunked %+v", eo), out)
					}
				}
			}
		}
	}
}

// TestFeaturesCancel checks that a cancelled context stops Features
// with ctx.Err().
func TestFeaturesCancel(t *testing.T) {
	dem := raster.NewFloat32(64, 64, make([]float32, 64*64))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := []Feature{{Op: RuggednessOptions{Radius: 3}, Dst: raster.NewFloat32Like(dem)}}
	if err := FeaturesTiled(ctx, out, dem, engine.Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("FeaturesTiled with a cancelled context = %v, want context.Canceled", err)
	}
}

func TestFeaturesPanics(t *testing.T) {
	r := raster.NewFloat32(20, 20, make([]float32, 400))
	o := func() raster.Float32Raster { return raster.NewFloat32Like(r) }
	tpi := RuggednessOptions{Type: RuggednessTPI, Radius: 3}
	mustPanic(t, "no outputs", func() { Features(nil, r) })
	mustPanic(t, "nil op", func() { Features([]Feature{{Dst: o()}}, r) })
	mustPanic(t, "output is the DEM", func() { Features([]Feature{{Op: tpi, Dst: r}}, r) })
	shared := o()
	mustPanic(t, "outputs share memory", func() {
		Features([]Feature{{Op: tpi, Dst: shared}, {Op: SlopeOptions{CellSize: 1}, Dst: shared}}, r)
	})
	mustPanic(t, "size", func() { Features([]Feature{{Op: tpi, Dst: raster.NewFloat32(19, 20, make([]float32, 380))}}, r) })
	mustPanic(t, "cell size", func() { Features([]Feature{{Op: SlopeOptions{}, Dst: o()}}, r) })
	mustPanic(t, "radius too large", func() {
		Features([]Feature{{Op: RuggednessOptions{Radius: MaxRadius + 1}, Dst: o()}}, r)
	})
	mustPanic(t, "no sinks", func() {
		_ = FeaturesChunked(context.Background(), nil, engine.NewMemorySource(r), engine.Options{})
	})
}

func TestRuggednessRadiusPanics(t *testing.T) {
	r := raster.NewFloat32(20, 20, make([]float32, 400))
	for _, radius := range []int{-1, MaxRadius + 1} {
		mustPanic(t, fmt.Sprintf("radius %d", radius), func() {
			Ruggedness(raster.NewFloat32Like(r), r, RuggednessOptions{Radius: radius})
		})
	}
}

// TestRoughnessSeparableIsBruteForce holds roughness at radii above 1,
// which runs focal's separable Max and Min, to the brute-force window
// kernel it replaces, bit for bit, on both focal backends, over NaN,
// ±Inf and runs of mixed signed zeros, at widths either side of a block.
func TestRoughnessSeparableIsBruteForce(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	defer focalrow.UseScalar(false)
	for _, scalar := range []bool{false, true} {
		focalrow.UseScalar(scalar)
		rng := rand.New(rand.NewPCG(75, 6))
		for _, w := range []int{17, roughnessBlock + 2*MaxRadius + 1, 2*roughnessBlock + 40} {
			dem := weightedOperand(rng, w, 2*MaxRadius+6, 50, true, false)
			for y := range dem.Height {
				for x := range 12 {
					dem.Data[dem.Index(x, y)] = [2]float32{0, negZero}[rng.IntN(2)]
				}
			}
			for r := 2; r <= MaxRadius; r++ {
				got := raster.NewFloat32Like(dem)
				Ruggedness(got, dem, RuggednessOptions{Type: RuggednessRoughness, Radius: r})
				rows := make([][]float32, 2*r+1)
				want := make([]float32, w-2*r)
				for y := r; y < dem.Height-r; y++ {
					for j := range rows {
						rows[j] = dem.Row(y - r + j)
					}
					stencil.RuggednessWindowRow(want, rows, stencil.RugRoughness)
					for i, v := range want {
						if g := got.Data[got.Index(i+r, y)]; !sameBits(g, v) {
							t.Fatalf("scalar=%v w=%d r=%d cell (%d, %d) = %v (%#x), want %v (%#x)",
								scalar, w, r, i+r, y, g, math.Float32bits(g), v, math.Float32bits(v))
						}
					}
				}
			}
		}
	}
}
