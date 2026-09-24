package graph

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/transfer"
)

// A graph's promise is that of every fused form in the engine (DESIGN.md
// §52, §55): each output is the bits the separate public calls write,
// Data and validity, and each summary is reduce.Stats of that raster, for
// every plan choice, entry point, tiling and worker count. The separate
// calls are the reference; each has its own, outside strata.

const w, h = 61, 37

var runs = []engine.Options{
	{Workers: 1},
	{TileWidth: 16, TileHeight: 16, Workers: 3},
	{TileWidth: 7, TileHeight: 5, Workers: 2},
	{TileWidth: 64, TileHeight: 1},
}

// operand is a random raster around base, with invalid cells when masked.
func operand(rng *rand.Rand, base float32, masked bool) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = base + 40*rng.Float32()
	}
	if masked {
		r.Valid = raster.NewMask(w * h)
		for y := range h {
			for x := range w {
				r.SetValid(x, y, rng.IntN(13) != 0)
			}
		}
	}
	return r
}

func blank(masked bool) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	if masked {
		r.Valid = raster.NewMask(w * h)
	}
	return r
}

func sameCells(t *testing.T, id string, got, want raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			valid := want.Valid == nil || want.IsValid(x, y)
			if want.Valid != nil && got.IsValid(x, y) != valid {
				t.Fatalf("%s: validity at (%d, %d) = %v, want %v", id, x, y, got.IsValid(x, y), valid)
			}
			g, v := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]
			if valid && math.Float32bits(g) != math.Float32bits(v) && !(g != g && v != v) {
				t.Fatalf("%s: Data at (%d, %d) = %v, want %v", id, x, y, g, v)
			}
		}
	}
}

func sameSummary(t *testing.T, id string, got, want reduce.Summary) {
	t.Helper()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("%s: summary %+v, want %+v", id, got, want)
	}
}

// check runs p in memory and chunked with every Options, and compares
// each output with want and each summary with stats.
func check(t *testing.T, id string, p *Plan, in map[string]raster.Float32Raster,
	want map[string]raster.Float32Raster, stats map[string]reduce.Summary, masked bool) {
	t.Helper()
	for _, eo := range runs {
		out := map[string]raster.Float32Raster{}
		for name := range want {
			out[name] = blank(masked)
		}
		res, err := p.Run(context.Background(), in, out, eo)
		if err != nil {
			t.Fatal(err)
		}
		how := fmt.Sprintf("%s masked=%v Run %+v", id, masked, eo)
		for name, r := range want {
			sameCells(t, how+" output "+name, out[name], r)
		}
		for name, s := range stats {
			sameSummary(t, how+" stats "+name, res.Stats[name], s)
		}

		srcs := map[string]engine.RasterSource{}
		for name, r := range in {
			srcs[name] = engine.NewMemorySource(r)
		}
		sinks := map[string]engine.RasterSink{}
		for name := range want {
			out[name] = blank(masked)
			sinks[name] = engine.NewMemorySink(out[name])
		}
		tmp := t.TempDir()
		res, err = p.RunChunked(context.Background(), srcs, sinks, ChunkedOptions{Engine: eo, TempDir: tmp})
		if err != nil {
			t.Fatal(err)
		}
		if left, _ := os.ReadDir(tmp); len(left) != 0 {
			t.Fatalf("%s: RunChunked left %d files in TempDir", id, len(left))
		}
		how = fmt.Sprintf("%s masked=%v RunChunked %+v", id, masked, eo)
		for name, r := range want {
			sameCells(t, how+" output "+name, out[name], r)
		}
		for name, s := range stats {
			sameSummary(t, how+" stats "+name, res.Stats[name], s)
		}
	}
}

// TestTerrainStack is "one read, every product": every derivative of one
// DEM, and a summary of one of them, from one pass that reads the DEM
// once.
func TestTerrainStack(t *testing.T) {
	sl := terrain.SlopeOptions{CellSize: 12.5, CellSizeY: 9, ZFactor: 1.5, Units: terrain.SlopePercent}
	as := terrain.AspectOptions{CellSize: 12.5, CellSizeY: 9, ZFactor: 1.5, ZeroForFlat: true}
	hs := terrain.HillshadeOptions{CellSize: 12.5, CellSizeY: 9, ZFactor: 1.5, Azimuth: 200, Altitude: 30}
	cu := terrain.CurvatureOptions{CellSize: 12.5}
	ru := terrain.RuggednessOptions{}
	g := New()
	dem := g.Input("dem")
	slope := Slope(dem, sl)
	g.Output("slope", slope)
	g.Output("aspect", Aspect(dem, as))
	g.Output("hillshade", Hillshade(dem, hs))
	g.Output("curvature", Curvature(dem, cu))
	g.Output("tri", Ruggedness(dem, ru))
	g.Stats("slope", slope)
	p := g.Plan(PlanOptions{})
	if len(p.passes) != 1 || !strings.Contains(p.String(), `input "dem" is read 1 time(s)`) {
		t.Fatalf("want one pass reading the DEM once:\n%v", p)
	}
	if !strings.Contains(p.String(), "is computed once for every product") {
		t.Fatalf("want slope, aspect and hillshade to share a gradient:\n%v", p)
	}
	for _, masked := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(1, 2))
		d := operand(rng, 800, masked)
		want := map[string]raster.Float32Raster{}
		for _, name := range []string{"slope", "aspect", "hillshade", "curvature", "tri"} {
			want[name] = blank(masked)
		}
		terrain.Slope(want["slope"], d, sl)
		terrain.Aspect(want["aspect"], d, as)
		terrain.Hillshade(want["hillshade"], d, hs)
		terrain.Curvature(want["curvature"], d, cu)
		terrain.Ruggedness(want["tri"], d, ru)
		check(t, "terrain stack", p, map[string]raster.Float32Raster{"dem": d}, want,
			map[string]reduce.Summary{"slope": reduce.Stats(want["slope"])}, masked)
	}
}

// TestLoneProductRunsItsOwnKernel checks the gradient rewrite: a product
// that is its gradient's only reader runs the standalone kernel, with no
// gradient stage.
func TestLoneProductRunsItsOwnKernel(t *testing.T) {
	g := New()
	g.Output("slope", Slope(g.Input("dem"), terrain.SlopeOptions{CellSize: 10}))
	p := g.Plan(PlanOptions{})
	if s := p.String(); strings.Contains(s, "Gradient") || len(p.passes[0].stages) != 1 {
		t.Fatalf("want one Slope stage and no gradient:\n%v", s)
	}
	for _, masked := range []bool{false, true} {
		d := operand(rand.New(rand.NewPCG(3, 4)), 100, masked)
		want := blank(masked)
		terrain.Slope(want, d, terrain.SlopeOptions{CellSize: 10})
		check(t, "lone slope", p, map[string]raster.Float32Raster{"dem": d},
			map[string]raster.Float32Raster{"slope": want}, nil, masked)
	}
}

// TestOverlay is "overlay to one output": several layers reclassified,
// rescaled, weighted, summed and masked, as one pass with one output,
// and a summary of the result.
func TestOverlay(t *testing.T) {
	breaks, values := []float32{810, 820, 830}, []float32{1, 2, 3, 4}
	g := New()
	a, b, m := g.Input("a"), g.Input("b"), g.Input("mask")
	ra := Rescale(Reclass(a, breaks, values), 0.25, 0)
	rb := Lookup(b, []float32{500, 540}, []float32{0, 1})
	sum := Add(Mul(ra, Clamp(rb, 0.2, 0.8)), Min(ra, rb))
	out := Mask(Max(sum, RescaleRange(a, 800, 840, 0, 1)), m)
	g.Output("suitability", out)
	g.Stats("suitability", out)
	p := g.Plan(PlanOptions{})
	if len(p.passes) != 1 {
		t.Fatalf("want one pass:\n%v", p)
	}
	for _, masked := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(5, 6))
		ai, bi, mi := operand(rng, 800, masked), operand(rng, 500, masked), operand(rng, 0, true)
		t1, t2, t3, t4, t5 := blank(masked), blank(masked), blank(masked), blank(masked), blank(masked)
		transfer.Reclass(t1, ai, breaks, values)
		transfer.Rescale(t1, t1, 0.25, 0)
		transfer.Lookup(t2, bi, []float32{500, 540}, []float32{0, 1})
		algebra.Clamp(t3, t2, 0.2, 0.8)
		algebra.Mul(t3, t1, t3)
		algebra.Min(t4, t1, t2)
		algebra.Add(t3, t3, t4)
		transfer.RescaleRange(t5, ai, 800, 840, 0, 1)
		algebra.Max(t3, t3, t5)
		want := blank(true)
		algebra.Mask(want, t3, mi)
		check(t, "overlay", p, map[string]raster.Float32Raster{"a": ai, "b": bi, "mask": mi},
			map[string]raster.Float32Raster{"suitability": want},
			map[string]reduce.Summary{"suitability": reduce.Stats(want)}, true)
	}
}

// TestStatsOnly is "straight to the answer": a chain that ends in
// statistics writes no raster.
func TestStatsOnly(t *testing.T) {
	g := New()
	dem := g.Input("dem")
	g.Stats("slope", Slope(dem, terrain.SlopeOptions{CellSize: 5}))
	g.Stats("dem", dem)
	p := g.Plan(PlanOptions{})
	for _, masked := range []bool{false, true} {
		d := operand(rand.New(rand.NewPCG(7, 8)), 300, masked)
		sl := blank(masked)
		terrain.Slope(sl, d, terrain.SlopeOptions{CellSize: 5})
		check(t, "stats only", p, map[string]raster.Float32Raster{"dem": d}, nil,
			map[string]reduce.Summary{"slope": reduce.Stats(sl), "dem": reduce.Stats(d)}, masked)
	}
}

// TestNormalizeBoundary checks a global operation under every boundary
// choice: the range found in one pass, the map in the next, and the
// result algebra.Normalize's.
func TestNormalizeBoundary(t *testing.T) {
	// The shapes of Normalize's input, and whether BoundaryAuto stores it:
	// one raster through per-cell steps is recomputed, two rasters or a
	// neighbourhood operation are stored.
	shapes := []struct {
		name   string
		stored bool
	}{{"one raster", false}, {"two rasters", true}, {"stencil", true}}
	for _, bnd := range []Boundary{BoundaryAuto, BoundaryRecompute, BoundaryCache} {
		for _, shape := range shapes {
			g := New()
			dem, wt := g.Input("dem"), g.Input("w")
			var x Node
			switch shape.name {
			case "one raster":
				x = Rescale(dem, 3, -2)
			case "two rasters":
				x = Mul(dem, wt)
			default:
				x = Slope(dem, terrain.SlopeOptions{CellSize: 10})
			}
			n := Normalize(x)
			g.Output("norm", n)
			g.Output("x", x)
			g.Output("scaled", Rescale(n, 2, 1))
			p := g.Plan(PlanOptions{Boundary: bnd})
			stored := strings.Contains(p.String(), "stored for the next pass")
			if want := bnd == BoundaryCache || bnd == BoundaryAuto && shape.stored; stored != want {
				t.Fatalf("boundary %v, %s: stored = %v:\n%v", bnd, shape.name, stored, p)
			}
			for _, masked := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(9, 10))
				d, wi := operand(rng, 100, masked), operand(rng, 1, masked)
				xr, nr, sr := blank(masked), blank(masked), blank(masked)
				in := map[string]raster.Float32Raster{"dem": d}
				switch shape.name {
				case "one raster":
					transfer.Rescale(xr, d, 3, -2)
				case "two rasters":
					algebra.Mul(xr, d, wi)
					in["w"] = wi
				default:
					terrain.Slope(xr, d, terrain.SlopeOptions{CellSize: 10})
				}
				algebra.Normalize(nr, xr)
				transfer.Rescale(sr, nr, 2, 1)
				check(t, fmt.Sprintf("normalize %v %s", bnd, shape.name), p, in,
					map[string]raster.Float32Raster{"norm": nr, "x": xr, "scaled": sr}, nil, masked)
			}
		}
	}
}

// TestAlonePass checks an operation that cannot be a stage: focal.Mean
// asks for scratch, so it runs as a pass of its own over a stored input,
// and a later fused pass reads its result.
func TestAlonePass(t *testing.T) {
	box := focal.BoxOptions{Radius: 2}
	g := New()
	a, b := g.Input("a"), g.Input("b")
	mean := FocalMean(Mul(a, b), box)
	g.Output("out", Sub(mean, a))
	g.Output("raw", FocalMax(a, box))
	p := g.Plan(PlanOptions{})
	if s := p.String(); !strings.Contains(s, "cannot be a pipeline stage") || strings.Count(s, "(alone,") != 2 {
		t.Fatalf("want two passes of their own and a stored product:\n%v", s)
	}
	for _, masked := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(11, 12))
		ai, bi := operand(rng, 10, masked), operand(rng, 2, masked)
		prod, mr, want, raw := blank(masked), blank(masked), blank(masked), blank(masked)
		algebra.Mul(prod, ai, bi)
		focal.Mean(mr, prod, box)
		algebra.Sub(want, mr, ai)
		focal.Max(raw, ai, box)
		check(t, "alone", p, map[string]raster.Float32Raster{"a": ai, "b": bi},
			map[string]raster.Float32Raster{"out": want, "raw": raw}, nil, masked)
	}
}

// TestOutputsThePipelineCannotWrite checks the copies: an input written
// as an output, a value written under two names, and a value that is an
// output and also read by a neighbourhood stage in the same pass.
func TestOutputsThePipelineCannotWrite(t *testing.T) {
	g := New()
	a, b := g.Input("a"), g.Input("b")
	x := Mul(a, b)
	g.Output("a", a)
	g.Output("x", x)
	g.Output("x2", x)
	g.Output("slope", Slope(x, terrain.SlopeOptions{CellSize: 3}))
	g.Output("conv", Convolve(a, focal.WeightsOptions{Radius: 1, Weights: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}}))
	p := g.Plan(PlanOptions{})
	for _, masked := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(13, 14))
		ai, bi := operand(rng, 10, masked), operand(rng, 2, masked)
		xr, sr, cr := blank(masked), blank(masked), blank(masked)
		algebra.Mul(xr, ai, bi)
		terrain.Slope(sr, xr, terrain.SlopeOptions{CellSize: 3})
		focal.Convolve(cr, ai, focal.WeightsOptions{Radius: 1, Weights: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}})
		check(t, "copies", p, map[string]raster.Float32Raster{"a": ai, "b": bi},
			map[string]raster.Float32Raster{"a": ai, "x": xr, "x2": xr, "slope": sr, "conv": cr}, nil, masked)
	}
}

// TestCommonSubexpressions checks that building the same operation twice
// gives one node.
func TestCommonSubexpressions(t *testing.T) {
	g := New()
	a := g.Input("a")
	first, second := Slope(a, terrain.SlopeOptions{CellSize: 1}), Slope(a, terrain.SlopeOptions{CellSize: 1})
	if first != second {
		t.Fatal("the same Slope twice gave two nodes")
	}
	if Slope(a, terrain.SlopeOptions{CellSize: 1}) == Slope(a, terrain.SlopeOptions{CellSize: 2}) {
		t.Fatal("Slopes with different options gave one node")
	}
	if g.Input("a") != a {
		t.Fatal("the same input twice gave two nodes")
	}
}
