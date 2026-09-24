package graph

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/resample"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/transfer"
)

// Grids in the graph (DESIGN.md §55): values on declared grids, Resample
// as a grid change between fused passes. The reference is the same as
// everywhere in the package, the separate public calls, here with
// resample.Resample among them.

// fine is a 10 m grid of w×h cells, coarse a 20 m grid over most of it,
// and shifted a 7 m grid offset from both, reaching past fine's edge.
var (
	fine    = raster.Grid{Width: w, Height: h, ResolutionX: 10, ResolutionY: -10, OriginX: 1000, OriginY: 2000, CRS: raster.CRS{Code: "EPSG:25833"}}
	coarse  = raster.Grid{Width: 29, Height: 17, ResolutionX: 20, ResolutionY: -20, OriginX: 1010, OriginY: 1990}
	shifted = raster.Grid{Width: 90, Height: 50, ResolutionX: 7, ResolutionY: -7, OriginX: 1003, OriginY: 1996}
)

func blankOn(g raster.Grid, masked bool) raster.Float32Raster {
	r := raster.NewFloat32(g.Width, g.Height, make([]float32, g.Width*g.Height))
	if masked {
		r.Valid = raster.NewMask(g.Width * g.Height)
	}
	return r
}

func operandOn(rng *rand.Rand, g raster.Grid, base float32, masked bool) raster.Float32Raster {
	r := blankOn(g, masked)
	for i := range r.Data {
		r.Data[i] = base + 40*rng.Float32()
	}
	if masked {
		for y := range g.Height {
			for x := range g.Width {
				r.SetValid(x, y, rng.IntN(13) != 0)
			}
		}
	}
	return r
}

// resampled is resample.Resample of src (on sg) onto dg, into a masked
// raster.
func resampled(dg, sg raster.Grid, src raster.Float32Raster, m resample.Method) raster.Float32Raster {
	out := blankOn(dg, true)
	resample.Resample(raster.NewDataset(dg, out), raster.NewDataset(sg, src), resample.Options{Method: m})
	return out
}

// checkGrids is check for outputs of any size: each output is allocated
// like its want, masked where want is.
func checkGrids(t *testing.T, id string, p *Plan, in map[string]raster.Float32Raster,
	want map[string]raster.Float32Raster, stats map[string]reduce.Summary) {
	t.Helper()
	like := func(r raster.Float32Raster) raster.Float32Raster {
		o := raster.NewFloat32(r.Width, r.Height, make([]float32, r.Width*r.Height))
		if r.Valid != nil {
			o.Valid = raster.NewMask(r.Width * r.Height)
		}
		return o
	}
	for _, eo := range runs {
		out := map[string]raster.Float32Raster{}
		for name, r := range want {
			out[name] = like(r)
		}
		res, err := p.Run(context.Background(), in, out, eo)
		if err != nil {
			t.Fatal(err)
		}
		how := fmt.Sprintf("%s Run %+v", id, eo)
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
		for name, r := range want {
			out[name] = like(r)
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
		how = fmt.Sprintf("%s RunChunked %+v", id, eo)
		for name, r := range want {
			sameCells(t, how+" output "+name, out[name], r)
		}
		for name, s := range stats {
			sameSummary(t, how+" stats "+name, res.Stats[name], s)
		}
	}
}

// TestResampleThenTerrain: a DEM resampled to a coarser grid, and a
// terrain stack on that grid, with the resampled DEM written too and a
// summary of it. The resampling is a pass of its own; the stack is one
// fused pass on the coarse grid reading it stored.
func TestResampleThenTerrain(t *testing.T) {
	sl := terrain.SlopeOptions{CellSize: 20}
	hs := terrain.HillshadeOptions{CellSize: 20, Azimuth: 200}
	for _, masked := range []bool{false, true} {
		g := New()
		dem := g.InputOn("dem", fine)
		c := Resample(dem, coarse, resample.Options{Method: resample.Average})
		g.Output("coarse", c)
		g.Stats("coarse", c)
		g.Output("slope", Slope(c, sl))
		g.Output("hillshade", Hillshade(c, hs))
		p := g.Plan(PlanOptions{})
		s := p.String()
		if len(p.passes) != 2 || p.passes[0].kind != passAlone || p.passes[1].grid != p.passes[0].grid ||
			!strings.Contains(s, "resample.Resample(Method=Average)") || !strings.Contains(s, "grid 2: 29×17 from (1010, 1990) by (20, -20)") {
			t.Fatalf("want a resampling pass and one fused pass on the coarse grid:\n%s", s)
		}
		d := operandOn(rand.New(rand.NewPCG(7, 8)), fine, 800, masked)
		rc := resampled(coarse, fine, d, resample.Average)
		if !masked {
			// Average over a source without a mask that covers the grid
			// leaves every cell valid: the separate calls need no mask.
			rc.Valid = nil
		}
		want := map[string]raster.Float32Raster{"coarse": rc, "slope": blankOn(coarse, masked), "hillshade": blankOn(coarse, masked)}
		terrain.Slope(want["slope"], rc, sl)
		terrain.Hillshade(want["hillshade"], rc, hs)
		checkGrids(t, fmt.Sprintf("resample then terrain, masked=%v", masked), p,
			map[string]raster.Float32Raster{"dem": d}, want, map[string]reduce.Summary{"coarse": reduce.Stats(rc)})
	}
}

// TestChangeDetection is the change-detection chain across grids: a
// second date on a shifted 7 m grid, resampled onto the first date's
// grid, differenced, normalised and thresholded, with statistics of the
// difference. The first date's grid reaches past the second's, so the
// resampled values have invalid cells even when neither date has any.
func TestChangeDetection(t *testing.T) {
	for _, b := range []Boundary{BoundaryAuto, BoundaryRecompute, BoundaryCache} {
		for i, m := range []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic} {
			afterMasked := i != 1
			g := New()
			before := g.InputOn("before", fine)
			after := g.InputOn("after", shifted)
			ag, ok := before.Grid()
			if !ok || ag != fine {
				t.Fatalf("Grid() = %+v, %v", ag, ok)
			}
			diff := Sub(Resample(after, ag, resample.Options{Method: m}), before)
			g.Stats("diff", diff)
			g.Output("change", Clamp(Normalize(diff), 0.25, 0.75))
			p := g.Plan(PlanOptions{Boundary: b})
			rng := rand.New(rand.NewPCG(9, 10))
			bi, ai := operandOn(rng, fine, 800, false), operandOn(rng, shifted, 805, afterMasked)
			ra := resampled(fine, shifted, ai, m)
			d := blankOn(fine, true)
			algebra.Sub(d, ra, bi)
			n := blankOn(fine, true)
			algebra.Normalize(n, d)
			change := blankOn(fine, true)
			algebra.Clamp(change, n, 0.25, 0.75)
			checkGrids(t, fmt.Sprintf("change detection %v %v, after masked=%v", b, m, afterMasked), p,
				map[string]raster.Float32Raster{"before": bi, "after": ai},
				map[string]raster.Float32Raster{"change": change},
				map[string]reduce.Summary{"diff": reduce.Stats(d)})
		}
	}
}

// TestPassPerGrid: products of one phase on two grids are two fused
// passes, one per grid, and a value on the undeclared grid next to them a
// third.
func TestPassPerGrid(t *testing.T) {
	g := New()
	dem := g.InputOn("dem", fine)
	other := g.Input("other")
	g.Output("fine", Rescale(dem, 2, 1))
	g.Output("coarse", Rescale(Resample(dem, coarse, resample.Options{Method: resample.Bilinear}), 2, 1))
	g.Output("undeclared", Rescale(other, 2, 1))
	p := g.Plan(PlanOptions{})
	grids := map[int]bool{}
	for _, ps := range p.passes {
		if ps.kind == passFused {
			grids[ps.grid] = true
		}
	}
	if len(p.passes) != 4 || len(grids) != 3 {
		t.Fatalf("want a resampling pass and three fused passes, one per grid:\n%v", p)
	}
	rng := rand.New(rand.NewPCG(11, 12))
	d, o := operandOn(rng, fine, 100, false), operandOn(rng, raster.Grid{Width: 5, Height: 3}, 0, false)
	rc := resampled(coarse, fine, d, resample.Bilinear)
	want := map[string]raster.Float32Raster{"fine": blankOn(fine, false), "coarse": blankOn(coarse, true), "undeclared": blankOn(raster.Grid{Width: 5, Height: 3}, false)}
	transfer.Rescale(want["fine"], d, 2, 1)
	transfer.Rescale(want["coarse"], rc, 2, 1)
	transfer.Rescale(want["undeclared"], o, 2, 1)
	checkGrids(t, "pass per grid", p, map[string]raster.Float32Raster{"dem": d, "other": o}, want, nil)
}

// TestGridChecks: values on different grids cannot be combined, a
// Resample needs a declared source grid, and bound rasters must have
// their grids' sizes; each panics where the mistake is.
func TestGridChecks(t *testing.T) {
	panics := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			t.Helper()
			v := recover()
			s, _ := v.(string)
			if !strings.HasPrefix(s, "graph: ") || !strings.Contains(s, want) {
				t.Errorf("%s: panic %v, want a graph: panic mentioning %q", name, v, want)
			}
		}()
		f()
	}
	g := New()
	a, b, u := g.InputOn("a", fine), g.InputOn("b", coarse), g.Input("u")
	panics("mixed grids", "different grids: grid 61×37", func() { Sub(a, b) })
	panics("declared and undeclared", "the undeclared grid", func() { Mask(a, u) })
	panics("resample of undeclared", "declare its input's grid with InputOn", func() { Resample(u, fine, resample.Options{}) })
	panics("redeclared", `input "a" is declared on grid 61×37`, func() { g.InputOn("a", coarse) })
	panics("bad grid", "sizes must be positive", func() { g.InputOn("z", raster.Grid{ResolutionX: 1, ResolutionY: 1}) })
	other := coarse
	other.CRS.Code = "EPSG:4326"
	panics("crs", "reprojection is not supported", func() { Resample(a, other, resample.Options{}) })
	panics("method", "unknown", func() { Resample(a, coarse, resample.Options{Method: 9}) })

	// Same geometry, a CRS on one side only: one grid.
	same := fine
	same.CRS = raster.CRS{}
	Sub(a, g.InputOn("c", same))
	// Resampled onto a's grid, b combines with a.
	sum := Add(a, Resample(b, fine, resample.Options{}))
	if gr, ok := sum.Grid(); !ok || gr != fine {
		t.Fatalf("sum lies on %+v", gr)
	}

	h := New()
	x := h.InputOn("x", fine)
	h.Output("y", Resample(x, coarse, resample.Options{}))
	p := h.Plan(PlanOptions{})
	panics("input size", `input "x" is 5×5, but it is declared on grid 61×37`, func() {
		_, _ = p.Run(context.Background(), map[string]raster.Float32Raster{"x": blankOn(raster.Grid{Width: 5, Height: 5}, false)},
			map[string]raster.Float32Raster{"y": blankOn(coarse, true)}, engine.Options{})
	})
	panics("output size", `output "y" is 61×37, but its value lies on grid 29×17`, func() {
		_, _ = p.RunChunked(context.Background(), map[string]engine.RasterSource{"x": engine.NewMemorySource(blankOn(fine, false))},
			map[string]engine.RasterSink{"y": engine.NewMemorySink(blankOn(fine, true))}, ChunkedOptions{})
	})
}

// TestCommonResample: the same Resample of the same value twice is one
// node, and one pass.
func TestCommonResample(t *testing.T) {
	g := New()
	a := g.InputOn("a", fine)
	r1 := Resample(a, coarse, resample.Options{Method: resample.Cubic})
	r2 := Resample(a, coarse, resample.Options{Method: resample.Cubic})
	r3 := Resample(a, coarse, resample.Options{Method: resample.Bilinear})
	if r1 != r2 || r1 == r3 {
		t.Fatalf("Resample nodes %v %v %v: want the first two equal and the third not", r1, r2, r3)
	}
}
