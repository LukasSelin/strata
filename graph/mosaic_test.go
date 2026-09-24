package graph

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/resample"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/transfer"
)

// Mosaics in the graph (DESIGN.md §55): resample.Mosaic as a grid change,
// held to the separate calls like every other node.

// Three tiles over and around fine (10 m, 61×37 from (1000, 2000)): the
// western part at 10 m on fine's lattice, the east at 20 m offset by half
// a cell, and a 7 m strip across the middle, over both.
var (
	tileWest   = raster.Grid{Width: 36, Height: 40, ResolutionX: 10, ResolutionY: -10, OriginX: 990, OriginY: 2010}
	tileEast   = raster.Grid{Width: 17, Height: 20, ResolutionX: 20, ResolutionY: -20, OriginX: 1285, OriginY: 2005}
	tileMiddle = raster.Grid{Width: 60, Height: 12, ResolutionX: 7, ResolutionY: -7, OriginX: 1150, OriginY: 1880}
)

// mosaicked is resample.Mosaic of srcs onto dg, into a masked raster.
func mosaicked(dg raster.Grid, srcs []raster.Dataset, m resample.Method) raster.Float32Raster {
	out := blankOn(dg, true)
	resample.Mosaic(raster.NewDataset(dg, out), srcs, resample.Options{Method: m})
	return out
}

// TestMosaicThenTerrain: three tiles at mixed resolutions laid onto one
// grid, written, summarised, and fed to a terrain stack on that grid.
// The mosaic is one pass reading every tile; the stack one fused pass.
func TestMosaicThenTerrain(t *testing.T) {
	sl := terrain.SlopeOptions{CellSize: 10}
	hs := terrain.HillshadeOptions{CellSize: 10, Azimuth: 135}
	for _, m := range []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic, resample.Average} {
		for _, masked := range []bool{false, true} {
			g := New()
			ins := []Node{g.InputOn("west", tileWest), g.InputOn("east", tileEast), g.InputOn("middle", tileMiddle)}
			dem := Mosaic(ins, fine, resample.Options{Method: m})
			g.Output("dem", dem)
			g.Stats("dem", dem)
			g.Output("slope", Slope(dem, sl))
			g.Output("hillshade", Hillshade(dem, hs))
			p := g.Plan(PlanOptions{})
			if s := p.String(); len(p.passes) != 2 || len(p.passes[0].sources) != 3 || !strings.Contains(s, "resample.Mosaic(Method=") {
				t.Fatalf("want one mosaic pass over three tiles and one fused pass:\n%s", s)
			}
			rng := rand.New(rand.NewPCG(13, 14))
			w, e, mi := operandOn(rng, tileWest, 800, masked), operandOn(rng, tileEast, 820, false), operandOn(rng, tileMiddle, 810, masked)
			d := mosaicked(fine, []raster.Dataset{
				raster.NewDataset(tileWest, w), raster.NewDataset(tileEast, e), raster.NewDataset(tileMiddle, mi)}, m)
			want := map[string]raster.Float32Raster{"dem": d, "slope": blankOn(fine, true), "hillshade": blankOn(fine, true)}
			terrain.Slope(want["slope"], d, sl)
			terrain.Hillshade(want["hillshade"], d, hs)
			checkGrids(t, fmt.Sprintf("mosaic then terrain %v masked=%v", m, masked), p,
				map[string]raster.Float32Raster{"west": w, "east": e, "middle": mi}, want,
				map[string]reduce.Summary{"dem": reduce.Stats(d)})
		}
	}
}

// TestMosaicOfComputedValues: tiles rescaled before they are laid
// together, so the mosaic reads values an earlier pass stores, and the
// mosaic differenced with an input on its own grid.
func TestMosaicOfComputedValues(t *testing.T) {
	for _, b := range []Boundary{BoundaryAuto, BoundaryCache} {
		g := New()
		west := Rescale(g.InputOn("west", tileWest), 0.5, 3)
		east := g.InputOn("east", tileEast)
		ref := g.InputOn("ref", fine)
		mos := Mosaic([]Node{west, east}, fine, resample.Options{Method: resample.Bilinear})
		diff := Sub(mos, ref)
		g.Output("diff", diff)
		g.Stats("diff", diff)
		g.Output("norm", Normalize(diff))
		p := g.Plan(PlanOptions{Boundary: b})
		rng := rand.New(rand.NewPCG(15, 16))
		w, e, r := operandOn(rng, tileWest, 800, false), operandOn(rng, tileEast, 400, true), operandOn(rng, fine, 400, false)
		ws := blankOn(tileWest, false)
		transfer.Rescale(ws, w, 0.5, 3)
		mo := mosaicked(fine, []raster.Dataset{raster.NewDataset(tileWest, ws), raster.NewDataset(tileEast, e)}, resample.Bilinear)
		d := blankOn(fine, true)
		algebra.Sub(d, mo, r)
		n := blankOn(fine, true)
		algebra.Normalize(n, d)
		checkGrids(t, fmt.Sprintf("mosaic of computed values, %v", b), p,
			map[string]raster.Float32Raster{"west": w, "east": e, "ref": r},
			map[string]raster.Float32Raster{"diff": d, "norm": n},
			map[string]reduce.Summary{"diff": reduce.Stats(d)})
	}
}

// TestMosaicCoverage: unmasked tiles that together cover the grid may be
// written without masks, all the way down the chain; tiles that leave a
// gap make the chain's values masked, and an output without a mask then
// panics, as resample.Mosaic would.
func TestMosaicCoverage(t *testing.T) {
	left := raster.Grid{Width: 30, Height: 37, ResolutionX: 10, ResolutionY: -10, OriginX: 1000, OriginY: 2000}
	right := raster.Grid{Width: 16, Height: 19, ResolutionX: 20, ResolutionY: -20, OriginX: 1290, OriginY: 2000}
	g := New()
	cover := Rescale(Mosaic([]Node{g.InputOn("left", left), g.InputOn("right", right)}, fine, resample.Options{Method: resample.Nearest}), 2, 0)
	g.Output("cover", cover)
	p := g.Plan(PlanOptions{})
	rng := rand.New(rand.NewPCG(17, 18))
	l, r := operandOn(rng, left, 0, false), operandOn(rng, right, 100, false)
	mo := mosaicked(fine, []raster.Dataset{raster.NewDataset(left, l), raster.NewDataset(right, r)}, resample.Nearest)
	mo.Valid = nil // covered: every cell valid, and no mask needed
	want := blankOn(fine, false)
	transfer.Rescale(want, mo, 2, 0)
	checkGrids(t, "covering tiles", p, map[string]raster.Float32Raster{"left": l, "right": r}, map[string]raster.Float32Raster{"cover": want}, nil)

	h := New()
	gap := Rescale(Mosaic([]Node{h.InputOn("left", left)}, fine, resample.Options{}), 2, 0)
	h.Output("gap", gap)
	q := h.Plan(PlanOptions{})
	mg := mosaicked(fine, []raster.Dataset{raster.NewDataset(left, l)}, resample.Nearest)
	wg := blankOn(fine, true)
	transfer.Rescale(wg, mg, 2, 0)
	checkGrids(t, "a gap", q, map[string]raster.Float32Raster{"left": l}, map[string]raster.Float32Raster{"gap": wg}, nil)
}

func TestMosaicChecks(t *testing.T) {
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
	a, b, u := g.InputOn("a", tileWest), g.InputOn("b", tileEast), g.Input("u")
	other := tileMiddle
	other.CRS.Code = "EPSG:4326"
	c := g.InputOn("c", other)
	panics("none", "of no sources", func() { Mosaic(nil, fine, resample.Options{}) })
	panics("undeclared", "undeclared grid (source 1)", func() { Mosaic([]Node{a, u}, fine, resample.Options{}) })
	panics("crs to dst", "reprojection is not supported", func() { Mosaic([]Node{a, c}, fine, resample.Options{}) })
	dstless := fine
	dstless.CRS = raster.CRS{}
	panics("crs between sources", `source 0 in CRS "EPSG:25833"`, func() {
		g2 := New()
		x := g2.InputOn("x", fine)
		y := g2.InputOn("y", other)
		Mosaic([]Node{x, y}, dstless, resample.Options{})
	})
	panics("method", "unknown", func() { Mosaic([]Node{a}, fine, resample.Options{Method: 9}) })
	panics("other graph", "different Graphs", func() { Mosaic([]Node{a, New().InputOn("z", tileEast)}, fine, resample.Options{}) })

	m1 := Mosaic([]Node{a, b}, fine, resample.Options{})
	if m2 := Mosaic([]Node{a, b}, fine, resample.Options{}); m1 != m2 {
		t.Fatal("the same mosaic twice is two nodes")
	}
	if m3 := Mosaic([]Node{b, a}, fine, resample.Options{}); m1 == m3 {
		t.Fatal("mosaics in two orders are one node")
	}
	if gr, ok := m1.Grid(); !ok || gr != fine {
		t.Fatalf("the mosaic lies on %+v", gr)
	}
}
