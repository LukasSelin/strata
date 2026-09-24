package graph_test

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/graph"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
	"github.com/LukasSelin/strata/terrain"
)

// A terrain feature stack: every product of one DEM in one pass, with
// Slope, Aspect and Hillshade sharing one gradient, and a summary of the
// slope folded into the same pass.
func ExamplePlan_terrainStack() {
	g := graph.New()
	dem := g.Input("dem")
	slope := graph.Slope(dem, terrain.SlopeOptions{CellSize: 10})
	g.Output("slope", slope)
	g.Output("aspect", graph.Aspect(dem, terrain.AspectOptions{CellSize: 10}))
	g.Output("hillshade", graph.Hillshade(dem, terrain.HillshadeOptions{CellSize: 10}))
	g.Output("tri", graph.Ruggedness(dem, terrain.RuggednessOptions{}))
	g.Stats("slope", slope)
	fmt.Print(g.Plan(graph.PlanOptions{}))
	// Output:
	// plan: 1 pass(es), boundary auto
	//   input "dem" is read 1 time(s)
	// pass 1 (fused, phase 0)
	//   read  %0 = input "dem"
	//   run   %1, %2 = terrain.Gradient(CellSize=10) ← %0, radius 1
	//   run   %3 = terrain.Slope(CellSize=10) ← %1, %2
	//   run   %4 = terrain.Aspect(CellSize=10) ← %1, %2
	//   run   %5 = terrain.Hillshade(CellSize=10) ← %1, %2
	//   run   %6 = terrain.Ruggedness ← %0, radius 1
	//   write %3 → output "slope", stats "slope"
	//   write %4 → output "aspect"
	//   write %5 → output "hillshade"
	//   write %6 → output "tri"
	// note: the gradient %1, %2 is computed once for every product that reads it
}

// Heat load and direct radiation next to slope and aspect: the planner
// runs all four from the one gradient they share, and the heat load at a
// coarser scale (FitRadius 4) from its own least-squares fit in the same
// pass. The DEM is a valley running east–west at 46°N whose sides rise 4
// m for every 3 across (53°): the north side faces south and the south
// side north, so the north side carries the heat.
//
// The DEM has a validity mask, all valid. Without one, every cell counts
// as valid, including the one-cell border each terrain output fills with
// NaN, and the summary of the heat load would be NaN.
func ExamplePlan_heatLoad() {
	const w, h = 40, 21
	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	for y := range h {
		for x := range w {
			dem.Data[y*w+x] = float32(500 + 40*math.Abs(float64(y-h/2)))
		}
	}
	dem.Valid = make([]uint64, raster.MaskWords(w*h))
	for i := range dem.Valid {
		dem.Valid[i] = ^uint64(0)
	}

	g := graph.New()
	d := g.Input("dem")
	heat := terrain.HeatLoadOptions{CellSize: 30, Latitude: 46}
	coarse := heat
	coarse.FitRadius = 4
	g.Output("slope", graph.Slope(d, terrain.SlopeOptions{CellSize: 30}))
	g.Output("aspect", graph.Aspect(d, terrain.AspectOptions{CellSize: 30}))
	hl := graph.HeatLoad(d, heat)
	g.Output("heatload", hl)
	g.Output("radiation", graph.HeatLoad(d, terrain.HeatLoadOptions{CellSize: 30, Latitude: 46, Radiation: true, Linear: true}))
	g.Output("heatload_r4", graph.HeatLoad(d, coarse))
	g.Stats("heatload", hl)
	p := g.Plan(graph.PlanOptions{})
	fmt.Print(p)

	out := map[string]raster.Float32Raster{}
	for _, name := range []string{"slope", "aspect", "heatload", "radiation", "heatload_r4"} {
		out[name] = raster.NewFloat32Like(dem)
	}
	res, err := p.Run(context.Background(), map[string]raster.Float32Raster{"dem": dem}, out, engine.Options{})
	if err != nil {
		panic(err)
	}
	at := func(name string, y int) float32 { return out[name].Data[y*w+w/2] }
	for _, side := range []struct {
		name string
		y    int
	}{{"north side", 4}, {"south side", 16}} {
		fmt.Printf("%s: slope %.2f°, aspect %.0f°, heat load %.4f, radiation %.4f MJ/cm²/yr\n",
			side.name, at("slope", side.y), at("aspect", side.y), at("heatload", side.y), at("radiation", side.y))
	}
	s := res.Stats["heatload"]
	fmt.Printf("heat load over %d valid cells: min %.4f, max %.4f\n", s.Count, s.Min, s.Max)
	// Output:
	// plan: 1 pass(es), boundary auto
	//   input "dem" is read 1 time(s)
	// pass 1 (fused, phase 0)
	//   read  %0 = input "dem"
	//   run   %1, %2 = terrain.Gradient(CellSize=30) ← %0, radius 1
	//   run   %3 = terrain.Slope(CellSize=30) ← %1, %2
	//   run   %4 = terrain.Aspect(CellSize=30) ← %1, %2
	//   run   %5 = terrain.HeatLoad(CellSize=30, Latitude=46) ← %1, %2
	//   run   %6 = terrain.HeatLoad(CellSize=30, Latitude=46, Radiation=true, Linear=true) ← %1, %2
	//   run   %9 = terrain.HeatLoad(CellSize=30, FitRadius=4, Latitude=46) ← %0, radius 4
	//   write %3 → output "slope"
	//   write %4 → output "aspect"
	//   write %5 → output "heatload", stats "heatload"
	//   write %6 → output "radiation"
	//   write %9 → output "heatload_r4"
	// note: the gradient %1, %2 is computed once for every product that reads it
	// north side: slope 53.13°, aspect 180°, heat load -0.0047, radiation 0.9092 MJ/cm²/yr
	// south side: slope 53.13°, aspect 0°, heat load -1.2254, radiation 0.1618 MJ/cm²/yr
	// heat load over 722 valid cells: min -1.2254, max -0.0047
}

// A canopy height chain: DSM − DTM, smoothed, classed into height bands,
// and a normalised height, ending in statistics rather than rasters. The
// smoothing cannot be a pipeline stage, so it is a pass of its own over a
// stored height, and Normalize ends a pass at the smoothed height.
func ExamplePlan_canopy() {
	g := graph.New()
	height := graph.Sub(g.Input("dsm"), g.Input("dtm"))
	smooth := graph.FocalMean(height, focal.BoxOptions{Radius: 1})
	g.Stats("classes", graph.Reclass(smooth, []float32{2, 10, 20}, []float32{0, 1, 2, 3}))
	g.Output("relative", graph.Normalize(smooth))
	fmt.Print(g.Plan(graph.PlanOptions{}))
	// Output:
	// plan: 3 pass(es), boundary auto
	//   input "dsm" is read 1 time(s)
	//   input "dtm" is read 1 time(s)
	// pass 1 (fused, phase 0)
	//   read  %0 = input "dsm"
	//   read  %1 = input "dtm"
	//   run   %2 = algebra.Sub ← %0, %1
	//   write %2 → stored
	// pass 2 (alone, phase 1)
	//   read  %2, stored
	//   run   %3 = focal.Mean(Radius=1) ← %2, radius 1
	//   write %3 → range for Normalize, stored
	// pass 3 (fused, phase 1)
	//   read  %3, stored
	//   run   %4 = transfer.Reclass([2 10 20] → [0 1 2 3]) ← %3
	//   run   %5 = algebra.Normalize ← %3
	//   write %4 → stats "classes"
	//   write %5 → output "relative"
	// note: %2 is stored: %3 = focal.Mean(Radius=1) reads it and cannot be a pipeline stage
	// note: %5 = algebra.Normalize ends a pass at %3, which is read stored
}

// Change detection across grids: a second date on a 7 m grid, resampled
// onto the first date's 10 m grid, differenced and normalised. The
// resampling is a grid change, a pass of its own; the difference is
// computed on the first date's grid, in one pass with its statistics.
func ExamplePlan_changeDetection() {
	utm := raster.CRS{Code: "EPSG:25833"}
	first := raster.Grid{Width: 4000, Height: 3000, ResolutionX: 10, ResolutionY: -10, OriginX: 500000, OriginY: 6400000, CRS: utm}
	second := raster.Grid{Width: 6000, Height: 4500, ResolutionX: 7, ResolutionY: -7, OriginX: 499993, OriginY: 6400003, CRS: utm}
	g := graph.New()
	before := g.InputOn("before", first)
	after := graph.Resample(g.InputOn("after", second), first, resample.Options{Method: resample.Bilinear})
	diff := graph.Sub(after, before)
	g.Stats("diff", diff)
	g.Output("change", graph.Normalize(diff))
	fmt.Print(g.Plan(graph.PlanOptions{}))
	// Output:
	// plan: 3 pass(es), boundary auto
	//   input "before" is read 1 time(s)
	//   input "after" is read 1 time(s)
	//   grid 1: 4000×3000 from (500000, 6400000) by (10, -10) in "EPSG:25833"
	//   grid 2: 6000×4500 from (499993, 6400003) by (7, -7) in "EPSG:25833"
	// pass 1 (alone, phase 0, grid 1)
	//   read  %1 = input "after"
	//   run   %2 = resample.Resample(Method=Bilinear) ← %1
	//   write %2 → stored
	// pass 2 (fused, phase 0, grid 1)
	//   read  %0 = input "before"
	//   read  %2, stored
	//   run   %3 = algebra.Sub ← %2, %0
	//   write %3 → stats "diff", range for Normalize, stored
	// pass 3 (fused, phase 1, grid 1)
	//   read  %3, stored
	//   run   %4 = algebra.Normalize ← %3
	//   write %4 → output "change"
	// note: %4 = algebra.Normalize ends a pass at %3: stored for the next pass (recomputing it would read 2 stored rasters again)
}
