package graph_test

import (
	"fmt"

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
