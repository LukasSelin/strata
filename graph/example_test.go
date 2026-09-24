package graph_test

import (
	"fmt"

	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/graph"
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
