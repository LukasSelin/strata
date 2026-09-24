// Package graph describes a raster workflow as a lazy graph of strata's
// own operations and plans how to run it: which operations share a pass
// over the data, where a pass must end, and what is written between
// passes (DESIGN.md §55).
//
// Most raster work is a chain of steps, and after a fused pass reads its
// input once, the writes are what cost (benchmarks/gdalsuite/WORKFLOW.md).
// So the planner's job is to read each input as few times as possible
// and write only what is asked for:
//
//   - Per-cell operations (algebra, transfer, Normalize's map) fuse into
//     one pass at no cost.
//   - Neighbourhood operations (terrain, Correlate, Convolve) fuse too,
//     at the price of a wider halo around each tile. Slope, Aspect and
//     Hillshade of one DEM share one gradient.
//   - Statistics fold into the pass that computes their value, as a side
//     output: asking only for statistics writes no raster.
//   - A global operation (Normalize, which needs the range of its whole
//     input) ends a pass. The planner then either computes its input
//     again in the next pass or stores it; PlanOptions.Boundary chooses.
//   - An operation that cannot be a pipeline stage yet (focal Mean, Min,
//     Max and CorrelateSeparable, which need scratch of their own) runs as
//     a pass of its own over a stored input.
//   - A grid change (Resample, Mosaic) is a pass of its own too,
//     between the fused passes on its sources' grids and on its own.
//     Values carry grids: InputOn declares an input's, and combining
//     values on two grids panics when the node is built. A fused pass
//     runs over one grid.
//
// Building a graph computes nothing:
//
//	g := graph.New()
//	dem := g.Input("dem")
//	slope := graph.Slope(dem, terrain.SlopeOptions{CellSize: 10})
//	g.Output("slope", slope)
//	g.Output("hillshade", graph.Hillshade(dem, terrain.HillshadeOptions{CellSize: 10}))
//	g.Stats("slope", slope)
//	plan := g.Plan(graph.PlanOptions{})
//	fmt.Print(plan) // the passes, and why
//	res, err := plan.RunChunked(ctx,
//		map[string]engine.RasterSource{"dem": demSource},
//		map[string]engine.RasterSink{"slope": slopeSink, "hillshade": shadeSink},
//		graph.ChunkedOptions{})
//
// # Guarantees
//
// Every output is the bits the separate operations would write, Data and
// validity, and every summary is reduce.Stats of that raster, for every
// plan choice and every engine.Options: a plan is an optimisation, and the
// separate calls are its reference. Operations are the ones in the
// packages they are named after, with the same options, checks and
// panics, raised when the node is built rather than when the plan runs.
//
// Building the same operation on the same values twice returns the same
// Node, so a subexpression is computed once however often it is named.
//
// A Plan is immutable: one Plan runs any number of times, concurrently if
// wanted, over different rasters of any size.
package graph
