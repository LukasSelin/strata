// Package engine configures tiled, parallel execution of raster
// operations (DESIGN.md §22–§26). It holds what callers pass to every
// tiled operation; the operations themselves are typed entry points in
// their own packages, such as terrain.SlopeTiled and algebra.AddTiled:
//
//	err := terrain.SlopeTiled(ctx, dst, dem, terrain.SlopeOptions{CellSize: 30},
//		engine.Options{})
//
// Entry points live with their operations, not here, so that this package
// never imports a package that imports it, and so that adding an
// operation does not grow this package. The kernel machinery that runs
// them (tile and band planning, halos, edges, validity and cancellation)
// is internal until its shape settles.
//
// # Guarantees
//
// A tiled entry point takes the same operands, applies the same checks
// and writes the same bits (Data and validity) as its plain counterpart,
// for every Options value: every tile size and every worker count. Tiles
// do not change edges: a cell on a tile boundary reads its neighbours
// from the rasters passed in, and only the edge of those rasters gets the
// operation's edge policy (for terrain, NaN and invalid).
//
// # Workers
//
// An entry point splits its tiles into bands of whole rows of about 1<<16
// cells and runs the bands on Options.Workers goroutines: 0 means
// runtime.GOMAXPROCS(0), 1 runs everything on the calling goroutine and
// starts none, and there are never more workers than bands, so a raster
// of up to 1<<16 cells runs on the calling goroutine. The goroutines are
// started by the call and have all exited when it returns, whether it
// finishes, is cancelled or panics. A panic in a worker is re-raised on
// the calling goroutine with the same value. Operations and kernels start
// no goroutines of their own (§26): the plain functions, such as
// terrain.Slope, run on the calling goroutine.
//
// # Choosing tiles
//
// For rasters in memory, leave TileWidth and TileHeight at 0. Workers
// share a full-width tile's rows, so it already runs in parallel, and it
// is the fastest shape with any number of workers. A narrower tile makes
// the kernels work on short rows, which pays their per-row cost over
// fewer cells, reads memory in short streams, and splits pointwise
// operations into one vector call per row: in the SIMD build, 256×256
// tiles are 15–52% slower than the default on one worker, depending on
// the operation and raster size (benchmarks/engine/RESULTS.md). Small tiles
// are still correct, bit for bit. Tile size matters for sources and sinks
// that read and write a tile at a time (§24, §27), which are planned.
//
// # Cancellation
//
// Workers check ctx before each band, never per cell, and stop taking
// bands once ctx is done; a band a worker has taken is written completely,
// Data and validity. The call returns ctx.Err() after every worker has
// stopped, unless every band was written. So after a cancelled call every
// output cell either holds its final value and validity or is untouched,
// the finished cells are whole bands in plan order (tiles row-major, rows
// top-down within a tile) up to some band, and cells outside the outputs
// are never touched. With W workers at most W-1 bands start after ctx is
// done. A context that is already done writes nothing. Only cancellation
// is reported as an error; programming errors panic, as in the rest of
// strata.
//
// Larger-than-memory sources and sinks (§24) are planned.
package engine
