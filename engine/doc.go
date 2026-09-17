// Package engine configures tiled execution of raster operations
// (DESIGN.md §22–§25). It holds what callers pass to every tiled
// operation; the operations themselves are typed entry points in their
// own packages, such as terrain.SlopeTiled and algebra.AddTiled:
//
//	err := terrain.SlopeTiled(ctx, dst, dem, terrain.SlopeOptions{CellSize: 30},
//		engine.Options{TileWidth: 512, TileHeight: 512})
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
// for every Options value. Tiles do not change edges: a cell on a tile
// boundary reads its neighbours from the rasters passed in, and only the
// edge of those rasters gets the operation's edge policy (for terrain,
// NaN and invalid).
//
// Entry points check ctx between bands of rows of about 1<<16 cells,
// never per cell, and return ctx.Err() as soon as they find ctx done. A
// band is written completely, Data and validity, before the next check,
// so after a cancelled call every output cell either holds its final
// value and validity or is untouched, and cells outside the outputs are
// never touched. A context that is already done writes nothing. Only
// cancellation is reported as an error; programming errors panic, as in
// the rest of strata.
//
// This version runs every tile on the calling goroutine and starts no
// goroutines. Worker pools (STRATA-9) and larger-than-memory sources and
// sinks (§24) are planned.
package engine
