// Package engine configures tiled, parallel execution of raster
// operations (DESIGN.md §22–§27). It holds what callers pass to every
// tiled operation: Options, and the RasterSource and RasterSink
// interfaces with their memory and raw float32 file implementations. The
// operations themselves are typed entry points in their own packages.
// Tiled entry points, such as terrain.SlopeTiled and algebra.AddTiled,
// run over rasters in memory:
//
//	err := terrain.SlopeTiled(ctx, dst, dem, terrain.SlopeOptions{CellSize: 30},
//		engine.Options{})
//
// Chunked entry points, such as terrain.SlopeChunked and
// algebra.ClampChunked, run over sources and sinks a tile at a time, in
// memory bounded by the tile size and worker count rather than the
// raster (see Sources and sinks):
//
//	demFile, err := engine.OpenRawFile("dem.f32", os.O_RDONLY, 0, 0)
//	...
//	slopeFile, err := engine.CreateRawFile("slope.f32", 4*20000*20000, 0o644, 0)
//	...
//	in := engine.NewRawSource(demFile, 20000, 20000, engine.RawOptions{})
//	out := engine.NewRawSink(slopeFile, 20000, 20000, engine.RawOptions{})
//	err := terrain.SlopeChunked(ctx, out, in, terrain.SlopeOptions{CellSize: 30},
//		engine.Options{TileHeight: 256})
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
// operation's edge policy (for terrain, NaN and invalid). A chunked entry
// point stores in its sinks the bits the plain function would write into
// rasters holding the sources' data, for every Options value as well.
//
// # Workers
//
// A Tiled entry point splits its tiles into bands of whole rows of about 1<<16
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
// are still correct, bit for bit. Tile size matters for Chunked entry
// points, which read and write a tile at a time; see Sources and sinks.
//
// # Cancellation
//
// In a Tiled entry point, workers check ctx before each band, never per cell, and stop taking
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
// # Sources and sinks
//
// A RasterSource reads any window of a raster into a caller's buffer, and
// a RasterSink stores a window; both are safe for concurrent use
// (DESIGN.md §24). MemorySource and MemorySink wrap a raster.Float32Raster.
// RawSource and RawSink read and write a raw little-endian float32 file
// through io.ReaderAt and io.WriterAt, with no validity or with a fill
// value (RawOptions). They read a full-width window into a buffer without
// row padding in calls of about 1 MiB, and anything else a row per call.
// Open files for them with OpenRawFile, which gives one file several
// handles: calls on a single *os.File queue behind each other, which made
// 12 workers 1.6× slower on Windows (benchmarks/chunked/RESULTS.md).
// Create output files with CreateRawFile, which sizes the file at once
// and, for concurrent writers, maps it into memory: the operating system
// serialises writes into one file through system calls (Linux takes the
// file's lock for each), and copies into a mapping run in parallel
// (benchmarks/rawio/RESULTS.md). Format adapters implement the same
// interfaces (DESIGN.md §34).
//
// Validity crosses the interfaces explicitly. A source that is not Masked
// has every cell valid, and reads set all of a destination's bits; a
// Masked source reads validity into a destination that must have a mask.
// A sink that is not Masked cannot store validity, so a chunked call with
// a Masked source panics unless every sink is Masked, as a Tiled call
// panics for a dst without a mask.
//
// A chunked call plans tiles in row-major order, like a Tiled one, and
// runs whole tiles on its workers: 0 means runtime.GOMAXPROCS(0), 1 runs
// on the calling goroutine, and there are never more workers than tiles.
// A worker reads its tile and the halo around it (the kernel's radius,
// clipped to the raster) from every source into buffers of its own, runs
// the operation over the tile's bands on those buffers, and hands the
// tile to a writer goroutine of its own, which writes it to every sink
// while the worker reads and computes the next tile (write-behind). Bands
// are not shared between workers, so the tile count, not the band count,
// bounds the parallelism: a single tile runs on one worker.
//
// Write-behind means a chunked call keeps up to two cores busy per
// worker, one computing and one writing, and starts a goroutine per
// worker even with Workers 1. A write waiting in a system call holds no
// P (no GOMAXPROCS slot), so it overlaps computing even under
// GOMAXPROCS=1, but the writer needs a P between its calls and may wait
// a scheduler time slice for one: the overlap is partial there, and full
// with a P to spare (benchmarks/rawio/RESULTS.md).
//
// # Memory
//
// Each worker allocates its buffers once per call and reuses them for
// every tile: one of (TileWidth+2r)×(TileHeight+2r) cells per input, for
// radius r, and two of TileWidth×TileHeight per output, one computed
// while the other is written. An operand with
// validity also gets a mask, and its buffer's rows are padded to a
// multiple of 64 cells so that each row's bits start a word (DESIGN.md
// §23); a buffer without a mask has no padding, so full-width rows are
// consecutive in memory as they are in a file. The
// engine keeps nothing else per cell, per tile or per band, and pools
// nothing, so a call's working memory is about
//
//	Workers × (TileWidth+2r) × (TileHeight+2r) × Σ(bytes per cell)
//
// with 4 bytes per float32 input plus 1/8 per mask, and twice that per
// output, whatever the size of the raster (DESIGN.md §27). Sources and
// sinks may add their own: the memory and raw implementations read and
// write straight into and out of the buffers and allocate nothing. The
// pages of a mapped RawFile are the operating system's file cache, like
// the pages a write system call fills, and not the process's memory: the
// system writes them back and reclaims them as it needs.
//
// An operation built from a chain of kernels (DESIGN.md §52) adds its own
// working memory, one allocation per call shared out between the workers.
// It is sized by the largest span one kernel call covers, which is a
// band and not a tile — about 1<<16 cells — so it is roughly
// Workers × 256 KiB per value the chain keeps between its steps,
// whatever the tile size. That term does not depend on the raster
// either, so the bound above still holds; it simply has one more
// operand-shaped piece in it. The zero Options is one tile, the
// whole raster, per worker, so set TileHeight (and TileWidth for very wide
// rasters) for rasters larger than memory. Full-width tiles of a few
// hundred rows are a good default, for the reasons in Choosing tiles and
// because a source reads them with few calls: 1024×1024 tiles of a
// 20000-wide file take a read and a write call per row and ran 3–4×
// slower (benchmarks/chunked/RESULTS.md).
//
// # Traffic
//
// Options.Stats, when not nil, receives how many bytes a call moved at
// each stage: what its sources delivered, what its kernels read and
// wrote, and what its sinks took. Stats.Amplification divides that by
// what the work strictly needed — each input cell read once, each output
// cell written once — so a pointwise Tiled call is 1.0 and the same
// operation Chunked is 2.0, because every cell goes through a buffer on
// the way in and another on the way out.
//
//	var s engine.Stats
//	err := terrain.SlopeChunked(ctx, out, in, terrain.SlopeOptions{CellSize: 30},
//		engine.Options{TileHeight: 256, Stats: &s})
//	fmt.Println(s.Amplification(), s.Halo(1))
//
// It is an out-parameter rather than a setting: the counters are kept
// whether or not one is passed, so a measured call runs the same code as
// an unmeasured one. They count the bytes the engine moves between
// stages, not the bytes that reach memory — a tile buffer that stays in
// cache is counted twice although DRAM saw it once — which is what makes
// them a property of the structure rather than of the machine. See Stats.
//
// # Cancellation and errors in chunked calls
//
// Workers check ctx before each tile, stop taking tiles once ctx is done
// or a tile has failed, and finish a tile they have taken: its reads, its
// computation and its writes to every sink. A write fails behind its
// worker, which has taken its next tile by then, and stops every worker
// taking tiles the moment it fails. Sources and sinks are passed
// context.WithoutCancel(ctx), so cancellation never cuts a tile short;
// with W workers at most W-1 tiles start after ctx is done, and
// cancellation waits, per worker, for at most one tile's reads and
// computation and two tiles' writes. The call returns once every write
// has finished.
//
// The call returns nil when every tile is written; the first error of a
// source or sink, wrapped with the operand and the window position and
// matching the original error with errors.Is; or ctx.Err() if ctx was
// done with tiles left. A panic in a worker is re-raised on the calling
// goroutine, and sink contents are then unspecified. What an error or a
// cancellation leaves in the sinks is defined in whole tiles, in plan
// order. The tiles workers took form a prefix of the plan, and every
// later tile is untouched: nothing written, in any sink. Every tile of
// the prefix has been written completely to every sink (Data and
// validity), except a tile whose read failed, which is untouched, and a
// tile whose write failed, whose cells are unspecified in every sink. So
// after a cancellation the sinks hold a prefix of whole tiles, and after
// an error that prefix with a hole for each failed tile: one, or up to
// two per worker if tiles taken before the first error fail too. Cells
// outside the sinks' regions are never touched.
package engine
