// Package exec runs raster kernels for the tiled entry points of the
// operation packages (package engine documents the public contract): it
// plans tiles and bands of rows, builds each call's halo, writes edges,
// computes validity masks, checks cancellation and runs bands on
// workers, so that kernels only compute values (DESIGN.md §22, §23, §25,
// §26). ProcessN runs over rasters in memory; ProcessChunked runs the same
// kernels over engine.RasterSources and engine.RasterSinks a tile at a
// time, in bounded memory (DESIGN.md §24, §27). Reduce and
// ReduceChunked are the fold counterparts, which return a value instead
// of filling an output (DESIGN.md §49).
//
//	err := exec.Process(ctx, dst, dem, slopeKernel{...}, opts)
//	err := exec.ProcessChunked(ctx, sinks, sources, slopeKernel{...}, opts)
//	n, err := exec.Reduce(ctx, src, countOp{}, opts)
//
// The kernel interface is internal (DESIGN.md §22): package algebra and
// package terrain implement kernels with unexported types and expose
// typed entry points such as terrain.SlopeTiled and terrain.SlopeChunked,
// which take engine.Options.
//
// # Kernels, spans and windows
//
// A Kernel declares a Radius and an Arity and fills a Span: a rectangle
// of output cells, given as one raster view per output, with a Window of
// one view per input covering the span grown by Radius on every side.
// Views are ordinary raster.Float32Rasters sharing the operands' memory,
// so a kernel uses Row, Index and Stride as it would on a whole raster.
//
// Process calls a kernel once per rectangle, not once per row. A
// rectangle is the natural unit at every level of DESIGN.md §6's
// raster → tile → row → span → vector path: Process hands a kernel a
// band of whole rows of one tile, and the kernel walks its rows and hands
// row slices to vector code. Row kernels (internal/stencil) wrap in a
// three-line loop, and pointwise kernels keep algebra's fast path of one
// vector call over all cells when the views are compact (Stride ==
// Width), which a per-row interface could not offer. Bands, the unit of
// work between cancellation checks and of scheduling across workers, hold
// about 1<<16 cells (at least one row) and are invisible to kernels
// except through Span.X, Span.Y and the span size.
//
// Kernels with several operands take them as slices in one call rather
// than as nested single-operand kernels: algebra's binary kernels have
// two inputs and terrain's gradient kernel two outputs. One call seeing
// every operand keeps the interface at three methods, lets a multi-output
// kernel compute shared work once per cell (Gradient's dx and dy share
// the neighbourhood loads), and leaves room for operation fusion
// (DESIGN.md §29): a fused pipeline is a single kernel whose inputs are
// the pipeline's leaves and whose outputs are its sinks, running its
// stages over scratch rows inside Process.
//
// # Edges and halos
//
// Process calls a kernel only for output cells whose whole neighbourhood
// lies inside the input rasters passed to it, and reads the halo of every
// span from those rasters, so a cell on a tile or band boundary gets the
// same value as on a whole-raster run. Only the cells within Radius of
// the edge of the rasters passed in, the true edge, get the edge policy:
// the kernel's edge value in Data (NaN unless the kernel is an
// EdgeKernel) and, in outputs with a mask, a cleared validity bit. A
// raster no larger than 2·Radius in a dimension is all edge. Kernels
// never implement boundary logic (DESIGN.md §23). The edge is that of the
// rasters passed in even when they are windows whose parent has data
// beyond them; widening the input to the parent and narrowing the output
// is the caller's choice.
//
// This is package terrain's policy, which is why terrain's plain
// functions also run through Process, with zero Options.
//
// # Validity
//
// Process computes output masks; kernels compute only Data. An output
// cell is valid iff every input cell in its (2r+1)×(2r+1) neighbourhood
// is valid in every input with a mask, and it is not an edge cell. For
// radius 0 that is algebra's AND of the inputs. Data under an invalid
// cell is whatever the kernel wrote there: every cell is computed, valid
// or not, and validity is never inferred from Data (STRATA-3). A
// ReachKernel narrows that per output and input: each masked input is
// eroded by the distance at which the output reads it, or not ANDed at
// all, which is how a Pipeline that multiplies a stencil's result by a
// pointwise input keeps the pointwise input's NoData from spreading
// (DESIGN.md §52). Kernels whose validity rule is different in kind,
// such as a focal mean that skips invalid cells, need an extension of
// this interface.
//
// Masks are processed in words, apart from the arithmetic: radius 0 uses
// raster's range functions, with one pass over the whole span when every
// operand is compact and in place when an output shares its input's bits;
// larger radii use stencil.ErodeBox, which ANDs the 2r+1 halo rows of
// every masked input and shrinks each row by 2r cells with shifts.
//
//   - If no input and no output has a mask, no mask work is done.
//   - If no input has a mask but an output does, its non-edge cells are
//     marked valid and its edge cells invalid, so stale bits from earlier
//     use do not survive.
//   - If any input has a mask, every output must have one, or Process
//     panics: it cannot attach a mask that a parent raster sharing the
//     output's Data would see.
//
// Only output cells are written, in Data and in Valid; row padding and a
// parent's cells outside an output window are left alone.
//
// # Operands
//
// All operands must have the same Width and Height; strides, windows and
// mask offsets may differ. Outputs must not share memory with each other.
// An output of a radius-0 kernel may be exactly the same cells (and bits)
// as an input, to compute in place, when the kernel has a single output;
// otherwise it must not overlap an input at all, which is checked exactly
// for equal strides as in package algebra. An output of a kernel with
// radius > 0 must not share any Data span or validity bit span with an
// input, as in package terrain, because the kernel reads neighbours of
// cells already written. Inputs may overlap each other freely.
//
// # Tiles, bands and workers
//
// engine.Options.TileWidth and TileHeight split the raster into tiles in
// row-major order (0 means the raster's width or height), and each tile
// into bands of max(1, (1<<16)/TileWidth) whole rows. The bands, numbered
// in that order, are the plan. Workers run them: 0 means
// runtime.GOMAXPROCS(0), and never more workers than bands. With one
// worker every band runs on the calling goroutine, which starts no
// goroutines and takes no locks. With more, the calling goroutine and
// Workers-1 goroutines started by the call take bands from the plan in
// order, and the call returns only after all of them have stopped.
// Options never change the result, only which cells are written when and
// by which goroutine.
//
// Kernels run concurrently on disjoint bands. Validity masks are
// processed in words, and neighbouring bands can share a word (cells side
// by side in a row, the end of one row and the start of the next when
// Stride is not a multiple of 64, an input's bits and an output's in one
// mask), so with several workers all mask work, reads and writes, runs
// under one lock per call, after the band's Data. Mask work is small
// next to Data work: with masks, 12 and 24 workers scale within 10% of
// the same runs without (benchmarks/engine/RESULTS.md).
//
// # Tile shapes and performance
//
// For in-memory rasters the default, one tile as wide as the raster, is
// the fastest shape, with one worker or many, and narrower tiles only add
// cost. Workers share the rows as bands, so full-width strips already
// spread the work; a narrow tile gives the same kernel fewer cells per
// call. Row kernels pay a fixed cost per row (5–30 ns in the SIMD build;
// see stencil's BenchmarkRowWidth), which 254-cell rows amortise over
// fewer cells than 4094-cell ones, and a pointwise kernel's views stop
// being compact, so it makes one vector call and one mask operation per
// row instead of one per band, and short rows read memory in short
// streams the prefetcher has not seen. Measured in the SIMD build on one
// worker, 256×256 tiles cost 15–21% for Slope, 21–40% for Hillshade and
// 23–35% for Clamp at 1024² and 4096², and 36–52% at 16384², while
// full-width tiles stay within 3% of the plain functions
// (benchmarks/engine/RESULTS.md). Small tiles remain correct, bit for bit,
// and the engine does not widen them: TileWidth is the caller's statement
// of what one call may cover, which matters for sources that read tiles
// into buffers (DESIGN.md §24, §27).
//
// # Cancellation
//
// Each worker checks ctx before it takes a band (never per cell or inside
// a kernel), stops taking bands once ctx is done, and always finishes a
// band it has taken. A band is written completely: its Data, its edges
// and its validity. Process returns ctx.Err() once every worker has
// stopped, if any band was not written, and nil otherwise. So after a
// cancelled call every output cell either holds its final Data and
// validity or is untouched, and the finished bands are a prefix of the
// plan, with any number of workers: with one tile, a prefix of rows. With
// W workers, at most W-1 kernel calls start after ctx is done (the bands
// other workers took just before). A context that is done before the
// call writes nothing. Only cancellation is reported as an error;
// programming errors panic, like package raster.
//
// A panic in a kernel stops the other workers taking bands; Process waits
// for all of them and panics on the calling goroutine with the same value
// (the worker's stack is not preserved). Output cells are then
// unspecified.
//
// # Chunked execution
//
// ProcessChunked plans tiles as ProcessN does, but its unit of work
// across workers is a tile, not a band. Each worker allocates, once per
// call, one buffer per input covering a tile grown by the radius (at most
// (TileWidth+2r)×(TileHeight+2r) cells) and one per output covering a
// tile. The operands that have validity, a Masked source's input and a
// Masked sink's output, get a mask starting at bit 0 and a Stride rounded
// up to a multiple of 64 (DESIGN.md §23, §27); the others are compact, so
// that sources and sinks can move consecutive full-width rows in one call
// and pointwise kernels keep their whole-span path. For each tile the worker
// reads the tile's cells grown by the radius and clipped to the raster
// from every source, then runs every band of the tile, in order, through
// the same band code as ProcessN, with the buffers as operands placed at
// their raster positions: a job's dst and src start at (dx, dy) and (sx,
// sy), Span.X and Span.Y are raster positions, and only cells within the
// radius of the raster's edge get the edge policy. Cells on a tile's edge
// read their neighbours from the halo in the buffer. So the kernel sees
// the same neighbourhoods, computes the same Data and the engine the same
// validity as in a whole-raster ProcessN call, and the worker hands the
// finished tile to its writer goroutine, which writes it to every sink
// while the worker goes on (write-behind; see RunUnitsBehind). Each
// worker has two sets of output buffers, one computed while the other is
// written. The buffers belong to one worker, so its bands take no mask
// lock.
//
// Workers claim tiles through the same scheduler as bands: they check ctx
// before each tile and always finish a tile they have claimed, so reads
// and writes get context.WithoutCancel(ctx). A read or write error stops
// the workers claiming tiles and is returned, wrapped with the operand
// and position; a write error, which happens on a writer, stops them
// through the context the scheduler checks. See package engine for what
// the sinks then hold. Only
// memory sources and sinks can be checked for shared memory: a memory
// sink must not share Data or validity words with another memory sink or
// with a memory source, because other tiles would read or rewrite them
// concurrently (a radius-0 kernel in place included); ProcessN is the way
// to compute in place.
//
// # Allocation
//
// A call allocates a handful of small slices (views, mask regions and
// one row of scratch words per worker, in one backing array each) and
// its goroutines, and nothing per tile, band, row or cell, and pools
// nothing (DESIGN.md §37). ProcessChunked also allocates each worker's
// tile buffers, one Data and one mask array per worker, about ten small
// slices per worker, and a writer per worker (a goroutine, two channels
// and its views); nothing per tile. Kernels should not allocate in Process
// either, since it runs once per band. Callers that cannot afford even
// per-call allocations, such as package algebra's plain functions, call
// their vector kernels directly instead.
//
// # Reductions
//
// Reduce and ReduceChunked are the fold side of the engine (DESIGN.md
// §49): the same tiles, bands and workers, but no outputs and no sinks,
// and a value rather than a raster. A Reducer declares how many inputs it
// takes, folds a rectangle of Cells into a partial of its own type, and
// combines two partials; the engine gives every worker a zero partial,
// folds the plan's bands into them, and combines them on the calling
// goroutine once every worker has stopped. Combine must be associative
// and commutative with the zero partial as its identity, because the
// order it sees depends on the tiling and the worker count and the answer
// must not.
//
//	total, err := exec.Reduce(ctx, src, countOp{}, opts)
//	total, err := exec.ReduceChunked(ctx, sources, countOp{}, opts)
//
// A reduction has radius 0, so none of the halo, edge or erosion
// machinery above applies: a neighbourhood reduction is a map pass
// followed by a fold. It reads validity and never writes any, so unlike
// job.band it takes no mask lock — the input words that neighbouring
// bands share are only read, and Cells.ValidBits hands a reducer the AND
// of every masked input over a run of up to 64 cells without materialising
// anything. Inputs may overlap each other, and each other's validity
// words, freely; there is no output for them to collide with.
//
// Cancellation is where a fold differs from a map. A cancelled ProcessN
// leaves a prefix of finished cells in its outputs, which is useful; a
// fold over an unknown subset of a raster is not, so a cancelled or
// failed reduction returns ctx.Err(), or the source's error, and the zero
// partial — never what it had folded so far. A call whose every band
// finishes before ctx is done still returns its value, as ProcessN does.
//
// ReduceChunked holds one tile-sized buffer per input per worker and
// nothing else: with no halo and no output buffers its memory is strictly
// below ProcessChunked's for the same Options.
//
// # Concurrency
//
// Process and Reduce are the only places in this module that start
// goroutines (DESIGN.md §26). Kernels must be safe to call concurrently on disjoint
// spans, and must not start goroutines themselves. Each worker has its
// own views and scratch words; nothing a kernel is handed is shared with
// another worker except the operands' memory, and in ProcessChunked not
// even that. Sources and sinks must be safe for concurrent reads and for
// concurrent writes of disjoint regions (engine.RasterSource and
// engine.RasterSink).
package exec
