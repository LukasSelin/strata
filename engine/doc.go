// Package engine runs raster kernels: it plans tiles and bands of rows,
// builds each call's halo, writes edges, computes validity masks and
// checks cancellation, so that kernels only compute values
// (DESIGN.md §22, §23, §25). This version runs on the calling goroutine; tiles
// on a worker pool are STRATA-9.
//
//	err := engine.Process(ctx, dst, dem, terrain.SlopeKernel(opts), engine.Options{})
//
// # Kernels, spans and windows
//
// A Kernel declares a Radius and an Arity and fills a Span: a rectangle
// of output cells, given as one raster view per output, with a Window of
// one view per input covering the span grown by Radius on every side.
// Views are ordinary raster.Float32Rasters sharing the operands' memory,
// so a kernel uses Row, Index and Stride as it would on a whole raster.
//
// The engine calls Process once per rectangle, not once per row. A
// rectangle is the natural unit at every level of DESIGN.md §6's
// raster → tile → row → span → vector path: the engine hands a kernel a
// band of whole rows of one tile, and the kernel walks its rows and hands
// row slices to vector code. Row kernels (internal/stencil) wrap in a
// three-line loop, and pointwise kernels keep algebra's fast path of one
// vector call over all cells when the views are compact (Stride ==
// Width), which a per-row interface could not offer. For STRATA-9 a tile
// with its halo is exactly a span and its window, so workers need nothing
// new from kernels. Bands, which bound the work between cancellation
// checks, hold about 1<<16 cells (at least one row) and are invisible to
// kernels except through Span.X, Span.Y and the span size.
//
// Kernels with several operands take them as slices in one call rather
// than as nested single-operand kernels: algebra.AddKernel has two
// inputs and terrain.GradientKernel two outputs. One call seeing every
// operand keeps the interface at three methods, lets a multi-output
// kernel compute shared work once per cell (Gradient's dx and dy share
// the neighbourhood loads), and leaves room for operation fusion
// (DESIGN.md §29): a fused pipeline is a single kernel whose inputs are
// the pipeline's leaves and whose outputs are its sinks, running its
// stages over scratch rows inside Process.
//
// # Edges and halos
//
// The engine calls a kernel only for output cells whose whole
// neighbourhood lies inside the input rasters passed to Process, and
// reads the halo of every span from those rasters, so a cell on a tile
// or band boundary gets the same value as on a whole-raster run. Only the
// cells within Radius of the edge of the rasters passed in, the true
// edge, get the edge policy: the kernel's edge value in Data (NaN unless
// the kernel is an EdgeKernel) and, in outputs with a mask, a cleared
// validity bit. A raster no larger than 2·Radius in a dimension is all
// edge. Kernels never implement boundary logic (DESIGN.md §23). The edge
// is that of the rasters passed in even when they are windows whose
// parent has data beyond them; widening the input to the parent and
// narrowing the output is the caller's choice.
//
// This is package terrain's policy, which is why terrain's functions now
// run through the engine and give the same bits as before, and the same
// bits under any Options.
//
// # Validity
//
// The engine computes output masks; kernels compute only Data. An
// output cell is valid iff every input cell in its (2r+1)×(2r+1)
// neighbourhood is valid in every input with a mask, and it is not an
// edge cell. For radius 0 that is algebra's AND of the inputs. Data under
// an invalid cell is whatever the kernel wrote there: every cell is
// computed, valid or not, and validity is never inferred from Data
// (STRATA-3). Kernels whose validity rule is different, such as a focal
// mean that skips invalid cells, need an extension of this interface.
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
//     panics: the engine cannot attach a mask that a parent raster sharing
//     the output's Data would see.
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
// # Options and cancellation
//
// Options.TileWidth and TileHeight split the raster into tiles, processed
// in row-major order; Workers is validated and otherwise ignored until
// STRATA-9, so every tile runs on the calling goroutine. Options never
// change the result, only the order in which cells are written.
//
// Process checks ctx before every band (never per cell or inside a
// kernel) and returns ctx.Err() as soon as it finds ctx done. A band is
// written completely before the next check: its Data, its edges and its
// validity. So after a cancelled call every output cell either holds its
// final Data and validity or is untouched, and the finished cells are
// whole bands: with one tile, a prefix of rows. A context that is done
// before the call writes nothing. Cells outside the outputs are never
// touched. Only cancellation is reported as an error; programming errors
// panic, like package raster.
//
// # Allocation
//
// A call allocates a handful of small slices (views, mask regions, one
// row of scratch words) and nothing per tile, band, row or cell, and
// pools nothing (DESIGN.md §37). Kernels should not allocate in Process
// either, since it runs once per band. Callers that cannot afford even per-call
// allocations, such as package algebra's functions, call their vector
// kernels directly instead.
//
// # Concurrency
//
// Neither the engine nor any kernel in this module starts goroutines.
// Kernels must nevertheless be safe to call concurrently on disjoint
// spans, which STRATA-9 will do.
package engine
