// Package algebra applies pointwise operations to whole rasters: the
// arithmetic of Add, Sub, Mul, Min, Max and Clamp, Mask, which narrows
// validity, and Normalize, which maps the valid cells' range onto [0, 1]
// after a reduction pass over src. Each operation writes into a
// caller-supplied dst, allocates nothing (except Normalize, in its
// reduction), and runs the internal/vec kernels over flat spans,
// so it picks up the SIMD backend without exposing it. Mask has no
// kernel of its own: copying cells is a memmove, which already runs at
// memory speed.
//
// # Operands
//
// dst and every input must have the same Width and Height. Any Stride,
// window, or window of a window is accepted, and each operand is read and
// written through its own layout. Mismatched dimensions and inconsistent
// rasters (see raster.Float32Raster.Validate) panic, like internal/vec and
// raster. Only dst's cells are written: row padding and the cells of a
// parent raster outside a dst window are left alone, in Data and in Valid.
//
// dst may be the same raster as an input, so dst == a computes in place,
// and it may be a window disjoint from the inputs' cells, such as a
// neighbouring tile of the same raster. dst sharing some but not all
// cells with an input, such as two windows of one raster offset by a
// cell, gives undefined results; the operations detect this and panic.
// (For operands with different strides over the same memory the check is
// conservative and also rejects interleaved but disjoint cells.)
//
// # Values
//
// Every cell is computed, valid or not, with the semantics of Go's +, -,
// * and builtin min and max: NaN propagates and min(-0, +0) is -0. Clamp
// computes min(max(v, lo), hi), so lo > hi yields hi. Mask computes
// nothing: it writes src's cells bit for bit. Data under a cleared
// validity bit is unspecified, on input and on output.
//
// # Validity
//
// A result cell is valid iff it is valid in every input, where an input
// with a nil mask is valid everywhere. Validity is never inferred from
// Data: a NaN with its bit set is an ordinary value (STRATA-3).
//
//   - If every input and dst have nil masks, no mask work is done.
//   - If every input has a nil mask but dst has one, dst's cells are
//     marked valid, so stale bits from earlier use do not survive.
//   - If any input has a mask, dst must have one too, or the operation
//     panics. An operation cannot attach a mask to dst itself, because a
//     parent raster sharing dst's Data would never see it. Allocate dst
//     with raster.NewFloat32Like, or set Valid on the root raster before
//     windowing.
//
// Mask is the operation for narrowing validity: Mask(dst, src, mask)
// writes src's values with validity valid(src) AND valid(mask). Only
// mask's validity is read, never its values, so the cells under a mask
// raster may hold anything; it must still be a raster of dst's size whose
// cells do not partly overlap dst's, like any input. Producing a mask
// from values (Threshold, Compare) is a later operation.
//
// Masks are combined up to 64 bits at a time, and in whole-word loops
// when the operands' bit offsets are word-aligned.
//
// # Tiled execution
//
// AddTiled, SubTiled, MulTiled, MinTiled, MaxTiled, MaskTiled,
// ClampTiled and NormalizeTiled run the same operations in tiles on engine.Options.Workers goroutines (by
// default one per GOMAXPROCS) with a context, and return ctx.Err() if
// cancelled (see package engine). They apply the same operand, in-place
// and validity rules (and also reject a dst whose mask bits partly
// overlap an input's) and give the same bits for every tiling and worker
// count. The plain functions other than Normalize do not go through the
// engine: they keep their promise to allocate nothing, which the
// engine's per-call setup cannot.
//
// AddChunked, SubChunked, MulChunked, MinChunked, MaxChunked,
// MaskChunked, ClampChunked and NormalizeChunked read their inputs from
// engine.RasterSources and write dst to an engine.RasterSink a tile at a
// time, so rasters larger than memory, such as raw float32 files, run in
// Workers × tile buffers (DESIGN.md §27), with the same values and
// validity. MaskChunked reads the mask source like any other, so a raw
// mask file carries validity only through engine.RawOptions.Fill. They cannot run in place over
// memory: a memory sink sharing memory with a memory source panics.
// NormalizeChunked reads its source twice, once to find the range and
// once to map it, so it costs two passes over a file.
package algebra
