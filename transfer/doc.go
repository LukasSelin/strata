// Package transfer turns a computed surface into a factor or a class: a
// step function over breakpoints (Reclass), a bounded piecewise-linear
// curve through knots (Lookup), and the affine map between two ranges
// (Rescale, RescaleRange). Each operation writes into a caller-supplied
// dst, allocates nothing, and runs a flat-span kernel, so it picks up
// the SIMD backend where there is one without exposing it.
//
// These are the operations an environmental model is built from. A
// wildfire risk index, for instance, weights terrain by a slope factor
// and an aspect factor and then cuts the result into risk classes:
//
//	terrain.Slope(slopeDeg, dem, terrain.SlopeOptions{CellSize: 10})
//	terrain.Aspect(aspectDeg, dem, terrain.AspectOptions{CellSize: 10})
//
//	transfer.Lookup(slopeF, slopeDeg, slopeX, slopeY)    // steepness -> factor
//	transfer.Lookup(aspectF, aspectDeg, aspectX, aspectY) // exposure  -> factor
//	algebra.Mul(risk, slopeF, aspectF)
//	transfer.Reclass(class, risk, classBreaks, classValues)
//
// The factors are the curves, the classes are the table, and the model
// lives in the numbers rather than in this package. See the examples.
//
// # Operands
//
// dst and src must have the same Width and Height. Any Stride, window,
// or window of a window is accepted, and each operand is read and
// written through its own layout. Mismatched dimensions and inconsistent
// rasters (see raster.Float32Raster.Validate) panic, like internal/vec
// and raster. Only dst's cells are written: row padding and the cells of
// a parent raster outside a dst window are left alone, in Data and in
// Valid.
//
// dst may be the same raster as src, so dst == src computes in place,
// and it may be a window disjoint from src's cells, such as a
// neighbouring tile of the same raster. dst sharing some but not all
// cells with src gives undefined results; the operations detect this and
// panic. (For operands with different strides over the same memory the
// check is conservative and also rejects interleaved but disjoint
// cells.)
//
// # Tables
//
// Reclass takes breaks and values, with one more value than breaks;
// Lookup takes knots as parallel xs and ys of equal length, at least
// one. The x side — breaks, xs — must be strictly increasing and must
// not hold NaN, and Lookup's xs must be finite. Equal neighbours would
// make a class or a segment unreachable, and an infinite knot would
// flatten the segments touching it. The y side — values, ys — is
// unconstrained: it need not be monotone, which is what lets a curve
// rise and fall, and any value in it, NaN included, is an ordinary
// value.
//
// Class intervals are half-open upward. values[i] covers
// [breaks[i-1], breaks[i]), so a cell exactly on a break takes the class
// above it, the first class runs down to minus infinity and the last up
// to plus infinity. Published classification tables are written that way
// ("class 5: index 21.3 and above"), so a table typed in from one means
// what it says; it also lets a raster of integer codes be reclassified
// with the codes themselves as breaks, rather than with halves.
//
// A table is read, never written or copied, and is not retained after
// the call returns. A Tiled or Chunked run holds it for the duration of
// the call and every worker reads it, so a caller must not mutate a
// table while a run is in flight: that is a data race, and it would
// break the guarantee that the result does not depend on how the raster
// was split.
//
// Validating a table is a pass over a handful of elements and allocates
// nothing. It happens once per call, before any cell is written, so a
// rejected table leaves dst untouched.
//
// # Values
//
// Every cell is computed, valid or not. Data under a cleared validity
// bit is unspecified, on input and on output.
//
// A NaN cell gives that same NaN back from every operation in this
// package, bit for bit. This is worth stating because it is not what the
// arithmetic alone does: every IEEE comparison against NaN is false, so
// a table search that only counts would place a NaN above the whole
// table and report the top class — silently turning missing-looking data
// into the highest risk. Reclass and Lookup test for NaN before
// searching. Validity is still never inferred from Data: a NaN with its
// bit set is an ordinary value (STRATA-3).
//
// The infinities need no special case; they fall in the outer class, or
// clamp to the end of a curve. A negative zero and a positive zero
// compare equal, so they always share a class and hit the same knot.
//
// Lookup reproduces every knot exactly, for every table: a cell equal to
// a knot's x returns that knot's y directly rather than interpolating
// from it, which would give NaN where the neighbouring y is NaN or
// infinite, and a positive zero where it is a negative zero. Nothing
// else is exact. Recovering a cell through a divide and a multiply loses
// an ulp in general, so Lookup with ys equal to xs is the identity only
// to within rounding — the exception being the curve through (0, 0) and
// (1, 1), which is exactly algebra.Clamp to [0, 1].
//
// A curve's segment value is y0 + t*(y1-y0), so a segment whose rise
// overflows float32 gives an infinity between two finite knots. Keeping
// a table's ys within a range whose span is representable avoids it; a
// factor curve, whose ys are small multipliers, never comes near.
//
// Rescale rounds the multiply and the add separately: a*src + b is never
// evaluated as one fused multiply-add. Go permits that fusion and arm64
// takes it where amd64 does not, so the result would otherwise depend on
// the machine (docs/adr/0001-simd-backend.md). The same applies to the
// segment value inside Lookup. Reclass does no arithmetic at all — only
// comparisons and a table index — so it is identical across
// architectures by construction.
//
// RescaleRange resolves its two intervals into Rescale's coefficients
// once per call, rounding each once, and then is Rescale — exactly
// Rescale, bit for bit, with those coefficients. Its endpoints therefore
// land near outLo and outHi rather than on them, to float32 precision
// relative to the output span: the offset absorbs the scaled inLo, so an
// endpoint close to zero can be several of its own ulps out once that
// cancels. A zero inLo is exact, since the offset is then outLo itself.
// Defining the operation the other way round, as outLo + (v-inLo)*scale,
// buys no more exactness and costs a third rounding on every cell and a
// second kernel.
//
// Lookup is the operation whose endpoints are exact, for every interval:
// the two-knot curve through (inLo, outLo) and (inHi, outHi) returns
// both knots bit for bit, and clamps rather than extrapolating outside
// them. That is the real choice between the two, more than the clamp.
//
// # Validity
//
// A result cell is valid iff the cell it came from is valid, where a src
// with a nil mask is valid everywhere. No operation here narrows
// validity: Reclass and Lookup are total, every cell having a class and
// every curve being bounded, so there is no out-of-domain case to mark.
// Use algebra.Mask to narrow validity afterwards.
//
//   - If src and dst have nil masks, no mask work is done.
//   - If src has a nil mask but dst has one, dst's cells are marked
//     valid, so stale bits from earlier use do not survive.
//   - If src has a mask, dst must have one too, or the operation panics.
//     An operation cannot attach a mask to dst itself, because a parent
//     raster sharing dst's Data would never see it. Allocate dst with
//     raster.NewFloat32Like, or set Valid on the root raster before
//     windowing.
//
// # Tiled execution
//
// ReclassTiled, LookupTiled, RescaleTiled and RescaleRangeTiled run the
// same operations in tiles on engine.Options.Workers goroutines (by
// default one per GOMAXPROCS) with a context, and return ctx.Err() if
// cancelled (see package engine). They apply the same operand, table,
// in-place and validity rules (and also reject a dst whose mask bits
// partly overlap src's) and give the same bits for every tiling and
// worker count. The plain functions do not go through the engine: they
// keep their promise to allocate nothing, which the engine's per-call
// setup cannot.
//
// ReclassChunked, LookupChunked, RescaleChunked and RescaleRangeChunked
// read src from an engine.RasterSource and write dst to an
// engine.RasterSink a tile at a time, so rasters larger than memory,
// such as raw float32 files, run in Workers × tile buffers (DESIGN.md
// §27), with the same values and validity. They cannot run in place over
// memory: a memory sink sharing memory with a memory source panics.
//
// # Backends
//
// Rescale and RescaleRange run internal/vec.Affine, which has an AVX2
// kernel in a GOEXPERIMENT=simd build on a CPU that has it, agreeing
// with the scalar one bit for bit. Reclass and Lookup are scalar in
// every build: their inner loop is a search over a small table rather
// than a lane-parallel map, and whether a vector form of it pays is a
// question for a benchmark and not yet an answered one (DESIGN.md §50).
package transfer
