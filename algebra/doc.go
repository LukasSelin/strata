// Package algebra applies pointwise arithmetic to whole rasters: Add, Sub,
// Mul, Min, Max and Clamp. Each operation writes into a caller-supplied
// dst, allocates nothing, and runs the internal/vec kernels over flat
// spans, so it picks up the SIMD backend without exposing it.
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
// computes min(max(v, lo), hi), so lo > hi yields hi. Data under a cleared
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
// Masks are combined up to 64 bits at a time, and in whole-word loops
// when the operands' bit offsets are word-aligned.
//
// # Kernels
//
// ClampKernel, AddKernel, SubKernel, MulKernel, MinKernel and MaxKernel
// return the operations as radius-0 kernels for package engine, which
// applies the same operand, in-place and validity rules (and also rejects
// a dst whose mask bits partly overlap an input's) and gives the same
// bits in any tiling. The functions themselves do not go through the
// engine: they keep their promise to allocate nothing, which a kernel
// behind an interface cannot.
package algebra
