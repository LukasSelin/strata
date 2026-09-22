// Package curve provides the table-driven flat-slice kernels of package
// transfer: Reclass, a step function over breakpoints, and Lookup, a
// bounded piecewise-linear curve through knots.
//
// It is separate from internal/vec because its hot loop is a per-cell
// search over a small table rather than a lane-parallel map, so it is
// optimized and measured as its own exercise, and because vec's
// kernelSet is a backend-swap table: two permanently scalar fields in it
// would make vec.Backend's answer false for two of its members.
//
// There is no SIMD backend and therefore no backend machinery. Whether a
// vector Reclass or Lookup is worth its complexity is a question for
// bench_test.go and benchmarks/transfer/RESULTS.md, not an assumption
// (DESIGN.md §50). The kernels are named scalar* so that adding one later
// only adds files, as internal/vec and internal/stencil do, and renames
// nothing.
//
// # Tables
//
// A table is one or two float32 slices held by the caller and read, never
// written, copied or retained after the call. Package transfer validates
// them once per call, before any cell is written; the kernels here check
// only what they index by.
//
// Reclass takes breaks and values with len(values) == len(breaks)+1, and
// Lookup takes knots xs and ys of equal length. Both expect the x side —
// breaks, xs — strictly increasing and free of NaN, which is what makes
// the forward scan correct; neither rechecks it per row.
//
// # Values
//
// A NaN cell gives that same NaN back from both kernels, which is not
// what the search alone would do: every comparison against NaN is false,
// so the scan would place a NaN above the whole table. Both kernels
// therefore test for NaN before searching (DESIGN.md §50).
//
// Like internal/vec, exported functions panic on operand errors.
//
//strata:kernel
package curve

// Reclass computes dst[i] = values[k], where k is the number of breaks
// less than or equal to src[i], and NaN where src[i] is NaN. breaks must
// be strictly increasing and free of NaN, and values must hold one more
// element than breaks.
func Reclass(dst, src, breaks, values []float32) {
	requireEqualLen(dst, src)
	if len(values) != len(breaks)+1 {
		panic("curve: values must hold one more element than breaks")
	}
	scalarReclassFloat32(dst, src, breaks, values)
}

// Lookup computes dst[i] by linear interpolation between the knots
// (xs[j], ys[j]) bracketing src[i], clamped to ys[0] at or below xs[0]
// and to the last y at or above the last x, and NaN where src[i] is NaN.
// xs must be strictly increasing and finite, and ys must be the same
// length.
func Lookup(dst, src, xs, ys []float32) {
	requireEqualLen(dst, src)
	if len(xs) != len(ys) {
		panic("curve: xs and ys must have equal length")
	}
	if len(xs) == 0 {
		panic("curve: the table needs at least one knot")
	}
	scalarLookupFloat32(dst, src, xs, ys)
}

func requireEqualLen(dst, src []float32) {
	if len(src) != len(dst) {
		panic("curve: dst and src must have equal length")
	}
}
