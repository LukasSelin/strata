// Package algebra is the algebra category of the project benchmark suite
// (DESIGN.md §38, STRATA-10). It holds no code of its own: bench_test.go
// times the six v0.1 operations of strata/algebra (Add, Sub, Mul, Min,
// Max, Clamp) through their public API, over the benchmarks/internal/suite
// matrix:
//
//	Benchmark<Op>/size=<256|1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=1
//
// Operands are compact rasters of uniform values in [-100, 100). Clamp
// uses [-50, 50], so about half its cells are clamped. With mask=on every
// input has an independent mask with about 10% of cells invalid and dst
// has its own mask. With mask=off no operand has a mask, which is
// strata/algebra's no-mask-work fast path.
//
// The package-level micro-benchmarks in algebra/bench_test.go stay where
// they are: they compare internal paths (whole raster, per row, strided).
// This suite measures what a caller sees, at the §38 sizes and backends.
//
// Fixtures hold only the operands an operation uses. At 16384² the
// measured peak is 3.15 GiB of private memory for the two-input operations
// (three 1 GiB float32 operands plus 32 MiB masks) and 2.12 GiB for Clamp,
// so 16384² is skipped under -short. See benchmarks/README.md for how to
// run it and RESULTS.md for numbers.
package algebra
