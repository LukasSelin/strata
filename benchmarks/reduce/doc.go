// Package reduce is the reductions category of the project benchmark
// suite (DESIGN.md §38, §49). It holds no code of its own: bench_test.go
// times strata/reduce's Count, MinMax, Sum and Stats through their Tiled
// public API, over the benchmarks/internal/suite matrix:
//
//	Benchmark<Op>/size=<256|1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=<1|cores|threads>
//
// The operand is a compact DEM (suite.FillDEM). With mask=on it has a mask
// with about 10% of cells invalid. A reduction reads one raster and
// writes nothing, so GB/s is 4 bytes per cell, plus 1/8 with a mask,
// against the machine's read bandwidth rather than a copy's.
//
// The backend level switches internal/vec, which MinMax and Stats use for
// the extremes, and internal/accum, which Sum and Stats use for the exact
// sums, together.
//
// RESULTS.md holds this suite's numbers, after the accumulator decision
// of §49 whose benchmarks are in internal/accum.
package reduce
