// Package transfer is the transfer-function category of the project
// benchmark suite (DESIGN.md §38, §50). It holds no code of its own:
// bench_test.go times strata/transfer's Reclass, Lookup, Rescale and
// RescaleRange through their plain public API, over the
// benchmarks/internal/suite matrix:
//
//	Benchmark<Op>/size=<256|1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=1
//
// The operand is what the chain of §50 feeds these operations: the slope,
// in degrees, of the suite DEM (suite.FillDEM through terrain.Slope,
// computed once per size, before timing), about 0–25° with a NaN border
// one cell wide. It is a smooth field with the DEM's noise differentiated
// into it, so neighbouring cells share a class about three times in four
// — neither the uniform noise of transfer's own benchmarks nor a sorted
// ramp. Reclass is a five-class scale over four breaks and Lookup a
// five-knot factor curve, fitted to that range, the table sizes a model
// uses. Rescale and RescaleRange are the pointwise
// reference: the bandwidth-bound speed the table-driven pair would reach
// if the search cost nothing.
//
// Every operation reads one float32 raster and writes one, so GB/s is 8
// bytes per cell, plus 2/8 with masks: the source's read and the
// output's written. With mask=on the source has a mask with about 10% of
// cells invalid and the output its own.
//
// The backend level switches internal/curve, which Reclass and Lookup
// run, and internal/vec, which the Rescales run, together.
//
// At 16384² each float32 operand is 1 GiB; the fixture holds two, plus
// the DEM while the slope is computed. 16384² is skipped under -short.
// See benchmarks/README.md for how to run the suite and RESULTS.md for
// numbers.
package transfer
