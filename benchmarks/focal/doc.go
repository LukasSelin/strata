// Package focal is the focal category of the project benchmark suite
// (DESIGN.md §38, §53). It holds no code of its own: bench_test.go times
// the operations of strata/focal through their plain public API, over
// the benchmarks/internal/suite matrix, at several radii:
//
//	Benchmark<Op>R<r>/size=<256|1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=1
//
// with <Op> one of Correlate (full (2r+1)² weights), Gaussian
// (CorrelateSeparable with Gaussian taps of sigma r/2), Mean and Min at
// r = 1, 2, 3 and 5, and Max at r = 3 (Min's mirror image, as a check).
// The radius is part of the operation's name rather than a level of its
// own, so each radius is a separate workload for stratabench's §28
// classification and its name pattern needs no change.
//
// The question the category answers is DESIGN.md §28's prediction that
// convolution is compute-bound like the terrain kernels: its work per
// cell grows with the radius while its memory traffic does not. Every
// operation reads the input and writes the output once, 8 bytes per cell
// (8.25 with masks) whatever the radius; the other rows of a
// neighbourhood come from cache. So GB/s falls as the radius grows if
// the kernel is the limit, and holds if memory is.
//
// The input is the terrain category's DEM (suite.FillDEM). Correlate's
// weights are not all equal, so no kernel can treat it as a box. With
// mask=on the input has a mask with about 10% of cells invalid and the
// output its own.
//
// The plain functions run through the engine's one-worker path, which
// allocates a few slices per call, and the separable kernels borrow one
// row of scratch from a pool; see TestAllocs.
//
// At 16384² each float32 operand is 1 GiB, so a case holds 2 GiB and 64
// MiB of masks. 16384² is skipped under -short. See benchmarks/README.md
// for how to run the suite and RESULTS.md for numbers.
package focal
