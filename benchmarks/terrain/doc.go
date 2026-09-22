// Package terrain is the terrain category of the project benchmark suite
// (DESIGN.md §38, §42). It holds no code of its own: bench_test.go times
// the four v0.1 operations of strata/terrain (Gradient, Slope, Aspect,
// Hillshade), and Curvature (profile, the kind with the most work per
// cell), through their public API, over the
// benchmarks/internal/suite matrix:
//
//	Benchmark<Op>/size=<256|1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=1
//
// These are the plain functions: one goroutine, the whole raster at once,
// the internal/stencil kernels. The engine category (benchmarks/engine)
// measures what tiles and workers do to Slope and Hillshade; this one
// measures the kernels every path runs, for every operation, at every
// §38 size.
//
// The DEM is the engine category's (suite.FillDEM): a smooth surface of
// 800 ± 300 with slopes of a few percent plus uniform noise of ±1, so
// data-dependent branches (Hillshade's clamps, Aspect's flat cells)
// behave as on real terrain rather than on white noise. Slope is in
// degrees and Aspect a compass bearing, the defaults, so both pay for the
// arctangent; Hillshade is lit from the gdaldem default (315°, 45°). With
// mask=on the DEM has a mask with about 10% of cells invalid and every
// output has its own.
//
// Gradient writes two outputs, so it touches three float32 operands per
// cell (12 bytes, 12.375 with masks) where the others touch two (8 and
// 8.25).
//
// The plain functions run through the engine's one-worker path, which
// allocates a few slices per call (never per tile), not per cell; see
// TestAllocs.
//
// At 16384² each float32 operand is 1 GiB. The measured peak over a whole
// run is 3.15 GiB of private memory: Gradient's fixture, the DEM plus dx
// and dy, with 96 MiB of masks. The single-output operations hold two
// operands and 64 MiB of masks. 16384² is skipped under -short. See
// benchmarks/README.md for how to run the suite and RESULTS.md for
// numbers.
package terrain
