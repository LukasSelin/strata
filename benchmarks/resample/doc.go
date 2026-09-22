// Package resample is the resampling category of the project benchmark
// suite (DESIGN.md §38, §54). It holds no code of its own: bench_test.go
// times strata/resample's plain Resample, one goroutine over the whole
// raster, for every method at four scale factors, over the
// benchmarks/internal/suite matrix:
//
//	Benchmark<Method><Scale>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=1
//
// Scale names the output resolution against the source's: Up2 and Up4
// are 2× and 4× as many cells per axis, Down2 half as many, and Down1p37
// a non-integer 1/1.37, where Bilinear, Cubic and Lanczos widen their
// kernels. N is the side of the output, and throughput counts output
// cells, since that is what a caller asks for; the source is N/2 or N/4
// on a side upsampling and 2N or 1.37N downsampling. Sizes stop where the
// source would pass 16384² (4096 for Down2).
//
// BytesPerCell is the source cells each output cell accounts for plus the
// output cell itself, 4 bytes each, and an eighth of a byte per cell for
// masks with mask=on: 4·(1 + s²), where s is the source cells per output
// cell along an axis. It is the traffic of reading each source cell once;
// the intermediate between the passes, one row of it per source row the
// output reaches, is not counted.
//
// mask=on runs for Bilinear and Lanczos only, over the suite's random
// mask with about 10% of source cells invalid, so every footprint has an
// invalid cell and every band takes the masked path: three horizontal
// passes (value, valid values, valid weight) instead of one. It is the
// worst case, not the clustered NoData of real rasters, whose valid
// footprints take the unmasked path.
//
// BenchmarkCubicDirect<Scale> and BenchmarkLanczosDirect<Scale> time
// resamp.Direct2D, the cell-by-cell evaluation that gives the separable
// passes' bits by recomputing every tap row's sum for every output cell,
// against the same fixtures (DESIGN.md §54). It is scalar only: its lanes
// would need a gather per tap, which simd/archsimd does not have and which
// is slow on the Zen 2 the other numbers come from.
//
// See benchmarks/README.md for how to run the suite and RESULTS.md for
// numbers.
package resample
