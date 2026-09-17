// Package engine is the engine category of the project benchmark suite
// (DESIGN.md §26, §28, §38, STRATA-9). It holds no code of its own:
// bench_test.go times terrain.Slope, terrain.Hillshade and algebra.Clamp
// through their public API, plain and through the engine, over the
// benchmarks/internal/suite matrix with worker and tile levels:
//
//	Benchmark<Op>/size=<1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=<W>/tiles=<T>
//
// W is 1, one per physical core and one per logical CPU (suite.Workers);
// the scalar backend runs with one worker only, as DESIGN.md §43 compares
// scalar, SIMD and SIMD with workers.
// T is the tile shape:
//
//   - plain: the plain function (terrain.Slope, algebra.Clamp), with no
//     tiles and one goroutine, run at workers=1 only. It is the
//     whole-raster reference the tiled runs are compared with.
//   - strips: the Tiled function with default engine.Options: one tile
//     the width of the raster, divided into bands of whole rows of about
//     1<<16 cells, which the workers share.
//   - 256x256: the Tiled function in 256×256 tiles, each divided into
//     bands the same way.
//
// The DEM is a smooth surface (800 ± 300 with slopes of a few percent)
// plus uniform noise of ±1. Clamp clamps it to [700, 900]. With mask=on
// the DEM has a mask with about 10% of cells invalid and dst has its own
// mask. Every operation reads the DEM and writes dst: two float32
// operands, 8 bytes per cell (8.25 with masks).
//
// 256² is not run: it is a single band, which the engine runs on the
// calling goroutine whatever Workers says.
//
// At 16384² the fixture is two 1 GiB float32 operands plus two 32 MiB
// masks; measured peak private memory of the whole run is 2.14 GiB, so
// 16384² is skipped under -short. Scaling runs must not be pinned to one
// CPU. See benchmarks/README.md for how to run the suite and RESULTS.md
// for numbers.
package engine
