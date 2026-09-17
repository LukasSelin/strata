// Package chunked is the chunked category of the project benchmark suite
// (DESIGN.md §24, §27, §38): bounded-memory execution over sources and
// sinks. It holds no code of its own: bench_test.go times
// terrain.SlopeChunked, terrain.HillshadeChunked and algebra.ClampChunked
// through their public API over the benchmarks/internal/suite matrix with
// worker and tile levels:
//
//	Benchmark<Op>/size=<1024|4096|16384>/mask=<off|on>/backend=<scalar|simd>/workers=<W>/tiles=<T>
//
// W is 1, one per physical core and one per logical CPU (suite.Workers);
// the scalar backend runs with one worker only. T is the tile shape:
//
//   - plain: the plain function (terrain.Slope, algebra.Clamp) on rasters
//     in memory, at workers=1 only: the whole-raster reference, with no IO.
//   - strips256: the Chunked function from a raw float32 file to another
//     (engine.RawSource and engine.RawSink over engine.RawFile, one handle
//     per logical CPU) in full-width tiles of 256 rows.
//   - 256x256: the same in 256×256 tiles.
//   - memstrips256: the Chunked function over engine.MemorySource and
//     engine.MemorySink on the in-memory rasters, in full-width tiles of
//     256 rows: the tile buffers' cost without files.
//
// The DEM is benchmarks/engine's (800 ± 300 with gentle slopes and ±1
// noise). With mask=on the DEM has a mask with about 10% of cells invalid;
// its raw file holds the fill value -9999 under them, and the sink writes
// -9999 under invalid output cells. GB/s counts 8 bytes per cell (8.25
// with masks), the DEM and the output, although chunked calls copy both
// into and out of their buffers. The fixture writes its files to a
// temporary directory, which Matrix.Release removes after each size.
//
// At 16384² the fixture is two 1 GiB float32 rasters and two 32 MiB masks
// in memory and three 1 GiB files in the OS cache; all Slope cases at
// 16384² peak at 3.69 GiB of private memory, with the buffers of 24
// workers. 16384² is skipped under -short. benchmarks/cmd/stratademo runs
// the §43 demo whose results RESULTS.md also records.
package chunked
