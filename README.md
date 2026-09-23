# strata

[![CI](https://github.com/LukasSelin/strata/actions/workflows/ci.yml/badge.svg)](https://github.com/LukasSelin/strata/actions/workflows/ci.yml)

A SIMD-accelerated spatial compute engine for Go, focused on large raster
and environmental array workloads.

> Fast numerical computing for spatial data in Go.

strata processes large spatial datasets without GDAL, Python, or cgo in the
hot compute path. The emphasis is compute: memory layout, SIMD, streaming,
bounded-memory execution, and concurrency.

It is **not** a GDAL rewrite, a general-purpose GIS suite, a vector geometry
engine, or a file-format compatibility project.

## Status

Pre-release. The v0.1 scope — float32 rasters, windows, validity bitmaps,
the scalar and AVX2 backends, pointwise algebra, terrain derivatives, and
tiled and bounded-memory execution — is implemented and measured, but the
API is not stable and nothing is tagged yet. An arm64 NEON backend has
since joined the AVX2 one. v0.2 is under way: the fold
side of the engine and `reduce.Count`/`MinMax` have landed, and `Sum` and
`Stats` follow once their accumulator is benchmarked. The `transfer`
package has landed alongside them, so a computed surface can now be
turned into a factor or a class, and the `focal` package adds
convolution and focal statistics for any radius up to 8. The engine also counts the bytes each
call moves (`engine.Stats`), and it can run a chain of pointwise kernels
as a single pass over each tile. That chain (the `Pipeline`) is internal
for now. Two things judge the library from outside: an acceptance
harness checks it against numpy and `gdaldem`, and a benchmark times it
against `gdaldem`.

The status table at the top of [DESIGN.md](DESIGN.md) summarizes where
everything stands. In DESIGN.md, §42 has the milestone checklist, §49
covers reductions, §50 transfer functions, §51 the traffic counter, §52
pipelines, §53 focal operations, and §45 the roadmap.

## Installation

```bash
go get github.com/LukasSelin/strata
```

Requires Go 1.27 or later.

## Usage

Compute slope from an elevation raster held in memory:

```go
dem := raster.NewFloat32(width, height, data)
slope := raster.NewFloat32Like(dem)

terrain.Slope(slope, dem, terrain.SlopeOptions{CellSize: 10})
```

Run the same operation in parallel tiles:

```go
err := terrain.SlopeTiled(ctx, slope, dem,
    terrain.SlopeOptions{CellSize: 30}, engine.Options{})
```

Or stream it over a raster larger than memory, reading and writing a raw
little-endian float32 file a tile at a time:

```go
demFile, err := engine.OpenRawFile("dem.f32", os.O_RDONLY, 0, 0)
// ...
in := engine.NewRawSource(demFile, 20000, 20000, engine.RawOptions{})
out := engine.NewRawSink(slopeFile, 20000, 20000, engine.RawOptions{})

err = terrain.SlopeChunked(ctx, out, in,
    terrain.SlopeOptions{CellSize: 30}, engine.Options{TileHeight: 256})
```

Or read a GeoTIFF or Cloud Optimized GeoTIFF straight from the file,
through the `cog` adapter, its own module
(`go get github.com/LukasSelin/strata/cog`):

```go
f, err := os.Open("dem.tif")
// ...
file, err := cog.Open(f)                          // any io.ReaderAt
dem, err := file.Source(cog.SourceOptions{})      // band 0, full resolution
cell := dem.Grid().ResolutionX

err = terrain.SlopeChunked(ctx, out, dem,
    terrain.SlopeOptions{CellSize: cell}, engine.Options{TileHeight: 256})
```

Or turn a surface into a classification — here a five-class fire-risk
scale from a slope factor, with the breakpoints and the curve supplied
by the caller:

```go
transfer.Lookup(factor, slope,
    []float32{0, 5, 10, 20, 30, 40},      // degrees
    []float32{1, 1.1, 1.3, 1.9, 3, 4.5})  // spread factor, flat past the last knot

transfer.Reclass(class, factor,
    []float32{1.2, 1.6, 2.2, 3}, []float32{1, 2, 3, 4, 5})
```

Or reduce it to numbers instead of another raster, streaming the same
way:

```go
mn, mx, count, err := reduce.MinMaxChunked(ctx, in, engine.Options{TileHeight: 256})
```

The plain, `Tiled`, and `Chunked` forms of an operation produce
bit-for-bit identical results for every tile size and worker count.

## Packages

| Package   | Contents |
| --------- | -------- |
| `raster`  | `Float32Raster`, grids, windows, and the validity bitmap. |
| `algebra` | Pointwise `Add`, `Sub`, `Mul`, `Min`, `Max`, `Clamp`, `Mask`, each allocation-free and writing into a caller-supplied destination. |
| `terrain` | Terrain derivatives from Horn's 3×3 gradient: `Gradient`, `Slope`, `Aspect`, `Hillshade`; profile, plan and mean `Curvature` from the Zevenbergen–Thorne quadratic; and `Ruggedness` (TRI, Riley or Wilson; TPI; roughness), bit-identical to gdaldem's. |
| `focal`   | Neighbourhood operations of radius 1 to 8: `Correlate` and `Convolve` with a caller's weights, `CorrelateSeparable` (with `Gaussian` taps), and focal `Mean`, `Min`, `Max`. The same bits for every tile size, worker count and backend. |
| `transfer` | Turns a computed surface into a factor or a class: `Reclass` over breakpoints, `Lookup` along a bounded piecewise-linear curve, `Rescale` and `RescaleRange`. |
| `resample` | Resamples between grids in the same CRS, gdalwarp's conventions: `Nearest`, `Bilinear`, `Cubic`, `Lanczos`, `Average`, with NoData renormalised as gdalwarp does. |
| `reduce`  | Folds a raster to numbers over its valid cells: `Count`, `MinMax`. The same bits for every tile size, worker count and backend. |
| `engine`  | Execution options, the `Stats` traffic counter, and the `RasterSource` / `RasterSink` interfaces with memory and raw float32 file implementations. |
| `cog`     | A separate module: GeoTIFF and COG files as a `RasterSource`. Classic and BigTIFF, tiles and strips, integer and float samples, LZW/Deflate/PackBits/ZSTD, overviews, NoData as validity, geotransform and EPSG code. Pure Go, and bit-identical to GDAL's reading of 98 test files. |

Each package's doc comment is the reference for its operand rules, validity
semantics, edge handling, and cancellation behaviour.

## SIMD

Kernels dispatch at runtime. Building with `GOEXPERIMENT=simd` runs
vectorized kernels that agree bit-for-bit with the scalar ones: AVX2 on
amd64 CPUs that have it, and NEON on every arm64 CPU (Apple Silicon,
Graviton, Ampere). Every other build runs scalar.

```bash
GOEXPERIMENT=simd go build ./...
```

## Benchmarks

From `benchmarks/engine/RESULTS.md`, for a 4096 × 4096 raster with no mask,
in millions of cells per second:

|            | scalar | SIMD | SIMD/scalar | SIMD, 12 workers | SIMD, 24 workers |
| ---------- | -----: | ---: | ----------: | ---------------: | ---------------: |
| Slope      |    164 |  686 |       4.19× |             2644 |             2592 |
| Hillshade  |    209 | 1247 |       5.97× |             2654 |             2642 |
| Clamp      |    924 | 2303 |       2.49× |             2812 |             2733 |

Pointwise algebra is memory-bandwidth-bound from 4096² on. The suites live
under `benchmarks/`, each with its own `RESULTS.md`, and run through
`stratabench`.

Those numbers all compare strata with strata. For an outside one,
[`benchmarks/gdal/`](benchmarks/gdal/RESULTS.md) times the same
operations against GDAL's `gdaldem` in the same container, on a 126.9M
cell raster, and then differences the files it timed. Single-threaded on
one core, strata computes slope, aspect and hillshade in 2.57 s against
gdaldem's 9.39 s; on 12 workers, 1.33 s. On the scalar kernels — a build
without `GOEXPERIMENT=simd` — it is roughly a tie, so the advantage is
AVX2, not the language.

## Testing

Beyond unit tests, the suite runs fuzz tests, metamorphic relations (also
as `rapid` property tests), goroutine-leak checks with `goleak`, IO fault
injection, bounds-check elimination assertions, and `golangci-lint`.

```bash
go test ./...
```

Those tests share the library author's understanding of the problem.
[`acceptance/`](acceptance/README.md) is the independent check. It is a
separate module that drives strata through its public API. A numpy
program written from published definitions then judges the results.
`gdalcheck.sh` differences strata's output against `gdaldem` on a real
raster, and `cogcheck.sh` requires the `cog` reader to read every cell of
98 GDAL-written GeoTIFFs exactly as GDAL does.

CI runs the suite on Linux, Windows, and macOS, and runs the tests, the
race detector, and `golangci-lint` in both the default and the
`GOEXPERIMENT=simd` build, since the vector kernels are behind a build tag
and a default build never compiles them.

## Documentation

[DESIGN.md](DESIGN.md) is the design record: goals, architecture,
and the reasoning behind each decision. Section numbers are cited from code
comments and are stable. Architecture decision records live in
[docs/adr/](docs/adr/).

## Licence

No licence has been chosen yet, so the usual default applies: all rights
reserved. The source is readable here, but it is not yet licensed for
reuse, modification, or redistribution.
