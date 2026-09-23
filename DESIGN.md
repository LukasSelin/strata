# SIMD-First Spatial Compute Engine

This is the design record of strata's compute layer: memory layout,
kernels, SIMD, the execution engine, validity, and how each is tested and
measured. Formats, domain algorithms and GIS workflows sit outside it
(§7) and appear here only where they constrain the core.

Section numbers in this document are cited from code comments, ADRs and
benchmark results. Add new material inside existing sections or at the end
rather than renumbering. That is why §49–§52, which specify work that
landed after v0.1, follow the long-term sections §46–§48, and why a
section whose subject left the core keeps its number and a short note.
When a section's Status line changes, update the status table below as
well.

Decisions recorded elsewhere and summarized here:

- [ADR 0001](docs/adr/0001-simd-backend.md): SIMD backend technology (STRATA-2).
- [ADR 0002](docs/adr/0002-cog-adapter.md): the first format adapter, GeoTIFF/COG: its own module, its own parser, GDAL as the judge (§34, §35).
- [benchmarks/nodata/RESULTS.md](benchmarks/nodata/RESULTS.md): NoData representation (STRATA-3).
- [benchmarks/algebra/RESULTS.md](benchmarks/algebra/RESULTS.md): first benchmark suite results (STRATA-10).
- [benchmarks/chunked/RESULTS.md](benchmarks/chunked/RESULTS.md): bounded-memory execution and the §43 demo.
- [benchmarks/engine/RESULTS.md](benchmarks/engine/RESULTS.md): worker scaling and tile shape (STRATA-9).
- [benchmarks/terrain/RESULTS.md](benchmarks/terrain/RESULTS.md): the terrain kernels on one worker.
- [benchmarks/focal/RESULTS.md](benchmarks/focal/RESULTS.md): the focal kernels by radius, and §28's convolution prediction (§53).
- [benchmarks/gdal/RESULTS.md](benchmarks/gdal/RESULTS.md): strata timed against `gdaldem`, the outside speed baseline (§38).
- [benchmarks/cog/RESULTS.md](benchmarks/cog/RESULTS.md): the GeoTIFF/COG reader's decode speed against GDAL, and slope over a COG against `gdaldem` (§34).
- [benchmarks/resample/RESULTS.md](benchmarks/resample/RESULTS.md): resampling, separable against direct 2-D, on NEON (§54).
- [acceptance/README.md](acceptance/README.md): black-box checks against numpy, `gdaldem` and GDAL's own GeoTIFF reading, the outside correctness oracles (§39).
- [tools/herbie/RESULTS.md](tools/herbie/RESULTS.md): Herbie's rewrites of the kernel formulas, triaged (§39).

Where things stand, as of 2026-09-23. Each section's own **Status** line is
the detailed record; this table only points at it.

| Area | § | Status |
|---|---|---|
| `Float32Raster`, windows, validity bitmap | §9, §21, §31 | done |
| Scalar backend, amd64 AVX2 backend (`GOEXPERIMENT=simd`) | §14–§17 | done |
| arm64 NEON backend (`GOEXPERIMENT=simd`, STRATA-11) | §14, §17 | done: `vec`, `stencil`, `accum` |
| `algebra`: Add, Sub, Mul, Min, Max, Clamp, Mask, Normalize | §18 | done |
| `terrain`: Gradient, Slope, Aspect, Hillshade, Curvature, Ruggedness | §20 | done |
| Engine: tiled, multi-worker, halos | §22–§26 | done |
| Engine: chunked, bounded memory, memory and raw file IO | §24, §27 | done |
| First validation target: 20000² DEM | §43 | done |
| Traffic counter (`engine.Stats`) | §51 | done |
| Reductions: Count, MinMax, the fold driver | §49 | done |
| Reductions: exact accumulator (`internal/accum`) and its decision | §49 | done |
| Reductions: Sum, Stats, `benchmarks/reduce` suite | §49 | done |
| `transfer`: Reclass, Lookup, Rescale, RescaleRange | §50 | done; vector table kernels and `benchmarks/transfer` open |
| `focal`: Correlate, Convolve, CorrelateSeparable, Mean, Min, Max | §53 | done; median, skip-invalid statistics and r = 3's residual 64 KiB-stride loss open |
| `Pipeline`, radius 0, internal | §52 | done |
| `Pipeline`: radius > 0, several outputs, public `Kernel` | §52 | not started |
| Register-level operation fusion | §29 | measured, not built: about 5% out of cache (`benchmarks/fusion`) |
| N-dimensional arrays | §10 | not started (v0.3) |
| Point clouds | §11 | not started (v0.7) |
| Format adapters: GeoTIFF/COG read (`cog` module) | §34, §35 | done: identical to GDAL on 98 files, from disk and over HTTP range requests (`cog.HTTPReaderAt`), timed against it (`benchmarks/cog`); writing open |
| Format adapters: Zarr, LAS/LAZ, … | §34, §35 | not started |
| CRS contract: one CRS per computation, labels checked where grids meet | §36 | done; reprojection is the caller's preprocessing |
| `resample`: same-CRS grid resampling, Nearest to Average | §54 | done; Mode and mosaics open |
| Publishing: module path, README, CI | §42 | done |
| Publishing: licence, first tag | §42 | not started |

## 1. Project Goal

Build a Go-native, SIMD-accelerated compute layer for large spatial
arrays: rasters first, N-dimensional arrays and point batches later.

> High-throughput numerical computation over large spatial datasets using Go-native memory layouts, SIMD, chunking, and concurrency.

The first concrete target:

> A Go backend should be able to process large spatial datasets efficiently without needing GDAL, Python, or cgo in the hot compute path.

strata is not a GDAL rewrite, a GIS suite, a vector geometry engine, a
wrapper around GDAL/OGR, a file-format project, or a replacement for
scientific Python as an analysis environment.

## 2. Positioning

> Fast numerical computing for spatial data in Go.

Not "GDAL for Go". The emphasis is compute, memory layout, bounded-memory
streaming, SIMD, concurrency and reusable kernels, not format breadth,
desktop GIS workflows, feature/layer abstractions or GIS compatibility.

## 3. Why This Project Exists

A Go program that needs serious raster math today either links a native
stack through cgo (GDAL, PDAL) or calls out to a Python service. strata
puts that computation in-process, behind ordinary Go calls, with SIMD and
workers underneath:

> Serious spatial numerical processing inside a normal Go application, with minimal deployment complexity.

**SIMD is a build-time opt-in.** Go's SIMD packages are still experimental
(ADR 0001). A default `go build` produces a pure-Go, cgo-free binary that
runs the scalar kernels on every architecture. Building with
`GOEXPERIMENT=simd` (Go ≥ 1.27) enables the SIMD kernels. The experiment is
a whole-build flag that the application sets, not something strata's
`go.mod` can turn on. Documentation, examples and published benchmark
numbers must say which mode they use.

## 4. Evidence of Need

The signals behind the project, none of them a request for this exact
architecture:

- **Pure-Go demand.** Go users repeatedly remove cgo and native
  dependencies for cross-compilation, static binaries, containers and
  simpler CI, and pure-Go geospatial projects exist for that reason.
- **SIMD over slices.** Spatial workloads are `[]float32` arithmetic,
  comparisons and reductions, which is what vector execution is for.
- **Go is already used for geospatial systems** (GDAL bindings, H3 ports,
  LAS readers, raster packages). What is missing is the numerical layer.

The hypothesis the early releases test:

> A shared compute layer across raster, multidimensional arrays, point clouds, and future spatial representations is useful enough to be preferable to a collection of specialized libraries.

## 5. Primary User Stories

### 5.1 Go backend processing a large raster

The primary v0.x use case. A backend computes slope, aspect, hillshade,
normalization, threshold masks or environmental indices over a large
raster, in a single Go binary with bounded memory, workers and SIMD, and
no Python service or GDAL dependency:

```go
dem := raster.NewFloat32(width, height, data) // from any source adapter
slope := raster.NewFloat32Like(dem)

terrain.Slope(slope, dem, terrain.SlopeOptions{CellSize: 10})
```

Operations write into a caller-supplied destination and allocate nothing
(§37). For rasters larger than memory, the same operation runs tile by
tile through the execution engine, reading windows from a source (§24,
§25).

### 5.2 Other workloads the core must serve

- **Environmental arrays** such as `temperature[time,y,x]`: map, reduce,
  normalize, threshold, combine. Needs N-D arrays (§10, v0.3).
- **Remote-sensing indices** such as NDVI = (NIR − Red) / (NIR + Red):
  pointwise arithmetic and a natural case for fusion (§29, §52).
- **Point clouds**: attribute filters, transforms, aggregation and
  rasterization over point batches (§11, v0.7).

strata supplies the arithmetic, stencils, reductions and scheduling. The
domain workflow — a fire-risk model, a hydrological network, a canopy
product — belongs to the module that imports strata (§7).

## 6. Core Architectural Principle

The central compute flow:

> Source → chunk → tile/span → vector kernel → sink

For rasters: raster → tile → band of rows → span → SIMD vector. For point
clouds, later: point stream → batch → attribute columns → SIMD filter or
transform.

## 7. Scope Boundary

strata is the compute layer. The core understands:

```text
Array  Raster  Grid  Shape  Stride  Window  Span
Chunk  Tile  Mask  Kernel  PointBatch
```

It does not depend on storage formats (GeoTIFF, COG, LAS, Shapefile,
GeoPackage, FileGDB, PostGIS, GDAL datasets). Those are adapters (§34).

**Domain algorithms live in modules that import strata.** The core keeps
operations that are domain-neutral numerical building blocks: pointwise
algebra (§18), neighbourhood stencils (§20, and `focal`'s convolution and
focal statistics, §53), reductions (§49) and transfer functions (§50),
whose models live in the caller's tables. `terrain` is
the one domain package in the core, because it is the proving consumer
for stencils and halos and the operation the gdaldem comparison checks;
it grows only local derivatives of a DEM.

Anything with non-local dependencies or domain conventions of its own —
hydrology (depression filling, flow direction, accumulation, watersheds),
fire-behaviour models, classification schemes — belongs in a separate
module. Where such a module cannot do what it needs through the public
API (a kernel it cannot run on the engine, an element type strata lacks),
that is evidence for what strata publishes next (§9, §22, §52), and it is
recorded here when it happens.

## 8. Spatial Representations

The engine will eventually serve three computational families, each with
its own API rather than one generic abstraction: **dense** (raster,
voxel, N-D array), **sparse** (point clouds, meshes) and **graph**
(networks). They share low-level infrastructure — workers, buffers, SIMD
primitives, masks, reductions — where that fits (§12).

## 9. Raster Model

The first supported spatial representation is raster. It is implemented in
package `raster` (STRATA-4).

The compute type is `float32`, as a concrete type rather than a generic
`Raster[T Number]`:

```go
type Float32Raster struct {
    Data []float32

    Width  int
    Height int
    Stride int

    Valid       []uint64 // validity bitmap, nil = every cell valid (§31)
    ValidOffset int      // bit index in Valid of Data[0]
}
```

- **Layout.** Row-major: cell (x, y) is `Data[y*Stride+x]`. `Stride ≥ Width`
  allows row padding, and `len(Data) == (Height-1)*Stride + Width`.
- **No NoData value.** Validity lives in `Valid`, never in `Data` (§31).
- **Constructors** `NewFloat32`, `NewFloat32Stride`, `NewFloat32Like`.
  Invalid arguments panic because they are programming errors.
  `Validate` checks a hand-built raster without panicking.
- **Other data types.** Storage may keep integer or float64 data, but
  source adapters convert it to `float32` plus a validity mask at the IO
  boundary. A generic element type is deferred to the Array work (v0.3,
  §10). At that point, decide whether `Float32Raster` becomes a 2D
  specialization of `Array`.

Spatial metadata stays separate from the numbers, so kernels never see it:

```go
type Grid struct {
    Width, Height            int
    ResolutionX, ResolutionY float64 // signed, as in a GDAL geotransform
    OriginX, OriginY         float64 // outer corner of cell (0, 0)
    CRS                      CRS     // opaque label, §36
}

type Dataset struct {
    Grid   Grid
    Raster Float32Raster
}
```

`Grid.Window` and `Dataset.Window` shift the origin to match a raster window
(§21).

## 10. Multidimensional Arrays

Environmental data is often not purely two-dimensional: `[time,y,x]`,
`[level,time,y,x]`, `[scenario,time,y,x]`. A future core array may look
like:

```go
type Array[T Number] struct {
    Data   []T
    Shape  []int
    Stride []int
}
```

This aligns with chunked storage such as Zarr. Arrays need the same
validity bitmap as rasters (§31). `Stride` here is per dimension, while
`Float32Raster.Stride` is the row stride in elements. The element-type
decision made here also decides integer and float64 rasters (§9), which
importing modules need for counts and labels (§7).

Status: not started (v0.3).

## 11. Point-Cloud Model

Point clouds use a column-oriented representation (Structure of Arrays),
never a `[]Point` of structs, so attribute filters, distances, masks and
transforms map onto SIMD:

```go
type PointCloud struct {
    X, Y, Z        []float32
    Intensity      []uint16
    Classification []uint8
}
```

Coordinates may need `float64`, or `float32` offsets from a batch origin,
for large coordinate ranges (§39). That decision belongs to v0.7.

Status: not started (v0.7).

## 12. Dense vs Sparse Scheduling

Dense data (raster, voxel, N-D array) is scheduled in tiles, chunks and
halos over strides. Sparse data (point clouds, meshes) is scheduled in
batches and streams over attribute columns and spatial partitions. The
engine shares workers, buffers, SIMD primitives, masks, reductions and
the chunk lifecycle between them without forcing identical high-level
APIs. In particular, dense sources are random-access windowed readers and
sparse sources are streams (§24).

**Kernels are span-level.** What the two schedules share is the layer
below them, so that layer must not know which one called it. A kernel
package works on spans — slices of cells or mask words — and scalars:
`vec.Add(dst, a, b []float32)`, `accum.Sum.Add(xs []float32)`,
`stencil.HornSlopeRow(dst, r0, r1, r2 []float32, …)`. It never takes a
`raster.Float32Raster`, a grid, a source, a context or a worker, and it
starts no goroutines. A raster row and a point-batch column are then the
same argument, and only the engine schedules. The fold driver is the one
raster-shaped seam (`exec.Cells`); point batches get a counterpart that
feeds the same reducers, not a generalised `Cells`.

`lint/kernelborder` enforces this. A package declares itself a kernel
package with a `//strata:kernel` line above its package clause
(`internal/vec`, `accum`, `stencil`, `curve` and `focalrow` do), and
then:

```text
K0  only kernel packages import simd/archsimd (§14); benchmarks/ exempt
K1  kernel packages import a short standard-library allowlist,
    simd/archsimd and other kernel packages, nothing else
K2  exported kernel functions take and return span-level types: basic
    types, slices, arrays, pointers and funcs of them, type parameters
    with a type-set constraint, and kernel packages' own types
K3  kernel packages have no go statements, channels or select
```

None of the rules applies to `_test.go` files. A new kernel package,
`internal/focalrow` (§53) or a resampling one, registers by carrying the
marker; one that imports `simd/archsimd` without it fails K0, which is
how `focalrow` was found when this check first ran over it, and
`internal/resamp` when it met the check on master: its band driver,
which reads `raster` masks and takes the mask lock, moved to `resample`.

## 13. SIMD-First Design

Algorithms should process contiguous spans of values rather than individual cells.

Avoid:

```go
func SlopeAt(r Raster, x, y int) float32
```

Prefer:

```go
func Slope(dst, src Raster)
```

Internally:

```text
span
  ↓
SIMD load
  ↓
vector arithmetic
  ↓
SIMD store
```

This should be designed into algorithms from the start.

## 14. SIMD Must Remain Internal

Experimental SIMD APIs should never leak into the public API.

The layering is:

```text
algebra, reduce,       transfer         terrain     (later: array, pointcloud)
transfer.Rescale       Reclass, Lookup     │
   │                      │             raster
   ▼                      ▼                ▼
internal/vec           internal/curve   internal/stencil
(pointwise, folds)     (table search,   (neighbourhood row kernels,
   │                    scalar only)     mask erosion)
   │                                       │
   └──────────────────┬────────────────────┘
          ├── simd/archsimd, amd64 AVX2    (GOEXPERIMENT=simd)
          ├── simd/archsimd, arm64 NEON    (GOEXPERIMENT=simd, STRATA-11)
          └── scalar                       (canonical, every build)
```

Per ADR 0001:

- All SIMD code is written in Go with `simd/archsimd`. There is no
  assembly.
- The fixed-width `archsimd` package is used, not the portable `simd`
  package. Portable `simd` picks 128-bit lanes on many AVX2 CPUs, and a
  library cannot override that. Revisit with Go 1.28.
- `archsimd` may only be imported under `internal/`, in files built with
  `//go:build goexperiment.simd && <arch>`.

This protects users from changes in the underlying SIMD API.

## 15. Scalar Correctness Path

Every SIMD operation should have an equivalent scalar implementation.

Scalar implementations provide:

```text
reference correctness
fallback
portability
debugging
comparison tests
```

SIMD should be an execution backend, not a different semantic path. SIMD
kernels agree with scalar **bit for bit**: any NaN matches any NaN, but
+0 ≠ −0. Scalar references use explicit `float32(...)` conversions to
prevent FMA fusion, and SIMD `min`/`max` are corrected to Go's builtin
semantics for NaN and signed zero (ADR 0001).

## 16. Vector Kernel Layer

Pointwise kernels live in `internal/vec` (implemented):

```go
func Add(dst, a, b []float32)
func Sub(dst, a, b []float32)
func Mul(dst, a, b []float32)
func Div(dst, a, b []float32)

func AddScalar(dst, src []float32, value float32)
func MulScalar(dst, src []float32, value float32)

func Min(dst, a, b []float32)
func Max(dst, a, b []float32)

func Clamp(dst, src []float32, min, max float32)

func Abs(dst, src []float32)
func Sqrt(dst, src []float32)

// dst[i] = a*src[i] + b, the kernel of transfer.Rescale (STRATA-13,
// §50). The multiply and the add round separately: it is never a fused
// multiply-add, which is what keeps the scalar reference canonical on
// arm64 as well (ADR 0001).
func Affine(dst, src []float32, a, b float32)

// Folds. acc is the running value, so an empty src returns it and a
// caller starts from +Inf or -Inf and needs no empty-input case
// (STRATA-12, §49).
func ReduceMin(acc float32, src []float32) float32
func ReduceMax(acc float32, src []float32) float32
```

Neighbourhood row kernels live in `internal/stencil` (implemented). Each
one takes the three rows around an output row:

```go
func HornGradientRow(dx, dy, r0, r1, r2 []float32, kx, ky float32)
func HornSlopeRow(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool)
func HornAspectRow(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool)
func HornHillshadeRow(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32)
func ZTCurvatureRow(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32, kind CurvatureKind)
func RuggednessRow(dst, r0, r1, r2 []float32, kind RuggednessKind)

func Erode3x3(...) // validity of radius-1 outputs, word-level
func ClearBorder(...)
```

Package `focal`'s row kernels, for any radius, live in
`internal/focalrow` (§53): a 2-D weighted row (`CorrelateRow`), and the
column and row passes of the separable operations (`ColumnCorrelate`,
`ColumnSum`, `ColumnMin`, `ColumnMax`, `RowCorrelate`, `RowMean`,
`RowMin`, `RowMax`). They take their input rows as one slice and a
stride rather than a slice of rows, which would escape through the
dispatch variables and allocate per band.

Future operations:

```text
Select
Blend
Compare
ReduceSum
Dot
Hypot
```

`FMA` is deliberately absent rather than pending. A kernel that rounds a
multiply and an add together is exactly what §15 forbids in a scalar
reference, since arm64 would fuse where amd64 would not; `Affine` is the
non-fused form, and §50 has the argument.

`ReduceSum` is deliberately not here yet. An order-independent sum needs
an accumulator that does not round as it goes (§49), and its per-cell
work is a scatter, which is hostile to vectorization; an inexact vector
sum would fold its lanes in an order the scalar loop does not and so
break §15 by construction. `ReduceMin` and `ReduceMax` have no such
problem, which is why they could land first.

## 17. Backend Model

Current layout:

```text
internal/vec/                      internal/stencil/
├── scalar.go                      ├── horn.go        (scalar + dispatch)
├── dispatch.go                    ├── aspect.go
├── simd_amd64.go                  ├── mask.go
└── simd_arm64.go                  ├── simd_amd64.go
                                   └── simd_arm64.go
```

- `internal/accum` has the same pair, with the block loop they share in
  `blocks.go`, and so has `internal/focalrow` (§53), after a scalar
  `focalrow.go`. There is no `simd.go` (portable SIMD), per §14.
- **Dispatch.** Kernels are package-level function variables. They start
  scalar, and `init` swaps them for SIMD versions when the build has
  `GOEXPERIMENT=simd` and, on amd64, the CPU has AVX2. This is gated on
  AVX2, not just AVX. On arm64 NEON is baseline, so there is no check.
- **Equivalence tests** are written once, in each package's
  `simd_test.go`, and run against whichever backend the architecture has;
  a small per-architecture test file names that backend's kernels.
- **Backend switching.** Each kernel package exports `Backend() string`
  (`"avx2"`, `"neon"` or `"scalar"`) and `UseScalar(bool)`, so benchmarks
  and tests can run both backends in one binary.
- There is no per-call interface dispatch.
- The kernel-writing rules (bounds-check-free loops, `ClearAVXUpperBits`
  before scalar tails, splitting kernels into `*Lanes` functions, explicit
  NaN handling) are in ADR 0001 and benchmarks/nodata/RESULTS.md. The
  NEON files keep the same loop shapes and the `*Lanes` split, minus the
  AVX-specific parts, so each reads side by side with its amd64 twin.

## 18. Raster Algebra

Package `algebra` exposes pointwise operations over whole rasters. The
plain functions are one file, `algebra.go`, and the engine entry points
another, `tiled.go`; it needs no per-operation files.

Implemented (STRATA-5):

```go
algebra.Add(dst, a, b)
algebra.Sub(dst, a, b)
algebra.Mul(dst, a, b)
algebra.Min(dst, a, b)
algebra.Max(dst, a, b)
algebra.Clamp(dst, src, lo, hi)

// Mask writes src into dst with validity = valid(src) AND valid(mask).
// Only mask's validity is read, never its values; a nil mask on either
// input means all valid. Producing masks from values (Threshold, Compare)
// is a later operation.
algebra.Mask(dst, src, mask)
```

And one that is not pointwise:

```go
// Normalize maps the valid cells' [min, max] onto [0, 1] and returns
// the min and max it used.
lo, hi := algebra.Normalize(dst, src)
```

`algebra.Scale(dst, src, 1.25)` stood here and is not being added.
`transfer.Rescale(dst, src, a, b)` subsumes it (§50), over a single
`vec.Affine` kernel rather than a `MulScalar` pass and an `AddScalar`
one; two spellings of one operation is what this section exists to
prevent.

`Normalize` is not a pointwise operation. It needs global statistics (min
and max, or mean and standard deviation, over valid cells). Under chunked
execution that means a reduction pass over every tile before the map pass.

It is min-max normalisation, computed as `(v - lo) / (hi - lo)` in float32:
a subtraction then a division (`vec.SubDiv`), not `transfer.Rescale`'s
single multiply-add. The multiply-add rounds `1/(hi - lo)` and `-lo/(hi -
lo)` into coefficients, so the maximum lands near 1 rather than on it.
Measured on elevation-like data, about one raster in three came out with
a maximum other than 1, half of those above it, by up to 64 ulps. The
difference-then-quotient form maps `lo` to +0 and `hi` to exactly 1, and
since both steps round monotonically, every valid cell lands in [0, 1]. It
is also, bit for bit, what NumPy computes for
`(x - x.min()) / (x.max() - x.min())` on a float32 array, which is what
`acceptance/check.py` holds it to. The division costs more than a
multiply, but a pass that writes every cell is bound by memory bandwidth.

Degenerate ranges are not special-cased: a constant raster gives 0/0,
which is NaN, and a NaN or infinity among the valid cells propagates as
IEEE arithmetic has it. `Normalize` returns `lo` and `hi` so a caller can
detect those cases, or map a result back. Standardising by mean and
standard deviation (`reduce.Stats`) would be a separate operation and is
not added.

Status: done. `Normalize`, `NormalizeTiled` and `NormalizeChunked`; the
chunked form reads its source twice (§49).

Semantics, as in the `algebra` package documentation:

- `dst` may alias an input exactly. Partial overlaps panic.
- Any stride or window is accepted.
- Every cell is computed with Go's `+ - * min max` semantics.
- An output cell is valid iff it is valid in every input, which is all
  `Mask` does: it copies `src` and intersects the two validities.

## 19. Avoid Arbitrary Callbacks in Hot Paths

Generic APIs such as:

```go
raster.Map(src, func(v float32) float32 {
    return ...
})
```

may be convenient but can inhibit optimization.

Explicit operations are easier to:

```text
vectorize
fuse
benchmark
specialize
```

Generic callbacks may exist as convenience functionality, but should not
define the performance model. The rule concerns per-cell callbacks. A
callback per row or per tile, such as terrain's internal row driver or the
engine's kernel interface, costs nothing measurable.

## 20. Terrain Package

Terrain processing is the first major real-world consumer.

Implemented (STRATA-6, STRATA-7):

```text
terrain/
├── doc.go
├── gradient.go    Gradient(dx, dy, dem, GradientOptions)
├── slope.go       Slope(dst, dem, SlopeOptions)       degrees | radians | percent
├── aspect.go      Aspect(dst, dem, AspectOptions)
├── hillshade.go   Hillshade(dst, dem, HillshadeOptions)
├── curvature.go   Curvature(dst, dem, CurvatureOptions)   profile | plan | mean
├── ruggedness.go  Ruggedness(dst, dem, RuggednessOptions) TRI | TRI Wilson | TPI | roughness
└── stencil.go     shared row driver, edge and validity policy
```

`terrain` stays limited to local derivatives of a DEM; flow routing and
everything built on it are a separate module's (§7).

- **Method.** Gradient, Slope, Aspect and Hillshade use Horn's 3×3
  gradient, computed a whole row at a time. Each SIMD lane loads the
  three neighbouring rows at offsets and evaluates the stencil for 8
  cells at once. Curvature has the same shape but its own derivatives:
  the Zevenbergen–Thorne quadratic (p, q, r, s, t, reading the centre
  cell too) and Florinsky's normal-section profile, plan and mean
  curvature, positive where convex. It is the terrain kernel with the
  most arithmetic per cell and no arctangent: two divisions and a
  square root.
- **Ruggedness.** Riley's and Wilson's terrain ruggedness index, the
  topographic position index and roughness (max − min) of the 3×3
  window, in elevation units, with no cell size or ZFactor. Unlike the
  other kernels, whose conventions are gdaldem's but whose rounding is
  their own, these reproduce gdaldem's arithmetic operation for
  operation (apps/gdaldem_lib.cpp, float32 input): each difference
  rounded to float32, sums folded left to right in row-major order,
  "/ 8" as "· 0.125", and Riley's squares, sum and root in float64
  before one rounding to float32. The results are therefore
  bit-identical to gdaldem's, which the acceptance harness checks cell
  for cell on a real raster, not within a tolerance. Riley's float64
  root makes it the slowest of the four: in BenchmarkRowWidth on
  4094-cell rows with AVX2, about 1.7 ns/cell against 0.3–0.6 for the
  others, on the Ryzen 9 3900X. A float32 sum would be cheaper
  but disagrees with gdaldem in the last bit. Roughness uses Go's min
  and max (a NaN anywhere gives NaN); the AVX2 kernel uses VMAXPS and
  VMINPS as they are and repairs the difference once, rather than each
  comparison.
- **Conventions.** Conventions are gdaldem-compatible: compass bearings,
  gdaldem-style `ZFactor`, and positive cell sizes.
- **Edges.** The one-cell border of the rasters passed in gets NaN and
  cleared validity bits. For a window, that is the window's edge, not the
  parent raster's edge. Reading halos from the parent is the engine's job
  (§23).
- **Validity.** An output cell is valid iff its whole 3×3 neighbourhood is
  valid, centre included. This is computed by word-level erosion.

Terrain is useful because it exercises:

```text
neighbor access
overlapping loads
edge handling
floating-point math
SIMD
tiling
halo management
```

## 21. Windows and Views

Raster windows are views rather than copies.

A window is not a separate type. It is a `Float32Raster` that shares its
parent's `Data` and `Valid`, keeps the parent's `Stride`, and points
`ValidOffset` at its own first cell:

```go
window := r.Window(x, y, width, height)
```

Because windows are rasters, every operation accepts them, and windows can
be windowed again. Nothing is copied or allocated.

A validity mask must be attached to the root raster before windowing. A
mask attached to a window is not seen by its parent.

This is particularly important for large datasets and neighborhood kernels.

## 22. Kernel Abstraction

Once multiple algorithms exist, introduce a common kernel concept.

Conceptually:

```go
type Kernel interface {
    Radius() int

    Process(
        dst Span,
        src Window,
    )
}
```

Examples:

```text
Clamp       radius 0
Normalize   radius 0 (after a reduction pass, §18)
Slope       radius 1
Hillshade   radius 1
5×5 filter  radius 2    (focal.Correlate, §53)
```

The execution engine can then handle boundaries and halos generically.

**For v0.1 the kernel interface is internal** (`internal/exec`). Users
call typed entry points (§25). STRATA-8 settled a first shape:

- `Radius()`, `Arity() (inputs, outputs int)` and `Process(dst Span, src
  Window)`, plus an optional `Edge() float32` for the value written at
  the true raster edge (NaN by default).
- A `Span` is a rectangle of output cells, one raster view per output; a
  `Window` holds one view per input, grown by the radius on every side.
  The engine calls `Process` once per band of rows of a tile, so row
  kernels loop over rows and pointwise kernels keep one vector call per
  contiguous span.
- Several inputs or outputs (`Add`, `Gradient`) are slices in one call,
  which keeps fusion (§29) possible: a fused pipeline is one kernel.
- Kernels write only Data. The engine derives validity: the AND of every
  masked input over the (2r+1)×(2r+1) neighbourhood, with word-level
  erosion.

It stays internal until worker pools, sources and fusion have exercised
it. Worker pools (§26) and sources (§24) now have, and so has tile-level
fusion, as the radius-0 `Pipeline` (§52); §52's "Where it lives" is why
the contract is still not published.

`Reducer` is the fold counterpart (§49): inputs and no outputs, radius
fixed at 0, and a value rather than a raster. It is a separate interface
rather than a kernel whose arity is `(n, 0)`, so `Kernel`'s "at least one
output" rule stays true and the map path grows no branch for it.

## 23. Halo Handling

Chunked neighborhood operations require surrounding data.

For example:

```text
┌───────────┬───────────┐
│           │           │
│  tile A   │  tile B   │
│           │           │
└───────────┴───────────┘
```

A slope calculation at the edge of tile A requires data from tile B.

The runtime should automatically build halo regions:

```text
XXXXXXXXXXXX
X..........X
X.. tile ..X
X..........X
XXXXXXXXXXXX
```

Algorithms should not implement tile-boundary coordination individually.

- **Building the halo.** The engine reads each tile together with a halo
  of width `Radius()`. Halo cells that fall outside the dataset are marked
  invalid (zero mask bits). This makes the true raster edge behave like
  the edge of a whole-raster call, with no per-kernel border code.
- **Contract.** Tiled output must equal the whole-raster call on the same
  data bit for bit, in Data and in validity, for every tile size and
  worker count. Tests enforce this (§39): `internal/exec`'s
  `TestTilesAndWorkers` runs every tile width and height in {1, 7, 64,
  256, full, larger than the raster} with 1, 2, 3 and GOMAXPROCS workers,
  for Clamp, Add, Slope, Aspect, Hillshade, Gradient, a radius-2
  kernel and three focal kernels (a radius-3 Correlate, a radius-2
  CorrelateSeparable, which uses scratch, and a radius-4 Max, §53), with
  and without masks, on windows whose stride is not a
  multiple of 64.
- **Buffers.** Tile and halo buffers of operands with validity are
  allocated with `Stride` rounded up to a multiple of 64, so each row's
  mask bits start on a word boundary and mask copies are plain word
  copies (benchmarks/nodata/RESULTS.md). Buffers without validity are
  compact (`Stride == Width`), so a raw file's consecutive full-width
  rows are read and written in one call and pointwise kernels keep their
  whole-span path.

Status: done. Tiles of in-memory rasters read their halos as views of
the input (STRATA-8), with any number of workers (STRATA-9), and the
engine writes the edge policy itself instead of calling the kernel for
edge cells. Chunked execution (§24, §27) copies each tile with its halo
into per-worker buffers and runs the same band code on them, with the
buffers placed at their raster positions, so only the raster's own edge
gets the edge policy. `internal/exec`'s `TestChunkedTilesAndWorkers`
runs the same matrix as `TestTilesAndWorkers` through memory sources and
sinks.

## 24. Chunk-Oriented Execution

The engine should understand chunks rather than files.

**Dense data uses random-access windowed sources.** The engine plans the
tiles itself, so it can read each tile and its halo independently and in
parallel. Settled shape, in package `engine`:

```go
type RasterSource interface {
    Size() (width, height int)
    Masked() bool // false: every cell is valid
    // ReadWindow fills dst's cells (Data, and validity if dst has a mask)
    // from the dst.Width×dst.Height region at (x, y). Safe for concurrent
    // calls.
    ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error
}

type RasterSink interface {
    Size() (width, height int)
    Masked() bool // whether it stores validity
    // WriteWindow stores src's cells at (x, y). Safe for concurrent calls
    // on disjoint regions. It may overwrite the Data of src's invalid
    // cells (a fill value), which §31 leaves unspecified.
    WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error
}
```

- Sources read into caller-supplied buffers, so workers reuse them
  (§37). Reads and writes touch only the window's cells, never row
  padding or bits outside it.
- **Validity crosses explicitly.** A source that is not `Masked` sets
  every bit of a masked destination; a `Masked` source panics on a
  destination without a mask, and a sink that is not `Masked` panics on
  a masked `src`, rather than dropping validity. A chunked call with a
  `Masked` source panics unless every sink is `Masked`, like the Tiled
  functions for a dst without a mask.
- **Errors.** IO failures are returned; the engine wraps them with the
  operand and window position (`errors.Is` still matches). Programming
  errors (regions outside the raster, invalid rasters) panic.
- **Raw files have no validity of their own.** By default every cell is
  valid. `RawOptions{Fill, HasFill}` declares a NoData value as an IO
  adapter would (§31): reading marks cells equal to it invalid (any NaN
  for a NaN fill), writing puts it under invalid cells. A valid cell
  holding the fill value does not survive a round trip. There is no
  sidecar mask: that would be a format.
- **File handles.** Calls on one `*os.File` queue: Go serialises
  `ReadAt`/`WriteAt` per handle on Windows (a lock and two seeks per
  call), and the OS may too. `engine.OpenRawFile` opens a file with
  several handles and spreads calls over them: 1.6× faster with 12
  workers. Raw sources and sinks move full-width windows in 1 MiB calls
  (17–21% faster than a call per row) and other windows a row per call;
  joining short rows across the gaps between them costs more than the
  calls it saves.

**Sparse data uses streams**, from v0.7 on:

```go
type PointSource interface {
    Next(context.Context) (PointBatch, error)
}
```

v0.1 sources and sinks:

```text
memory (a Float32Raster)
raw little-endian float32 file, row-major, via io.ReaderAt / io.WriterAt
```

The raw file is not a format parser (§41). It is the minimum needed for a
larger-than-memory demo (§43).

Status: done for dense data. `MemorySource`, `MemorySink`, `RawSource`,
`RawSink` and `RawFile` are in package `engine`; chunked entry points
run over them (§25, §27).

Later sources and sinks are adapters (§34): Zarr, GeoTIFF, COG, LAS/LAZ,
object storage, generated data. This keeps computation independent from
storage. The first, a GeoTIFF/COG source, is the `cog` module (§34).

## 25. Execution Engine

The execution engine owns:

```text
chunk scheduling
tile planning
worker scheduling
halo construction
temporary buffers
backend dispatch
bounded-memory execution
```

Internally:

```go
exec.Process(ctx, dst, src, kernel, opts) error // internal/exec
exec.Reduce(ctx, src, reducer, opts) (P, error) // the fold, §49
```

Publicly, for v0.1, typed entry points take a source, a sink, the
operation's own options and the engine options. They live in the
operation's own package, next to the plain function, and package `engine`
holds only what every entry point shares (`Options`, later sources and
sinks). That way the engine never imports a package that imports it, and
adding an operation does not grow the engine:

```go
err := terrain.SlopeTiled(ctx, dst, dem, terrain.SlopeOptions{CellSize: 30},
    engine.Options{TileWidth: 512, TileHeight: 512})
```

STRATA-8 added `AddTiled`, `SubTiled`, `MulTiled`, `MinTiled`, `MaxTiled`,
`ClampTiled`, `GradientTiled`, `SlopeTiled`, `AspectTiled` and
`HillshadeTiled` over in-memory rasters (`CurvatureTiled` and `RuggednessTiled` came later), and `MaskTiled` followed with
`Mask` (§18). They give the same bits as the plain functions for every
`Options`. The terrain functions run the same kernels as one tile; the
algebra functions stay direct to keep their zero allocations.

The Chunked functions (`SlopeChunked`, `AspectChunked`,
`HillshadeChunked`, `GradientChunked`, `CurvatureChunked`, `RuggednessChunked`, `AddChunked`, `SubChunked`,
`MulChunked`, `MinChunked`, `MaxChunked`, `MaskChunked`, `ClampChunked`)
take sources and sinks instead of rasters and run with bounded memory
(§27), through `exec.ProcessChunked`. Their sinks receive the bits the
plain function would write, for every `Options`.

The later packages follow the same pattern: `transfer` has `ReclassTiled`,
`LookupTiled`, `RescaleTiled`, `RescaleRangeTiled` and their `Chunked`
counterparts (§50), and `reduce` has `CountTiled`, `CountChunked`,
`MinMaxTiled` and `MinMaxChunked`, which return a value instead of
filling a sink (§49). `focal` has a Tiled and a Chunked form of each of
its six operations (§53).

Configuration:

```go
type Options struct {
    TileWidth  int
    TileHeight int
    Workers    int
    Stats      *Stats // optional traffic counter, §51
}
```

The zero value is the default for in-memory rasters: one tile as wide as
the raster and `Workers` = `GOMAXPROCS`. The engine plans tiles in
row-major order and splits each into bands of whole rows of about 2¹⁶
cells, the unit of scheduling and cancellation. Narrow tiles are correct
but slower, because row kernels pay a fixed cost per row and pointwise
kernels lose their one-call-per-band fast path: on one worker 256×256
tiles cost 15–52% over the default, depending on operation and size, and
the default is the fastest shape for every worker count
(benchmarks/engine/RESULTS.md). Measuring this found that the SIMD row
kernels paid about 65 ns per row for an SSE/AVX transition (legacy SSE
instructions while upper YMM bits were dirty), now removed; see the
kernel-writing rules in docs/adr/0001-simd-backend.md. Tile size matters
for chunked calls, which hold a tile per worker in memory (§27).

The engine returns errors for IO and cancellation, and panics on
programming errors, as the rest of strata does. Workers check the context
before each band and finish a band they have started, so a cancelled call
leaves every output cell either final or untouched, and the finished bands
are a prefix of the plan whatever the worker count. A kernel panic in a
worker is re-raised on the calling goroutine after every worker has
stopped.

For chunked calls, cancellation and errors work in whole tiles. Workers
check the context before each tile, stop taking tiles after an error,
and finish a tile they have taken (sources and sinks get
`context.WithoutCancel`). So the tiles taken form a prefix of the plan,
later tiles are untouched in every sink, and every tile of the prefix is
written completely, except a tile whose read failed (untouched) or whose
write failed (unspecified). After a cancellation the sinks hold a prefix
of whole tiles.

Status: done. Tiled execution over in-memory rasters, on one or many
workers (STRATA-8, STRATA-9); chunked execution over sources and sinks;
the fold driver (§49); the traffic counter (§51); and the radius-0
`Pipeline` (§52).

## 26. Parallelism Model

There are three natural levels of parallelism.

**IO parallelism.** Multiple chunks may be loaded concurrently.

**Worker parallelism.** Chunks or tiles may be assigned to goroutines.

**Instruction parallelism.** Each worker uses SIMD.

```text
Dataset
   │
   ▼
Chunks
   │
   ├──── Worker ─── SIMD
   ├──── Worker ─── SIMD
   ├──── Worker ─── SIMD
   └──── Worker ─── SIMD
```

Low-level kernels remain synchronous. `internal/vec`, `internal/stencil`,
`internal/curve`, `internal/focalrow`, `algebra`, `terrain`, `focal`,
`transfer` and `reduce` create no
goroutines of their own; their `Tiled` and `Chunked` entry points get
workers from the engine.

Concurrency belongs in the execution engine.

Reductions (§49) are the first driver with no lock at all: they read
validity and never write it, so the mask lock below has nothing to guard.

**Implementation (STRATA-9).** `internal/exec` starts `Workers − 1`
goroutines per call and uses the calling goroutine as the last worker;
there is no global pool of goroutines. Workers take bands from the plan
in order with an atomic counter, each with its own views and erosion
scratch, and the call joins them before it returns. Kernels run
concurrently on disjoint bands.
Validity words can be shared between bands (cells side by side, row ends
when the stride is not a multiple of 64, inputs and outputs in one mask),
so all mask work runs under one lock per call, after the band's Data.
With masks, workers scale within 10% of the same runs without.

Measured scaling (benchmarks/engine/RESULTS.md, 12-core Zen 2, dual-channel
DDR4-3200): at 1024², in cache, 12 workers run Slope 5.5× and Hillshade
4.4× faster than one. From 4096² every operation flattens at 2.4–2.9
billion cells/s, 19–23 GB/s of memory traffic: Slope 3.7×, Hillshade
2.1×, Clamp 1.2–1.7×. 24 workers (SMT) are no faster than 12.

## 27. Bounded-Memory Execution

Large spatial datasets should never require full materialization.

The preferred model:

```text
read chunk
   ↓
construct halo
   ↓
process
   ↓
write result
   ↓
release buffer
   ↓
next chunk
```

This should work for datasets much larger than system memory.

Peak working memory should be about:

```text
Workers × (TileWidth + 2r) × (TileHeight + 2r) × Σ(operand bytes per cell)
```

This bound does not depend on the dataset size. The engine demo must
report measured peak memory against it (§43).

**Implementation.** `exec.ProcessChunked` gives each worker, once per
call, one buffer per input of at most (TileWidth+2r)×(TileHeight+2r)
cells (clipped to the raster), one per output of TileWidth×TileHeight,
and a mask per operand with validity. A worker reads a tile and its halo
from every source, runs the tile's bands in order through the in-memory
band code on its buffers (no mask lock: nothing is shared), and writes
the tile to every sink. The unit of parallelism is the tile: there are
never more workers than tiles, so the zero `Options` (one tile) runs one
worker with the whole raster in memory. Set `TileHeight`.

**Measured** (benchmarks/chunked/RESULTS.md, 12-core Zen 2, DDR4-3200):

- Every chunked run's peak private bytes are its bound plus 14–22 MiB of
  process overhead. On a 20000² DEM (1.49 GiB raw file), full-width
  strips of 256 rows peak at 54 MiB on 1 worker, 489 MiB on 12 and 964
  MiB on 24 (bounds 39, 472 and 945); 1024×1024 tiles on 12 workers peak
  at 113 MiB at 4096² and 114 MiB at 20000² (bound 96). The whole raster
  in memory peaks at 3076 MiB.
- Throughput is copy-bound: from 12 workers Slope, Hillshade and Clamp
  all run at 770–845 M cells/s from raw files and about 1.1 billion from
  memory sources, whatever their compute cost, because every cell is
  copied into a buffer and out again. Tile shape matters for files: a
  source or sink makes a call per row of a narrow tile, so 1024×1024
  tiles of a 20000-wide file run at about 200 M cells/s with 12 workers
  and 256×256 tiles at 65–85. Use full-width strips.

## 28. Memory Bandwidth Awareness

Not every operation will scale dramatically with SIMD.

Simple operations such as:

```go
dst[i] = src[i] * 2
```

may quickly become memory-bandwidth-bound.

This has been measured (benchmarks/algebra/RESULTS.md):

- From 4096² on, every algebra operation is capped at about 22 GB/s of
  single-threaded memory traffic.
- SIMD adds only 1.1–1.2× to Add, Sub and Mul there, although the CPU
  computes 4–6 billion cells/s while the operands fit in cache.
- At 16384² throughput drops further, likely from TLB and prefetcher
  effects. That argues for tiled execution even for pointwise
  operations.
- Workers hit the same wall (benchmarks/engine/RESULTS.md): from 4096²
  Slope, Hillshade and Clamp all flatten at 19–23 GB/s, whatever their
  single-worker compute cost, and SMT adds nothing. Full-width strips
  stream memory better than 256×256 tiles at every worker count.
- Chunked execution copies each cell into a tile buffer and out again
  (benchmarks/chunked/RESULTS.md), so it hits the wall sooner: every
  operation flattens at about 800 M cells/s from raw files and 1.1
  billion from memory sources with 12 workers.

The terrain kernels are the other case (benchmarks/terrain/RESULTS.md).
Gradient, Slope, Aspect and Hillshade hold their throughput from 256² to
16384² on one core, because a 3×3 stencil with an arctangent or a square
root per cell does enough work per byte that memory keeps up: Slope,
Aspect and Hillshade ask for 4–11 GB/s of the 22 a core can pull, so the
kernel is the limit. Gradient, which writes two outputs and so moves 12
bytes per cell, runs at 13–20 GB/s and is the one terrain operation near
the wall. SIMD is worth 8.7× to Aspect, 5.7× to Hillshade, 4.1× to Slope
and 2.6× to Gradient at 4096². That is why workers scale these operations
where they barely scale the algebra ones.

Convolution and resampling do, and both are now measured on AVX2 and
NEON: convolution in §53 and benchmarks/focal/RESULTS.md, by radius
(compute-bound at every radius and size, since the AVX2 column pass stopped
colliding in L2 at 64 KiB row strides), and resampling in §54 and
benchmarks/resample/RESULTS.md, where every method at every scale is
compute-bound, asking at most 8 GB/s. Interpolation and point-cloud
filtering should behave like the terrain kernels too, and are not
measured yet.
Benchmarks distinguish compute-bound, cache-bound, memory-bound and
IO-bound operations.

`stratabench` classifies each measured operation this way (§38).

## 29. Operation Fusion

A chain of pointwise operations should run as one kernel, with the values
between them in registers.

Example:

```text
Normalize
   ↓
Scale
   ↓
Clamp
```

Instead of three full passes:

```text
load → normalize → store

load → scale → store

load → clamp → store
```

the engine supports:

```text
SIMD load
   ↓
normalize
   ↓
scale
   ↓
clamp
   ↓
SIMD store
```

For many workloads this matters more than SIMD alone.

Two earlier decisions are what make it writable at all:

- SIMD kernels are Go (ADR 0001), so a fused kernel is Go, not assembly.
- Validity is a separate bitmap (§31), so a fused data kernel is plain
  branch-free arithmetic. Validity is computed once for the whole chain,
  which §52 already does.

### What it is on top of

§52's `Pipeline` runs a chain of kernels over one span, so a tile is
loaded once instead of once per operation. What it does not remove is the
values between the stages: each one is a span-sized buffer the engine
lends, written by one stage and read by the next. A span is up to
`bandCells` = 65536 cells, so each is up to 256 KiB — one Zen 2 core's
whole L2 — and a five-stage chain has four of them. That traffic is
exactly what `engine.Stats` does not count (§51), which is why a staged
pipeline reads 28 B/cell while moving considerably more.

Fusion removes them. The engine's structure does not change at all: a
`Pipeline` still satisfies `Kernel`, still declares radius 0, and still
lets the engine derive its validity in one pass. Only `Process` changes,
from "run each stage over the span" to "run the whole chain over a few
cells at a time".

### The left-deep cut

The running value is a register only if there is one of it. So this cut
fuses a **left-deep** chain and nothing else:

```text
((a · b) · c) · d          fused
 (a · b) · (c · d)         staged: two live intermediates
```

Concretely, a `Pipeline` lowers when every stage is a `FusableKernel`
with one output, stage 0 starts from a pipeline input, every later stage
takes the stage before it as its *first* operand, every second operand is
a pipeline input, and the last stage produces the output. A pipeline that
fails any of those runs staged, which is the reference the fused form is
tested against; nothing panics, because a diamond is a legal pipeline and
not a mistake.

That is not a narrow case. A weighted factor product is exactly this
shape, and so is any chain of arithmetic with a `Clamp` or a `Rescale` on
the end.

Operands are never swapped, even for an operation that would commute. A
caller who wants `src - acc` is asking for a different chain and should
say so, and the rule costs nothing: the shapes the typed entry points
build are left-deep already.

### The form

`internal/vec` holds the evaluator, because it already has both
backends, the `kernelSet` swap table, and `min8`/`max8` — the two
functions that reproduce Go's builtin `min` and `max` over NaN and signed
zeros, which a fused chain needs as much as `Min` and `Max` do.

```go
type Op uint8 // Add Sub Mul Div Min Max AddScalar MulScalar Affine Clamp Abs Sqrt

// Step is one operation of a Chain: what to do, which of Run's slices a
// binary op reads, and the immediates of the ops that take them.
type Step struct {
    Op  Op
    Src int
    K   [2]float32
}

type Chain struct{ ... }

func NewChain(inputs, first int, steps []Step) *Chain
func (c *Chain) Run(dst []float32, srcs [][]float32)
```

`exec.FusableKernel` is how a kernel names itself as a step:

```go
type FusableKernel interface {
    Kernel
    Fuse() (step vec.Step, ok bool)
}
```

The `ok` result is for a kernel whose arity depends on its parameters and
so is a step for some of them and not others. A kernel that says `true`
and whose arity contradicts its operation is a programming error and
panics.

`algebra`'s `binaryOp` had to learn which operation it is. It carried a
bare `func(dst, a, b []float32)`, which has no identity a fusion pass can
switch on, so it now carries the `vec.Op` beside the function: the
function is what `Process` calls and the op is what a `Pipeline` fuses
on. `Clamp` and `transfer.Rescale` implement `Fuse` too. `Reclass` and
`Lookup` do not and will not as written: their inner scan over a table is
per cell, data-dependent and breaks early, which is not a step a lane of
a vector can take (`internal/curve`).

### One extension to the contract

`Chain.Run` takes its operands as `[][]float32`, which `Process` may not
allocate per band (§26). So `ScratchSize` and `Scratch` gain `Runs`, for
the same reason they gained `Views`: slice headers a kernel hands on and
cannot make for itself. A fused pipeline asks for `Runs` and **no cells
at all**, which is the claim this section makes, as an assertion.

### The two backends are not the same shape, on purpose

The vector backends are the ones this section is named for. They carry
the accumulators in registers for the whole chain — four vectors at a
time, so 32 cells on AVX2 and 16 on NEON, each input loaded once, `dst`
stored once, nothing else written anywhere. AVX2 follows that file's
existing rules: the operand array is filled before the first 256-bit
instruction, immediates are broadcast at the top of the lane function,
and `ClearAVXUpperBits` comes before the scalar tail (ADR 0001,
`internal/stencil/simd_amd64.go`). NEON needs none of that, and `min4`
and `max4` are already Go's builtins where `min8` and `max8` are a
repair.

The scalar backend carries the value through a block of 2048 cells
instead. One cell at a time with the operation dispatched per cell would
put a switch and two bounds checks against one multiply; a block puts
them against 2048. Each step is then the very kernel the unfused chain
would have called, over the same cells in the same order, which makes the
scalar chain the staged chain's bits by construction and leaves
`internal/vec`'s tight loops as they were. It is not a register, but it
is 8 KiB rather than 256, and that is where the traffic was.

The two must still agree bit for bit, which is what the tests are for.

### Bit-exactness has three landmines

1. **No FMA contraction.** `OpAffine` keeps `float32(acc*K0) + K1`, the
   explicit conversion `scalarAffineFloat32` has, and the AVX2 step is
   `VMULPS` then `VADDPS`. A running value crossing a multiply and an add
   in a register is exactly where a compiler would contract them, and
   arm64 would where amd64 would not.
2. **`min`/`max` are not the hardware's.** `min8`/`max8`, never `.Min`
   and `.Max`.
3. **Bounds checks.** The chain evaluators' tightest loop is over the
   chain's *operations*, not its cells, so the checks it leaves are
   amortised over a block or a lane rather than paid per element. They
   are named in `internal/vec/bce_test.go`'s allow list with that reason,
   and the element loops they call stay checked everywhere else.

### What it is worth

Not a counter figure. `engine.Stats` reads 28 B/cell for a six-input
product whether the stages are staged or fused — it never counted the
scratch — so what fusion changes is time, and §51's number simply stops
being optimistic.

`benchmarks/fusion` is where it is measured, because the spike that asked
the question was already there. It runs §52's six-factor product four
ways — five chained `MulTiled` calls, a staged `Pipeline`, that same
`Pipeline` lowered, and a `Fused` kernel written by hand as a generator
would have had to emit it — and `TestFormsAgree` holds all four to the
same bits.

4096², mask off, NEON, Mcells/s, medians of 3, Apple M4 (10 cores):

| workers | tiles | chained | staged | lowered | fused | lowered ÷ staged | lowered ÷ fused |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | strips | 1170 | 1171 | 1562 | 2079 | 1.33× | 0.75 |
| 1 | 256×256 | 209 | 274 | 354 | 437 | 1.29× | 0.81 |
| 10 | strips | 1581 | 2928 | 3188 | 3502 | 1.09× | 0.91 |
| 10 | 256×256 | 925 | 1159 | 1769 | 1961 | 1.53× | 0.90 |

So the lowering is worth 9–53% over the staged pipeline, and it collects
three quarters to nine tenths of what hand-written code gets. The gap to
hand-written is the dispatch: the lane loop runs a switch and a slice
advance per operation, where the hand-written kernel has the chain in its
instruction stream. That is the argument for a generator, and it is now a
number rather than a hope — 10 to 25%, on this machine, for this chain.

### Four vectors at a time, which is not a detail

The first version of the lane loop carried one vector, and it was
*slower* than the staged pipeline it replaced: 840 Mcells/s against 1171
on one worker. Dispatching an operation costs about as much as performing
it on four lanes, so a chain that pays that per vector spends more on
deciding than on arithmetic, and carrying the value in a register buys
nothing.

Carrying `chainWide` = 4 vectors divides the dispatch by four and took
the same case to 1562. Four accumulators are four registers of sixteen on
amd64 and of thirty-two on arm64, so nothing spills, and what is left
over — fewer than 32 cells of a span — goes to the scalar block
evaluator rather than to a second copy of the switch.

This is the §51 lesson again: the design said "in a register" and was
right about the memory, and a cheap measurement caught that the
instruction it cost was the thing that mattered.

### Where these numbers sit against the spike's

`benchmarks/fusion/RESULTS.md` measured chained, staged and hand-fused on
a Zen 2 with AVX2 and concluded that out of cache register-level fusion
was worth about 5%, and that the large in-cache gaps were the
`Pipeline`'s per-call scratch allocation rather than fusion. Both of its
caveats have since moved: scratch comes from a pool now, and these
numbers are a different machine, with far more memory bandwidth per core,
where the same product is much less memory-bound and so has much more to
gain from not moving the intermediates. Neither run is wrong; they are
two machines, and the Zen 2 column of this table is not yet filled in.

What both agree on is the ordering: tile-level fusion (§52) is the larger
win, and register-level fusion is the smaller one on top.

### Where it lives

`internal/vec/chain.go`, `chain_amd64.go`, `chain_arm64.go` and
`scalarChainFrom` in `scalar.go`; `FusableKernel` and `Scratch.Runs` in
`internal/exec`; `Pipeline.lower` and `processFused` in
`internal/exec/pipeline.go`; `Fuse` methods in `algebra` and `transfer`.
The `lowered` arm of `benchmarks/fusion` is what measures it.

Nothing is public. A caller still cannot build a `Pipeline` (§52), so
fusion is reached only through the typed entry points that will be
written for the shapes that want it. Publishing `Kernel` is the same
separate decision it was.

### Testing

The staged pipeline is the reference, and it has a reference of its own:
§52's `unfused`, which runs each stage as its own whole-raster call. So
every test here is one comparison — build the same pipeline with fusion
on and with it off, and require the same bits.

- `TestPipelineFusedMatchesStaged` and its chunked twin run the §23
  matrix — every tile size, worker count, windowed and compact operands,
  masked and not — on every backend the build has.
- `TestPipelineFusedEveryOp` runs a chain using every operation a chain
  can hold, where the unary and immediate steps differ most between the
  two backends.
- `TestPipelineFallsBackOffTheCut` names the five shapes that do not
  lower — diamond, right-deep, an unfusable stage, a kernel too wide to
  be a step, an output that is not the last stage's — and requires each
  to run staged and still equal its unfused form.
- `TestPipelineFusedAsksNoCells` is the claim: `ScratchSize{Runs: n}`.
- `requireFused` in §52's own tests keeps them covering the fused path
  rather than quietly falling back to the staged one.
- In `internal/vec`: `TestChainMatchesStaged` over random programs,
  random data including every float32 class, and lengths straddling the
  lane widths, the unrolled group and the scalar block;
  `TestChainIsNotFused` for the two multiply-add shapes;
  `TestSIMDChainMatchesScalar` for the lane loop, on whichever backend
  the build has.
- In `benchmarks/fusion`, `TestFormsAgree` holds the lowered chain to the
  same bits as the chained calls, the staged pipeline and the
  hand-written kernel, on every backend, tile shape and worker count the
  benchmarks use.

The acceptance harness (§39) is byte-identical to the parent commit,
which is the check that a refactor reaching into `algebra` and `transfer`
changed nothing a caller can see.

Status: done, for the left-deep cut, on all three backends. Still to do:
the Zen 2 AVX2 run, which is the suite's headline machine (§38) and the
one the spike's own numbers came from; a chain with two live
intermediates, which needs a register file and is worth what it measures;
a generator, which this now prices at 10–25% for a chain of multiplies
and which nothing yet asks for; and the typed entry point that would let
a caller reach any of this.

Status: partly done. Tile-level fusion of radius-0 chains is done: the
internal `Pipeline` (§52) runs a chain of kernels on each tile while it is
loaded, and computes validity once for the chain. Its intermediates still
go through scratch memory, so register-level fusion, which removes them,
is not started, and neither is a public way to build a chain.

Register-level fusion has been measured before being built.
`benchmarks/fusion` runs §52's six-factor product as five chained calls,
as a `Pipeline` and as a hand-written fused kernel, all bit-identical.
From 4096² up the fused kernel is 0.96–1.11× the `Pipeline` in default
strips, and both reach the DRAM limit, about 26 GB/s on 12 workers: the
scratch traffic between stages stays in cache. Chained to `Pipeline` is
the large step, 1.6–2.6×. At 1024² the fused kernel wins by up to 5.5×,
but that measures the `Pipeline` allocating its scratch per worker on
every call (12.6 MB/op on 12 workers), not fusion. So a generator for
pointwise chains is not worth building on this evidence. Measure again
for stencil or longer chains, whose intermediates would spill out of
cache, and at in-cache sizes once scratch is not allocated per call.
The run is one `-count 3` on an unquiet machine; see
`benchmarks/fusion/RESULTS.md` for the numbers and their caveats.

## 30. Streaming Pipelines

Pipeline-style processing — source, several operations, sink, with no
materialised intermediates — is the raster form of what §52 builds: a
chain of kernels run on each tile while it is loaded. Point-cloud
pipelines (reader, filter, transform, aggregate, rasterize) follow the
same idea over batches once §11 exists.

Status: partly done for rasters. The radius-0 `Pipeline` (§52) runs a
chain of pointwise stages from one source read to one sink write, inside
the engine only. Stencil stages, a public pipeline API and point-cloud
streams are not started.

## 31. NoData and Validity

**Decided (STRATA-3, benchmarks/nodata/RESULTS.md): a separate validity
bitmap, and no NoData value.**

```go
type Float32Raster struct {
    Data        []float32
    // ...
    Valid       []uint64 // nil = every cell valid
    ValidOffset int
}
```

Candidates benchmarked were sentinel values, NaN, and a validity bitmap.
The rules that follow from the decision:

1. **Bit layout.** Bit `ValidOffset+i` of `Valid` is the validity of
   `Data[i]`. Bits are LSB first and a set bit means valid. This is
   Arrow-compatible.
2. **Validity is never inferred from Data.** A NaN or −9999 with its bit
   set is an ordinary value. Data under a cleared bit is unspecified.
3. **Kernels compute every cell unconditionally.** They derive output
   validity with word-level operations: an AND of the inputs for
   pointwise operations, and erosion by the radius for stencils.
4. **A nil mask means all valid.** If every input has a nil mask, no mask
   work is done at all.
5. **Fill values belong to IO adapters.** `GDAL_NODATA`, `_FillValue`,
   `missing_value` and valid ranges become `Valid` on read. Writers put a
   fill value under cleared bits.
6. **Invalid results are not produced implicitly.** Operations that
   should create NoData (division by zero, out-of-domain input) produce
   ordinary IEEE values, which stay valid. An explicit opt-in operation
   can invalidate non-finite results later.

Measured cost: about the same as NaN on pointwise operations, and 0–20%
more on the 3×3 slope. Sentinel values cost 14–25% more. Only the bitmap
has none of the correctness hazards of the other two, and it works
unchanged for integer data.

## 32. Point-Cloud to Raster Workflows

Crossing representations — rasterizing point batches into a DEM or a
canopy-height grid — is a core operation once point batches exist (v0.7):
an aggregation kernel from sparse input to a dense output. The LiDAR →
DEM → terrain workflow built on it is a caller's workflow and the §44
demo, not core scope.

## 33. 3D and Voxel Direction

Raster is the first compute model, not the permanent boundary. Voxel
grids (`fuel_density[x,y,z]`, `moisture[x,y,z]`) are dense 3-D arrays and
should fall out of §10 rather than need an engine of their own. Status:
v0.10 experimentation.

## 34. IO Architecture

Formats are adapters, never the core. An adapter implements the source
and sink interfaces of §24 and converts its format's data types and fill
values into `float32` plus validity at the boundary (§9, §31).

- An adapter that needs cgo — GDAL, which would give access to its
  format ecosystem — lives in its own module, so the core stays cgo-free.
  GDAL is a data source, not a foundation.
- Candidate adapters: GeoTIFF, COG, Zarr, NetCDF, LAS/LAZ.
- **Each adapter is a Go module of its own, in this repository**
  (`cog/`, like `acceptance/` and `lint/`), decided with the first one
  ([ADR 0002](docs/adr/0002-cog-adapter.md)). The core's go.mod stays
  free of format libraries, and an adapter reaches strata only through
  its public API, so a gap in the source and sink interfaces shows up as
  a gap rather than a workaround.
- **An adapter is judged by the tool the format is known from.** Its own
  tests share its author's reading of the specification, so its evidence
  of correctness comes from outside: for GeoTIFF, GDAL's reading of the
  same files, exactly (`acceptance/cogcheck.sh`).

Status: GeoTIFF/COG reading is done: `cog.Open` over an `io.ReaderAt`,
`File.Source` as an `engine.RasterSource` for one band of one resolution
level, with a byte-bounded cache of decoded blocks per source and, for a
pixel-interleaved file, one of compressed blocks that its band sources
share, so each block is fetched once for all bands. It reads classic and
BigTIFF, tiles and strips, chunky and planar, 8/16/32-bit integers and
32/64-bit floats, none/LZW/Deflate/PackBits/ZSTD with predictors 2 and 3,
overviews, sparse blocks, GDAL NoData (compared in the native type) and
the geotransform and EPSG code. It is bit-identical to GDAL 3.14 on 98
files, 58.5M cells, and all eight of `cogsabotage.py`'s planted defects
fail that comparison. `cog.NewHTTPReaderAt` reads the same files from a
URL (S3, GCS, any HTTPS server, presigned or public) with range requests
and the standard library only: a 64 KiB header prefetch that serves Open,
then one request per block past it, 206 required, bounded retries of 429,
5xx and truncated bodies, at most 8 requests in flight, and If-Match on
the ETag so a replaced file fails instead of mixing versions. Read through
it from nginx, all 98 files are again identical to GDAL's reading, and
every file but one costs exactly one prefetch plus one request per stored
block past it, pixel-interleaved ones included: 8,344 requests and
165.7 MiB for 165.7 MiB of files, where one fetch per band had cost
10,814 and 217.0 MiB (`acceptance/coghttpcheck.sh`). Its speed against
GDAL is measured in [benchmarks/cog/RESULTS.md](benchmarks/cog/RESULTS.md):
on one core GDAL reads a float32 COG 1.4–2.1× faster (libdeflate, against
Go's inflate, is most of the gap), yet slope over a COG still beats
`gdaldem slope` on the same file by 1.8–2.6×, and by 4.3–5.6× on 12
workers. Open: writing (a COG sink), internal masks, and a default cache
that scales with the file's block rows for many workers.

## 35. Use Existing Format Libraries Where Possible

Adapters wrap existing format libraries (a Zarr package into `Array`, a
LAS/LAZ library into point batches) rather than reimplementing parsers.
strata's differentiator is computation, not parsing.

The GeoTIFF adapter is the exception, recorded in
[ADR 0002](docs/adr/0002-cog-adapter.md): no Go library reads a
window's blocks of a floating-point TIFF, and GDAL means cgo. So `cog`
parses the container itself and wraps a library only for LZW, Deflate
and ZSTD (`klauspost/compress`). An exception to this rule needs an outside
judge, and GDAL is that judge here (§34).

## 36. CRS and Reprojection

Do not attempt to replace PROJ. Reprojection is the caller's
preprocessing, done before data reaches strata (with gdalwarp, PROJ or a
format adapter), not part of execution.

### The contract

1. **One CRS per computation.** Every input to one operation, and its
   output, is in the same CRS. strata never transforms coordinates.
2. **Grid operations assume a projected CRS with ground units.** Cell
   sizes, distances and neighbourhoods are taken as lengths on the ground,
   in the same unit on both axes. A raster in a geographic CRS (degrees)
   must be projected first; `terrain` would otherwise return wrong slopes
   and aspects without an error, since it cannot tell.
3. **The CRS is a label.** `raster.CRS` is opaque (`Code string`, for
   example `"EPSG:25833"`). A `Grid` carries it along, and nothing
   interprets it: no parsing, no lookup of units or axis order.
4. **Operations that see grids check labels.** Wherever an operation takes
   two or more `raster.Grid`s, it panics unless their CRSs match
   (`raster.CRS.Matches`): equal codes, or either code empty. Empty means
   unknown, and the caller vouches for it. Codes are compared as strings,
   so `"EPSG:25833"` and `"urn:ogc:def:crs:EPSG::25833"` do not match;
   normalise codes at the IO boundary.
5. **Operations that take bare rasters cannot check.** `algebra`, `focal`,
   `terrain`, `transfer`, `reduce` and the engine see `Float32Raster`s and
   `engine.RasterSource`s, never grids (§9), so for them the contract
   is the caller's to keep. They require equal dimensions, not equal grids.

Today `resample` is the one operation that takes grids, and it checks
(§54). Mosaics, alignment helpers (v0.8) and point rasterization (v0.7)
will take grids too, and must check the same way.

### If transformation is ever needed

Reprojecting inside strata would save a full pass over the data when it
is fused with the computation after it. That is an optimisation to make
once a workload shows it matters, not a gap. If it is built, it goes
through a narrow abstraction over coordinate columns rather than point
structs (§11), backed by PROJ through an adapter:

```go
type Transformer interface {
    Transform(src, dst CRS, x, y []float64) error // in place
}
```

The outside reference would be `gdalwarp -et 0` (exact), since gdalwarp's
default transformer interpolates between exact points along each row
and is off by up to 0.125 source pixels.

## 37. Memory Management

Large workloads may create substantial temporary allocations.

Avoid allocating complete temporary rasters for every operation:

- **Today.** Every `algebra` and `terrain` operation writes into a
  caller-supplied destination. The benchmark suite checks that operations
  make 0 allocations.
- **Done.** Chunked execution allocates tile and halo buffers once per
  worker per call and reuses them across tiles, with no pooling (§27).
  Raw sources and sinks read and write straight into them and allocate
  nothing.
- **Done.** A `ScratchKernel`'s working memory (§52) is pooled across
  calls, in `internal/exec/scratch.go`: one `sync.Pool` per kind and size
  class, a block per worker lent for the call and returned when its
  workers have stopped, never zeroed. That is the one pool, and it is
  allowed by the rule below because scratch's ownership is settled: the
  engine owns it, lends it for one call, and the contract already says
  its contents are unspecified and that a kernel must not keep it. Tile
  buffers are still per call.

Per-worker scratch for pipeline stages exists (§52). Workspaces shared
across calls, or arena-like temporaries, may follow; do not introduce
aggressive pooling before ownership semantics are stable.

## 38. Benchmarks

Benchmarks should be part of the project's identity.

The suite is implemented (STRATA-10). `benchmarks/README.md` covers running
it and adding categories.

```text
benchmarks/
├── README.md
├── internal/suite/     shared harness: sizes, backend switching, metrics
├── cmd/stratabench/    go test -bench output → §42 headline, speedups, §28 class
├── algebra/            implemented, RESULTS.md
├── engine/             implemented (STRATA-9): Slope, Hillshade, Clamp by workers and tiles, RESULTS.md
├── chunked/            implemented: the same over raw files with bounded memory; RESULTS.md with the §43 demo
├── terrain/            implemented: Gradient, Slope, Aspect, Hillshade plain, RESULTS.md
├── focal/              implemented: Correlate, Gaussian, Mean, Min, Max by radius, RESULTS.md (§53)
├── resample/           implemented: every method at 2×, 4×, ½, 1/1.37, against
│                        direct 2-D, RESULTS.md (AVX2 and NEON), §54
├── gdal/               implemented: the same operations timed against gdaldem, RESULTS.md
├── nodata/             STRATA-3 spike, not part of the suite
├── reduce/             implemented: Min, MinMax, Sum, Stats against read bandwidth, RESULTS.md (§49)
├── fusion/             implemented: register-level fusion measured before building it, RESULTS.md (§29)
└── transfer/           planned: lands with a vector Reclass or Lookup (§50)
```

`algebra.Mask` has no suite benchmark yet (§42), and `transfer`'s
numbers in §50 come from the package's own `bench_test.go`.

Benchmark names:

```text
Benchmark<Op>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=<W>[/tiles=<T>]
```

Metrics: ns/cell, cells/s, GB/s, allocations, SIMD/scalar speedup,
parallel scaling and peak memory.

- Peak memory is not measured by the harness's benchmarks. The algebra,
  engine and terrain suites' peaks were measured by hand and documented in
  their `doc.go`. `suite.ProcessMemory` reads the OS counters, and
  `cmd/stratademo` reports each run's peak against the §27 bound from a
  child process of its own (§43).
- Worker scaling compares SIMD with 1 worker, with one worker per physical
  core, and with one per logical CPU (`suite.Workers`). Only categories
  that run through the engine (`benchmarks/engine`) have `workers` above
  1; they add a `tiles` level for the tile shape.
- Published numbers come from a `GOEXPERIMENT=simd` build (§3).
- **An external baseline.** The suite's speedups are all measured against
  strata's own scalar kernels, which says how much the lanes are worth
  but nothing about whether the whole thing is fast. `benchmarks/gdal`
  answers that against another program: it times `gdaldem` and strata on
  the same raster, in the same container, under the same timer, and then
  differences the files it timed, so the number is a speed at the same
  answer. `gdaldem` is single-threaded, so the one-worker row is the
  like-for-like one; the rest show what the engine adds. The suite's
  scalar/SIMD numbers remain the internal measure, and the two are not
  interchangeable.

Raster sizes: 256², 1024², 4096² and 16384².

## 39. Correctness Testing

Every SIMD implementation should be compared against its scalar equivalent,
bit for bit (§15). Tests run in every build configuration: default, and
`GOEXPERIMENT=simd` on both amd64 (AVX2) and arm64 (NEON).

Test:

```text
ordinary values
NaN
infinity
negative values
zero, including -0 vs +0
NoData
odd dimensions
non-contiguous stride
vector tails
tiny arrays
tile boundaries
halo boundaries
mixed validity
stale mask bits in dst
```

For the engine, also test that tiled output equals the whole-raster call
for every tile size (including tiles smaller than the halo) and every
worker count (§23).

Beyond the table-driven tests, every package that takes shapes, offsets,
options or float bits from callers has native Go fuzz tests (`Fuzz*` in
`fuzz_test.go`, decoding inputs through `internal/fuzzdata`). They check
against exact references (math/big shapes, bit-at-a-time masks, brute-force
overlap, naive kernels, the scalar backend) and that every rejection is a
panic with the package's own message, raised before anything is written,
never a runtime error. Reductions are checked the same way: `FuzzReduce` against a plain loop
over the cells, and `FuzzReduceRelations` against the relations of §49.
Metamorphic targets (`Fuzz*Relations` in
`metamorphic_test.go`, with transformations in `internal/rastertest`) check
relations between results instead, which catch answers that are wrong but
in range: symmetries of the grid, translation and scaling of a DEM,
algebraic identities, and the locality and NoData rules of stencils, each
side run through its own execution path and memory layout. Horn's fixed
evaluation order makes most of them bit-exact. Their seed corpora run with
`go test`; to fuzz one target, in both builds:

```text
go test ./terrain -run '^$' -fuzz '^FuzzTerrain$' -fuzztime 5m
GOEXPERIMENT=simd go test ./terrain -run '^$' -fuzz '^FuzzTerrain$' -fuzztime 5m
```

Every metamorphic target also runs as a property test, under rapid, the
other dependency: `TestTerrainRelations`, `TestAlgebraRelations`,
`TestProcessRelations`, `TestReduceRelations` and `TestTransferRelations`
drive the bodies of their packages' `Fuzz*Relations` targets with
rapid's generators in place of a fuzz input. Both drivers go
through `fuzzdata.Source`, an interface over the values a test builds its
operands from, and report through `rastertest.TB`, the part of testing.TB
the relations use and rapid.T also has, so each relation has one
implementation and two ways of searching for a case that breaks it:
`internal/fuzzdata` decodes the values from the bytes of a corpus entry,
and `internal/rapidsource` draws each one from rapid, which shrinks a
failing case draw by draw and prints what is left, along with a seed that
reruns it. Fuzzing searches deeper, for as long as it is given; rapid runs
with `go test`, needs no corpus, and says what broke rather than which
bytes broke it. Three mutations show the difference: swapping the two axis
scales in `terrain/stencil.go`, which the range checks of `FuzzTerrain`
cannot see, fails within ten cases; a `Min` that answers the wrong operand
for a NaN shrinks to a 1×1 raster; and eroding validity by one less than
the radius shrinks to a 3×5 raster with one input and one output.

The raw source and sink are tested over files that fail partway through
a call. `internal/faultio` puts `testing/iotest`'s wrappers under the
`io.ReaderAt` and `io.WriterAt` they work through, and chooses by file
offset which calls suffer. Reads that come back in pieces (`HalfReader`,
`OneByteReader`, `DataErrReader`) must give exactly the cells a whole
read gives, in the same number of calls, for grouped and per-row reads
and with and without a fill value; reads that fail (`ErrReader`), time
out (`TimeoutReader`) or run off the end of a short file must reach the
caller as that error, named with the rows being read. The same faults
run under `ProcessChunked` with several workers, on top of the per-tile
injection of `failingSource`, and leave no goroutine behind. Two of
them are contract violations a real file can commit: a `ReadAt` that
returns a short count without an error, and a write that loses its tail
and reports success (`TruncateWriter`). The first was already refused;
the second was not, because `RawSink` checked the error and not the
count, and now fails with `io.ErrShortWrite` instead of losing the rows
silently.

The tests of `internal/exec`, the only package that starts goroutines, and
of `engine` fail if any test leaves a goroutine behind (goleak, one of
the module's two dependencies, both test-only). `TestNoLeaksOnFailure` ends
calls in every early way (kernel, source and sink panics, IO errors,
cancellation while workers are mid-call) with several workers.

The cancellation tests run in `testing/synctest` bubbles, where time is
virtual. A band or tile costs a tick (a kernel call or a read that
sleeps) and the stop — a cancellation, a deadline, or a failing read or
write — lands half a tick into a round, while every worker is asleep
inside its unit of work. What the scheduler would otherwise decide is
then exact: with W workers a stop in round k leaves exactly W·(k+1) units
claimed and finished, and none started after it, instead of the
"at most W-1" bound these tests could assert before.

`TestNoBoundsChecksInLoops`, in internal/vec and internal/stencil,
compiles its package with the compiler's optimization log (`-json`,
which reports every instance rather than one line per source position)
and fails if a bounds check survives inside a tightest loop, one with no
loop of its own, whose every iteration pays for it. The shared parsing
is `internal/bcecheck`. A check per element costs as much as the
arithmetic it guards: the vec kernels reslice their operands to the
length of the slice they range over, and the stencil row kernels read
their 3×3 window through the shifted views of `hornViews`, indexed with
the loop variable alone, because the compiler cannot prove `r0[i+1]` and
`r0[i+2]` in bounds from a range over a slice two cells shorter and
charged two checks per cell for them. That is worth 2 to 7% on gradient
and slope and 1.9% on the package's benchmark geomean. Checks outside
loops, and in loops that contain a loop, run once per call, row or chunk
and are allowed.

Two checks stay, each because it was measured, not assumed:
`scalarHornHillshadeRow` keeps its two per cell, since it holds the light
vector as well as the gradient and the views spill registers there (6.4
against 4.6 ns/cell for a 255-cell row, 33 to 56% slower across
`BenchmarkRowWidth`); and stencil's mask.go works a word at a time, where
a check costs a 64th as much and the word indices come from bit offsets a
caller chose. The test names both, so removing one is a change to it.

The module is clean under staticcheck with every check enabled, and under
gosec, in both builds. Both tools must be built with this module's Go
version, or they cannot read its packages:

```text
GOTOOLCHAIN=go1.27.0 go install honnef.co/go/tools/cmd/staticcheck@latest
GOTOOLCHAIN=go1.27.0 go install github.com/securego/gosec/v2/cmd/gosec@latest
staticcheck -checks all ./...
gosec -exclude=G103,G404 ./...
```

golangci-lint (`.golangci.yml`) runs the standard linters with
staticcheck's full set, plus gosec, errorlint, unconvert and nolintlint.
Like the tools above it must be built with this module's Go version, and
is run for both builds:

```text
GOTOOLCHAIN=go1.27.0 go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.13.2
golangci-lint run ./...
GOEXPERIMENT=simd golangci-lint run ./...
```

G103 (every use of unsafe) and G404 (math/rand) are excluded: unsafe is
confined to address comparisons in internal/overlap and algebra, the byte
views of raw IO in engine, and Windows system calls in the benchmark
suite, each reviewed; math/rand only makes test fixtures. Other accepted
findings carry a `#nosec` comment giving the reason. Neither tool
detected the overflow in `raster.Validate` found by fuzzing: gosec's
integer overflow rule covers conversions, not arithmetic.

CI (`.github/workflows/ci.yml`) runs all of this on every push and pull
request: build, vet and test on Linux, Windows and macOS; the same under
`GOEXPERIMENT=simd` on Linux (amd64, AVX2) and macOS (arm64, NEON);
`go test -race`; golangci-lint; and `kernelborder` (§12), a custom
analyzer in the separate `lint/` module, run as a vet tool. The race and
lint jobs run in both builds, and in the SIMD build for arm64 too (lint
cross-compiled), so the vet tool sees each backend's files:

```text
(cd lint && go build -o /tmp/kernelborder ./cmd/kernelborder)
go vet -vettool=/tmp/kernelborder ./...
GOEXPERIMENT=simd GOARCH=amd64 go vet -vettool=/tmp/kernelborder ./...
GOEXPERIMENT=simd GOARCH=arm64 go vet -vettool=/tmp/kernelborder ./...
```

**The outside opinion.** Everything above is written by whoever wrote
the library, against the same understanding of the problem, so a
misunderstanding passes it. `acceptance/` is the independent check. It
is a separate module that uses strata only through its public API and
writes every input and output to plain files. A numpy program written
from published definitions (`check.py`, Horn's kernel as gdaldem
documents it, shaded relief as the cosine between the light and the
surface normal, correlation and convolution as scipy.ndimage defines
them) then judges them in 600 checks. The checks cover terrain, focal,
algebra and reduce, in all three forms, on three synthetic DEMs, with
tolerances derived from the float32 error bound rather than tuned.
`sabotage.py` injects plausible defects one at a time and requires each to
turn the result red. `gdalcheck.sh` runs `gdaldem` on a real raster in
Docker and differences it against strata's `Chunked` output. `transfer`
is not covered by the harness yet.

Every new public operation gets a reference in `acceptance/` as well as
tests here; an operation with no outside reference is not done.

`tools/herbie/` runs [Herbie](https://herbie.uwplse.org/) on the kernel
formulas, restricted to operations a SIMD kernel can mirror without FMA,
to find rewrites that are more accurate or cheaper. It is a search, not a
check, and runs by hand; [tools/herbie/RESULTS.md](tools/herbie/RESULTS.md)
triages the latest run.

Point batches, when they exist, add empty batches, attribute-length
mismatches, sparse masks and large coordinate ranges to the table above.

## 40. Package Layout

Current:

```text
strata/
├── raster/                    implemented
│   ├── raster.go              Float32Raster, constructors, Row, Index, Validate
│   ├── window.go
│   ├── mask.go                bitmap helpers, range AND/copy/fill
│   └── grid.go                Grid, CRS placeholder, Dataset
│
├── algebra/                   implemented
│   ├── doc.go
│   ├── algebra.go
│   └── tiled.go               tiled entry points and their kernels
│
├── reduce/                    implemented (STRATA-12, §49)
│   ├── doc.go
│   └── reduce.go              Count, MinMax, their engine entry
│                               points and their reducers
│
├── transfer/                  implemented (STRATA-13, §50)
│   ├── doc.go
│   ├── transfer.go            Reclass, Lookup, Rescale, RescaleRange,
│   │                           table checks and the three kernels
│   └── tiled.go               tiled and chunked entry points
│
├── focal/                     implemented (§53)
│   ├── doc.go
│   ├── focal.go               MaxRadius, the engine calls, option checks
│   ├── correlate.go           Correlate, Convolve
│   ├── separable.go           CorrelateSeparable, Gaussian
│   └── box.go                 Mean, Min, Max
├── resample/                  implemented (§54)
│   ├── doc.go
│   ├── resample.go            Method, Options, Resample, ResampleTiled,
│   │                           ResampleChunked and their checks
│   ├── band.go                bands over the resamp kernels, validity
│   └── tiled.go               tiling, bands, the chunked driver
│
├── terrain/                   implemented
│   ├── gradient.go
│   ├── slope.go
│   ├── aspect.go
│   ├── hillshade.go
│   ├── curvature.go
│   ├── ruggedness.go
│   └── stencil.go
│
├── engine/                    public engine configuration
│   ├── engine.go              Options (STRATA-8)
│   ├── source.go              RasterSource / RasterSink, memory source and sink
│   ├── raw.go                 raw float32 file source and sink, RawOptions
│   ├── rawfile.go             RawFile: one file, several handles
│   └── stats.go               Stats, the traffic counter (§51)
│
├── internal/
│   ├── vec/                   implemented: scalar.go, dispatch.go, simd_amd64.go
│   ├── stencil/               implemented: horn.go, aspect.go, curvature.go, rugged.go,
│   │                           mask.go, simd_amd64.go, simd_arm64.go
│   ├── focalrow/              implemented (§53): focalrow.go (scalar, dispatch),
│   │                           simd_amd64.go, simd_arm64.go
│   ├── accum/                 implemented (§49): exact float32 Sum and Moments,
│   │                           accum.go, moments.go, result.go, simd_amd64.go
│   ├── summary/               implemented (§49): the Stats reducer and Runs,
│   │                           shared by reduce and, later, terrain
│   ├── curve/                 implemented (§50): curve.go, scalar.go
│   │                           table-driven Reclass and Lookup, scalar only
│   ├── pointwise/             implemented (§50): operand checks and validity
│   │                           for the radius-0 packages' plain functions
│   ├── resamp/                implemented (§54), kernel package: table.go (tap
│   │                           tables), scalar.go, dispatch.go,
│   │                           simd_amd64.go, simd_arm64.go
│   ├── exec/                  kernel machinery (STRATA-8)
│   │   ├── kernel.go          Kernel, Span, Window
│   │   ├── process.go         Process, operand checks
│   │   ├── tile.go            tile and band planning, cancellation
│   │   ├── halo.go            halos, edges, validity
│   │   ├── worker.go          workers, band and tile scheduling, mask lock (STRATA-9)
│   │   ├── chunked.go         ProcessChunked: per-worker tile buffers, sources and sinks
│   │   ├── reduce.go          Reducer, Cells, Reduce (STRATA-12, §49)
│   │   ├── reducechunked.go   ReduceChunked: per-worker tile buffers
│   │   ├── pipeline.go        Pipeline, Stage, NewPipeline: radius 0 (§52)
│   │   ├── units.go           RunUnits: the scheduler, for packages that plan
│   │   │                       their own units (§54)
│   │   └── workspace.go       planned
│   ├── overlap/               Data and mask overlap checks
│   │
│   │                          test support (§39):
│   ├── bcecheck/              bounds checks left in tight loops
│   ├── faultio/               fault-injecting io.ReaderAt / io.WriterAt
│   ├── fuzzdata/              fuzz input decoding, the fuzzdata.Source interface
│   ├── rapidsource/           fuzzdata.Source drawn from rapid
│   └── rastertest/            grid symmetries and comparison helpers
│
├── cog/                       GeoTIFF/COG source, a separate module (§34, ADR 0002)
│   ├── doc.go                 scope, values and validity, robustness
│   ├── tiff.go                header, IFD chain, tag values, read bounds
│   ├── ifd.go                 an IFD as an image: layout, samples, blocks
│   ├── geo.go                 GeoKeys: geotransform, PixelIsPoint, EPSG code
│   ├── decode.go              decompression, predictors, float32 and validity
│   ├── cache.go               byte-bounded LRU of decoded blocks
│   └── source.go              Open, File, Source (an engine.RasterSource)
│
├── benchmarks/                implemented (§38)
├── acceptance/                black-box checks, a separate module (§39)
├── lint/                      custom analyzers, a separate module (§39)
│   └── kernelborder/          kernel packages stay span-level (§12)
├── tools/herbie/              Herbie search for float rewrites (§39)
└── docs/adr/
```

Later: `array/` (v0.3) and `pointcloud/` (v0.7), and more format
adapters (§34). Domain packages other than `terrain` live in other modules (§7).

## 41. Explicit Non-Goals for v0.1

Do not build, in the core:

```text
vector geometry, polygon overlay, vectorization of rasters
format parsers: GeoTIFF, COG, LAS, Shapefile, GeoPackage, PostGIS
CRS transformation, a PROJ replacement
domain algorithms beyond local terrain derivatives (§7)
point-cloud and voxel engines (until v0.7 / v0.10)
generic (non-float32) raster types (until v0.3)
ARM64 SIMD kernels (STRATA-11)
distributed execution, GPU, CUDA, OpenCL
full GIS workflows
```

Formats may become adapters later (§34); domain algorithms are other
modules' business.

## 42. Initial Milestone — v0.1

The first release should validate this hypothesis:

> Go SIMD + flat spatial memory + chunked execution can form a strong foundation for high-performance environmental computation.

Support:

```text
Float32 raster                              done (STRATA-4)
2D windows                                  done (STRATA-4)
validity bitmap                             done (STRATA-3/4)
scalar backend                              done
SIMD backend (amd64 AVX2, GOEXPERIMENT)     done (STRATA-2)
backend dispatch                            done

Add                                         done (STRATA-5)
Subtract                                    done
Multiply                                    done
Clamp                                       done
Min                                         done
Max                                         done
Mask                                        done

Gradient                                    done (STRATA-6)
Slope                                       done (STRATA-6)
Aspect                                      done (STRATA-7)
Hillshade                                   done (STRATA-7)

single-thread processing                    done
tiled entry points (single thread)          done (STRATA-8)
multi-worker tile processing                done (STRATA-9)
halo handling                               done: in memory and copied buffers, any worker count
bounded-memory tiled execution              done (§27; 20000² demo, §43)
memory and raw float32 file source/sink     done (§24)

benchmark suite                             done: algebra (STRATA-10), engine (STRATA-9), chunked, terrain
```

Arm64 builds run the scalar kernels in v0.1. The NEON backend came after
it (STRATA-11).

Example:

```go
dem := raster.NewFloat32(
    width,
    height,
    data,
)

slope := raster.NewFloat32Like(dem)

terrain.Slope(
    slope,
    dem,
    terrain.SlopeOptions{
        CellSize: 10,
    },
)
```

Benchmark headline, as printed by `stratabench`
(benchmarks/engine/RESULTS.md), for a 4096 × 4096 raster with no mask, in
M cells/sec, the worker columns running the strips shape:

```text
              scalar      SIMD  SIMD/scalar   SIMD + 12 workers   SIMD + 24 workers
Slope            164       686        4.19×                2644                2592
Hillshade        209      1247        5.97×                2654                2642
Clamp            924      2303        2.49×                2812                2733
```

The other categories measure the same kernels on one worker, pinned:
`terrain` adds Gradient (490 → 1282 M cells/s, 2.6×) and Aspect (61.6 →
536, 8.7×), and agrees with the plain numbers above within 4%; `algebra`
reports its six operations, which are memory-bandwidth-bound from 4096²
on (§28). `Mask`, added after those runs, has no benchmark yet.

Status: done. Every item above is implemented and measured, and the §43
validation target ran. Of what publishing needed beyond this list, the
fetchable module path (`github.com/LukasSelin/strata`), the README and CI
are done. A licence has not been chosen yet — until one is, nothing may
import strata for reuse — and nothing is tagged for importing modules to
pin (§7, §45).

## 43. First Validation Target

The first project demo should be concrete:

> Process a 20,000 × 20,000 DEM using bounded memory, tiled execution, multiple workers, and SIMD.

The input is a raw float32 file (about 1.49 GiB) read through the v0.1
file source (§24), so the demo exercises real bounded-memory reads rather
than an in-memory raster.

The benchmark should compare:

```text
scalar

SIMD

SIMD + 1 worker / physical cores / logical CPUs
```

and report:

```text
cells/sec
GB/sec
peak memory (measured, against the §27 bound)
speedup
```

The same run should also verify that the tiled output equals the
whole-raster result (§23). The whole-raster reference may be computed
once on a machine with enough memory.

This validates the architecture before expanding scope.

Status: done. `benchmarks/cmd/stratademo` generates the DEM, computes
each operation's reference in memory, runs every chunked case in a
child process and compares its output file with the reference, cell for
cell. On a 20000² DEM (1.49 GiB) with SIMD: Slope at 276 M cells/s on 1
worker and 832 on 12 (121 scalar), Hillshade 334 and 829, Clamp 389 and
843, in 256-row strips; 12 workers peak at 489 MiB against a §27 bound of
472 MiB, while the whole raster in memory takes 3076 MiB; and all 126
runs, at 4096² and 20000², equal the whole-raster result.

## 44. Second Validation Target

The second proof tests cross-representation usefulness: stream a large
LAZ file as point batches, filter ground and vegetation, rasterize a DEM
and a canopy-height grid, run slope and aspect, and write the results —
in one Go binary, with bounded memory, workers and SIMD.

It needs point batches (v0.7), a rasterization kernel (§32) and a LAZ
adapter (§34). The demo program itself need not live in the core module.

## 45. Development Roadmap

**v0.1: Raster compute foundation** (§42, scope complete, not tagged)

```text
Float32Raster, windows, validity bitmap                 done
SIMD (amd64) + scalar fallback                          done
algebra + terrain kernels                               done
tiled multi-worker engine with halos                    done
bounded memory over windowed sources (memory, raw file) done
benchmarks                                              done
module path, README, CI                                 done
licence, first tag                                      open
```

**v0.2: Reductions and statistics** (§49)

```text
Min, Max, MinMax, Count over valid cells                done (STRATA-12)
the fold driver, tiled and chunked                      done (STRATA-12)
transfer: Reclass, Lookup, Rescale, RescaleRange (§50)  done (STRATA-13)
Sum, Mean, Stats                                        open
order-independent accumulation: the same bits for       open
  every tiling, worker count and backend
algebra.Normalize on top of the reduction pass          done
benchmarks/reduce, against the §28 bandwidth ceiling    open
```

The transfer family is not a reduction, but it lands in v0.2 for the same
reason reductions do: both are what turn a computed surface into
something to act on. `Normalize` was to write through `Rescale`, and
does not: its endpoints need a subtraction and a division, not a
multiply-add (§18).

**v0.3: Array foundation**

```text
N-dimensional arrays (generic element type decided here, §9)
strides
views
axis reductions (§49 extended to N-D)
broadcast-style operations
```

**v0.4: Streaming and pipelines**

```text
general Source / Sink APIs beyond raw files
streaming sources for sparse data
workspace reuse
pipeline execution          partly done: the radius-0 Pipeline, internal (§52);
                            radius > 0 and several outputs remain
deciding whether to publish Kernel (§22, §52)
```

**Later milestones**

```text
v0.5   Zarr adapter: chunk-native N-D datasets (§34)
v0.6   GeoTIFF / COG adapters (§34)
       partly done: reading (`cog` module, ADR 0002); writing open
v0.7   point batches: SoA, filters, reductions, rasterization (§11, §32)
v0.8   resampling, alignment, mosaics, interpolation
       partly done: same-CRS resampling (§54); reprojection is the
       caller's preprocessing (§36)
v0.9   fusion beyond hand-built pipelines: lazy planning, scheduling;
       register-level fusion is built for the left-deep pointwise cut
       (§29); a chain with two live intermediates remains
v0.10  voxel grids as 3-D arrays (§33)
```

**Not tied to a milestone:** a licence and tagged releases, so domain
modules can import and pin strata (§7, §42); and revisiting portable
`simd` with Go 1.28 (ADR 0001). The ARM64 NEON kernels (STRATA-11), the
traffic counter (§51), the outside benchmark against `gdaldem` (§38),
the acceptance harness (§39) and the focal operations (§53) landed this
way and are done; a focal median and statistics that skip invalid cells
are open (§53).

**v1.0**

A stable:

> SIMD-accelerated spatial compute engine for Go

with strong raster, environmental-array, and streaming foundations.

## 46. Long-Term Identity

strata should sit closer to NumPy/xarray + raster algebra + PDAL-style
streaming + Zarr + Go SIMD than to QGIS + GDAL + OGR. The goal is not to
own spatial data formats or domain workflows; it is to be the efficient
compute layer that formats feed and domain modules import.

## 47. Long-Term Architecture

```text
 format adapters (§34)          GeoTIFF · Zarr · LAZ · raw
          │
          ▼
       chunks ──► representations: Raster · Array · PointBatch
                          │
                          ▼
                  execution engine (§25)
                     SIMD │ scalar
                          │
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─  strata's public API
                          ▼
 domain modules       terrain (in core) · hydrology · remote sensing · wildfire
```

## 48. Core Architectural Principles to Protect

The following principles should survive every future expansion:

- Formats are adapters, not the core.
- Spatial representations remain specialized.
- Execution infrastructure is shared.
- Data should be processed in chunks rather than fully materialized.
- Hot paths should operate on predictable, contiguous memory.
- SIMD should remain an implementation detail behind stable APIs.
- Scalar execution defines correctness.
- Validity is carried beside the data, never encoded in it.
- The project should optimize numerical spatial computation, not recreate the entire GIS ecosystem.
- Domain algorithms live in modules that import strata (§7).

The shortest expression of the architecture is:

> Sources provide chunks. Representations organize spatial data. The engine schedules work. Kernels operate on spans or batches. SIMD performs the numerical work.

## 49. Reductions

Every operation so far is raster → raster. Nothing in the API produces a
number, so a caller cannot ask for the mean of a slope raster, the range
of a DEM, or how many cells of a chunked output are valid, without
writing the loop themselves. Three things need one:

- `algebra.Normalize` (§18), which is a map pass over statistics the
  engine has to gather first.
- Contrast stretching, classification breakpoints and quantiles, which
  are what turn a computed surface into something to look at or act on.
- Checking a chunked run: today the only way to verify a 20000² output
  file (§43) is to read it back in full.

Reductions are also the first operation whose result does not live in a
raster, so they settle how the engine returns a value rather than filling
a sink.

### Operations

```go
type Summary struct {
    Count  int64   // valid cells
    Sum    float64
    Mean   float64
    StdDev float64 // population
    Min    float32
    Max    float32
}

reduce.Count(src raster.Float32Raster) int64
reduce.MinMax(src raster.Float32Raster) (min, max float32, count int64)
reduce.Sum(src raster.Float32Raster) (sum float64, count int64)
reduce.Stats(src raster.Float32Raster) Summary
```

The result type is `Summary`, not `Stats`: a package cannot hold both a
type `Stats` and a function `Stats`, which is what the first draft of
this section asked for. `Count` is the cheapest of these — a popcount
over mask words — and answers the third motivation above directly.

with `Tiled` and `Chunked` counterparts beside them, as in `algebra` and
`terrain`:

```go
reduce.CountTiled(ctx, src, engine.Options{}) (int64, error)
reduce.CountChunked(ctx, src engine.RasterSource, engine.Options{}) (int64, error)
reduce.MinMaxTiled(ctx, src, engine.Options{}) (min, max float32, count int64, err error)
reduce.MinMaxChunked(ctx, src engine.RasterSource, engine.Options{}) (min, max float32, count int64, err error)
reduce.SumTiled(ctx, src, engine.Options{}) (sum float64, count int64, err error)
reduce.SumChunked(ctx, src engine.RasterSource, engine.Options{}) (sum float64, count int64, err error)
reduce.StatsTiled(ctx, src, engine.Options{}) (Summary, error)
reduce.StatsChunked(ctx, src engine.RasterSource, engine.Options{}) (Summary, error)
```

All of these and their `Tiled` and `Chunked` forms exist.

`Mean` and `StdDev` are in `Summary` rather than functions of their own.
Both come from the same pass: the exact accumulator keeps the sum of
squares beside the sum (`accum.Moments`), which is cheaper than a second
pass and, unlike Welford, exact. `Mean` is the exact sum over `Count`,
correctly rounded, which `Sum/Count` in float64 would not be.
`Summary` is computed by `internal/summary`, not in `reduce`, so that a
neighbourhood statistic can fold a kernel's output with the same reducer
and return the same type: the planned terrain statistics (§20).
Histograms and quantiles are a later operation: they return a vector
rather than a scalar, and their bin edges are a policy question of their
own.

Named operations, not a per-cell callback (§19). `Sum` returns `float64`
whatever the input's element type: a float32 accumulator loses the answer
on a raster of any size, and the result is one number, so the wider type
costs nothing.

### Validity and values

Only valid cells take part, as everywhere else (§31). A nil mask means
every cell counts. `Count` is always returned beside the value, so an
empty reduction is visible rather than disguised: with no valid cells
`Sum` is 0 and `Min`, `Max` and `Mean` are NaN.

Validity is still never inferred from data. A NaN in a valid cell is an
ordinary value: it propagates into `Sum` and `Mean`, and through `Min`
and `Max` with the semantics of Go's builtins, as in `algebra`. A caller
who wants NoData skipped clears the bit; a caller who writes NaN into a
valid cell gets NaN out. -0 and +0 follow the builtins too.

Go's builtins do not say *which* NaN comes back, though, and that is a
hole in the guarantee below rather than a detail. `min` and `max` return
a NaN when either operand is one, but the payload that survives depends
on which operand it was — so a scalar fold and an eight-lane one carry
different payloads out of a raster holding more than one, and so do two
different tilings. A reduction therefore returns the **canonical quiet
NaN**, not a payload copied out of the data. It costs one branch per
call, since the fold propagates NaN by itself. `reduce`'s
`TestNaNIsCanonical` fails in both builds without it.

### Determinism

The engine's promise is that a tiled or chunked call gives the same bits
as its plain counterpart, for every tile size and worker count (§25). A
reduction cannot keep that promise by fixing an evaluation order, because
float addition is not associative and tiles do not cut the raster in the
same places for every `Options` value: two tilings sum different subsets
before combining them.

The same problem appears one level down. A SIMD reduction keeps several
accumulator lanes and adds them at the end, so it sums in a different
order from the scalar loop, and §15 requires the two to agree bit for
bit.

So the guarantee is stated on the value rather than on the order:

> A reduction returns the correctly rounded result of the exact
> arithmetic over the valid cells, for every tiling, worker count and
> backend.

Exact accumulation is order-independent by construction, which makes
tiling, worker count and vector lanes free: partials combine in any
order, and no plan-order bookkeeping is needed. It is also the most
accurate answer available, which suits a project whose scalar path
defines correctness.

`Min` and `Max` are associative and commutative already, NaN and -0
included, so they carry none of this and landed first (STRATA-12): the
vector kernels keep eight accumulator lanes and combine them in an order
the scalar loop does not, and still agree bit for bit, which is §15 met
by construction rather than by matching an evaluation order. `Count` is
integer addition and is free for the same reason.

The accumulator representation is a benchmarked decision, recorded like
the NoData choice (STRATA-3) in `benchmarks/reduce/RESULTS.md`:

```text
float64 accumulator               fast, inexact, order-dependent
Neumaier compensated float64      near-exact, still order-dependent
error-free transformation (2Sum)  exact pair, order-dependent carry
binned / superaccumulator         exact and order-independent
```

Only the last satisfies the guarantee as written. What it costs per cell,
against a pass that is otherwise pure memory bandwidth (§28), is what the
benchmark has to show. If it is too expensive, the answer is to weaken
the guarantee to a documented, plan-independent evaluation order and say
so — not to let the result depend on `Options`.

**Decision: binned, and the guarantee stands.** `internal/accum` keeps one
`int64` bin per float32 exponent and adds significands to them, so partials
are integers and combine exactly. `Moments` adds the squares beside them
for Mean, Variance and StdDev. Its AVX2 and NEON backends add 64-cell
blocks in registers after shifting them onto a common exponent, and put
the same integer into the bins as the scalar loop, so the backends agree
by construction. On one core that makes the exact sum as fast as a plain
float64 loop, and sum plus squares as fast as Neumaier. With workers the
exact sum reaches 85% of read bandwidth. The numbers, and what lost
(more bin sets did not help: the scalar loop is instruction-bound, not
waiting on stores), are in `benchmarks/reduce/RESULTS.md`. Sum, Mean and
Variance are correctly rounded. StdDev is the 256-bit root of the exact
variance: within one ulp and still a function of the values alone.

### Engine

A reduction is a new driver shape in `internal/exec` rather than a new
kernel over the existing one: there is no `dst` and no sink, each band or
tile yields a partial, and the partials combine into one result on the
calling goroutine once every worker has stopped. Operand checks, tile and
band planning, halos (radius 0), the worker pool and the panic re-raise
are the ones already there.

**Implementation (STRATA-12).** `Kernel` is untouched, including its "at
least one output" rule: a `Reducer` is a different interface, not an
`Arity(n, 0)` kernel, so nothing in the map path grows a branch for it.

```go
type Reducer[P any] interface {
    Inputs() int
    Fold(p *P, src Cells)
    Combine(a *P, b P)
}
```

The partial is a type parameter, not an `any`: it is folded once per
band, so an interface would cost nothing in time (§19) but would box
every partial. `Combine` must be associative and commutative with the
zero `P` as its identity, which is where the guarantee above is actually
enforced — the engine hands each worker a zero partial and combines them
in an order that depends on `Options`.

`Cells` is the fold's `Span`: a rectangle, one view per input, and
`ValidBits(x, y, k)`, which returns the AND of every masked input over a
run of up to 64 cells (`raster.MaskBits`, exported for it). Validity
stays the engine's job and values stay the reducer's, as on the map side.
Nothing is materialised, so a reduction allocates only views.

Two things fall out of there being no output. A reduction takes **no mask
lock**: `job.maskLock` exists because bands *write* validity bits that
share words, and a fold only reads them. And inputs may overlap each
other, and each other's validity words, freely — there is no output for
them to collide with, so the whole overlap half of `check` is gone.

`runWorkers` needed no change at all: it already passes the worker index
to the work function, which is the per-worker partial's slot. Those slots
are padded to a cache line, since a partial is written once per band and
would otherwise be false-shared between neighbouring workers.

Cancellation differs from a map pass. A cancelled chunked map leaves a
prefix of whole tiles in its sinks, which is useful. A partial sum over
an unknown subset of a raster is not, so a cancelled or failed reduction
returns `ctx.Err()`, or the source's error, and no value.

A chunked `Normalize` is therefore two passes over the source: reduce,
then map. That is a full extra read of the file, and the clearest case
for fusion (§29) so far — worth recording now, not worth building yet.

### Testing

As §39, against an exact reference: `math/big.Float` accumulates the
valid cells at enough precision to round once, which is the definition
above, so the fuzz target compares against the answer rather than against
another implementation of the same mistake. Beyond the table in §39, test
sums that cancel catastrophically (large values either side of a small
one), sums that overflow float32 but not float64, all-invalid and
single-valid rasters, and rasters whose plan splits a run of cells that
the plain path sums consecutively.

Metamorphic relations (`FuzzReduceRelations`): a permutation of the cells
reduces to the same bits, `Sum(a) + Sum(b)` equals `Sum(a+b)` under exact
accumulation, scaling by a power of two scales the sum exactly, and `Min`
and `Max` commute with the grid symmetries in `internal/rastertest`. Each
side runs through its own execution path, as elsewhere.

A reduction writes nothing per cell, so `benchmarks/reduce` reports GB/s
against the machine's measured read bandwidth (§28, §38) rather than
cells/s alone: the interesting number is how close an exact accumulator
stays to a bandwidth-bound pass.

`FuzzReduce` is the reference comparison and `FuzzReduceRelations` the
metamorphic one; `TestReduceRelations` runs the same bodies under rapid.
`internal/exec`'s `TestReduceTilesAndWorkers` and
`TestReduceChunkedTilesAndWorkers` run the §23 matrix for folds against a
reducer whose partial is a position-mixing XOR, so a cell read twice,
skipped, or read at the wrong place all fail — with no arithmetic of its
own to be wrong. `TestNoLeaksOnReduceFailure` and
`TestReduceChunkedRawFaults` add the leak and IO-fault halves, both
checking the rule above that no value comes back with an error.

Status: partly done. `Count`, `MinMax` and the fold driver done (STRATA-12): the
`Reducer`/`Cells` shape, `Reduce` and `ReduceChunked`, `vec.ReduceMin`
and `vec.ReduceMax`, and package `reduce`. The accumulator decision and
`internal/accum` done, with `benchmarks/reduce/RESULTS.md`. `Sum`,
`Stats` and `Summary` done, on `accum` and `internal/summary`, with the
`benchmarks/reduce` suite; its numbers are in the same RESULTS.md. A
masked fold packs the valid cells of partly valid mask words into a
per-worker buffer (`summary.Runs`), so the vector blocks apply to
scattered NoData too. `MinMax` does not use it yet and still walks
such words cell by cell. `algebra.Normalize` done, on `MinMax` (§18).

## 50. Transfer Functions

Every operation so far computes a physical quantity: a sum, a gradient, a
slope angle, an extent. None of them turns one into a judgement. A
wildfire risk model over Swedish terrain wants a slope factor, an aspect
factor and a fuel factor, each a bounded curve applied per pixel, and
then *brandriskklass* 1–5, a reclassification of the combined surface.
The module can produce every input to that model — `terrain.Slope`,
`terrain.Aspect`, `algebra.Mul` — and can do nothing with them.

Those factors are not three algorithms. They are two, plus a scaling:

```go
transfer.Reclass(dst, src, breaks, values)   // a step function over breakpoints
transfer.Lookup(dst, src, xs, ys)            // a bounded piecewise-linear curve
transfer.Rescale(dst, src, a, b)             // dst = a·src + b
transfer.RescaleRange(dst, src, inLo, inHi, outLo, outHi)
```

The model then lives in the caller's tables, and the package stays
domain-neutral, which §7's scope boundary asks for. There is no `fire`
package and no fire-risk demo; the chain appears once, as an `Example`.

`Rescale` subsumes the `algebra.Scale(dst, src, 1.25)` §18 has carried as
a to-do, so `Scale` is not added to `algebra`: two spellings of one
operation is what §18 exists to prevent. §49 already named
"classification breakpoints" as one of the things that turn a computed
surface into something to act on, so this is adjacent to v0.2 rather than
new territory, but it is new scope.

### Conventions

**Class intervals are half-open upward.** `values[i]` covers
`[breaks[i-1], breaks[i])`, so a cell exactly on a break takes the class
above it, the first class runs down to −∞ and the last up to +∞. Three
reasons, the third the one that decided it:

- Published tables are written that way — "klass 4: FWI 11.2 och uppåt" —
  so a table typed in from one means what it says.
- The unbounded classes end up at the ends of the scale, where a risk
  scale wants them.
- A raster of integer land-cover codes can be reclassified with the codes
  themselves as breaks. Upper-inclusive would force 2.5, 40.5, 41.5, which
  is a trap.

**A table is read, never copied.** Validation is a pass over a handful of
elements, so it costs nothing and happens on every path, before any cell
is written; a rejected table leaves `dst` untouched. The cost is that a
`Tiled` or `Chunked` run holds the table while every worker reads it, so
a caller must not mutate one mid-call. That is the concrete form of §22's
"no mutable state" rule for kernels, and the first time the module has
had to state it, because these are its first kernels taking anything but
slices and scalars.

**The x side is strictly increasing and free of NaN**, and `Lookup`'s
knots are finite; the y side is unconstrained. A curve that rises and
falls is the point — an aspect factor peaks on the south-facing slopes —
so `ys` cannot be required monotone.

### Values

**A NaN cell gives that same NaN back.** This is not what the arithmetic
alone does, and the failure it prevents is the sharpest argument for
settling semantics before writing kernels: every IEEE comparison against
NaN is false, so a search that only counts breaks at or below a cell
places NaN above the whole table and reports **the top class**. A model
would silently turn missing-looking data into the highest fire risk.
`Reclass` and `Lookup` test for NaN before searching — one compare on top
of an n-compare scan, so a 1/n overhead not worth being clever about.
Validity is still never inferred from Data (§31): a NaN with its bit set
is an ordinary value.

Neither operation narrows validity, because both are total: every cell
has a class and every curve is bounded. There is no out-of-domain case,
so §22's "kernels whose validity rule is different need an extension of
this interface" does not bite, and `algebra.Mask` remains the way to
narrow validity.

**`Lookup` reproduces every knot exactly, for every table.** A cell equal
to a knot's x returns that knot's y directly rather than interpolating
from it. That short circuit is not an optimisation: `t` is 0 at the lower
knot, but `0·(y1-y0)` is NaN when the next y is NaN or infinite, so one
undefined knot would swallow the defined one below it, and a −0 would
come back +0. The fuzzer found it; reasoning had not.

Nothing else is exact. Recovering a cell through a divide and a multiply
loses an ulp, so `Lookup` with `ys == xs` is the identity only to within
rounding — the fuzzer produced −185.16173 coming back as −185.16174. The
one bit-exact bridge to `algebra` is the curve through (0,0) and (1,1),
which is `algebra.Clamp` to [0, 1].

A segment's value is `y0 + t·(y1-y0)`, so a rise that overflows float32
gives an infinity between two finite knots. Also from the fuzzer, also
documented rather than defended against: a factor curve never comes near.

`RescaleRange` resolves its two intervals into `Rescale`'s coefficients
once per call and is then exactly `Rescale`. Its endpoints therefore land
*near* their targets, to float32 precision relative to the output span,
not on them — a zero `inLo` is the exception. The obvious alternative,
`outLo + (v-inLo)·scale`, buys no more exactness and costs a third
rounding on every cell and a second kernel. The operation whose endpoints
*are* exact is the two-knot `Lookup`, which is the substantive difference
between the two, more than the clamp.

### Determinism

Go permits fusing a multiply-add into one FMA, and arm64 takes it where
amd64 does not — and CI runs `macos-latest`. ADR 0001 already requires
explicit `float32(...)` conversions in scalar references for that reason
(§15). Two expressions here are exposed: `Rescale`'s `a·v + b`, and
`Lookup`'s segment value. `RescaleRange`'s offset is a third, in float64,
once per call. The AVX2 `Affine` is correspondingly `VMULPS` then
`VADDPS`, never `VFMADD`, and the NEON one `VFMUL` then `VFADD`, never
`VFMLA`: the compiler does not fuse archsimd's `Mul` and `Add`.

`Reclass` does no arithmetic at all — comparisons and a table index — so
it is identical across architectures by construction. It is the only
member of the family that needs none of this machinery.

`vec.TestAffineIsNotFused` and `curve.TestLookupIsNotFused` pin it with
inputs where the two answers differ, each asserting first that its
fixture still distinguishes them. A local pre-flight for what CI checks
on arm64:

```bash
GOOS=linux GOARCH=arm64 go build -gcflags=-S ./internal/vec ./internal/curve ./transfer | grep -E 'FMADD|FMSUB'
```

and for the NEON kernels, which must print nothing:

```bash
GOEXPERIMENT=simd GOOS=linux GOARCH=arm64 go build -gcflags=-S ./internal/vec ./internal/stencil 2>&1 | grep -E 'VFMLA|VFMLS'
```

### Where the kernels live

`Rescale`'s kernel is `vec.Affine`: a lane-parallel map, so it belongs in
`internal/vec` with the dispatch table and the AVX2 backend. `MulScalar`
then `AddScalar` would have read and written the whole span twice.

`Reclass` and `Lookup` live in a new `internal/curve`, not in `vec`.
`vec`'s `kernelSet` is a backend-swap table and `vec.Backend()` answers
for all of it; two permanently scalar fields would make that answer
false, and `TestBackendSelection` compares the SIMD set field by field
against a literal, which would then have to name scalar functions. Their
loop is also a per-cell search rather than a map, so it is optimised and
measured on its own terms, and its bounds-check record is its own. This
is `internal/stencil`'s precedent (§17): a family with a different kernel
shape gets a package.

`internal/curve` has no backend machinery until it has a backend, but its
kernels are named `scalar*` so that adding one only adds files. The
operand checks and validity bookkeeping the plain path needs moved from
`algebra` into `internal/pointwise`, so the two packages that document
the same operand rules share one implementation of them.

### The scan

The search is a forward scan that stops at the first table entry above
the cell. `BenchmarkTableSize` in `internal/curve` measures it against a
binary search and against a branchless count of the whole table, over
uniformly spread cells, cells clustered in one class, and cells outside
the table.

On spread cells — what a computed surface produces — the scan wins at
every size that matters: 105 against 83 Mcells/s at four breaks, 90
against 62 at eight, 72 against 47 at sixteen (Zen 2, scalar, ±2%). A
binary search only overtakes past about sixteen breaks, and then only on
clustered or out-of-range cells whose branches it predicts. The
branchless count — the form one reaches for expecting mispredictions to
dominate — is the slowest of the three beyond two breaks: never stopping
early costs more than the branches it avoids. Real tables are four to
eight breaks or five to ten knots.

Cells are not assumed sorted. One exploitation that would be safe,
because the answer is a pure function of the cell, is to try the previous
cell's segment first: adjacent pixels are strongly correlated, so it
would usually hit in two compares. It adds a loop-carried dependency and
data-dependent branching, so it is a measurement, not an assumption, and
not made yet.

### Performance and the open question

At 4096², from `transfer`'s own benchmarks:

```text
                scalar    AVX2
Rescale           1290    2011 Mcells/s
RescaleRange      1313    2099
Reclass            107     107
Lookup              87      88
```

`Rescale` sits in the band §38 measures `Clamp` in, and is
bandwidth-bound by 4096² like the rest of the pointwise algebra. The
table-driven pair are an order of magnitude behind it and identical in
both builds, which is the whole open question: whether a vector `Reclass`
or `Lookup` — unrolled compare-and-accumulate over a bounded table,
blends rather than gathers — is worth its complexity and its
bit-equivalence tests. That is a decision for a measurement, and these
numbers are the baseline it has to beat. `benchmarks/transfer` lands with
it, as `benchmarks/reduce` lands with `Sum` (§49).

### Testing

As §39. The reference matrix runs every operation over every layout, size
and mask combination against a per-cell reference written from the
documentation; `TestPathsAgree` compares plain, `Tiled` and `Chunked` bit
for bit over the tiling grid; `FuzzTransfer` runs each operation on a
path the input chooses, so the three-form agreement is fuzzed rather than
only tabulated. `FuzzTransferPanics` requires every rejected table to
panic with this package's own message, before any cell is written, which
is what keeps the table checks ahead of the kernels.

Metamorphic relations (`FuzzTransferRelations`, and
`TestTransferRelations` under rapid): every class is one of the values or
the cell's own NaN; a monotone curve is bounded by its own knots; all
three operations commute with the grid symmetries in
`internal/rastertest`; a curve whose ys are its knot indices meets
`Reclass` over the same breaks at every knot; and `Rescale` by 1 and −0
is the bit identity, which pins that the addition is real arithmetic
rather than an elided no-op.

Two of the sharpest rules in this section came from the fuzzers rather
than from reasoning — the knot short circuit, and the segment overflow —
and one test bug did too: an insertion sort in a relation silently
no-ops around a NaN, so the relation read a nonsense range off its own
fixture. Rapid shrank that one; the corpus had not found it.

Status: done. `vec.Affine` with its AVX2 kernel, `internal/curve`,
`internal/pointwise`, and package `transfer` with all four operations in
all three forms. `benchmarks/transfer` and any vector backend for the
table kernels are open, in that order.

## 51. Traffic Accounting

Throughput cannot say why a call is slow. The chunked path runs Slope,
Hillshade and Clamp at the same 770–845 M cells/s although their compute
costs differ fivefold (§27, benchmarks/chunked/RESULTS.md), which is the
signature of a fixed per-cell tax — but reading that off three figures
that happen to coincide is an inference, and an inference cannot be
falsified by a change that claims to remove the tax.

`engine.Stats`, passed as `Options.Stats`, counts the Data each stage
moved: what the sources delivered, what the kernels read and wrote, what
the sinks took, with `Cells`, `Tiles`, `Bands`, and `Ideal` — the bytes a
pipeline that touched every byte once would have moved, `Cells × 4 ×
(inputs + outputs)`. `Amplification` is `Total / Ideal`.

It is an out-parameter rather than a return value because twenty typed
entry points would otherwise change signature for a diagnostic, and the
counters are kept whether or not a `Stats` is passed, so a measured call
runs the same code as an unmeasured one. A call adds to the `Stats` it is
given, so one can total a pipeline of calls of different arities.

Two deliberate limits. It does not count validity: a mask is one bit
against a float32's 32, so at most 3% of the traffic, and counting it
exactly through `ErodeBox` and the word-level range operations would
thread bookkeeping into every branch of `halo.go` for a term smaller than
the run-to-run variance of the benchmarks it would inform. And it counts
what the engine moves between stages, not what reaches memory — a tile
buffer that stays in L2 is counted when the source fills it and again
when the kernel reads it. That is the intended reading: how many times
the engine handles each byte is what its structure decides and what §29
or a windowed tile would change; how much of that reaches DRAM is a
hardware profiler's question, and the two are most useful together.

### What the first run said

Every pointwise chunked call is exactly 2.00×, independent of size, tile
shape and worker count: the source fills a buffer, the kernel reads it,
the kernel writes a buffer, the sink drains it, where the work needs two
touches. `Gradient`, with two outputs, lands at 2.01, so the arity
accounting holds.

The halo runs the other way from the obvious guess:

| Slope, 4096², chunked | B/cell | Amplification | halo |
|---|---:|---:|---:|
| whole | 16.5 | 2.06 | 11.1% |
| full-width strips of 256 | 16.5 | 2.07 | 11.1% |
| 1024 × 1024 tiles | 16.1 | 2.02 | 3.1% |
| 256 × 256 tiles | 16.1 | 2.01 | 1.4% |

Small tiles move **less** traffic and run 4–10× slower (§27). The halo is
therefore not why they are slow; the per-row source and sink calls are,
exactly as §27 says. What governs the halo is band *aspect ratio*, not
tile size: `bandCells / tileW` rows, so a 4096-wide tile gets 16-row
bands and a 256:1 perimeter-to-area ratio, where a 256×256 tile is one
square band. Full-width strips buy their call-count advantage with about
10% extra halo traffic, and the trade is overwhelmingly worth it.

`bandCells` is not the lever for that: 1<<16 cells × 4 bytes × 2 operands
is 512 KiB, one Zen 2 core's L2 exactly (§28), so raising it trades halo
for cache misses. Decoupling the IO tile from the compute tile is — IO
wants full-width for few calls, compute wants square for small perimeter,
and today they are one parameter.

Status: done. `engine.Stats`, `Options.Stats`, counters in all four
drivers (`ProcessN`, `ProcessChunked`, `Reduce`, `ReduceChunked`), and
`internal/exec/stats_test.go`. Measured cost against the parent commit
over Slope and Clamp at `-count 6`: geomean −0.67%, within the machine's
noise.

## 52. Pipelines

A chain of operations pays the per-tile cost once per operation. Five
chained `algebra.MulTiled` calls over 4096², the shape of a weighted
factor product, move 60 bytes per cell where one pass over the same six
inputs and one output would move 28 — and every individual call is
already optimal, so `Stats.Amplification` reads 1.00 for all five (§51).
Chunked it is worse: `Mul` costs 24 B/cell there, so the same chain is
120 against the same 28.

That waste is between calls, not inside them, and nothing in the engine
can see it. The fix is not a new execution model. It is to run the whole
chain on a tile while the tile is loaded, instead of the whole raster per
operation:

```text
for op { for tile { read; compute; write } }     ->  now
for tile { read; for op { compute }; write }     ->  wanted
```

`chunkJob.tile` already reads, computes and writes in exactly that order.
It runs one kernel.

### Shape

A `Pipeline` is a `Kernel` whose `Process` runs other kernels. That is the
whole design: every driver, tiling rule, halo, worker, cancellation path
and counter keeps working, because nothing above `Kernel` learns that a
pipeline exists.

```go
// Stage is one operation of a Pipeline: a kernel and where its inputs
// come from.
type Stage struct {
    Kernel Kernel
    // In names the kernel's inputs by value id, and must have exactly
    // the kernel's input count.
    In []int
}

// Pipeline runs a chain of kernels over one span, keeping the values
// between them in scratch rather than in rasters.
//
// Values are numbered in one sequence: ids [0, Inputs) are the
// pipeline's own inputs, and each stage appends its outputs in order.
// A stage may only name values already defined, so a Pipeline is a DAG
// by construction with no cycle check to write.
type Pipeline struct {
    Inputs  int
    Stages  []Stage
    // Outputs names the values the pipeline writes, in the order its
    // Span expects them.
    Outputs []int
}

func (p *Pipeline) Radius() int                  { ... }
func (p *Pipeline) Arity() (inputs, outputs int) { return p.Inputs, len(p.Outputs) }
func (p *Pipeline) Process(dst Span, src Window) { ... }
```

Value numbering rather than a `(stage, output)` pair because it makes the
ordering constraint structural: an id that is not yet defined cannot be
named, so "defined before use" is a comparison against the stage's own
first output id and needs no traversal. It is also what lets one input
feed several stages, and a stage's output feed both a later stage and
`Outputs`, with no special case.

### Radius composes by suffix sum

To write a W×H span of the last stage, the stage before it must produce a
span grown by that stage's radius, and so on back to the sources. With
stage radii r₁…rₙ, let

```text
R_k = r_k + r_{k+1} + … + r_n
```

Stage k then reads a window of (W+2R_k)×(H+2R_k) and writes a span of
(W+2R_{k+1})×(H+2R_{k+1}), the last stage writes W×H, and
`Pipeline.Radius()` is R₁.

Radii add along a chain, so a deep chain of stencils grows its halo
quickly. A pipeline is not a way to make radius free; it is a way to stop
reloading a tile. §51's halo table is where that trade shows up, and an
all-pointwise pipeline has R₁ = 0 and no halo at all — which is why it is
the first case to build.

### Scratch: the one contract extension

`Process` needs somewhere to put the values between stages, and the
`Kernel` contract forbids mutable state so that concurrent calls on
disjoint spans stay safe. Allocating inside `Process` would also break
"bands allocate nothing" (§26).

So the engine allocates it, the way it already carves tile buffers per
worker:

```go
// ScratchKernel is a Kernel that needs working memory of its own. The
// engine allocates it once per worker and passes the same memory to
// every Process call on that worker, so a kernel that keeps nothing
// between calls stays safe for concurrent spans.
type ScratchKernel interface {
    Kernel
    // Scratch returns the cells and mask words one Process call needs
    // for a span of at most w×h.
    Scratch(w, h int) (cells, words int)
}
```

`Span` gains `Scratch []float32` and `ScratchBits []uint64`, empty for a
kernel that does not ask. A `Pipeline` asks for the sum, over its
intermediate values v produced by stage k, of (W+2R_{k+1})(H+2R_{k+1})
cells plus the mask words where validity is carried — exactly computable
from the radii, with nothing allocated to plan it.

This is the only change to the contract. Everything else about `Kernel`,
`Span` and `Window` stands.

### Edges compose; edge *values* do not

The engine never calls `Process` for an output cell whose neighbourhood
leaves the rasters, and fills those cells with the kernel's `Edge()`
itself. So inside `Process` every window cell exists and no stage meets a
raster boundary: the pipeline's composite R₁-wide border is the engine's
business, exactly as a single kernel's r-wide border is.

Validity agrees with running the stages separately: n erosions of width
r_k leave the same invalid ring as one of width R₁.

Data does not, when a stage declares a non-NaN `Edge()`. Run separately,
the outer rₙ ring gets the last stage's edge value and the ring inside it
gets whatever arithmetic on the previous stage's edge cells produced;
fused, the whole R₁ ring gets the pipeline's edge value. For NaN they
agree, because NaN propagates, and every kernel in the tree declares NaN
today (`terrain/stencil.go`). So:

> A non-final stage that implements `EdgeKernel` with a non-NaN edge
> panics. Lifting that means deciding which of the two readings is
> correct, and no caller is asking.

### Validity, for an all-pointwise pipeline, is one pass

Per stage, validity is what it already is: the AND of masked inputs for
radius 0, an erosion for radius r. But AND is associative and idempotent,
so when every stage is pointwise, each value's validity is the AND of the
masked *pipeline inputs it transitively depends on* — and an output
depending on all of them is the AND of all of them. The pipeline computes
output validity once, from the inputs, rather than once per stage: n
pointwise stages do one mask pass, not n.

`halo.go` already has the word loops for it (`andBits`, `copyBits`) as
free functions over views. The radius > 0 path wants `erodedValidity`
extracted from `job` the same way, and that refactor is the bulk of the
work beyond the pointwise case.

### What it is worth, as the counter will read it

Five chained Muls over 4096², as one `Pipeline` of 6 inputs and 1 output:

| | today | as a Pipeline |
|---|---:|---:|
| Tiled, B/cell | 60 | **28** |
| Tiled, Amplification | 1.00, ×5 calls | **1.00, ×1 call** |
| Chunked, B/cell | 120 | **56** |
| Chunked, Amplification | 2.00, ×5 calls | **2.00, ×1 call** |

Amplification does not move, and should not: each call was always
optimal. `Total` per cell is the figure that halves, which is what §51
was built to expose.

Two honest limits. The scratch traffic between stages is not counted —
the counter sees a pipeline as one kernel with its declared arity — so 28
B/cell is the DRAM figure only while the intermediates stay in cache.
That understatement was the argument for §29 on top, which has since
landed: register-level fusion removes the intermediates rather than
relegating them to L2, so a pipeline that lowers now moves what it says
it moves. And a
pipeline does nothing about the chunked 2.00×, which is the source and
sink copy. It makes that copy carry more work, which is the most that can
be done for a file.

### Where it lives

`Kernel` is in `internal/exec` and stays there until its shape settles
(package `engine` documentation). A `Pipeline` there can compose strata's
own kernels — a `terrain` stencil feeding `algebra` arithmetic — behind a
typed public entry point, which is how every other operation is exposed,
and needs no decision about publishing the contract.

It cannot be built by a caller. A caller who wants a weighted factor
product gets a typed entry point for that shape instead, which needs no
public `Kernel` and is the smaller commitment. Publishing `Kernel`, `Span`
and `Window` is a separate decision with its own consequences; `Pipeline`
is evidence for it rather than a reason to take it now.

### Testing

The §23 matrix, as everywhere else: every tile size and worker count
against the same pipeline run as separate calls over whole rasters, bit
for bit, Data and validity. That reference is the point — a pipeline that
does not equal its unfused form is a bug, and the unfused form has a
reference of its own already.

Add to it: a pipeline of one stage must equal that stage, which catches
wiring; a pipeline whose stages are permuted into another valid order
must give the same result; and `Scratch` must be exactly enough, checked
by allocating exactly what it asks for and letting the race detector and
a poisoned tail find an overrun. §49's position-mixing trick applies here
too — a stage that reads the wrong value, or reads one twice, fails a
hash it cannot accidentally satisfy.

Status: partly done. The radius-0 cut is done: `Pipeline`, `NewPipeline`, the
`ScratchKernel`/`ScratchSize`/`Scratch` contract extension and
`Span.Scratch`, per-worker scratch in `job.allocScratch` sized by
`plan.spanSize` and `chunkJob.spanSize` and pooled across calls, and
`internal/exec/pipeline_test.go`.

The acceptance test passes at both numbers: a six-input product over
256² moves 60 B/cell as five chained calls and 28 as one pipeline,
Amplification exactly 1.00; chunked it is 56 against the chained 120,
Amplification exactly 2.00. `TestPipelineMatchesUnfused` and
`TestPipelineChunkedMatchesUnfused` run the §23 matrix against the same
stages as separate whole-raster calls, windowed and compact, masked and
not, bit for bit including validity — which is where the "every input
must be reachable from the output" rule earns itself: without it the
engine's single AND over the declared inputs would invalidate cells the
unfused chain keeps.

`TestScratchBoundsEverySpan` is the one that checks arithmetic rather
than behaviour. A tile clipped at the raster's edge builds its own plan,
so a narrower tile takes taller bands and can cover more cells than a
full one; the spy kernel records the largest span it is actually given
and fails if the engine sized its scratch for less.

`Span.Scratch` is a pointer because the first version was not. Three
slice headers by value cost 8% on `Slope/4096/tiles256`, where a Span is
built for every one of a great many small bands; by pointer the same
matrix is geomean −0.68% against the parent commit, which is the
machine's noise. That is the §51 counter's sibling lesson — a cheap
measurement caught a cost that was invisible in the design.

Scratch is pooled across calls (`internal/exec/scratch.go`, §37),
because allocating it per call cost more than the pipeline saved. Every
call made and zeroed a fresh block per worker — 1 MiB for the four
intermediates of `benchmarks/fusion`'s five-Mul chain in default strips,
12 MiB on 12 workers — and at 1024², where the operands are near cache,
that allocation was the call: the pipeline ran slower on 12 workers
than on one. Now each worker takes a block from a `sync.Pool` per kind
and size class (eight to an octave, so at most 1/8 over; the usual
spans are powers of two and waste nothing) and gives it back when the
call's workers have stopped. Nothing is zeroed, which the contract
already allowed. A `sync.Pool` rather than a cache on the `Pipeline`
because it serves every `ScratchKernel` and both drivers, is already
safe for concurrent calls, and is emptied by the collector; rather
than caller-supplied scratch because that would put ownership in the
public API for a saving the engine can make alone. The slices lent are
still exactly the lengths asked for, capacity included, so an overrun
still trips a bounds check; views are cleared on the way back so a
pooled block does not keep a caller's rasters alive; and tests poison
every block lent (NaN cells, patterned bits, junk views), so a kernel
that reads scratch before writing it fails every test, not only a pool
miss. `TestScratchIsReused` checks that a scratch call allocates no more
than the same call of a plain kernel, and `TestPipelineConcurrentCalls`
runs one `Pipeline` from eight goroutines through both drivers.

Measured with `BenchmarkMulPipeline` (AVX2, 12-core Zen 2, -count 5,
median):

| 1024², default strips | before | after |
|---|---:|---:|
| 1 worker, M cells/s | 516 | 547 |
| 12 workers, M cells/s | 417 | 1065 |
| 12 workers, B/op | 12.3 MiB | 13.5 KiB |

From 4096² the call is bandwidth-bound and the change is within noise
(an interleaved re-run of the 1-worker cases: +6% strips, the rest
unchanged); B/op drops from 1–12 MiB to 13–75 KiB, which is about one
pool miss per benchmark run spread over its few dozen calls.

Still to do, in the order the spec gives them: radius > 0 stages, which
need `erodedValidity` extracted from `job` and the suffix-sum window;
more than one output; and the decision about publishing `Kernel`, which
is what would let a caller build one of these. Register-level fusion on
top of the radius-0 cut is §29, and is done.

## 53. Focal Operations

§28 predicted that convolution would behave like the terrain kernels:
compute-bound, where SIMD and workers pay, rather than capped by memory
bandwidth like the algebra. Its cost per cell grows with the
neighbourhood, (2r+1)² products for a full kernel, so if any operation
turns work into speed it is this one. §7 puts domain-neutral
neighbourhood stencils in the core and keeps `terrain` to derivatives of
a DEM, so these operations are a package of their own, `focal`.

### API

Every operation has a plain function, a `Tiled` form and a `Chunked`
form, shaped as `terrain`'s are (§25):

```go
focal.Correlate(dst, src, focal.WeightsOptions{Radius: 2, Weights: w})    // 25 weights, row-major
focal.Convolve(dst, src, focal.WeightsOptions{Radius: 2, Weights: w})     // the same, rotated 180°
focal.CorrelateSeparable(dst, src, focal.SeparableOptions{Radius: 3, Row: g, Col: g})
focal.Gaussian(3, 1.5)                                                    // 7 taps for Row and Col
focal.Mean(dst, src, focal.BoxOptions{Radius: 1})
focal.Min(dst, src, focal.BoxOptions{Radius: 1})
focal.Max(dst, src, focal.BoxOptions{Radius: 1})
```

- **Correlate and Convolve are both public, with one kernel.** A
  function called Convolve that does not rotate its weights surprises
  anyone coming from scipy or a textbook, and one that does surprises
  anyone coming from GIS "focal weights" or OpenCV's `filter2D`, which
  are correlations. So each name means what it says. Convolve rotates the
  weights once, when the call starts, and the hot loop always correlates.
  Since the sum is ordered by the *source* cells it reads (below),
  `Convolve(w)` is bit for bit `Correlate(rot180(w))` by construction.
  The separable form is only `CorrelateSeparable`: for symmetric taps,
  Gaussian and box included, the two coincide.
- **The radius is required, 1 to `MaxRadius = 8`.** There is no default,
  unlike terrain's zero-means-default options: a Mean whose radius was
  forgotten would quietly be 3×3. Radius 0 would be a different engine
  path (in place is allowed at radius 0). The cap is the engine's
  cancellation granularity: bands are about 2¹⁶ cells (§25), and at r=8
  Correlate's 289 products per cell make a band tens of milliseconds of
  scalar work. A per-kernel band size would lift it; nobody is asking.
- **Weights must be finite.** A NaN weight is a programming error, and
  panics with `focal:` before anything is written, as terrain's unusable
  scales do. Data is another matter: NaN and ±Inf in valid cells flow
  through IEEE arithmetic (§31), and zero weights are applied, not
  skipped, so a zero weight on an infinite cell gives NaN — numpy's and
  scipy's answer too.

### Validity: the whole neighbourhood

An output cell is valid iff it is not an edge cell and all (2r+1)² cells
of its neighbourhood are valid, whatever their weights: terrain's rule,
applied by the engine's erosion (§22, `stencil.ErodeBox`, any radius),
with no mask code in `focal`.

The alternative — a focal mean of the valid cells only, or weights
renormalised over them — is what many GIS tools do, and it is not built.
It needs the kernel to read input validity and write its own output
validity, which the `Kernel` contract forbids (`internal/exec/doc.go`:
"Kernels whose validity rule is different ... need an extension of this
interface"). The extension is not large: `Window` would expose the masks
it already carries, and an optional interface would let a kernel own its
output validity. But it gives up §31's one-pass word-level validity, and
the SIMD kernels would need per-lane validity words instead of plain
arithmetic. It should come with a caller who needs it, as its own
operations (`MeanValid`, say), rather than as a mode of these.

### Evaluation order, and why nothing slides

Tiled, Chunked, scalar, AVX2 and NEON give the same bits because every
cell's terms are folded in one fixed order:

- A weighted sum starts from its first product, not from +0, and adds
  each later one: `acc = float32(acc + float32(w·v))`, never an FMA
  (§15). Starting from +0 would turn a sum of −0 products into +0.
- Correlate's terms are in row-major order of the input cells.
- The separable operations fold each column of the neighbourhood first,
  top to bottom, then the column results left to right (below for why
  in that order). The result is the same sum as Correlate's with the
  outer product of the taps, not the same bits, and the documentation
  says so. On data whose sums are exact, the two agree to the sign of a
  zero, and the metamorphic tests hold them to that.
- Mean is the separable box sum, then a division by (2r+1)², not a
  multiplication by its reciprocal (§18's argument for Normalize).
- Min and Max use Go's builtin `min` and `max`, which are associative and
  commutative with NaN and signed zero included, so their order does not
  matter at all: the separable result is the brute-force 2-D result bit
  for bit.

A running sum — add the column entering the window, subtract the one
leaving — would make Mean and box filters O(1) per cell at any radius.
It is not used, because its rounding depends on where the running sum
started, which is the start of a band or tile: tiled output would stop
equalling whole-raster output (§23). Every cell is summed afresh, O(r²)
for Correlate and O(r) per pass for the rest. Min and Max are different:
van Herk/Gil-Werman gives O(1) per cell with exact arithmetic, so its
result cannot depend on where a band starts. It is a later optimisation
for large radii, not a contract question.

### Separable kernels: column first, one row of scratch

A separable kernel is one `ScratchKernel` of radius r (§52), not a
two-stage pipeline: it does both passes inside one `Process` call, and
no radius > 0 `Pipeline` is needed. The order of the passes matters:

- **Row pass first** would filter every window row horizontally into
  scratch, (H+2r) rows for an H-row span, then filter those columns. A
  band is about 2¹⁶ cells of whole rows, so a 4096-wide raster gets
  16-row bands, and each band would redo the row pass for its 2r halo
  rows: 1.6× the row-pass work at r=5, and (2r+1)× for one-row bands.
  Its scratch, W·(H+2r) cells, is 425 KB at 4096×16, r=5: about an L2.
- **Column pass first**, the one built, folds the 2r+1 rows under each
  output row into one row of W+2r column results, then folds each run of
  2r+1 of those into an output cell. The only repeated work is the 2r
  column results at the ends of each row, (W+2r)/W: 1.002 at 4096 wide.
  Band height stops mattering, and scratch is one row, W+2r cells, which
  stays in L1.

Mean, Min and Max are the same kernel with sums, minima or maxima in
place of weighted sums.

Building it found an engine bug. `ScratchKernel.Scratch(w, h)` is asked
for the largest span a call can produce, and `ProcessChunked` answered
with the *largest-area* span among its full and clipped tiles. A clipped
tile is narrower and so gets taller bands, so the largest-area span can
be narrower than the widest one: a 2500-wide raster in 1000-wide,
256-tall tiles has full-tile bands of 1000×65 and last-column bands of
500×131, and was asked for 500 columns. `Pipeline`'s scratch scales with
area, so it never showed; a kernel whose scratch is a row panicked out of
range. `chunkJob.spanSize` now bounds each side separately (the widest
span's width, the tallest span's height), and `TestScratchBoundsEverySpan`
checks both sides as well as the area.

Two limits remain. Pooled scratch sits outside §27's memory bound, as a
`Pipeline`'s does: a row per worker here, which is negligible. And
`Pipeline` passes no `Scratch` to its stages, so a focal kernel cannot
be a `Pipeline` stage yet; the radius > 0 `Pipeline` (§52) has to pass
it on.

### Kernels

The row kernels are in `internal/focalrow`, with the dispatch and
`Backend`/`UseScalar` of every kernel package (§17), not in
`internal/stencil`, whose kernels and tests are built around Horn's 3×3
and the masks:

- **Scalar** kernels loop over terms outside and cells inside,
  accumulating in `dst` — the same per-cell order as a register
  accumulator, with every cell loop indexed by its loop variable alone,
  so they carry no bounds checks (§39).
- **AVX2 and NEON** kernels keep one output cell's accumulator per lane.
  A block is four vectors (32 cells on AVX2, 16 on NEON) with an
  accumulator each, so four add chains hide one's latency, and each term
  costs one weight broadcast and one bounds check per block: the
  compiler cannot bound a term's offset, so converting it to an array
  pointer is checked, and the four loads are constant slices of that
  pointer, which are not. The BCE test names the four lane functions and
  says why. Weights are broadcast from memory: on amd64 that is a VEX
  `VMOVSS` and `VBROADCASTSS`, so no legacy SSE enters the loop (ADR
  0001). Neither backend emits a fused multiply-add; the §50 disassembly
  check covers `internal/focalrow`.
- **Composing from `vec`** — a `MulScalar` then an `AddScaled` per term,
  over whole rows — was considered and not built. It is BCE-clean and
  needs two new kernels, but every term becomes a pass that loads and
  stores the accumulator row, three memory operations per vector per
  term against one, and the point of this section is to measure what
  convolution costs, not what that composition costs.

Building the AVX2 min found a hazard in the exact `min8` recipe
(ADR 0001) that `internal/vec` shares. It restores NaN lanes from the
first operand only, which assumes `x.Min(y)` becomes `VMINPS` with x
first. Inside a fold (`acc = min8(acc, load)`) the compiler treated
`Min` as commutative and swapped the operands for its register
allocation, and min(+Inf, NaN) came out +Inf. `focalrow`'s `min8` now
repairs from both operands in one blend: x|y is NaN when either is, and
the right signed zero when they are equal. `max8` repairs NaN lanes with
x+y, since the AND it needs for equal lanes can turn a NaN into ±Inf.
`internal/vec`'s `min8` and `max8` were changed to the same forms
alongside, and ADR 0001's rule with them.

### Testing

- `internal/focalrow`: the scalar kernels against a per-cell definition,
  bit for bit, over hazards, odd lengths and strides; the SIMD kernels
  against scalar for every length to 120 and every neighbourhood to 17
  cells, with the cells they must not read poisoned differently in the
  two runs (`simd_test.go`, shared by both architectures); min and max
  on every ordered pair of hazards; `FuzzFocalRows`; the BCE test.
- `focal`: every operation against a naive per-cell reference in the
  documented order, bit for bit, with edges and validity, on windows
  whose stride and mask offset are not the parent's; hand-worked 3×3
  cases for orientation and rotation; plain, Tiled and Chunked over a
  grid of tilings and worker counts for r ∈ {1, 2, 3, 5}; scalar against
  SIMD; panics before any write. `FuzzFocal` covers the same with fuzzed
  operations, options and float bits.
- Metamorphic relations (`FuzzFocalRelations`, and `TestFocalRelations`
  under rapid), each side through its own path, tiling and layout, all
  bit-exact: crop, locality of one changed cell and of one invalidated
  cell, data under invalid cells, power-of-two scaling of data and of
  weights, negation (to the sign of a zero), Min(−x) = −Max(x),
  Min(x+c) = Min(x)+c, and the eight grid symmetries for Min and Max.
  On data whose sums are exact (small integers, integer weights) every
  order gives the same sum, so every operation is also invariant under
  the symmetries with its weights mapped, separable equals Correlate
  with the outer product, and Mean equals a unit-weight Correlate
  divided by (2r+1)². Fuzzing found that the separable relations hold
  only to the sign of a zero, since a transposition or an outer product
  factors a zero term differently; both corpus entries are kept.
- The §23 matrix in `internal/exec` runs three focal kernels — Correlate
  at r=3 with asymmetric weights, CorrelateSeparable at r=2, and Max at
  r=4, which leaves one interior row of the 300×9 window — under poisoned
  scratch and one-row bands, which only the engine's own tests can set.
- Mutations of `focal` — Convolve without its rotation, a radius declared
  one short, the separable taps swapped, Mean multiplying by the
  reciprocal, a sum started from +0 — each fail the tests.
- `acceptance/` judges every operation, in all three forms, on its three
  DEMs against the definition as shifted sums in float64 (Min and Max
  exactly), with the dot-product error bound `(m+1)·2⁻²⁴·Σ|w||z|`, and
  cross-checks that reference against `scipy.ndimage.correlate` and
  `convolve` when scipy is installed. Errors land at 0.05–0.12× the
  bound. Its border and erosion checks now take each operation's radius.
  `sabotage.py` adds seven focal defects, all caught. A Mean multiplying
  by a rounded reciprocal is not among them: it is an ulp off, inside
  any bound a float64 reference can justify, so the unit tests pin
  Mean's division instead.

### Benchmarks

`benchmarks/focal` times Correlate, CorrelateSeparable with Gaussian
taps and Mean and Min at radii 1, 2, 3 and 5, and Max at 3, through the
plain API at the §38 sizes, masked and not, on both backends. The radius
is part of the operation's name (`CorrelateR3`), so each radius gets its
own §28 class from `stratabench` without a change to its name pattern.
Every operation moves 8 bytes per cell whatever its radius: the other
rows of the neighbourhood come from cache.

**§28's prediction holds**, on AVX2 (the Zen 2 desktop, one pinned
core) as on NEON (Apple M4), benchmarks/focal/RESULTS.md, at every
radius and size. Figures are AVX2 at 4096², unmasked.

- Correlate costs what its products cost: 0.96, 1.92, 3.44 and 7.98 ns
  per cell at r = 1, 2, 3 and 5, a flat 0.066–0.077 ns per product from
  r = 2 (0.053–0.064 on NEON), so its throughput falls as (2r+1)² and its
  memory demand with it, from 8.3 GB/s at r = 1 to 1.0 at r = 5.
- The separable forms grow linearly: Gaussian 0.75 → 1.65 ns per cell
  from r = 1 to 5, Mean 0.71 → 1.58, Min 0.82 → 2.45. At r = 5 separable
  is 4.8× cheaper than the full kernel (6× on NEON).
- The highest demand in the suite is 15.1 GB/s (Mean at r = 1, 1024²),
  under a core's 22.
- AVX2 is worth 2.9–5.5× over scalar where the scalar numbers are
  stable, NEON 3.3–6.0×: more than the lanes alone would give, because
  the scalar kernels accumulate through memory a term at a time (the
  canonical order, bounds-check free) where the lanes hold several
  accumulators in registers. Eight lanes give no more than four; the
  M4's core is 1.2–2.6× faster in absolute terms. Scalar Correlate and
  Mean are not stable: the same source ran 11–43% slower after
  unrelated commits moved `internal/focalrow`'s functions by 32 bytes,
  which puts their ratios up to 7.5× in the current run.
- **A 64 KiB row stride, fixed.** At 16384², and not at 8192², 12288²,
  16320² or 16448², Gaussian, Mean and Min at r = 5 ran at 42–56% of
  their 4096² speed (Gaussian 602 → 252 M cells/s). Timing the column
  pass with the stride varied alone puts the cause in Zen 2's L2 (512
  KiB, 8-way, sets repeating every 64 KiB): the loss repeats every
  64 KiB of stride, rows a line or two apart modulo 64 KiB lose too,
  strides that are multiples of 4 KiB but not 64 KiB do not (ruling out
  the L1 and 4K aliasing), and it starts between 7 and 9 rows, at the 8
  ways; no counters were read. Narrow windows of wide rasters, as Tiled
  hands kernels, are exposed as much as whole rasters. The AVX2 column
  passes, and Correlate's 2-D pass, now fold rows that collide (more
  than seven within 1 KiB of one another modulo 64 KiB) a group of at
  most seven at a time over chunks of 4096 cells, accumulating through
  the destination row in the scalar kernels' per-cell order, so no bit
  changes (`TestAliasedStride`). Those rows now run at 86–93% of their
  4096² speed, 3–10% short of 16320²; other strides keep the single
  pass, which the fold would slow by 4–19%. r = 3 (7 rows, under the
  threshold) still loses about 20% at 16384², and folding does not
  recover it.

So workers should scale these operations as they scale Slope (§26), and
further, since they ask for less bandwidth per core; that is not
measured yet.

Status: done: `focal` (Correlate, Convolve, CorrelateSeparable,
Gaussian, Mean, Min, Max, plain, Tiled and Chunked), `internal/focalrow`
(scalar, AVX2, NEON), the §23 matrix entries, `benchmarks/focal` with
AVX2 (Zen 2) and NEON (Apple M4) runs, and the acceptance checks. The
`chunkJob.spanSize` fix is in, and so is the fold of the AVX2 column
pass at 64 KiB-aliased strides (r = 5 now within 3–10% of neighbouring
strides, r = 8 within 7–21%). Open: r = 3's residual loss of about 20% at a 64 KiB stride,
whose cause folding does not reach, and L2 counters to confirm the set
conflicts directly (uProf, or an elevated `wpr`); a focal median
(sorting networks up to 5×5, a selection algorithm beyond); statistics over valid cells only, with the
interface extension above; van Herk/Gil-Werman for Min and Max at large
radii; rectangular and per-axis radii, whose erosion would need to
follow the footprint; and focal kernels as `Pipeline` stages.

## 54. Resampling

Status: done for grids in one CRS: `resample.Resample`, `ResampleTiled` and
`ResampleChunked` with Nearest, Bilinear, Cubic, Lanczos and Average,
scalar-canonical passes with AVX2 and NEON kernels that match them bit for
bit, checked against a float64 reference and against gdalwarp
(`acceptance/`), measured in `benchmarks/resample`. Open: Mode, mosaics
and alignment helpers, and the AVX2 numbers from the Zen 2 machine.
Reprojection is not planned; it is the caller's preprocessing (§36).

### Scope

Resampling between two `raster.Grid`s in the same CRS: a different
resolution, a different origin, or both, with axis-aligned cells of any
sign on either axis. That is v0.8's first item (§45) and needs no CRS
transformation (§36): the source coordinate of an output cell's centre
is an affine function of its column alone along x and of its row alone
along y. Reprojection breaks that and is not attempted. Grids whose CRSs
do not match (`raster.CRS.Matches`) panic.

### Separable, table-driven, gather-free

Because the mapping is separable, every interpolating method is too: its
2-D weight is the product of a weight per column and a weight per row.
`internal/resamp` builds, once per call, a tap table per axis: for each
output index, the run of source indices it reads and their weights,
computed in float64 and rounded to float32. The filter then runs as two
passes:

1. horizontal: every source row the output reaches is resampled into an
   intermediate, `t[j][c] = Σ_k wx[c][k] · src[j][first_c + k]`;
2. vertical: each output row blends intermediate rows,
   `out[r][c] = Σ_k wy[r][k] · t[first_r + k][c]`.

The vertical pass is a lane-parallel multiply-accumulate across columns,
the ideal SIMD shape. The horizontal pass is the hard one: neighbouring
output columns read unrelated source offsets, so lanes over columns need a
gather per tap — `VPGATHERDD` is slow on Zen 2, and `simd/archsimd` has no
gather at all (go1.27). Instead the SIMD kernels transpose L source rows
(L = 8 on AVX2, 4 on NEON) into a column-major scratch, in which each tap
of each output column is one contiguous vector of L rows; accumulate with
the rows in the lanes; and transpose each L×L block of output columns
back. The transposes are VPUNPCK/VPERM2F128 on AVX2 and ZIP1/ZIP2 on NEON,
all from archsimd; a last partial block uses masked loads and stores, so
AVX2's lane loops stay in VEX instructions (ADR 0001). This works the same
for upsampling and for downsampling of any factor, and every lane performs
the scalar pass's operations in the scalar pass's order.

### Canonical arithmetic

The bits are defined by the scalar passes (§15): products rounded
separately (`float32(w*x)`, never a fused multiply-add), taps summed in
increasing source index starting from the first product (not from +0,
which would turn −0 into +0), the horizontal pass before the vertical.
`TestPassesAreNotFused` pins the rounding with a fixture that first proves
it distinguishes fused from unfused, and a build of the package for
arm64 and for amd64 with FMA contains no fused instruction; the float64
table code routes its products through a non-inlined multiply, since a
`float64(...)` conversion does not stop Go fusing float64 operands, so the
tables and the ties they decide are the same bits on every architecture.

`resamp.Direct2D` evaluates the same sums cell by cell, recomputing each
tap row's horizontal sum rather than reusing it. It gives the separable
passes' bits exactly (`TestSeparableIsDirect`), so separability changes
the cost and not the answer; it is the tests' reference and the
benchmarks' direct contender.

A cell centred exactly on a source centre has one tap of weight 1:
Lanczos is exactly zero at the non-zero integers, taps of weight zero are
trimmed, and coordinates within 1e-9 of a cell centre or edge are snapped
for the weights (not for the centre cell, which decides Nearest and
validity and must round as gdalwarp's does). The identity grid is
therefore a bit-exact copy under every method, NaN payloads, infinities
and −0 included.

### gdalwarp's conventions, measured

gdalwarp's same-CRS behaviour is the outside reference, so its
conventions were measured before anything was written, with gdalwarp
3.12.1 through GDAL's Python bindings (`-et 0`, `-wt Float64`) on small
synthetic rasters: ramps, quadratics and impulses whose results can be
inverted for the weights. What it does, and strata now does:

- Geometry: an output cell is its centre, mapped by the inverse source
  geotransform, `u = inv0 + X·inv1` with `inv1 = 1/res`; source cell i
  covers [i, i+1). Nearest takes `floor(u)`, with no epsilon: at 2×
  downsampling, where every centre lands on a source edge, it takes the
  cell after the edge.
- Weights: `K((i + 0.5 − u)/s)` over |·| < support, for the tent, Keys'
  cubic with a = −0.5, and the Lanczos-3 window. When an axis downsamples,
  s is its source cells per output cell, which stretches the kernel so it
  averages: for Bilinear and Cubic once 1/s < 0.95, for Lanczos once s > 1,
  per axis. Average weighs by area of overlap. Weights are renormalised
  over the source cells inside the source, so the edge renormalises; an
  output cell whose centre lies outside the source is invalid (Average:
  one that overlaps no source cell).
- Cubic has two modes. With neither axis stretched it is gdalwarp's
  four-sample formula, which falls back to unstretched Bilinear for a
  cell whose 4×4 taps of non-zero weight lose one to the edge or, with a
  mask, include an invalid cell. With either axis stretched it is the
  general kernel, which renormalises. strata reproduces both: the fallback
  cells are few (a band two to three cells wide along edges and NoData)
  and are recomputed by a scalar bilinear after the passes.

### Validity: renormalise, as gdalwarp does

§31 left the choice open between "valid iff every contributing source
cell is valid" (the stencils' erosion) and renormalising over the valid
cells. Resampling renormalises, because that is gdalwarp's rule and
because erosion would cost a kernel's reach — twelve source cells for
Lanczos downsampling by four — around every NoData cell:

- an output cell is valid iff the source cell under its centre is valid
  (except under Average) and the weight of its valid source cells is
  positive;
- under Lanczos with a masked source, at any scale, at least half of the
  source cells its window reaches, weight zero included and cells outside
  the source excluded, must also be valid (measured: 6 of 12 at 2×
  downsampling, 12 of 24 at 4×, 3 of 6 along an axis at 1:1 or
  upsampling; a corner cell with most of its window outside the source
  stays valid), except a cell whose centre lies exactly on a source
  centre on both axes, which copies that cell. The first probes had
  missed the rule below 1:1, because the 1:1 probe was aligned, which is
  the exception; the gdalwarp comparison found it;
- its value is Σ w·v·x / Σ w·v over the valid cells, where v is 0 or 1.

Both sums are separable, so the masked path runs the same passes over
two more planes built from the footprint by selection, never by
multiplying (Data under a cleared bit may be Inf or NaN): the valid
values with +0 under cleared bits, and the validity as 1 or +0. A count
of valid taps per cell, small integers exact in float32 and summed by the
same vertical pass with unit weights, tells cells whose taps are all
valid, which keep the unmasked sum N. An all-valid mask therefore gives
the same bits as no mask. Masked sources are processed in chunks of 256
output columns, each testing its own footprint, so one invalid cell sends
only its chunk down the masked path. Invalid cells get NaN in Data.

Renormalising over the valid cells of a kernel with negative lobes is
ill-conditioned where the valid weight nearly cancels: the result can lie
far outside the inputs, as gdalwarp's does. The tests' tolerances are
derived from the sums (A/|D|, not a fixed ulp count) for that reason.

### Where strata departs from gdalwarp

Only where gdalwarp departs from its own definitions:

- the stretch: gdalwarp derives it for each warp chunk from pixel
  counts, the chunk's source window (clipped to the source) over its
  destination cells, not from the resolutions. The two agree for a
  destination inside the source on its cell edges, and nearly agree in
  the interior of a large warp, but where the destination overhangs the
  source, or gdalwarp splits a warp into chunks, its kernel width depends
  on the extent and on the chunking (measured to 6e-8: an 80-cell source
  under 59 cells of 1.37 is stretched by 80/59 = 1.356). strata keeps the
  resolution ratio, which is the geometry and the same for every tiling,
  as §23 needs. The acceptance cases compare a non-integer stretch on a
  grid where the two agree (1.25) and record the others as excluded;
- Lanczos downsampling by an odd integer factor (3, 5): gdalwarp gives the
  tap at an output centre about 83 times its weight, which looks like a
  defect in its optimised Lanczos path;
- Lanczos for 1 < s < 1.05, where gdalwarp mixes stretched and unstretched
  kernels;
- a source one cell wide or high, where gdalwarp's Bilinear degrades to
  Nearest;
- Average on output cells that extend past the source's edge, by about 1%.

Values otherwise agree with gdalwarp to within float32 rounding, and
validity exactly (`acceptance/gdalwarp_resample.py`).

### Execution

The engine's Kernel assumes one grid and a fixed symmetric halo (§22–§24);
resampling reads, for each output tile, a source footprint of its own
shape, two table lookups per axis. Rather than generalise `internal/exec`,
package resample plans its own units and runs them on the engine's
scheduler, exported for this as `exec.RunUnits` in a new file, the only
change to `internal/exec`:

- Tiled: engine.Options tiles split into bands of at least 64 rows (2¹⁷
  cells), each reading its footprint from the in-memory source. A band
  recomputes the horizontal pass for the source rows it shares with its
  neighbours, a few per cent at 64 rows; validity words are written under
  one lock per call, as §26's.
- Chunked: a tile is the unit, each worker holds one output tile, one
  footprint window and the intermediates, sized once for the largest
  footprint, and reads and writes under `context.WithoutCancel` (§25). The
  bound is in the package documentation; downsampling by a large factor
  wants small tiles, because the footprint grows with the factor.

Every cell's arithmetic depends only on the tables, so every tiling and
worker count gives the plain call's bits (`TestTiledAndChunkedMatch`,
`FuzzResample`).

### Measured

[`benchmarks/resample/RESULTS.md`](benchmarks/resample/RESULTS.md): AVX2
on the Zen 2 desktop, the suite's headline machine, and NEON on an Apple
M4. Figures below are AVX2 at 4096² unless marked.

- **Compute-bound, as §28 predicted,** on both machines: every method at
  every scale (2×, 4×, ½, 1/1.37) holds its throughput from 256² to the
  largest size, asking at most 7.7 GB/s on AVX2 (2–8 on NEON); the taps,
  not the memory, are the limit.
- **Separable against direct:** the passes beat `Direct2D` by 3.3–6.4×
  in the same scalar code (Cubic ×2 184 against 36.2 M cells/s, Lanczos
  ×2 138 against 21.5, Lanczos ½ 35.8 against 7.1) and by 8.7–17× once
  they vectorise, which the direct form cannot without a gather. NEON
  gave the same shape (3.3–6.5×, 11–15×). The two give the same bits, so
  nothing is traded for it.
- **SIMD:** AVX2 gives 2.3–3.2× on the interpolating methods (Average ×2
  1105 M cells/s, Bilinear ×2 811, Lanczos ×2 362, Lanczos ½ 100) and
  nothing to Nearest, a copy at about 400 M cells/s. That is no more than
  NEON's 2.0–3.3× on four lanes, against the expectation that eight
  lanes would give more: lane width is not what limits the passes. The
  M4's faster core runs them 1.6–1.7× faster in absolute terms (Lanczos
  ×2 596, Average ×2 1900, Nearest about 1 G cells/s).
- **Masks:** a source mask with 10% of cells invalid at random, which
  puts an invalid cell in every footprint, costs 5–9×: three horizontal
  passes, the counts and a per-cell finish. Clustered NoData costs only
  the chunks whose footprints it touches.
