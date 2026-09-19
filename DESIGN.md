# SIMD-First Spatial Compute Engine

Section numbers in this document are cited from code comments, ADRs and
benchmark results. Add new material inside existing sections or at the end
rather than renumbering.

Decisions recorded elsewhere and summarized here:

- [ADR 0001](docs/adr/0001-simd-backend.md): SIMD backend technology (STRATA-2).
- [benchmarks/nodata/RESULTS.md](benchmarks/nodata/RESULTS.md): NoData representation (STRATA-3).
- [benchmarks/algebra/RESULTS.md](benchmarks/algebra/RESULTS.md): first benchmark suite results (STRATA-10).
- [benchmarks/chunked/RESULTS.md](benchmarks/chunked/RESULTS.md): bounded-memory execution and the §43 demo.

## 1. Project Goal

Build a Go-native, SIMD-accelerated spatial compute engine for large environmental, geospatial, and scientific workloads.

The project should initially focus on raster and multidimensional environmental data, while leaving room for point clouds, voxels, and other spatial representations later.

The project is not intended to be:

- a GDAL rewrite
- a general-purpose GIS suite
- a vector geometry engine
- a wrapper around GDAL/OGR
- a file-format compatibility project
- a replacement for scientific Python as an analysis environment

Instead, the core focus is:

> High-throughput numerical computation over large spatial datasets using Go-native memory layouts, SIMD, chunking, and concurrency.

The first concrete target is:

> A Go backend should be able to process large spatial datasets efficiently without needing GDAL, Python, or cgo in the hot compute path.

## 2. Positioning

The project should not be positioned as:

> GDAL for Go

That would place it in direct competition with projects that primarily expose traditional GIS functionality.

Instead, position it as:

> A SIMD-accelerated spatial compute engine for Go, initially focused on large raster and environmental array workloads.

A shorter description:

> Fast numerical computing for spatial data in Go.

The emphasis should be on:

- compute
- memory layout
- streaming
- SIMD
- bounded-memory execution
- concurrency
- reusable spatial kernels

rather than on:

- file-format breadth
- desktop GIS workflows
- feature/layer abstractions
- GIS compatibility

## 3. Why This Project Exists

There are several recurring needs in the Go and geospatial ecosystems.

Developers often face one of these architectures:

```text
Go application
     │
     ├── cgo
     │    │
     │    ▼
     │   GDAL / PDAL / native library
     │
     └── HTTP / subprocess
          │
          ▼
      Python / NumPy / xarray
```

The project aims to make this possible instead:

```text
Go application
     │
     ▼
Spatial compute engine
     │
     ▼
SIMD + workers
     │
     ▼
CPU
```

The key value proposition is:

> Serious spatial numerical processing inside a normal Go application, with minimal deployment complexity.

**SIMD is a build-time opt-in.** Go's SIMD packages are still experimental
(ADR 0001). A default `go build` produces a pure-Go, cgo-free binary that
runs the scalar kernels on every architecture. Building with
`GOEXPERIMENT=simd` (Go ≥ 1.27) enables the SIMD kernels. The experiment is
a whole-build flag that the application sets, not something strata's
`go.mod` can turn on. Documentation, examples and published benchmark
numbers must say which mode they use.

## 4. Evidence of Need

The project is based on several existing signals rather than an assumption that users explicitly asked for this exact architecture.

### Strong signal: pure-Go alternatives are valuable

There is repeated demand in the Go ecosystem for removing cgo and native-library dependencies.

The reasons are usually:

```text
cross compilation
static binaries
containers
simpler deployment
fewer system dependencies
portable builds
easier CI
```

Geospatial projects are already appearing specifically to replace native dependencies with pure-Go implementations.

This suggests a real user need:

> "I want spatial functionality in Go without bringing a native geospatial stack into my deployment."

### Strong signal: SIMD numerical processing in Go

There is also clear interest in efficient numerical operations over Go slices.

Common workloads are naturally expressed as:

```text
[]float32
[]float64
[]uint16
```

with operations such as:

```text
add
multiply
clamp
compare
reduce
filter
normalize
```

Spatial data is a natural consumer of this type of vectorized execution.

### Moderate signal: geospatial backend workloads in Go

The existence of:

```text
GDAL bindings
H3 ports
LAS/LAZ libraries
raster packages
spatial databases
GIS wrappers
```

shows that Go is already used for geospatial systems.

The missing layer is a strong Go-native numerical compute foundation.

### Unproven hypothesis

The key hypothesis this project should test is:

> A shared compute layer across raster, multidimensional arrays, point clouds, and future spatial representations is useful enough to be preferable to a collection of specialized libraries.

The early releases should deliberately validate this.

## 5. Primary User Stories

The project should be designed around real jobs rather than around abstract capabilities.

### 5.1 Go backend processing a large raster

A backend developer needs to compute:

```text
slope
aspect
hillshade
normalization
threshold masks
terrain derivatives
environmental indices
```

over a large raster.

They want:

```text
single Go binary
bounded memory
parallel processing
SIMD
no Python service
no GDAL dependency
```

Example, for a raster that fits in memory:

```go
dem := raster.NewFloat32(width, height, data) // from any source adapter
slope := raster.NewFloat32Like(dem)

terrain.Slope(slope, dem, terrain.SlopeOptions{CellSize: 10})

writeResponse(slope)
```

Operations write into a caller-supplied destination and allocate nothing
(§37). For rasters larger than memory, the same operation runs tile by
tile through the execution engine, reading windows from a source (§24,
§25).

This should be the primary v0.x use case.

### 5.2 Environmental modelling

Users may work with:

```text
weather
terrain
forestry
wildfire
hydrology
snow
agriculture
remote sensing
oceanography
```

Typical data:

```text
temperature[time,y,x]

humidity[time,y,x]

wind_u[time,y,x]

wind_v[time,y,x]

elevation[y,x]
```

Typical operations:

```text
map
filter
reduce
resample
normalize
combine
terrain derivatives
thresholds
```

These workloads should map naturally onto the compute engine.

### 5.3 Remote sensing

Users should be able to compute operations such as:

```text
NDVI
NDMI
NBR
spectral masks
normalization
classification inputs
terrain correction
```

For example:

```text
       NIR - Red
NDVI = ─────────
       NIR + Red
```

This is a strong SIMD workload and a natural fit for fused execution later.

### 5.4 Point-cloud processing

Longer term, users should be able to stream point clouds and perform operations such as:

```text
height filtering
classification filtering
coordinate transforms
aggregation
voxel downsampling
rasterization
terrain extraction
canopy extraction
```

Potential workflow:

```text
LAS / LAZ
   │
   ▼
Point cloud
   │
   ├── ground filter
   ├── vegetation filter
   │
   ▼
Rasterize
   │
   ├── DEM
   └── canopy height
          │
          ▼
      terrain analysis
```

## 6. Core Architectural Principle

The central compute flow should be:

> Source → chunk → tile/span → vector kernel → sink

For raster workloads:

```text
Raster
  ↓
Tile
  ↓
Row
  ↓
Span
  ↓
SIMD vector
```

For point clouds:

```text
Point stream
   ↓
Chunk
   ↓
Attribute columns
   ↓
SIMD filter / transform
```

This should be the foundation around which the rest of the project evolves.

## 7. Scope Boundary

The compute engine should know as little as possible about specific storage formats.

The core should understand concepts such as:

```text
Array
Raster
Grid
Shape
Stride
Window
Span
Chunk
Tile
Mask
Kernel
PointBatch
```

It should not fundamentally depend on:

```text
Shapefile
GeoPackage
PostGIS
FileGDB
GDAL Dataset
LAS file
GeoTIFF file
```

Those should be adapters or external libraries.

## 8. Spatial Representations

The engine should eventually support multiple kinds of spatial data.

Do not force them into one generic abstraction.

Instead, distinguish between broad computational families.

```text
                     Spatial engine

          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        Dense          Sparse          Graph
        spatial        spatial         spatial

          │              │              │
     ┌────┴────┐    ┌────┴────┐      networks
     ▼         ▼    ▼         ▼
   Raster    Voxel Points    Mesh
```

These representations should share low-level compute infrastructure where appropriate, but retain domain-specific APIs.

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
    CRS                      CRS     // opaque placeholder, §36
}

type Dataset struct {
    Grid   Grid
    Raster Float32Raster
}
```

`Grid.Window` and `Dataset.Window` shift the origin to match a raster window
(§21).

## 10. Multidimensional Arrays

Environmental data is often not purely two-dimensional.

A future core array representation may look like:

```go
type Array[T Number] struct {
    Data   []T
    Shape  []int
    Stride []int
}
```

Examples:

```text
[y,x]

[time,y,x]

[z,y,x]

[scenario,time,y,x]
```

Typical environmental datasets include:

```text
temperature[time,y,x]

wind[level,time,y,x]

risk[scenario,time,y,x]

fuel[class,y,x]
```

This aligns naturally with chunked storage systems such as Zarr.

Arrays need the same validity bitmap as rasters (§31). Note that `Stride`
here is per dimension, while `Float32Raster.Stride` is the row stride in
elements.

Status: not started (v0.3).

## 11. Point-Cloud Model

Point clouds should use a column-oriented representation.

Avoid:

```go
type Point struct {
    X float32
    Y float32
    Z float32

    Intensity uint16
}
```

as the internal compute model.

Prefer Structure of Arrays:

```go
type PointCloud struct {
    X []float32
    Y []float32
    Z []float32

    Intensity      []uint16
    Classification []uint8
}
```

Coordinates may need `float64`, or `float32` offsets from a batch origin,
for large coordinate ranges (§39). That decision belongs to v0.7.

This enables operations such as:

```text
height filtering
distance calculations
mask generation
attribute filtering
coordinate transforms
aggregation
```

to map naturally onto SIMD.

Status: not started (v0.7).

## 12. Dense vs Sparse Scheduling

Raster and point-cloud workloads need different execution strategies.

Dense spatial data:

```text
Raster
Voxel
N-D Array
```

should be scheduled using:

```text
tiles
chunks
halos
strides
```

Sparse data:

```text
Point clouds
Meshes
```

should be scheduled using:

```text
batches
streams
attribute columns
spatial partitions
```

The engine should share:

```text
workers
buffers
SIMD primitives
masks
reductions
chunk lifecycle
```

without forcing identical high-level APIs. In particular, dense sources are
random-access windowed readers and sparse sources are streams (§24).

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
algebra        terrain        (later: array, pointcloud)
   │              │
   │           raster
   ▼              ▼
internal/vec   internal/stencil
(pointwise)    (neighbourhood row kernels, mask erosion)
   │              │
   └──────┬───────┘
          ├── simd/archsimd, amd64 AVX2    (GOEXPERIMENT=simd)
          ├── simd/archsimd, arm64 NEON    (planned, STRATA-11)
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

func Erode3x3(...) // validity of radius-1 outputs, word-level
func ClearBorder(...)
```

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
└── simd_amd64.go                  ├── mask.go
                                   └── simd_amd64.go
```

- `simd_arm64.go` follows with STRATA-11. There is no `simd.go` (portable
  SIMD), per §14.
- **Dispatch.** Kernels are package-level function variables. They start
  scalar, and `init` swaps them for SIMD versions when the build has
  `GOEXPERIMENT=simd` and the CPU has AVX2. This is gated on AVX2, not
  just AVX.
- **Backend switching.** Each kernel package exports `Backend() string`
  and `UseScalar(bool)`, so benchmarks and tests can run both backends in
  one binary.
- There is no per-call interface dispatch.
- The kernel-writing rules (bounds-check-free loops, `ClearAVXUpperBits`
  before scalar tails, splitting kernels into `*Lanes` functions, explicit
  NaN handling) are in ADR 0001 and benchmarks/nodata/RESULTS.md.

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

Later:

```go
algebra.Normalize(dst, src)     // needs reduce.MinMax over a whole
                                //   raster, which §49 has (STRATA-12)
```

`algebra.Scale(dst, src, 1.25)` stood here and is not being added.
`transfer.Rescale(dst, src, a, b)` subsumes it (§50), over a single
`vec.Affine` kernel rather than a `MulScalar` pass and an `AddScalar`
one; two spellings of one operation is what this section exists to
prevent.

`Normalize` is not a pointwise operation. It needs global statistics (min
and max, or mean and standard deviation, over valid cells). Under chunked
execution that means a reduction pass over every tile before the map pass.

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
└── stencil.go     shared row driver, edge and validity policy
```

Later: `ruggedness.go`, `curvature.go`.

- **Method.** All four operations use Horn's 3×3 gradient, computed a
  whole row at a time. Each SIMD lane loads the three neighbouring rows
  at offsets and evaluates the stencil for 8 cells at once.
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
5×5 filter  radius 2
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
it.

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
  data bit for bit, in Data and in validity, for every tile size, band
  shape and worker count. Tests enforce this (§39): `internal/exec`'s
  `TestTilesAndWorkers` runs every tile width and height in {1, 7, 64,
  256, full, larger than the raster}, over band shapes from one row to
  two-dimensional (§53), with 1, 2, 3 and GOMAXPROCS workers,
  for Clamp, Add, Slope, Aspect, Hillshade, Gradient and a radius-2
  kernel, with and without masks, on windows whose stride is not a
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

Potential later sources:

```text
Zarr
GeoTIFF
COG
LAS / LAZ
HTTP
S3
database
generated data
```

Potential later sinks:

```text
Zarr
GeoTIFF
COG
network stream
database
```

This keeps computation independent from storage.

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
`HillshadeTiled` over in-memory rasters, and `MaskTiled` followed with
`Mask` (§18). They give the same bits as the plain functions for every
`Options`. The terrain functions run the same kernels as one tile; the
algebra functions stay direct to keep their zero allocations.

The Chunked functions (`SlopeChunked`, `AspectChunked`,
`HillshadeChunked`, `GradientChunked`, `AddChunked`, `SubChunked`,
`MulChunked`, `MinChunked`, `MaxChunked`, `MaskChunked`, `ClampChunked`)
take sources and sinks instead of rasters and run with bounded memory
(§27), through `exec.ProcessChunked`. Their sinks receive the bits the
plain function would write, for every `Options`.

Configuration:

```go
type Options struct {
    TileWidth  int
    TileHeight int
    Workers    int
}
```

The zero value is the default for in-memory rasters: one tile as wide as
the raster and `Workers` = `GOMAXPROCS`. The engine plans tiles in
row-major order and splits each into bands of about 2¹⁶ cells, the unit
of scheduling and cancellation — whole tile rows for a pointwise kernel,
a squarer sub-rectangle for one with a radius (§53). Narrow tiles are correct
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

Status: tiled execution over in-memory rasters done, on one or many
workers (STRATA-8, STRATA-9). Chunked execution over sources and sinks
done.

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
`algebra` and `terrain` create no goroutines.

Concurrency belongs in the execution engine.

Reductions (§49) are the first driver with no lock at all: they read
validity and never write it, so the mask lock below has nothing to guard.

**Implementation (STRATA-9).** `internal/exec` starts `Workers − 1`
goroutines per call and uses the calling goroutine as the last worker;
there is no global pool. Workers take bands from the plan in order with an
atomic counter, each with its own views and erosion scratch, and the call
joins them before it returns. Kernels run concurrently on disjoint bands.
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

The rest of the list should benefit the same way, and is not measured yet:

```text
convolution
resampling
interpolation
remote-sensing indices
point-cloud filtering
coordinate transforms
```

Benchmarks should explicitly distinguish:

```text
compute-bound
cache-bound
memory-bound
IO-bound
```

`stratabench` classifies each measured operation this way (§38).

## 29. Operation Fusion

Operation fusion should be a major future optimization.

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

the engine should eventually support:

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

For many workloads this may be more important than SIMD alone.

The architecture should not block future fusion. Two earlier decisions
keep it open:

- SIMD kernels are Go (ADR 0001), so a fusion generator emits Go, not
  assembly.
- Validity is a separate bitmap (§31), so a fused data kernel is plain
  branch-free arithmetic. Validity is computed once for the whole chain.

## 30. Streaming Pipelines

The project should eventually support pipeline-style processing.

For point clouds:

```text
Reader
  ↓
Filter
  ↓
Transform
  ↓
Aggregate
  ↓
Rasterize
  ↓
Sink
```

For rasters:

```text
Source
  ↓
Normalize
  ↓
Index calculation
  ↓
Threshold
  ↓
Sink
```

The engine should ideally fuse compatible stages and avoid unnecessary materialization.

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

A major long-term strength should be crossing spatial representations.

Example:

```text
LiDAR
  │
  ├── ground points
  │      │
  │      ▼
  │     DEM
  │      │
  │      ├── slope
  │      └── aspect
  │
  └── vegetation points
         │
         ▼
      canopy height
      canopy density
```

This is directly useful for:

```text
wildfire
forestry
terrain analysis
drone mapping
infrastructure
```

and provides a natural reason for the project to support multiple spatial representations.

## 33. 3D and Voxel Direction

Spatial computing is increasingly not limited to 2D rasters.

The project should leave room for:

```text
point clouds
voxel grids
meshes
3D fields
```

Potential voxel datasets include:

```text
fuel_density[x,y,z]

temperature[x,y,z]

vegetation[x,y,z]

moisture[x,y,z]
```

This may become especially relevant for future fire, forestry, atmospheric, and remote-sensing workloads.

Raster should therefore be treated as the first compute model, not the permanent boundary.

## 34. IO Architecture

Formats should remain adapters.

Potential future modules:

```text
io/
├── geotiff/
├── cog/
├── zarr/
├── netcdf/
├── las/
└── gdal/
```

An optional GDAL adapter may eventually provide access to GDAL's large format ecosystem:

```text
GDAL
  │
  ▼
adapter
  │
  ▼
Raster / Array
  │
  ▼
Compute engine
```

GDAL is therefore a data source, not the architectural foundation. An
adapter that needs cgo lives in its own module, so the core stays cgo-free.

Adapters implement the source and sink interfaces of §24. They convert
their formats' data types and fill values into `float32` plus validity
(§9, §31).

## 35. Use Existing Format Libraries Where Possible

The project should avoid unnecessarily reimplementing formats.

Prefer:

```text
format library
     │
     ▼
common chunks / arrays
     │
     ▼
spatial compute
```

For example:

```text
existing Zarr package
        │
        ▼
      Array

LAS / LAZ library
        │
        ▼
   Point batches
```

The project's differentiator should remain computation rather than parsing.

## 36. CRS and Reprojection

Do not attempt to replace PROJ in the initial project.

Today `raster.CRS` is an opaque placeholder (`Code string`, for example
`"EPSG:25833"`). A `Grid` carries it along, but nothing interprets it.

When transformation is needed, define a narrow abstraction over coordinate
columns rather than point structs (§11):

```go
type Transformer interface {
    Transform(src, dst CRS, x, y []float64) error // in place
}
```

Native support may eventually include a useful subset (WGS84, Web Mercator,
UTM). More advanced transformations remain adapter-driven.

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

Possible future concepts:

```text
workspaces
chunk-local scratch buffers
arena-like temporary memory
buffer reuse
```

Example:

```go
type Workspace struct {
    Float32 []float32
}
```

Do not introduce aggressive pooling before ownership semantics are stable.

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
├── nodata/             STRATA-3 spike, not part of the suite
├── remote_sensing/
├── convolution/
├── pointcloud/
└── nd/
```

Benchmark names:

```text
Benchmark<Op>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=<W>[/tiles=<T>]
```

Metrics:

```text
ns/cell
cells/sec
points/sec
GB/sec
allocations
SIMD/scalar speedup
parallel scaling
peak memory
```

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

Raster sizes:

```text
256 × 256

1024 × 1024

4096 × 4096

16384 × 16384
```

Suggested point-cloud sizes:

```text
1M points

10M points

100M points
```

## 39. Correctness Testing

Every SIMD implementation should be compared against its scalar equivalent,
bit for bit (§15). Tests run in every build configuration: default,
`GOEXPERIMENT=simd`, and arm64 once STRATA-11 lands.

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
other dependency: `TestTerrainRelations`, `TestAlgebraRelations` and
`TestProcessRelations` drive the bodies of the three `Fuzz*Relations`
targets with rapid's generators in place of a fuzz input. Both drivers go
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

For point clouds also test:

```text
empty batches
attribute length mismatch
sparse masks
large coordinate ranges
```

## 40. Package Layout

Current, and planned for v0.1:

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
├── terrain/                   implemented
│   ├── gradient.go
│   ├── slope.go
│   ├── aspect.go
│   ├── hillshade.go
│   └── stencil.go
│
├── engine/                    public engine configuration
│   ├── engine.go              Options (STRATA-8)
│   ├── source.go              RasterSource / RasterSink, memory source and sink
│   ├── raw.go                 raw float32 file source and sink, RawOptions
│   └── rawfile.go             RawFile: one file, several handles
│
├── internal/
│   ├── vec/                   implemented: scalar.go, dispatch.go, simd_amd64.go
│   ├── stencil/               implemented: horn.go, aspect.go, mask.go, simd_amd64.go
│   ├── curve/                 implemented (§50): curve.go, scalar.go
│   │                           table-driven Reclass and Lookup, scalar only
│   ├── pointwise/             implemented (§50): operand checks and validity
│   │                           for the radius-0 packages' plain functions
│   ├── exec/                  kernel machinery (STRATA-8)
│   │   ├── kernel.go          Kernel, Span, Window
│   │   ├── process.go         Process, operand checks
│   │   ├── tile.go            tile and band planning, band shape (§53)
│   │   ├── halo.go            halos, edges, validity
│   │   ├── worker.go          workers, band and tile scheduling, mask lock (STRATA-9)
│   │   ├── chunked.go         ProcessChunked: per-worker tile buffers, sources and sinks
│   │   ├── reduce.go          Reducer, Cells, Reduce (STRATA-12, §49)
│   │   ├── reducechunked.go   ReduceChunked: per-worker tile buffers
│   │   └── workspace.go       planned
│   └── overlap/               Data and mask overlap checks
│
├── benchmarks/                implemented (§38)
└── docs/adr/
```

Later:

```text
array/
pointcloud/
voxel/
mesh/
remote/
io/
```

## 41. Explicit Non-Goals for v0.1

Do not initially build:

```text
Shapefile support
GeoPackage support
PostGIS integration
vector geometry
polygon overlay

GeoTIFF parser
COG parser
LAS parser

CRS transformation
PROJ replacement

point-cloud engine

voxel engine

generic (non-float32) raster types

ARM64 SIMD kernels

distributed execution

GPU acceleration

CUDA
OpenCL

full GIS workflows
```

These may become integrations later.

The first release should prove the compute foundation.

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

Arm64 builds run the scalar kernels in v0.1.

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

Status: every item above is implemented and measured, and the §43
validation target ran. What publishing v0.1 still needs is outside this
list: a fetchable module path (`go.mod` says `strata`), a licence, a
README and CI.

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

The second major proof should test cross-representation usefulness.

Potential demo:

```text
5 GB LAZ
   │
   ▼
stream point batches
   │
   ├── ground filter
   └── vegetation filter
          │
          ▼
      rasterization
          │
    ┌─────┴─────┐
    ▼           ▼
   DEM      canopy height
    │
    ▼
slope + aspect
    │
    ▼
Zarr / GeoTIFF
```

The important properties would be:

```text
one Go binary
bounded memory
parallel
SIMD accelerated
streaming
```

That would make the project's value proposition immediately understandable.

## 45. Development Roadmap

**v0.1: Raster compute foundation** (§42, scope complete)

```text
Float32Raster, windows, validity bitmap
SIMD (amd64) + scalar fallback
algebra + terrain kernels
tiled multi-worker engine with halos
bounded memory over windowed sources (memory, raw float32 file)
benchmarks
```

**v0.2: Reductions and statistics** (§49)

```text
Min, Max, MinMax, Count over valid cells       done (STRATA-12)
the fold driver, tiled and chunked             done (STRATA-12)
Sum, Mean, Stats
order-independent accumulation: the same bits for every tiling,
  worker count and backend
algebra.Normalize on top of the reduction pass
benchmarks against the §28 bandwidth ceiling
transfer: Reclass, Lookup, Rescale, RescaleRange (§50)  done (STRATA-13)
```

The transfer family is not a reduction, but it lands in v0.2 for the same
reason reductions do: both are what turn a computed surface into
something to act on, and `Rescale` is the building block `Normalize`
writes through.

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
pipeline execution
```

**v0.5: Zarr integration**

```text
chunk-native datasets
environmental time series
large N-D arrays
```

**v0.6: Raster interoperability**

```text
GeoTIFF
COG
external adapters
```

**v0.7: Point-cloud foundation**

```text
PointBatch
Structure of Arrays
filters
reductions
aggregation
rasterization
```

**v0.8: Spatial processing**

```text
resampling
alignment
mosaics
interpolation
```

**v0.9: Pipeline optimization**

```text
operation fusion
lazy planning
kernel scheduling
```

**v0.10: 3D experimentation**

```text
voxel grids
3D arrays
point-to-voxel aggregation
```

**Not tied to a milestone:** ARM64 NEON kernels (STRATA-11), and revisiting
portable `simd` with Go 1.28 (ADR 0001).

**v1.0**

A stable:

> SIMD-accelerated spatial compute engine for Go

with strong raster, environmental-array, and streaming foundations.

## 46. Long-Term Identity

The project should live closer to:

```text
NumPy / xarray
+
raster algebra
+
PDAL-style streaming
+
terrain processing
+
Zarr
+
Go SIMD
```

than to:

```text
QGIS
+
GDAL
+
OGR
```

The goal is not to own every spatial data format.

The goal is to become the efficient compute layer those formats can feed.

## 47. Long-Term Architecture

```text
                         Sources

          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        Zarr            GeoTIFF          LAZ
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                        Chunks
                          │
                          ▼
                   Spatial representations

             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Raster        Array       PointCloud
             │            │            │
             └────────────┼────────────┘
                          ▼
                    Execution engine
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
               SIMD             scalar
                          │
                          ▼
                     Domain layers

          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       terrain         remote          wildfire
                       sensing
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
    Count int64   // valid cells
    Sum   float64
    Min   float32
    Max   float32
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
reduce.MinMaxTiled(ctx, src, engine.Options{}) (min, max float32, count int64, err error)
reduce.MinMaxChunked(ctx, src engine.RasterSource, engine.Options{}) (min, max float32, count int64, err error)
reduce.StatsTiled(ctx, src, engine.Options{}) (Summary, error)
reduce.StatsChunked(ctx, src engine.RasterSource, engine.Options{}) (Summary, error)
```

`Mean` is `Sum/Count` and needs no pass of its own. Standard deviation
needs a second accumulator (sum of squares, or Welford) and can follow.
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

Status: `Count`, `MinMax` and the fold driver done (STRATA-12): the
`Reducer`/`Cells` shape, `Reduce` and `ReduceChunked`, `vec.ReduceMin`
and `vec.ReduceMax`, and package `reduce`. `Sum`, `Stats`, `Summary` and
the accumulator decision are next, then `algebra.Normalize`;
`benchmarks/reduce` lands with them, since what it has to measure is the
cost of an exact accumulator and a `Min` fold is pure bandwidth.

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
`VADDPS`, never `VFMADD`.

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
and here they are one parameter. §53 separates them, and reports what
that was worth.

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
That understatement is the argument for §29 on top: register-level fusion
removes the intermediates rather than relegating them to L2. And a
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

Status: the radius-0 cut is done. `Pipeline`, `NewPipeline`, the
`ScratchKernel`/`ScratchSize`/`Scratch` contract extension and
`Span.Scratch`, per-worker scratch in `job.allocScratch` sized by
`plan.spanSize` and `chunkJob.spanSize`, and
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

Still to do, in the order the spec gives them: radius > 0 stages, which
need `erodedValidity` extracted from `job` and the suffix-sum window;
more than one output; and the decision about publishing `Kernel`, which
is what would let a caller build one of these.

## 53. Band Shape

§51 measured the halo and found it was not where the intuition put it.
Small tiles move *less* traffic than large ones and run 4–10× slower, so
the halo is not what makes them slow. What governs the halo is not tile
size at all but the aspect ratio of the band — the rectangle one kernel
call covers — and the band's width was the tile's width, so one parameter
set both. §51 named the fix and this section takes it: separate the IO
tile from the compute tile.

### The arithmetic

A band of `bandW × bandH` cells reads a window of `(bandW+2r) × (bandH+2r)`,
so its halo is

```text
(bandW + 2r)(bandH + 2r)
------------------------ - 1
      bandW · bandH
```

With `bandW · bandH` fixed at `bandCells`, that is smallest for a square
and grows without bound as the band gets thin. Bands were whole tile rows,
`bandCells / tileW` of them, so the thinness was set by the tile's width:

| tile width | rows per band | halo at r=1 |
|---:|---:|---:|
| 1024 | 64 | 3.3% |
| 4096 | 16 | 12.5% |
| 16384 | 4 | 50.0% |

That is §51's table read the other way round. Its "1024 × 1024 tiles, 3.1%
halo" row is not a property of a 1024-wide *tile*; it is a property of a
1024-wide *band*, which a 1024-wide tile happened to force. A caller who
wanted the low halo had to take the narrow tile with it, and pay §27's
price for it: a source makes a call per row of a narrow window, and
1024×1024 tiles of a 20000-wide file run at 200 M cells/s against 800 for
full-width strips. The two wants are opposite — IO wants the tile full
width, compute wants the band square — and they were one number.

### Shape

The band is given its own size, and the IO tile keeps `TileWidth` and
`TileHeight`:

```go
type Options struct {
    TileWidth  int
    TileHeight int
    // ComputeWidth and ComputeHeight are the size in cells of the band
    // one kernel call covers inside a tile. 0 is the engine's choice.
    ComputeWidth  int
    ComputeHeight int
    ...
}

// bandShape is the size of the bands a tileW×tileH tile is split into.
func bandShape(tileW, tileH int, t tiling) (bandW, bandH int)
```

Nothing above `plan` learns that a band is no longer a whole row. A tile
is still read in one `ReadWindow` per source and written in one
`WriteWindow` per sink, whatever shape the bands inside it take, so the
call count — the thing §27 says decides file throughput — is untouched.
That is the decoupling: the same reads, a different halo.

Three bounds hold the rule in:

- **Radius 0 keeps whole rows.** There is no halo to save, and a band as
  wide as its tile keeps a pointwise kernel's views compact
  (`Stride == Width`), which is what lets it and `pointwiseValidity` take
  their whole-span paths. Reductions are radius 0 by definition (§49), so
  they are untouched.
- **A tile no taller than one whole-row band is left alone.** It is
  already a single band, and splitting it across its width would only add
  perimeter: a 4096×16 tile in bands of 1024×16 reads 12.7% where the
  whole tile reads 12.5%.
- **`minBandWidth` stops the rows getting short**, which is the next
  subsection.

### Why not square

A square band reads the least halo, and is the wrong shape anyway. §25's
measurement says why: a row kernel pays a fixed 5–30 ns per row, short
rows read memory in streams the prefetcher has not seen, and 256×256 tiles
— which is to say one 256×256 band — cost Slope 15–21% and Hillshade
21–40% against full-width ones. The halo a band saves is a few per cent of
one of the four stages a chunked call moves; the row cost is paid on every
row of every band. So the band is the squarest rectangle of `bandCells`
whose rows are still at least `minBandWidth` long, not the squarest
rectangle, and the tile is divided into equal columns rather than into
`minBandWidth` columns and a narrow remainder.

### What it was worth

`BenchmarkBandWidth` sweeps the band width over one full-width tile, so
every case reads and writes exactly the same cells through exactly the
same sources and only the rectangle one call covers changes. Slope,
Apple M4, scalar backend, median of 6 runs:

| Slope, ms | rows | 2048 | 1024 | 512 | 256 |
|---|---:|---:|---:|---:|---:|
| 4096², 1 worker | 52.2 | +2% | +2% | +5% | +13% |
| 4096², 12 workers | 8.3 | +6% | +4% | +10% | +17% |
| 4096², masked, 1 worker | 51.9 | +3% | +3% | +6% | +15% |
| 16384², 1 worker | 847.8 | −2% | −1% | +4% | +12% |
| 16384², 12 workers | 140.8 | +2% | +3% | +5% | +10% |
| 16384², masked, 1 worker | 911.4 | −7% | −6% | +0% | +17% |
| 16384², masked, 12 workers | 151.0 | −3% | +3% | +12% | +21% |

and the traffic those shapes move, which does not depend on the size, the
worker count or the mask (§51 does not count validity):

| | rows | 2048 | 1024 | 512 | 256 |
|---|---:|---:|---:|---:|---:|
| halo, 4096² | 12.4% | 6.3% | 3.2% | 1.9% | 1.5% |
| halo, 16384² | 50.0% | 6.3% | 3.3% | 1.9% | 1.5% |
| B/cell, 16384² | 10.00 | 8.25 | 8.13 | 8.08 | 8.06 |

The halo behaves exactly as the arithmetic says, and it is not a rounding
error: at 16384 wide, a fifth of everything a tiled call moves is halo,
and shaping the band removes it.

It buys nothing. At 4096² every shaped case is slower, by 2–4% for the
shape the rule would pick. At 16384² the two run at the same speed — 847.8
against 842.6 — while moving 10.00 and 8.13 bytes per cell, a 19%
difference in traffic that costs nothing and saves nothing. Only 16384²
masked on one worker shows a real gain, and its mirror at 12 workers does
not. Below 1024 the row cost takes over and every case is worse, up to
21%.

That is §51's own warning — "these are the bytes the engine moves between
stages, not the bytes that reach memory" — holding for the change §51
proposed. Neighbouring full-width bands share halo *rows* and run
back-to-back in plan order, so the re-read was already an L2 hit; shaping
the band converts it into shared halo *columns*, also an L2 hit, and the
cache absorbed it either way.

**So the rule is off.** `minBandWidth` sits above every real tile width,
bands are whole tile rows as they were, and `ComputeWidth`/`ComputeHeight`
are there for the caller who wants to measure it again. Turning it back on
is restoring one constant to 1024.

Two reasons that constant is worth leaving where it can be moved. The
measurement is from an Apple M4 in the scalar build — the SIMD path is
`amd64` only (§17) — where the 12-core Zen 2 every other figure in this
document comes from has a quarter of the L1d and a different prefetcher,
and the row cost the floor guards against is a SIMD-build number. And a
19% traffic reduction that is free today is not free on a machine or a
kernel that is bandwidth-bound rather than latency-bound; §29's fusion and
§52's pipelines both raise arithmetic intensity per byte moved, which is
the direction that would make this pay.

What would earn the default: Slope and Hillshade faster than whole-row
bands beyond the run-to-run noise at one worker *and* at twelve, at 4096²
*and* 16384², with Clamp — radius 0, and so untouched — flat.

### Where it lives

`engine.Options.ComputeWidth` and `ComputeHeight`; `bandShape`,
`minBandWidth`, `tiling` and the two-dimensional `plan` in
`internal/exec/tile.go`; `chunkJob.bandBounds` in `chunked.go`, which must
size `ErodeBox`'s scratch from the widest band of any tile — a tile
clipped at the raster's edge can band *wider* than a full one, since the
rule leaves a narrow tile whole; and an early-out in `job.band` for the
bands that have no edge cells, which a whole-row band rarely was and a
sub-rectangle usually is.

### Testing

`TestBandShape` pins the rule, and `TestBands` checks the numbering
against nested loops, taking the shape from `BandShape` so the rule is
written down once. `FuzzPlan` covers the four-level index — the corner is
a band clipped by the last tile column *and* the last tile row — over
fuzzed sizes, tiles, band areas, floors and radii. Band shape is a new
axis of §23's bit-for-bit matrix: `TestTilesAndWorkers` and
`TestChunkedTilesAndWorkers` rotate a low `minBandWidth` through the same
cross product, so two-dimensional bands are checked against the plain
whole-raster function for every operation, mask combination, window and
worker count. `TestStatsBandShapeCutsTheHalo` pins the mechanism in
hand-derived bytes — 1015808 of halo for whole rows against 242000 for
1024×64 over the same 4096×512 raster — and asserts that `SourceRead` and
`SinkWritten` do not move between them, which is the decoupling itself.
`TestChunkedClippedTileBandsWider` covers the wider-clipped-tile corner;
breaking `bandBounds` makes it fail inside `ErodeBox`.

Status: done, and off. The mechanism, the options and the tests are in;
the default is whole tile rows because `BenchmarkBandWidth` says shaping
the band cuts the halo by up to a factor of 15 and does not make the call
faster. This retires the lever §51 left open: not by pulling it, but by
measuring it.
