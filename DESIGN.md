# SIMD-First Environmental Raster Engine

## 1. Project Goal

Build a Go-native, SIMD-accelerated raster and multidimensional array compute engine optimized for large environmental, terrain, and geospatial workloads.

The project is not intended to be:

- a GDAL rewrite
- a general-purpose GIS suite
- a vector geometry engine
- a wrapper around GDAL/OGR
- a format-conversion toolkit

Instead, the core focus is:

High-throughput numerical computation over large spatial and environmental datasets using Go-native memory layouts, SIMD, chunking, and concurrency.

The initial emphasis should be on:

- dense numeric rasters
- multidimensional arrays
- SIMD-accelerated kernels
- terrain analysis
- tiled and chunked execution
- predictable memory access
- scalar fallbacks for correctness and portability

IO formats, CRS support, and external interoperability should remain separate layers.

## 2. Positioning

The project should not be positioned as:

> GDAL for Go

That would immediately place it in competition with projects such as Gogeo, godal, and go-gdal, where the main value proposition is exposing traditional GIS functionality through Go.

Instead, the project should be positioned as:

> A SIMD-accelerated raster and array compute engine for Go, optimized for large geospatial and environmental datasets.

A shorter version:

> Fast numerical computing for environmental rasters in Go.

## 3. Differentiation from Gogeo and GDAL

Gogeo is fundamentally GIS-oriented. Its abstractions revolve around concepts such as:

- layers
- features
- shapefiles
- FileGDB
- PostGIS
- GDAL datasets
- clipping
- intersection
- union
- raster conversion

Performance primarily comes from:

- tiled processing
- worker concurrency
- native GDAL/OGR operations

The architecture is approximately:

```text
Go application
      │
      ▼
    Gogeo
      │
      ▼
  GDAL / OGR
      │
      ▼
native GIS implementation
```

This project should instead have:

```text
Go application
      │
      ▼
 Raster / Array API
      │
      ▼
 Execution engine
      │
      ▼
 Vector kernels
      │
 ┌────┴────┐
 ▼         ▼
SIMD     scalar
      │
      ▼
     CPU
```

The fundamental difference is therefore:

> Gogeo is primarily a GIS interface.
> This project is primarily a numerical compute engine.

## 4. Scope Boundary

The compute engine should know as little as possible about traditional GIS concepts.

The core should understand:

```text
Array
Raster
Grid
Shape
Stride
Window
Span
Tile
Mask
Kernel
```

It should not fundamentally depend on:

```text
Shapefile
GeoPackage
PostGIS
FileGDB
OGR Feature
GDAL Dataset
```

Those belong in adapters.

For example:

```go
terrain.Slope(dst, dem)
```

should work regardless of whether `dem` originated from:

```text
GeoTIFF
COG
Zarr
NetCDF
memory
HTTP
S3
generated terrain
database
```

## 5. Core Design Principles

### 5.1 SIMD-first

Algorithms should be designed around processing contiguous groups of numeric values rather than individual cells.

Avoid APIs that encourage:

```go
func SlopeAt(r Raster, x, y int) float32
```

Prefer APIs that naturally process batches:

```go
func Slope(dst, src Raster)
```

Internally, the execution path should become:

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

The central computational model is:

> Raster → tile → span → vector kernel

### 5.2 SIMD implementation details must remain internal

Go's SIMD API should not leak into the public API.

Do not expose:

```go
simd.Float32s
```

to users. Instead:

```text
terrain
   │
raster
   │
algebra
   │
   ▼
internal/vector
   │
   ├── SIMD
   └── scalar
```

This protects the public API from changes in experimental SIMD support.

### 5.3 Scalar correctness is canonical

Every SIMD algorithm should have an equivalent scalar implementation.

The scalar path provides:

- correctness reference
- unsupported-platform fallback
- easier debugging
- deterministic comparison
- testing baseline

SIMD should be treated as an execution optimization, not as a different algorithm.

### 5.4 Contiguous memory matters

Hot processing paths should operate over flat slices. For example:

```go
[]float32
```

rather than:

```go
[]Cell
```

Avoid:

```go
type Cell struct {
    Elevation   float32
    Temperature float32
    Humidity    float32
}
```

Prefer Structure of Arrays:

```go
type Environment struct {
    Elevation   []float32
    Temperature []float32
    Humidity    []float32
}
```

This greatly improves vectorization and cache behavior.

### 5.5 Separate computation from storage

The compute engine should not care how data is persisted. Storage adapters should turn external datasets into processable chunks.

Conceptually:

```text
             Compute Engine
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
      GeoTIFF     Zarr     NetCDF
```

The existing Zarr project can become one storage backend rather than something duplicated inside this project.

## 6. Raster and Array Model

The project should begin with two related concepts:

- `Raster`
- `Array`

A raster adds spatial meaning. An array represents dense multidimensional numerical data.

### 6.1 Array

Eventually:

```go
type Array[T Number] struct {
    Data   []T
    Shape  []int
    Stride []int
}
```

Example shapes:

```text
[y, x]

[time, y, x]

[scenario, time, y, x]
```

This matters because environmental data is rarely purely two-dimensional. Examples:

```text
temperature[time,y,x]

wind_speed[time,y,x]

humidity[time,y,x]

fuel[y,x]

fire_intensity[scenario,time,y,x]
```

### 6.2 Raster

A raster is a spatial specialization:

```go
type Raster[T Number] struct {
    Data []T

    Width  int
    Height int
    Stride int

    NoData T
}
```

Initially, `float32` should be the primary processing type.

Potential optimized specialization:

```go
type Float32Raster struct {
    Data []float32

    Width  int
    Height int
    Stride int

    NoData float32
}
```

Storage may support many numeric types. Processing should strongly favor `float32`.

## 7. Spatial Metadata

Spatial metadata should remain separate from the raw numeric representation.

```go
type Grid struct {
    Width  int
    Height int

    ResolutionX float64
    ResolutionY float64

    OriginX float64
    OriginY float64

    CRS CRS
}
```

A higher-level dataset may combine both:

```go
type Dataset struct {
    Grid   Grid
    Raster Float32Raster
}
```

This allows low-level numerical algorithms to operate without knowing about CRS metadata.

## 8. Windows and Views

Avoid copying whenever possible. A `Window` should represent a view into an underlying raster:

```go
type Window[T Number] struct {
    Data []T

    Width  int
    Height int
    Stride int
}
```

For example:

```go
window := raster.Window(
    x,
    y,
    width,
    height,
)
```

Windows become especially important for terrain operations where neighboring cells are required.

## 9. Vector Kernel Layer

The vector layer should be one of the most carefully designed components.

Suggested package:

```text
internal/vec
```

Operations:

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
```

Possible future operations:

```text
FMA
Select
Blend
Compare
Mask
ReduceMin
ReduceMax
ReduceSum
Dot
```

## 10. Backend Model

Suggested internal layout:

```text
internal/vec/
├── scalar.go
├── simd.go
├── simd_amd64.go
├── simd_arm64.go
└── dispatch.go
```

Backend selection should happen outside hot loops. For example:

```go
var addFloat32 = scalarAddFloat32
```

Then swap in the SIMD implementation where supported. Avoid per-vector dynamic dispatch.

## 11. Raster Algebra

A high-level algebra package should expose operations over entire rasters or arrays.

```text
algebra/
├── add.go
├── multiply.go
├── clamp.go
├── normalize.go
├── minmax.go
└── mask.go
```

Examples:

```go
algebra.Add(dst, a, b)

algebra.Scale(dst, src, 1.25)

algebra.Clamp(dst, src, 0, 100)

algebra.Normalize(dst, src)

algebra.Mask(dst, src, mask)
```

These operations should map naturally to vector kernels.

## 12. Avoid Arbitrary Callbacks in Hot Paths

APIs like:

```go
raster.Map(src, func(v float32) float32 {
    return ...
})
```

are convenient but can prevent vectorization. They may exist for flexibility, but should not become the preferred performance API.

Explicit operations are easier to:

- vectorize
- fuse
- optimize
- benchmark
- reason about

## 13. Terrain Package

Terrain analysis should be the first major real-world consumer of the compute engine.

Suggested structure:

```text
terrain/
├── gradient.go
├── slope.go
├── aspect.go
├── hillshade.go
├── ruggedness.go
└── curvature.go
```

Initial functions:

```go
terrain.Gradient(...)

terrain.Slope(...)

terrain.Aspect(...)

terrain.Hillshade(...)
```

Terrain processing is a good architecture test because it requires:

- neighboring cells
- overlapping vector loads
- edge handling
- floating-point math
- SIMD
- tiled execution
- halo management

## 14. Example Terrain Kernel

A slope algorithm may operate on:

```text
z1 z2 z3
z4 z5 z6
z7 z8 z9
```

Instead of calculating one output cell at a time, calculate several adjacent cells in parallel.

Conceptually:

```text
row y-1
──────────────────────────

row y
──────────────────────────

row y+1
──────────────────────────

         ↓ SIMD windows

[x x x x x x x x]
[x x x x x x x x]
[x x x x x x x x]

         ↓

8 gradients simultaneously
```

This should be a defining optimization model for the project.

## 15. Kernel Abstraction

After several algorithms exist, introduce a reusable kernel concept.

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
Normalize   radius 0
Slope       radius 1
Hillshade   radius 1
5×5 filter  radius 2
```

The engine can then reason about boundary requirements generically.

## 16. Halo Handling

Chunked spatial computation requires neighboring data. For example:

```text
┌───────────┬───────────┐
│           │           │
│  tile A   │  tile B   │
│           │           │
└───────────┴───────────┘
```

Slope at the edge of tile A requires cells from tile B. Therefore the runtime should automatically create halo regions:

```text
XXXXXXXXXXXX
X..........X
X.. tile ..X
X..........X
XXXXXXXXXXXX
```

Algorithms should not need to manually implement cross-tile boundary logic.

## 17. Execution Engine

The execution engine owns:

- tile planning
- worker scheduling
- halo construction
- temporary buffers
- backend execution

Suggested API:

```go
engine.Process(
    ctx,
    src,
    dst,
    kernel,
)
```

Possible configuration:

```go
type Options struct {
    TileWidth  int
    TileHeight int
    Workers    int
}
```

## 18. Parallelism Model

The system has three natural levels of parallelism.

**IO parallelism** — Multiple chunks can be loaded concurrently.

**Worker parallelism** — Tiles can be processed by goroutines.

**Instruction parallelism** — Each worker processes many values through SIMD.

```text
Dataset
   │
   ▼
Tiles
   │
   ├──── Worker ─── SIMD
   ├──── Worker ─── SIMD
   ├──── Worker ─── SIMD
   └──── Worker ─── SIMD
```

Low-level kernels should not create goroutines. Concurrency belongs in the execution engine.

## 19. Memory Bandwidth Awareness

SIMD should not be treated as automatically synonymous with large speedups. For very simple operations such as:

```go
dst[i] = src[i] * 2
```

performance may quickly become limited by memory bandwidth. More computationally intensive workloads should benefit more strongly:

```text
slope
aspect
hillshade
convolution
resampling
interpolation
fire-behavior transforms
weather calculations
```

Benchmarks should therefore distinguish between:

- compute-bound
- cache-bound
- memory-bandwidth-bound
- IO-bound workloads

## 20. Operation Fusion

Operation fusion should be considered a future major feature. For example:

```text
Normalize
   ↓
Scale
   ↓
Clamp
```

A naive implementation produces multiple memory passes. A fused implementation becomes:

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

This may eventually provide larger gains than SIMD alone for many raster pipelines. The initial architecture should avoid decisions that make operation fusion impossible later.

## 21. NoData and Validity

NoData handling must be designed carefully. Checking:

```go
if value == NoData
```

inside every inner loop can harm vectorization. One possible model is a separate validity mask:

```go
type Raster struct {
    Data  []float32
    Valid []uint64
}
```

Conceptually:

```text
values

12 18 22 30 44 50 65 80

validity

 1  1  0  1  1  1  0  1
```

This design should be benchmarked against sentinel values and NaN-based approaches before stabilizing.

## 22. Chunking

Large datasets should never require full materialization in memory. A chunk should be processable independently:

```text
Large dataset

┌────┬────┬────┬────┐
│    │    │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
```

The execution engine should eventually support:

```text
read chunk
   ↓
add halo
   ↓
process
   ↓
write chunk
   ↓
release buffer
```

This naturally integrates with Zarr and cloud-native raster storage.

## 23. N-Dimensional Environmental Data

This should become an important differentiator from traditional GIS engines. A future array engine should comfortably support:

```text
weather[time,y,x]

risk[scenario,time,y,x]

wind[level,time,y,x]

fuel[class,y,x]
```

This aligns naturally with Zarr. The project could eventually become:

```text
                   Environmental data

                          Zarr
                           │
                           ▼
                 N-dimensional arrays
                           │
                           ▼
                   Compute engine
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          terrain        weather        fire
```

This is more aligned with environmental modelling than recreating classic desktop GIS workflows.

## 24. IO Architecture

Formats should be adapters. Eventually:

```text
io/
├── geotiff/
├── cog/
├── zarr/
├── netcdf/
└── gdal/
```

A GDAL adapter could even be supported optionally:

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

GDAL would therefore become:

> a source of data

rather than:

> the architecture underneath the project

## 25. CRS and Reprojection

Do not attempt to replace PROJ in the initial project. Define a narrow abstraction:

```go
type Transformer interface {
    Transform(
        src CRS,
        dst CRS,
        points []Point,
    ) error
}
```

Native support could eventually cover common cases such as:

```text
WGS84
Web Mercator
common UTM zones
```

More advanced transformations can come through adapters.

## 26. Memory Management

Large workloads can create significant temporary allocations. Avoid patterns where every operation allocates a full raster.

Eventually introduce reusable workspaces:

```go
type Workspace struct {
    Float32 []float32
}
```

Possible future concepts:

```text
arena-like temporary buffers
chunk-local buffers
scratch spans
buffer reuse
```

Do not introduce aggressive pooling before ownership semantics are clear.

## 27. Benchmarking

Benchmarks should be part of the project's identity.

Suggested benchmark categories:

```text
benchmarks/
├── algebra/
├── terrain/
├── convolution/
├── resampling/
└── nd/
```

Metrics:

```text
ns/cell
cells/sec
GB/sec
allocations
SIMD/scalar speedup
parallel scaling
```

Suggested dataset sizes:

```text
256 × 256

1024 × 1024

4096 × 4096

16384 × 16384
```

This reveals:

- cache behavior
- memory bandwidth effects
- worker scaling
- SIMD benefits

Benchmarks should compare:

```text
scalar

SIMD

SIMD + 1 worker

SIMD + physical cores

SIMD + logical cores
```

## 28. Correctness Testing

Every SIMD implementation should be validated against its scalar equivalent.

Test:

- ordinary values
- NaN
- infinities
- negative values
- zero
- NoData
- odd raster dimensions
- non-contiguous stride
- vector-width tails
- tiny rasters
- tile boundaries
- halo boundaries

Vector tails are particularly important. For example:

```text
SIMD SIMD SIMD SIMD scalar scalar scalar
```

The result must be equivalent regardless of vector width.

## 29. Suggested Package Layout

Initial structure:

```text
project/
├── array/
│   ├── array.go
│   ├── shape.go
│   └── view.go
│
├── raster/
│   ├── raster.go
│   ├── grid.go
│   ├── window.go
│   └── mask.go
│
├── algebra/
│   ├── add.go
│   ├── multiply.go
│   ├── clamp.go
│   ├── normalize.go
│   └── minmax.go
│
├── terrain/
│   ├── gradient.go
│   ├── slope.go
│   ├── aspect.go
│   └── hillshade.go
│
├── engine/
│   ├── engine.go
│   ├── tile.go
│   ├── halo.go
│   ├── worker.go
│   └── workspace.go
│
├── internal/
│   └── vec/
│       ├── scalar.go
│       ├── simd.go
│       ├── simd_amd64.go
│       ├── simd_arm64.go
│       └── dispatch.go
│
└── benchmarks/
```

## 30. Explicit Non-Goals for v0.1

Do not initially build:

```text
Shapefile support
GeoPackage support
PostGIS integration
vector geometry
polygon overlay

GeoTIFF parsing
COG
NetCDF

CRS transformation
PROJ replacement

distributed computing

GPU acceleration

CUDA
OpenCL

full GIS workflows
```

These are all potential future integrations. They should not distract from proving the compute architecture.

## 31. Initial Milestone — v0.1

The first release should prove one central hypothesis:

> Go SIMD + flat raster memory + tiled execution can form a strong foundation for environmental numerical computing.

Support:

```text
Float32 raster

2D windows

scalar backend

SIMD backend

backend dispatch

Add

Subtract

Multiply

Clamp

Min

Max

Gradient

Slope

Aspect

Hillshade

single-thread processing

multi-worker tile processing

halo handling

benchmark suite
```

Example usage:

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

Benchmark output should eventually communicate results in domain-relevant terms:

```text
4096 × 4096 DEM

scalar:
xxx M cells/sec

SIMD:
xxx M cells/sec

SIMD + workers:
xxx B cells/sec
```

## 32. Development Roadmap

**v0.1 — Compute foundation**

```text
Raster
SIMD
scalar fallback
terrain kernels
tile engine
benchmarks
```

**v0.2 — Array foundation**

```text
N-dimensional arrays
strides
views
reductions
broadcast-style operations
```

**v0.3 — Chunked execution**

```text
streaming chunks
workspace reuse
large datasets
chunk scheduling
```

**v0.4 — Zarr integration**

```text
existing Zarr library
chunk-native execution
time × y × x datasets
```

**v0.5 — Raster IO**

```text
GeoTIFF
COG
```

**v0.6 — Spatial processing**

```text
resampling
alignment
mosaics
interpolation
```

**v0.7 — CRS interoperability**

```text
basic native CRS
external transformer adapters
```

**v0.8 — Pipeline optimization**

```text
operation fusion
lazy execution
kernel planning
```

**v1.0**

A stable:

> SIMD-accelerated environmental raster and array compute engine for Go

with cloud-native storage integration and production-quality terrain processing.

## 33. Long-Term Identity

The project should live closer to:

```text
NumPy
+
raster algebra
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

The ideal ecosystem becomes:

```text
                    Storage

       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
     Zarr          GeoTIFF          COG
       │              │              │
       └──────────────┼──────────────┘
                      ▼
               Raster / Array
                      │
                      ▼
              Execution engine
                      │
              ┌───────┴───────┐
              ▼               ▼
            SIMD            scalar
                      │
                      ▼
               Domain packages

        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     terrain       weather        wildfire
```

That keeps the project closely aligned with spatial and fire-risk work while leaving higher-level proprietary risk modelling outside the open-source boundary.

The architectural principle to protect throughout development is:

> Storage provides chunks. The engine schedules tiles. Kernels operate on spans. SIMD executes the numerical work.

Everything else should be built around that.
