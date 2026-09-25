# ADR 0003: The Zarr adapter — a wrapped library, and a chunk cache from the start

- **Status:** Accepted
- **Date:** 2026-09-25
- **Related:** DESIGN.md §9, §10, §24, §31, §34, §35, §45 (v0.5); ADR 0002

## Context

Zarr stores N-dimensional arrays cut into chunks, each chunk encoded on
its own under a key of a store: a directory, a bucket, memory. It is how
climate, Earth-observation and model output increasingly arrive, and the
roadmap has a Zarr adapter at v0.5 (§45). §34 names it as a candidate
adapter, and §35 says adapters wrap existing format libraries.

`github.com/LukasSelin/zarr` is such a library: Zarr version 3, pure Go,
the standard library only, tagged v0.3.0. It reads and writes arrays and
groups, the integer, unsigned, float and bool data types, regular chunk
grids, both chunk key encodings, and the codecs bytes, gzip, crc32c and
sharding_indexed, with others registered from outside (zstd is a module
of its own). Its interop test has zarr-python write every case it reads,
and read every case it writes.

Its benchmarks (LukasSelin/zarr#14, BENCHMARKS.md) show the cost that
decides this adapter's design. Every read decodes whole chunks, however
little of them it needs. On a float32 raster with gzip level 5:

- one pixel costs 9.1 ms with 512 × 512 chunks, 2.3 ms with 256 × 256;
- a 3 × 3 window across a chunk corner costs 12.2 ms and decodes four
  chunks;
- a 256 × 256 window across four chunks costs 12.6 ms.

strata's engine reads a tile and its halo at a time (§24), so for any
focal or terrain operation neighbouring tiles ask for the same chunks
again and again. The cog adapter met the same problem with blocks and
solved it with a byte-bounded cache of decoded blocks, sharing a block's
load between the readers that want it at once (ADR 0002,
benchmarks/cog/RESULTS.md, "The cache").

## Decision

### 1. A module of its own, wrapping the library

The adapter is `github.com/LukasSelin/strata/zarr`, in `zarr/` with its
own `go.mod`, as `cog/` is (ADR 0002, §1). It requires
`github.com/LukasSelin/zarr v0.3.0` and resolves strata through
`replace github.com/LukasSelin/strata => ..`. The core module gains no
dependency.

The package is called `zarr`, like its directory, and imports the
library as `zarrv3`. A program that uses both, which is most, imports one
under another name; the documentation's examples do what the package
does. The alternative, a package name that differs from its directory
(`zarrsrc`), is surprising every time the import path is read.

Unlike cog, it parses nothing: store, metadata, chunk keys, codecs and
shards are the library's. §35 holds without exception, so the adapter
is about 600 lines, comments included and tests aside.

### 2. What it reads, and what it refuses

`zarr.Open(ctx, store, path, SourceOptions)` or `zarr.NewSource(array,
SourceOptions)` gives a `Source`, an `engine.RasterSource` over one y-x
plane of an array:

- **Two or more dimensions.** The last two are the raster's rows and
  columns. `SourceOptions.Index` fixes the ones before them, one index
  each, so a (time, band, y, x) array gives one source per time step and
  band. Fewer than two dimensions, an empty plane, or an Index of the
  wrong length or out of range is an error.
- **int8 to int64, uint8 to uint64, float32, float64.** Each element
  converts to float32 as IEEE 754 rounds, ±Inf past float32's range,
  as cog converts (§9). bool is refused: a mask is not an elevation, and
  nothing in strata computes on one yet.
- **The fill value is NoData** (§31, rule 5). Every Zarr array has a
  `fill_value`, which is what a chunk nobody wrote reads as. A cell is
  invalid where its element equals it, compared in the array's own type
  before the conversion, so an int64 or float64 fill that float32 cannot
  hold matches only itself (benchmarks/nodata/RESULTS.md). A NaN fill
  matches every NaN, whatever its payload, and 0 matches −0, as
  `engine.RawOptions` compares. Zarr's fill value is also often just a
  default, 0 for a count, so `SourceOptions.IgnoreFill` reads every cell
  as valid, and the source is then not Masked.
- **The comparison is exact.** cog follows GDAL's tolerance for NoData
  (ADR 0002, addendum) because GDAL is its judge. The fill value of a
  Zarr array is the element a writer put there, in the array's own type,
  and xarray masks `_FillValue` by equality, so exact is the right rule
  here.

Everything the library refuses (other chunk grids, storage transformers,
float16, complex and raw types, unregistered codecs) fails in
`zarrv3.OpenArray`, with the library's error.

### 3. A decoded-chunk cache from day one, shared with cog

A Source keeps decoded chunks, as float32 cells and a validity mask
(nil when every cell is valid), in a least-recently-used cache under a
byte budget, `SourceOptions.CacheBytes`. It is keyed by the chunk's
position in the plane's chunk grid. The default is cog's: 8 rows of
chunks, between 64 MiB and 1 GiB, since a row of chunks is what one band
of tiles needs. A negative value turns it off, and `CacheBytes()`
reports the bound as cog's does.

Loads are shared. A chunk asked for while another reader is loading it
waits for that load rather than decoding it again. A failed load is not
kept, so the next read tries again. A load that failed because its own
reader's context ended is not handed to the readers waiting on it
either: each of them, its own context live, loads the chunk itself.
`CacheStats()` counts hits, shared loads, loads, failures and
evictions.

The cache is cog's, moved rather than copied. `cog/cache.go` became
`internal/blockcache` in the core module, exported within it (`New`,
`WithHolds`, `Get`, `Peek`, `Limit`, `Stats`), and both adapters import
it. Go's internal rule goes by import path, so
`github.com/LukasSelin/strata/internal/blockcache` is importable from
`github.com/LukasSelin/strata/zarr` and `.../cog`, and from nothing
outside strata. ADR 0002's rule that an adapter reaches strata only
through its public API is about computation and IO, the interfaces that
show whether a source is complete; a cache is neither, and two copies of
a concurrent data structure would drift. The cost is that an adapter
must be released with the strata it was built against, which the
`replace` directives already make true. The move added what zarr needed
and cog can use: the counters, `Peek`, and the waiters' retry after a
cancelled loader.

A window that needs several chunks copies those the cache holds first,
with no goroutine (`Peek`), and then loads the rest concurrently, up to
`SourceOptions.ReadConcurrency` at a time (8 by default). Chunk reads
are `zarrv3.ReadChunk[T]`: one store read and one decode per chunk, or,
in a sharded array on a store with range reads, the shard's index and
the one chunk. The cache holds inner chunks, not shards.

What the cache buys, from benchmarks/zarr/RESULTS.md:

- every chunk is decoded once, where without the cache the engine's
  256-row strips decode each 2.875 times (256² chunks) or 3.75 times
  (512²);
- slope over a 4096² gzip float32 DEM on one core runs 2.5× faster
  (256² chunks) and 3.0× (512²), and 2.3–2.4× on four workers;
- a window the cache holds costs 0.1 µs for a pixel and 0.3 µs for a
  3 × 3 across a chunk corner, where decoding it costs 1.2–12 ms.

Reading a window's chunks concurrently matters as much on many workers:
with one chunk at a time, four workers ran slope over 512² chunks only
1.25× faster than one, waiting in the cache for the chunk rows another
worker was decoding.

### 4. Georeferencing: two GeoZarr attributes, both optional

Zarr has no georeferencing of its own. The source reads two attributes
of the array, from the GeoZarr conventions:

- `"spatial:transform"`: six numbers, the affine transform in
  rasterio's order, `[a, b, c, d, e, f]` with a cell corner (col, row) at
  x = a·col + b·row + c, y = d·col + e·row + f. b and d must be 0:
  `raster.Grid` cannot hold a rotation, and cog refuses one too.
  `"spatial:registration"`, if present, is `"pixel"` (the default: the
  transform places corners) or `"node"` (it places cell centres, and the
  origin moves half a cell).
- `"proj:code"`: an authority code such as `"EPSG:32633"`, taken as the
  grid's CRS label as it is (§36).

A malformed attribute is an error; an absent one is not. Without a
transform, `Grid` is the array's own cells, origin (0, 0), resolution
(1, 1), as cog reports an ungeoreferenced file.

The other candidate was a `transform` plus `crs_wkt` pair, as rioxarray
writes. It was not taken because `raster.CRS` is a code, not a WKT
(§36), so the WKT would have to be parsed or carried blind, and because
the GeoZarr attributes are the ones a Zarr v3 writer is being asked to
use. CF grid mappings and coordinate arrays, which most NetCDF-born Zarr
stores carry instead, are not read.

## Consequences

- A Zarr array is an `engine.RasterSource`, so every Chunked entry point
  runs over a y-x plane of it in memory bounded by the tiles and the
  cache. Slope and a focal mean over a Zarr source write the bits they
  write over a `MemorySource` of the same cells, for tiles that do and do
  not line up with the chunks (`TestUnderEngine`).
- cog's cache now lives in `internal/blockcache`, and cog imports it.
  cog's behaviour is unchanged; its counters are there for a later
  `cog.Source.CacheStats`.
- The adapter's own tests compare it with the library's whole-array
  `zarrv3.Read`, which zarr-python judges. No outside tool judges the
  adapter itself yet, as GDAL judges cog (§34): an acceptance check that
  has xarray read the same stores and compares every cell and mask bit
  is open.

### Known limits

- **One plane per source.** A source reads a y-x plane; the engine has
  no N-D sources yet (§10). A chunk that spans several indices of a
  leading dimension is decoded whole by each source that reads one of
  them, and cached by each.
- **A cache per source.** Sources over different planes of one array
  share nothing, not even the store reads of a chunk they both need.
  cog's File shares compressed blocks between band sources; a
  store-level cache of encoded chunks would do the same here.
- **Sharded arrays without range reads.** On a store that cannot read
  part of a value, each chunk read fetches its whole shard. The cache
  holds decoded inner chunks, so a shard is fetched once per chunk the
  source loads, not once per window.
- **Fill value only.** `_FillValue`, `missing_value`, `valid_range`
  and scale/offset attributes are not read; the fill value is the only
  NoData.
- **Reading only.** A Zarr sink is later work.
