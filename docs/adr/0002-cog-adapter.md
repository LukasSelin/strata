# ADR 0002: The first format adapter — GeoTIFF/COG, its own module, its own parser

- **Status:** Accepted
- **Date:** 2026-09-22
- **Related:** DESIGN.md §9, §24, §31, §34, §35, §45 (v0.6)

## Context

Until now the only file strata could read was a headerless raw float32
file (`engine.RawSource`). A real raster had to be converted first with
`gdal_translate -of ENVI`, which is exactly what `acceptance/gdalcheck.sh`
does. GeoTIFF, and its cloud-optimized profile COG, is the format most
rasters arrive in. The roadmap had it at v0.6 (§45).

DESIGN.md §34 left two questions for the first adapter to settle:

1. Do adapters live under `io/` in this repository, or in modules of
   their own?
2. §35 says adapters wrap existing format libraries rather than
   reimplementing parsers. Does that hold here?

## Decision

### 1. Every adapter is its own Go module, in this repository

The adapter is `github.com/LukasSelin/strata/cog`, in `cog/` with its own
`go.mod`. `acceptance/` and `lint/` are set up the same way.

- The core module keeps no dependencies beyond its test libraries.
  Reading LZW and ZSTD blocks needs `golang.org/x/image/tiff/lzw` and
  `github.com/klauspost/compress/zstd`. Anyone who imports strata only
  for computation should not pull them in, and later adapters (Zarr,
  LAS/LAZ) will bring heavier ones.
- The adapter uses strata only through its public API: `engine.RasterSource`,
  `raster.Float32Raster`, `raster.Grid`. That forces the source interface
  to be complete. If an adapter needed something internal, the interface
  would be the thing to fix.
- It lives in the repository rather than a repository of its own, so it
  is tested against the strata it sits next to. Until strata is tagged, it
  resolves strata through a `replace` directive, as `acceptance/` does.

### 2. strata writes its own minimal GeoTIFF parser

§35 does not hold for GeoTIFF, because no library fits:

- `golang.org/x/image/tiff` decodes a whole image into an `image.Image`.
  It cannot read one window's blocks, and it has no floating-point
  samples. Those are the two things a raster engine needs most.
- The other pure-Go TIFF packages parse tags but decode no data, or
  support a subset smaller than what is needed here.
- GDAL through cgo (`godal`) would do the job, but it brings a cgo and
  system-library dependency. §34 allows that only in a module of its own,
  and it would make "no GDAL in the path" untrue for strata's most common
  input format.

What strata needs from TIFF is small and fully specified. It needs the
container (TIFF 6.0 and BigTIFF, IFDs, tags) and the block layouts (tiles,
strips, chunky and planar). It needs five decompressors, of which it
writes only PackBits: the standard library supplies Deflate, and the two
dependencies supply LZW and ZSTD. It needs libtiff's two predictors, a
few integer and float sample types, and the handful of GeoTIFF keys that
give a geotransform and an EPSG code. The package is about 1600 lines,
comments included and tests aside. It still wraps existing libraries wherever one does part of
the job, so §35 holds as far as the libraries allow.

Its scope is reading; writing is later. The reader turns every sample
type into `float32` plus validity at the boundary, as §9 and §31 require.
It compares NoData in the file's native type, before the conversion,
because a fill value inexact in float32 would otherwise never match
(benchmarks/nodata/RESULTS.md).

### 3. GDAL judges it

A parser written in-house is only as right as its author's reading of the
specification. The unit tests share that reading: the test-only TIFF
writer was written by the same hand. So the evidence that the reader is
correct comes from outside it. `acceptance/cogcheck.sh` has GDAL write 98
files in a Docker container:

- COGs of every supported sample type, compression and predictor, with
  overviews and sparse blocks;
- BigTIFF;
- stripped and tiled GeoTIFFs, pixel- and band-interleaved, big-endian;
- PixelIsPoint.

GDAL records its own reading of every file as the truth. strata must
match it exactly: every cell's float32 bits, every validity bit, every
level's geotransform, and the EPSG code. `acceptance/cogsabotage.py`
plants eight plausible reader bugs in strata's output and confirms that
the comparison fails on each one.

That check has already caught one defect. The first draft held float64
values beyond float32's range at ±MaxFloat32, believing GDAL does the
same. GDAL gives ±Inf. The unit tests encoded the same wrong belief and
passed.

## Consequences

- `cog.Open(io.ReaderAt)` → `File` (levels, bands, grid, NoData) →
  `File.Source(SourceOptions)`, an `engine.RasterSource`. Every Chunked
  entry point therefore runs over a GeoTIFF in bounded memory.
- A per-source, byte-bounded LRU of decoded blocks means that tiles and
  halos which share a block decode it once.
- The source takes an `io.ReaderAt`, so a byte-range reader over HTTP or
  object storage will plug in without changing the adapter. Shipping one
  is a separate step.
- The package refuses what it does not read with an error that names it:
  JPEG, WebP, LERC and the other lossy or exotic compressions, 16-bit
  floats and sub-byte samples, and rotated geotransforms, which
  `raster.Grid` cannot hold. Masks (internal or `.msk`) and GDAL
  scale/offset metadata are not read.
- Later adapters follow the same pattern: a module in the repository, a
  wrapped library where one fits, and an outside tool as the judge.

## Addendum, 2026-09-23: files GDAL did not write

The 98 files above were all written by GDAL, so they could not show how
the reader fares with other writers. `acceptance/cogcorpus.sh` runs the
same comparison over a corpus of GeoTIFFs from libtiff, tiffcp,
rasterio and rio-cogeo, ERDAS, PCI, Intergraph and USGS software, GDAL's
own deliberately odd test files, and Landsat, Sentinel-2, Copernicus,
NASA, USGS, JAXA and ESA products (`acceptance/README.md`, "Reading
GeoTIFFs other software wrote"). It changed three decisions above:

- **Internal masks are read.** A transparency-mask IFD now gives its
  level's validity, as GDAL's internal mask does, because rio-cogeo and
  GDAL write them for masked COGs and ignoring one reads masked cells as
  valid.
- **NoData compares as GDAL's mask band compares it, not exactly.**
  GDAL truncates an integer NoData, ignores one outside the type's range,
  and matches floats within `ARE_REAL_EQUAL`'s 2·FLT_EPSILON·|a+b|. The
  first draft compared exactly in the sample type, and its unit test
  encoded that belief, as the first draft's float64 overflow test had
  encoded another. The corpus found it through a byte file with NoData
  12.5; GDAL's source gave the rest, and generated files confirm each
  rule.
- **The CRS is named more cautiously.** An EPSG code is reported only
  when no other key or citation redefines it, and deprecated codes as
  their replacements, from a table generated from PROJ's database
  (`cog/epsg_gen.py`). Where GDAL identifies a redefined CRS with a code
  through PROJ, cog reports none; the corpus counts those files
  separately rather than failing them.
