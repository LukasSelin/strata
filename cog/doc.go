// Package cog reads Cloud Optimized GeoTIFFs, and GeoTIFFs in general, as
// strata raster sources. It is strata's first format adapter (DESIGN.md
// §34): it turns a file's sample type and NoData value into float32 cells
// and a validity mask at the boundary, so the engine never sees the
// format.
//
//	f, err := os.Open("dem.tif")
//	...
//	file, err := cog.Open(f)
//	src, err := file.Source(cog.SourceOptions{})       // band 0, full resolution
//	grid := src.Grid()                                  // origin, resolution, EPSG code
//	err = terrain.SlopeChunked(ctx, sink, src,
//		terrain.SlopeOptions{CellSize: grid.ResolutionX}, engine.Options{})
//
// A Source is an engine.RasterSource, so every Chunked entry point runs
// over it in memory bounded by the tile size, not the file. It reads from
// an io.ReaderAt, which may be a local file or anything else that serves
// byte ranges, such as an HTTPReaderAt.
//
// # Reading over HTTP
//
// NewHTTPReaderAt reads a COG straight from a URL, on S3, GCS or any
// HTTPS server, public or presigned, with range requests and the
// standard library only:
//
//	r, err := cog.NewHTTPReaderAt(ctx, "https://bucket.s3.amazonaws.com/dem.tif", cog.HTTPOptions{})
//	...
//	file, err := cog.Open(r)
//
// It fetches the first 64 KiB when it is created. GDAL writes every IFD
// of a COG, and the tile offsets they hold, at the start of the file, so
// Open is served from that block. After that, each block a Source
// decodes is one range request: the cache means a block is fetched once
// per Source, not once per window. A Source reads one band, so each band
// of a pixel-interleaved file fetches the block again. Requests must be
// answered 206 with exactly the range asked for. Transient failures
// (transport errors, truncated bodies, 429, 5xx) are retried with
// backoff, a bounded number of times, and at most
// HTTPOptions.MaxConcurrent requests are in flight. With a strong ETag,
// every request carries If-Match, so a file replaced while it is read
// fails with ErrObjectChanged rather than mixing two versions. Errors are
// *HTTPError, naming the URL (without its query string, which in a
// presigned URL is a credential) and the bytes asked for.
// acceptance/coghttpcheck.sh reads all of acceptance/cogcheck.sh's files
// through it, from nginx, and GDAL finds them identical.
//
// # What it reads
//
//   - Classic TIFF and BigTIFF, in either byte order.
//   - Tiled and stripped images, chunky or planar (PlanarConfiguration 1
//     or 2), with any number of bands; a Source reads one.
//   - Unsigned and signed 8-, 16- and 32-bit integers, and 32- and 64-bit
//     floats.
//   - No compression, LZW, Deflate, PackBits and ZSTD, with no predictor,
//     horizontal differencing (2) or the floating-point predictor (3).
//   - Overviews: every reduced-resolution image (NewSubfileType bit 0) of
//     the same bands and type, as levels 1, 2, ... in file order, which in
//     a COG is largest first.
//   - Georeferencing: ModelTransformation, or ModelPixelScale with one
//     ModelTiepoint, with GDAL's half-cell shift for PixelIsPoint; and the
//     EPSG code of a projected or geographic CRS.
//   - GDAL's NoData value (the GDAL_NODATA tag).
//
// Anything else is refused by Open with an error that names it: JPEG,
// WebP, LERC and other compressions; other bit depths, including 16-bit
// floats; and rotated or sheared geotransforms, which strata's Grid cannot
// hold. Transparency masks (internal or .msk) and GDAL's scale, offset and
// other metadata are not read.
//
// # Values and validity
//
// Cells are converted to float32 by rounding to nearest. That is exact
// for 8- and 16-bit integers and for float32 files. 32-bit integers
// beyond ±2²⁴ lose precision. float64 values round, and those beyond
// float32's range become ±Inf. GDAL converts the same way, and
// acceptance/cogcheck.sh checks every one of these conversions against
// GDAL.
//
// With a NoData value, the source is Masked, and a cell holding that
// value is invalid (DESIGN.md §31). The comparison is made in the file's
// sample type, before the conversion. GDAL does the same, so a float32
// file with NoData 1e-9 marks the cells that hold float32(1e-9), and a
// float64 cell that only rounds to NoData stays valid. A NaN NoData
// matches every NaN. A NoData value that the sample type cannot hold
// matches no cell: a fractional or out-of-range value for integers, or a
// value beyond float32's range for float32. Blocks that are absent from
// the file (sparse, offset and byte count 0) read as NoData, or as 0
// without it, as GDAL reads them.
//
// # Robustness
//
// Open and ReadWindow never panic on a file's contents. Every count and
// size read from the file is bounded before memory is allocated for it:
// IFDs, entries, tag values, a block's cells (64M) and bytes (256 MiB),
// and a stream's decompressed output. A block's data must decode to at
// least the size its layout implies. ReadWindow panics only on
// programming errors, as every RasterSource does.
//
// # Performance
//
// A Source decodes a block once and keeps it in a cache bounded by
// SourceOptions.CacheBytes. Windows that touch the same block, such as a
// tile and its neighbours' halos, share the decode. Concurrent reads of
// one block wait for a single decode.
package cog
