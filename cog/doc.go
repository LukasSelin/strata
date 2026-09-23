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
// per Source, not once per window. A Source reads one band, and the
// sources of a pixel-interleaved file share its compressed blocks, up to
// 64 MiB, so a block every band needs is fetched once, not once per band.
// Requests must be
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
//   - No compression, LZW (libtiff's old style too), Deflate, PackBits
//     and ZSTD, with no predictor, horizontal differencing (2) or the
//     floating-point predictor (3), in either FillOrder.
//   - Overviews: every reduced-resolution image (NewSubfileType bit 0)
//     with the same bands, as levels 1, 2, ..., in GDAL's order: the
//     first image's SubIFDs, then the IFD chain. In a COG that is largest
//     first.
//   - Transparency masks (NewSubfileType bit 2) of 1- or 8-bit samples,
//     GDAL's internal masks: a level's mask gives its validity, in place
//     of NoData.
//   - Georeferencing: ModelPixelScale with the first ModelTiepoint, or
//     else ModelTransformation, with GDAL's half-cell shift for
//     PixelIsPoint and its north-up reading of a negative Y scale; and the
//     EPSG code of a projected or geographic CRS (see CRS below).
//   - GDAL's NoData value (the GDAL_NODATA tag), parsed as GDAL parses it.
//
// The block tags are read as libtiff reads them: tiles listed under the
// strip tags, lists longer or shorter than the block count (cut, or
// padded with absent blocks), uncompressed data without byte counts, a
// last row of tiles cut short at the image's edge, and an IFD chain that
// loops or breaks after the first image.
//
// Anything else is refused by Open with an error that says it is not
// supported and names it: JPEG, WebP, LERC, LZMA, JPEG XL, CCITT and the
// other compressions; other bit depths, including 16-bit floats and
// sub-byte samples; complex samples; colour that GDAL converts rather
// than reads as stored (YCbCr, and CMYK, CIELab and LogLuv at 8 bits);
// GDAL's NODATA_VALUES metadata, a NoData value across all bands; and
// rotated or sheared geotransforms, which strata's Grid cannot hold. An
// alpha band is read as a band, not as validity, and .msk and .ovr
// files, GDAL's .aux.xml, and scale, offset and other metadata are not
// read.
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
// A level with a transparency mask is Masked, and a cell is invalid
// where the mask is 0; NoData is then ignored, as GDAL ignores it.
// Otherwise, with a NoData value that the sample type can hold, the
// source is Masked, and cells compare with NoData as GDAL's NoData mask
// compares them (DESIGN.md §31):
//
//   - An integer NoData outside the type's range is no NoData at all, and
//     one inside it is truncated toward zero: NoData 12.5 on bytes marks
//     the cells holding 12.
//   - Floats match within GDAL's ARE_REAL_EQUAL tolerance,
//     2·FLT_EPSILON·|cell+NoData|, in float32 arithmetic for float32
//     samples: about four float32 ulps either side, and the same relative
//     width for float64. Near ±MaxFloat32 the float32 sum overflows, and
//     every cell whose sum with NoData overflows matches.
//   - A float32 file's NoData is rounded to float32 first, a value within
//     1e-10 of ±MaxFloat32 onto it and one beyond float32's range to ±Inf.
//   - A NaN NoData matches every NaN.
//
// The comparison is made on the file's sample, before the conversion to
// float32. Blocks that are absent from the file (a byte count of 0) read
// as NoData, or as 0 without it, as GDAL reads them.
//
// # CRS
//
// Grid.CRS is the EPSG code the GeoKeys name, as the code of a projected
// (ProjectedCSTypeGeoKey) or geographic (GeographicTypeGeoKey) model,
// when nothing else in the keys redefines it. It is empty for a
// user-defined or geocentric model, for a projected CRS whose keys also
// give a projection, geographic CRS, datum or ellipsoid, and for units
// other than the code's own in the unit keys or in an ERDAS IMAGINE or
// GDAL citation. GDAL identifies some of those as an EPSG code through
// PROJ's database, which cog does not carry; it reports none rather than
// risk a wrong one. A code EPSG has deprecated in favour of one other is
// reported as that replacement, as GDAL reports it. Only the horizontal
// CRS is named: vertical keys are ignored. The units and deprecations
// come from the EPSG registry, in epsg_table.go (see epsg_gen.py).
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
