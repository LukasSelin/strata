// Package zarr reads arrays of Zarr version 3 stores as strata raster
// sources (DESIGN.md §34, docs/adr/0003-zarr-adapter.md). It wraps
// github.com/LukasSelin/zarr, which reads the store, its metadata and
// its codecs, and turns an array's elements and fill value into float32
// cells and a validity mask at the boundary, so the engine never sees the
// format.
//
// The library's package is also called zarr, so a program that uses both
// imports one under another name; this package imports the library as
// zarrv3, and the examples do the same:
//
//	import zarrv3 "github.com/LukasSelin/zarr"
//
//	store := zarrv3.NewDirStore("dem.zarr")
//	src, err := zarr.Open(ctx, store, "elevation", zarr.SourceOptions{})
//	...
//	grid := src.Grid() // from spatial:transform and proj:code, if present
//	err = terrain.SlopeChunked(ctx, sink, src,
//		terrain.SlopeOptions{CellSize: grid.ResolutionX}, engine.Options{})
//
// A Source is an engine.RasterSource, so every Chunked entry point runs
// over it in memory bounded by the tile size and the cache, not the array.
//
// # What it reads
//
//   - Arrays of two or more dimensions. The last two are the raster's rows
//     and columns, y then x; SourceOptions.Index fixes the others, such as
//     a time step and a band of a (time, band, y, x) array.
//   - The data types int8, int16, int32, int64, uint8, uint16, uint32,
//     uint64, float32 and float64. bool is refused.
//   - Whatever the library reads: regular chunk grids, both chunk key
//     encodings, and the codecs bytes, gzip, crc32c and sharding_indexed,
//     plus any registered with zarrv3.RegisterCodec, such as the zstd of
//     github.com/LukasSelin/zarr/zstd.
//
// # Values and validity
//
// Elements convert to float32 as float32(v) does under IEEE 754: rounded
// to nearest, and ±Inf past float32's range. Integers wider than 24 bits
// and float64s lose precision, as they do in cog.
//
// Zarr gives every array a fill_value, which is what a chunk nobody wrote
// reads as. The source takes it as NoData (DESIGN.md §31, rule 5): a cell
// is valid unless its element equals the fill value, compared in the
// array's own type before the conversion, so an int64 or float64 fill that
// float32 cannot hold still matches only itself (benchmarks/nodata). A NaN
// fill matches every NaN, whatever its payload; float comparisons are
// IEEE's, so a fill of 0 matches -0 too. SourceOptions.IgnoreFill turns
// this off, for an array whose fill value is an ordinary value, such as a
// count whose fill is 0; the source is then not Masked.
//
// # Georeferencing
//
// Zarr has no georeferencing of its own. The source reads two attributes of
// the array, from the GeoZarr conventions, and both are optional:
//
//   - "spatial:transform", six numbers [a, b, c, d, e, f], the affine
//     transform in rasterio's order: a cell's corner (col, row) is at
//     x = a·col + b·row + c, y = d·col + e·row + f. b and d must be 0, since
//     raster.Grid cannot hold a rotation. "spatial:registration", if set,
//     is "pixel" (the transform places cell corners, the default) or
//     "node" (it places cell centres).
//   - "proj:code", an authority code such as "EPSG:32633", which becomes
//     the grid's CRS as it is.
//
// Without a transform, Grid is the grid of the array's own cells: origin
// (0, 0), resolution (1, 1), as cog reports an ungeoreferenced GeoTIFF.
//
// # The chunk cache
//
// Zarr decodes a chunk whole, however little of it a window needs, and
// the engine reads tiles with halos, so neighbouring tiles ask for the
// same chunks again and again. A Source keeps decoded chunks, as float32
// cells and validity, in a least-recently-used cache bounded by
// SourceOptions.CacheBytes, and a chunk asked for while another reader is
// decoding it waits for that decode rather than starting its own. A chunk
// that failed to load is not kept, so the next read tries again. A window
// that needs several chunks loads them concurrently, up to
// SourceOptions.ReadConcurrency at a time. CacheStats reports hits and
// loads.
//
// Writing is not supported yet.
package zarr
