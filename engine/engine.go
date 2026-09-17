package engine

// Options configures how a tiled operation divides and schedules its
// work. The zero value runs the whole raster as one tile on
// runtime.GOMAXPROCS(0) workers, which is the fastest setting for rasters
// in memory. Options never change the result.
type Options struct {
	// TileWidth and TileHeight are the tile size in cells. 0 means the
	// raster's width or height. Tiles are planned in row-major order and
	// split into bands of rows; see the package documentation for why
	// narrow tiles are slower.
	TileWidth  int
	TileHeight int
	// Workers is the number of goroutines that run bands, the calling
	// goroutine included. 0 means runtime.GOMAXPROCS(0); 1 starts no
	// goroutines. A call never uses more workers than it has bands.
	Workers int
}
