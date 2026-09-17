package engine

// Options configures how a tiled operation divides and schedules its
// work. The zero value runs the whole raster as one tile. Options never
// change the result.
type Options struct {
	// TileWidth and TileHeight are the tile size in cells. 0 means the
	// raster's width or height. Tiles are processed in row-major order.
	TileWidth  int
	TileHeight int
	// Workers is the number of goroutines that process tiles. 0 lets the
	// engine choose. This version runs every tile on the calling
	// goroutine whatever the value, so callers can already pass the
	// setting STRATA-9's worker pool will use.
	Workers int
}
