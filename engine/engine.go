package engine

// Options configures how a tiled operation divides and schedules its
// work. The zero value runs the whole raster as one tile on
// runtime.GOMAXPROCS(0) workers, which is the fastest setting for rasters
// in memory. Options never change the result.
type Options struct {
	// TileWidth and TileHeight are the tile size in cells: the IO tile,
	// the region a Chunked call reads from a source and writes to a sink
	// in one call. 0 means the raster's width or height. Tiles are planned
	// in row-major order; see the package documentation for why narrow
	// tiles are slower, and why full-width strips are usually right.
	TileWidth  int
	TileHeight int
	// ComputeWidth and ComputeHeight are the size in cells of the band
	// one kernel call covers inside a tile: the compute tile. 0, the
	// engine's choice, is whole tile rows, and is what to use unless you
	// are measuring. The two are separate from TileWidth and TileHeight
	// because the two want opposite shapes — a source is fastest reading
	// full-width strips, where a neighbourhood kernel reads the least
	// halo over a square — and separating them is what lets one call have
	// both. Measuring that is what set the default: shaping the band does
	// cut the halo, and does not make the call faster (DESIGN.md §53).
	// A 0 ComputeHeight with a ComputeWidth set is the number of rows
	// that fills a band.
	ComputeWidth  int
	ComputeHeight int
	// Workers is the number of goroutines that run bands, the calling
	// goroutine included. 0 means runtime.GOMAXPROCS(0); 1 starts no
	// goroutines. A call never uses more workers than it has bands.
	Workers int
	// Stats, when not nil, receives how many bytes the call moved at each
	// stage of its pipeline. It is an out-parameter, not a setting: it
	// changes neither the result nor the work, and the counters are kept
	// whether or not one is given, so a call measures the same code a call
	// without it runs. A call adds to it after every worker has stopped;
	// see Stats.
	Stats *Stats
}
