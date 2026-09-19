package exec

// SetBandCells sets the band size target and returns a function that
// restores it, so tests can force one-row bands.
func SetBandCells(n int) (restore func()) {
	old := bandCells
	bandCells = n
	return func() { bandCells = old }
}

// Bands returns the rectangles [x0, y0, x1, y1) of the bands Process plans
// for a w×h raster with the given tile size and kernel radius, in plan
// order.
func Bands(w, h, tileW, tileH, r int) [][4]int {
	p := newPlan(w, h, tiling{tileW: tileW, tileH: tileH, r: r})
	out := make([][4]int, p.bands)
	for i := range out {
		x0, y0, x1, y1 := p.band(i)
		out[i] = [4]int{x0, y0, x1, y1}
	}
	return out
}

// SetBandMinWidth sets the narrowest band the engine plans for itself and
// returns a function that restores it, so tests can force two-dimensional
// bands on small rasters, or force whole-row bands with a width no tile
// reaches.
func SetBandMinWidth(n int) (restore func()) {
	old := minBandWidth
	minBandWidth = n
	return func() { minBandWidth = old }
}

// BandShape is the shape the plan gives the bands of a tileW×tileH tile
// for a kernel of radius r.
func BandShape(tileW, tileH, r int) (bandW, bandH int) {
	return bandShape(tileW, tileH, tiling{r: r})
}
