package exec

// SetBandCells sets the band size target and returns a function that
// restores it, so tests can force one-row bands.
func SetBandCells(n int) (restore func()) {
	old := bandCells
	bandCells = n
	return func() { bandCells = old }
}

// Bands returns the rectangles [x0, y0, x1, y1) of the bands Process plans
// for a w×h raster with the given tile size, in plan order.
func Bands(w, h, tileW, tileH int) [][4]int {
	p := newPlan(w, h, tileW, tileH)
	out := make([][4]int, p.bands)
	for i := range out {
		x0, y0, x1, y1 := p.band(i)
		out[i] = [4]int{x0, y0, x1, y1}
	}
	return out
}

// SetPoisonScratch sets whether scratch is filled with garbage every time
// it is lent, and returns a function that restores the old setting.
func SetPoisonScratch(on bool) (restore func()) {
	old := poisonScratch
	poisonScratch = on
	return func() { poisonScratch = old }
}
