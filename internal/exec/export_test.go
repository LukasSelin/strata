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

// SetFusion turns register-level fusion on or off for the pipelines
// built while it is in effect, and returns a function that restores it.
// It exists so that a test can build the same pipeline both ways and
// compare them, which is how the fused form is checked against the
// staged one it must equal.
func SetFusion(on bool) (restore func()) {
	old := fusePipelines
	fusePipelines = on
	return func() { fusePipelines = old }
}

// Fused reports whether p lowered to a fused chain.
func (p *Pipeline) Fused() bool { return p.chain != nil }
