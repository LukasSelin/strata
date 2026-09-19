package exec

import (
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// band writes band i: its output cells [x0, x1) × [y0, y1), the kernel
// over the cells whose neighbourhood lies inside the rasters, with the
// halo read from the inputs beyond the band, and the edge value over the
// rest. It writes Data first and then, holding the mask lock, validity,
// so Data work runs concurrently across workers. Coordinates here are
// the raster's; the operands start at (dx, dy) and (sx, sy).
func (e *job) band(wk *worker, i int) {
	x0, y0, x1, y1 := e.plan.band(i)
	x0, y0, x1, y1 = x0+e.dx, y0+e.dy, x1+e.dx, y1+e.dy
	r := e.r
	ix0, ix1 := max(x0, r), min(x1, e.w-r)
	iy0, iy1 := max(y0, r), min(y1, e.h-r)
	interior := ix0 < ix1 && iy0 < iy1
	// The band's output cells, counted once however many outputs there
	// are, and the Data the edge policy writes into the cells outside the
	// interior. interior counts the cells the kernel itself writes, so the
	// two together are exactly Cells × 4 × outputs.
	cells := int64(x1-x0) * int64(y1-y0)
	inner := int64(0)
	if interior {
		inner = int64(ix1-ix0) * int64(iy1-iy0)
	}
	wk.stats.Cells += cells
	wk.stats.Bands++
	wk.stats.KernelWritten += (cells - inner) * bytesPerCell * int64(len(e.dst))
	if interior {
		e.interior(wk, ix0, iy0, ix1-ix0, iy1-iy0)
	}
	if r > 0 {
		e.edges(y0, y1, x0, x1, ix0, ix1, iy0, iy1, interior, false)
	}
	if !e.dstMasked {
		return
	}
	e.maskLock()
	defer e.maskUnlock()
	if interior {
		e.interiorValidity(wk, ix1-ix0, iy1-iy0)
	}
	if r > 0 {
		e.edges(y0, y1, x0, x1, ix0, ix1, iy0, iy1, interior, true)
	}
}

// edges writes the edge cells of each row of a band, the whole rows above
// and below the interior and the cells left and right of it: their Data,
// or with valid set their validity.
func (e *job) edges(y0, y1, x0, x1, ix0, ix1, iy0, iy1 int, interior, valid bool) {
	fill := func(y, x0, x1 int) {
		if valid {
			e.clearEdgeValid(y, x0, x1)
		} else {
			e.fillEdgeData(y, x0, x1)
		}
	}
	for y := y0; y < y1; y++ {
		if !interior || y < iy0 || y >= iy1 {
			fill(y, x0, x1)
			continue
		}
		fill(y, x0, ix0)
		fill(y, ix1, x1)
	}
}

// interior runs the kernel over the w×h cells at (x, y), whose
// neighbourhoods are all inside the inputs, leaving the worker's views
// set for interiorValidity.
func (e *job) interior(wk *worker, x, y, w, h int) {
	r := e.r
	// The window is the span grown by the radius, so its cells beyond the
	// span are the halo: read here and read again by whichever band or
	// tile owns them. Counting the window, not the span, is what makes
	// Stats.Halo the cost of the tiling rather than an estimate of it.
	wk.stats.KernelRead += int64(w+2*r) * int64(h+2*r) * bytesPerCell * int64(len(e.src))
	wk.stats.KernelWritten += int64(w) * int64(h) * bytesPerCell * int64(len(e.dst))
	for i, d := range e.dst {
		wk.dstViews[i] = d.Window(x-e.dx, y-e.dy, w, h)
	}
	for i, s := range e.src {
		wk.srcViews[i] = s.Window(x-r-e.sx, y-r-e.sy, w+2*r, h+2*r)
	}
	e.k.Process(
		Span{X: x, Y: y, Width: w, Height: h, Dst: wk.dstViews, Scratch: &wk.kscratch},
		Window{Radius: r, Src: wk.srcViews})
}

// interiorValidity sets the validity of the interior interior just wrote.
// Some output has a mask.
func (e *job) interiorValidity(wk *worker, w, h int) {
	switch {
	case len(e.masked) == 0:
		for _, d := range wk.dstViews {
			if d.Valid != nil {
				fillValid(d)
			}
		}
	case e.r == 0:
		e.pointwiseValidity(wk)
	default:
		e.erodedValidity(wk, w, h)
	}
}

// fillEdgeData writes the edge value into cells [x0, x1) of row y of
// every output.
func (e *job) fillEdgeData(y, x0, x1 int) {
	if x0 >= x1 {
		return
	}
	for _, d := range e.dst {
		start := (y-e.dy)*d.Stride + x0 - e.dx
		row := d.Data[start : start+x1-x0]
		for i := range row {
			row[i] = e.edge
		}
	}
}

// clearEdgeValid marks cells [x0, x1) of row y of every output with a
// mask invalid.
func (e *job) clearEdgeValid(y, x0, x1 int) {
	n := x1 - x0
	if n <= 0 {
		return
	}
	for _, d := range e.dst {
		start := d.ValidOffset + (y-e.dy)*d.Stride + x0 - e.dx
		switch {
		case d.Valid == nil:
		case n <= 8:
			// The left and right edges of every interior row: a few bits,
			// cheaper one at a time than through a range call.
			for b := start; b < start+n; b++ {
				d.Valid[b>>6] &^= 1 << uint(b&63)
			}
		default:
			raster.MaskFillRange(d.Valid, start, n, false)
		}
	}
}

// erodedValidity sets each output view's validity to the AND of every
// masked input over the (2r+1)×(2r+1) neighbourhood. It erodes once into
// the first output and copies the bits to the others. Every output has a
// mask here, because an input does.
func (e *job) erodedValidity(wk *worker, w, h int) {
	for i, j := range e.masked {
		s := wk.srcViews[j]
		wk.regions[i] = stencil.MaskRegion{Bits: s.Valid, Off: s.ValidOffset, Stride: s.Stride}
	}
	first := wk.dstViews[0]
	stencil.ErodeBox(stencil.MaskRegion{Bits: first.Valid, Off: first.ValidOffset, Stride: first.Stride},
		wk.regions, w, h, e.r, wk.scratch)
	for _, d := range wk.dstViews[1:] {
		for y := range h {
			raster.MaskCopyRange(d.Valid, d.ValidOffset+y*d.Stride,
				first.Valid, first.ValidOffset+y*first.Stride, w)
		}
	}
}

// pointwiseValidity sets each output view's validity to the AND of the
// masked inputs' bits for the same cells, with the word loops of package
// algebra: one range operation over the whole view when every operand is
// compact, one per row otherwise.
func (e *job) pointwiseValidity(wk *worker) {
	for i, d := range wk.dstViews {
		whole := compact(d)
		for _, j := range e.masked {
			whole = whole && compact(wk.srcViews[j])
		}
		same, rest := e.sameBits[i], e.masked
		if same < 0 {
			a := wk.srcViews[rest[0]]
			if len(rest) == 1 {
				copyBits(d, a, whole)
				continue
			}
			andBits(d, a, wk.srcViews[rest[1]], whole)
			rest = rest[2:]
		}
		for _, j := range rest {
			if j != same {
				andBits(d, d, wk.srcViews[j], whole)
			}
		}
	}
}

func compact(r raster.Float32Raster) bool { return r.Stride == r.Width }

func fillValid(d raster.Float32Raster) {
	if compact(d) {
		raster.MaskFillRange(d.Valid, d.ValidOffset, d.Width*d.Height, true)
		return
	}
	for y := range d.Height {
		raster.MaskFillRange(d.Valid, d.ValidOffset+y*d.Stride, d.Width, true)
	}
}

func copyBits(d, s raster.Float32Raster, whole bool) {
	if whole {
		raster.MaskCopyRange(d.Valid, d.ValidOffset, s.Valid, s.ValidOffset, d.Width*d.Height)
		return
	}
	for y := range d.Height {
		raster.MaskCopyRange(d.Valid, d.ValidOffset+y*d.Stride, s.Valid, s.ValidOffset+y*s.Stride, d.Width)
	}
}

func andBits(d, a, b raster.Float32Raster, whole bool) {
	if whole {
		raster.MaskAndRange(d.Valid, d.ValidOffset, a.Valid, a.ValidOffset, b.Valid, b.ValidOffset, d.Width*d.Height)
		return
	}
	for y := range d.Height {
		raster.MaskAndRange(d.Valid, d.ValidOffset+y*d.Stride,
			a.Valid, a.ValidOffset+y*a.Stride, b.Valid, b.ValidOffset+y*b.Stride, d.Width)
	}
}
