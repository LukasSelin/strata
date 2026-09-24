package exec

import (
	"math"

	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// band writes band i: its output cells [x0, x1) × [y0, y1), the kernel
// over the cells whose neighbourhood lies inside the rasters, with the
// halo read from the inputs beyond the band, and the edge value over the
// rest. It writes Data first and then, holding the mask lock, validity,
// so Data work runs concurrently across workers. Coordinates here are
// the raster's; the operands start at (dx, dy) and (sx, sy).
//
// When the outputs' edge rings differ (edgeW), the kernel runs over the
// cells outside the narrowest ring, with padded windows where they leave
// the rasters, and each output's own ring then gets the edge policy,
// overwriting whatever the kernel computed there from the padding.
func (e *job) band(wk *worker, i int) {
	x0, y0, x1, y1 := e.plan.band(i)
	x0, y0, x1, y1 = x0+e.dx, y0+e.dy, x1+e.dx, y1+e.dy
	lo := e.rmin
	ix0, ix1 := max(x0, lo), min(x1, e.w-lo)
	iy0, iy1 := max(y0, lo), min(y1, e.h-lo)
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
	if e.r > 0 {
		e.edges(x0, y0, x1, y1, false)
	}
	if !e.dstMasked {
		return
	}
	e.maskLock()
	defer e.maskUnlock()
	if interior {
		e.interiorValidity(wk, ix1-ix0, iy1-iy0)
	}
	if e.r > 0 {
		e.edges(x0, y0, x1, y1, true)
	}
}

// edges writes the cells of the band [x0, x1) × [y0, y1) that lie in
// each output's edge ring: their Data, or with valid set their validity.
func (e *job) edges(x0, y0, x1, y1 int, valid bool) {
	for o := range e.dst {
		b := e.r
		if e.edgeW != nil {
			b = e.edgeW[o]
		}
		if b == 0 {
			continue
		}
		ix0, ix1 := max(x0, b), min(x1, e.w-b)
		iy0, iy1 := max(y0, b), min(y1, e.h-b)
		interior := ix0 < ix1 && iy0 < iy1
		for y := y0; y < y1; y++ {
			if !interior || y < iy0 || y >= iy1 {
				e.fillEdge(o, y, x0, x1, valid)
				continue
			}
			e.fillEdge(o, y, x0, ix0, valid)
			e.fillEdge(o, y, ix1, x1, valid)
		}
	}
}

// fillEdge writes cells [x0, x1) of row y of output o as edge cells:
// their Data, or with valid set their validity.
func (e *job) fillEdge(o, y, x0, x1 int, valid bool) {
	if valid {
		e.clearEdgeValid(o, y, x0, x1)
	} else {
		e.fillEdgeData(o, y, x0, x1)
	}
}

// interior runs the kernel over the w×h cells at (x, y), whose
// neighbourhoods are all inside the inputs unless the outputs' edge
// rings differ, leaving the worker's views set for interiorValidity.
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
	if x-r < 0 || y-r < 0 || x+w+r > e.w || y+h+r > e.h {
		e.padWindows(wk, x, y, w, h)
	} else {
		for i, s := range e.src {
			wk.srcViews[i] = s.Window(x-r-e.sx, y-r-e.sy, w+2*r, h+2*r)
		}
	}
	e.k.Process(
		Span{X: x, Y: y, Width: w, Height: h, Dst: wk.dstViews, Scratch: &wk.kscratch},
		Window{Radius: r, Src: wk.srcViews})
}

// padWindows sets the worker's input views to the w×h span at (x, y)
// grown by the radius, for a span whose window leaves the rasters. Only
// a kernel whose outputs' edge rings differ is called for such a span.
// Each window is copied into the worker's pad buffer: cells inside the
// rasters with their Data and validity, cells outside NaN and invalid.
// No output cell outside its own ring reads a padded cell, by the
// definition of its ring, so padding changes no value that is kept.
func (e *job) padWindows(wk *worker, x, y, w, h int) {
	r := e.r
	ww, wh := w+2*r, h+2*r
	// The part of the window inside the rasters, in window coordinates.
	cx0, cy0 := max(0, r-x), max(0, r-y)
	cx1, cy1 := min(ww, e.w-(x-r)), min(wh, e.h-(y-r))
	nan := float32(math.NaN())
	n := ww * wh
	for i, s := range e.src {
		p := wk.pad[i]
		p.Width, p.Height, p.Stride = ww, wh, ww
		p.Data = p.Data[:n:n]
		for k := range p.Data {
			p.Data[k] = nan
		}
		if s.Valid != nil {
			raster.MaskFillRange(p.Valid, 0, n, false)
		} else {
			p.Valid = nil
		}
		for wy := cy0; wy < cy1; wy++ {
			sy := y - r + wy - e.sy
			sx := x - r + cx0 - e.sx
			copy(p.Data[wy*ww+cx0:wy*ww+cx1], s.Data[sy*s.Stride+sx:])
			if s.Valid != nil {
				raster.MaskCopyRange(p.Valid, wy*ww+cx0, s.Valid, s.ValidOffset+sy*s.Stride+sx, cx1-cx0)
			}
		}
		wk.srcViews[i] = p
	}
}

// interiorValidity sets the validity of the interior interior just wrote.
// Some output has a mask.
func (e *job) interiorValidity(wk *worker, w, h int) {
	for i, d := range wk.dstViews {
		v := &e.valid[i]
		switch {
		case d.Valid == nil:
		case v.from >= 0:
			s := wk.dstViews[v.from]
			copyBits(d, s, compact(d) && compact(s))
		case len(v.ins) == 0:
			fillValid(d)
		case e.r == 0:
			e.pointwiseValidity(wk, i)
		default:
			e.erodedValidity(wk, i, w, h)
		}
	}
}

// fillEdgeData writes the edge value into cells [x0, x1) of row y of
// output o.
func (e *job) fillEdgeData(o, y, x0, x1 int) {
	if x0 >= x1 {
		return
	}
	d := e.dst[o]
	start := (y-e.dy)*d.Stride + x0 - e.dx
	row := d.Data[start : start+x1-x0]
	for i := range row {
		row[i] = e.edge
	}
}

// clearEdgeValid marks cells [x0, x1) of row y of output o invalid, if
// it has a mask.
func (e *job) clearEdgeValid(o, y, x0, x1 int) {
	n := x1 - x0
	d := e.dst[o]
	if n <= 0 || d.Valid == nil {
		return
	}
	start := d.ValidOffset + (y-e.dy)*d.Stride + x0 - e.dx
	if n <= 8 {
		// The left and right edges of every interior row: a few bits,
		// cheaper one at a time than through a range call.
		for b := start; b < start+n; b++ {
			d.Valid[b>>6] &^= 1 << uint(b&63)
		}
		return
	}
	raster.MaskFillRange(d.Valid, start, n, false)
}

// erodedValidity sets output i's validity to the AND, over every masked
// input it reads, of that input's validity over the neighbourhood of its
// reach: the (2r+1)×(2r+1) box for a kernel that reads every input over
// its radius. An input of reach less than the radius is read from the
// middle of its window, which is grown by the full radius.
func (e *job) erodedValidity(wk *worker, i, w, h int) {
	v := &e.valid[i]
	regions := wk.regions[:len(v.ins)]
	for k, j := range v.ins {
		s := wk.srcViews[j]
		off := e.r - v.reach[k]
		regions[k] = stencil.MaskRegion{Bits: s.Valid, Off: s.ValidOffset + off*s.Stride + off, Stride: s.Stride}
	}
	d := wk.dstViews[i]
	stencil.ErodeReach(stencil.MaskRegion{Bits: d.Valid, Off: d.ValidOffset, Stride: d.Stride},
		regions, v.reach, w, h, wk.scratch)
}

// pointwiseValidity sets output i's validity to the AND of the masked
// inputs' bits it reads, for the same cells, with the word loops of
// package algebra: one range operation over the whole view when every
// operand is compact, one per row otherwise.
func (e *job) pointwiseValidity(wk *worker, i int) {
	d := wk.dstViews[i]
	ins := e.valid[i].ins
	whole := compact(d)
	for _, j := range ins {
		whole = whole && compact(wk.srcViews[j])
	}
	same, rest := e.sameBits[i], ins
	if same < 0 {
		a := wk.srcViews[rest[0]]
		if len(rest) == 1 {
			copyBits(d, a, whole)
			return
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
