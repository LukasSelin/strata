package engine

import (
	"strata/internal/stencil"
	"strata/raster"
)

// band writes output cells [x0, x1) × [y0, y1): the kernel over the cells
// whose neighbourhood lies inside the rasters, with the halo read from
// the inputs beyond the band, and the edge value over the rest.
func (e *exec) band(x0, y0, x1, y1 int) {
	r := e.r
	ix0, ix1 := max(x0, r), min(x1, e.w-r)
	iy0, iy1 := max(y0, r), min(y1, e.h-r)
	interior := ix0 < ix1 && iy0 < iy1
	if interior {
		e.interior(ix0, iy0, ix1-ix0, iy1-iy0)
	}
	if r == 0 {
		return // no edges
	}
	for y := y0; y < y1; y++ {
		if !interior || y < iy0 || y >= iy1 {
			e.fillEdge(y, x0, x1)
			continue
		}
		e.fillEdge(y, x0, ix0)
		e.fillEdge(y, ix1, x1)
	}
}

// interior runs the kernel over the w×h cells at (x, y), whose
// neighbourhoods are all inside the inputs, and sets their validity.
func (e *exec) interior(x, y, w, h int) {
	r := e.r
	for i, d := range e.dst {
		e.dstViews[i] = d.Window(x, y, w, h)
	}
	for i, s := range e.src {
		e.srcViews[i] = s.Window(x-r, y-r, w+2*r, h+2*r)
	}
	e.k.Process(Span{X: x, Y: y, Width: w, Height: h, Dst: e.dstViews}, Window{Radius: r, Src: e.srcViews})

	switch {
	case !e.dstMasked:
		// Nothing to record; the checks ensured no input has a mask.
	case len(e.masked) == 0:
		for _, d := range e.dstViews {
			if d.Valid != nil {
				fillValid(d)
			}
		}
	case r == 0:
		e.pointwiseValidity()
	default:
		e.erodedValidity(w, h)
	}
}

// fillEdge writes the edge value into cells [x0, x1) of row y of every
// output and marks them invalid.
func (e *exec) fillEdge(y, x0, x1 int) {
	if x0 >= x1 {
		return
	}
	for _, d := range e.dst {
		start := y*d.Stride + x0
		row := d.Data[start : start+x1-x0]
		for i := range row {
			row[i] = e.edge
		}
		switch {
		case d.Valid == nil:
		case len(row) <= 8:
			// The left and right edges of every interior row: a few bits,
			// cheaper one at a time than through a range call.
			for b := d.ValidOffset + start; b < d.ValidOffset+start+len(row); b++ {
				d.Valid[b>>6] &^= 1 << uint(b&63)
			}
		default:
			raster.MaskFillRange(d.Valid, d.ValidOffset+start, len(row), false)
		}
	}
}

// erodedValidity sets each output view's validity to the AND of every
// masked input over the (2r+1)×(2r+1) neighbourhood. It erodes once into
// the first output and copies the bits to the others. Every output has a
// mask here, because an input does.
func (e *exec) erodedValidity(w, h int) {
	for i, j := range e.masked {
		s := e.srcViews[j]
		e.regions[i] = stencil.MaskRegion{Bits: s.Valid, Off: s.ValidOffset, Stride: s.Stride}
	}
	first := e.dstViews[0]
	stencil.ErodeBox(stencil.MaskRegion{Bits: first.Valid, Off: first.ValidOffset, Stride: first.Stride},
		e.regions, w, h, e.r, e.scratch)
	for _, d := range e.dstViews[1:] {
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
func (e *exec) pointwiseValidity() {
	for i, d := range e.dstViews {
		whole := compact(d)
		for _, j := range e.masked {
			whole = whole && compact(e.srcViews[j])
		}
		same, rest := e.sameBits[i], e.masked
		if same < 0 {
			a := e.srcViews[rest[0]]
			if len(rest) == 1 {
				copyBits(d, a, whole)
				continue
			}
			andBits(d, a, e.srcViews[rest[1]], whole)
			rest = rest[2:]
		}
		for _, j := range rest {
			if j != same {
				andBits(d, d, e.srcViews[j], whole)
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
