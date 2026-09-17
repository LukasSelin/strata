// Package rastertest holds raster transformations and comparisons for the
// module's metamorphic tests: tests that run an operation on related
// inputs and check that the results relate as the mathematics says, such
// as a slope that is unchanged when its DEM is mirrored.
package rastertest

import (
	"math"

	"strata/internal/fuzzdata"
	"strata/raster"
)

// Dihedral is one of the eight symmetries of the square grid, acting on a
// raster as t(r)(x, y) = r(u, v): with (x0, y0) = (x, y) mirrored
// horizontally by FlipX and vertically by FlipY in the result's
// coordinates, (u, v) is (y0, x0) if Swap is set (a transposition) and
// (x0, y0) otherwise.
type Dihedral struct{ Swap, FlipX, FlipY bool }

// All returns the eight symmetries, the identity first.
func All() []Dihedral {
	var out []Dihedral
	for i := range 8 {
		out = append(out, Dihedral{i&4 != 0, i&1 != 0, i&2 != 0})
	}
	return out
}

// Size returns the size of the result of t on a w×h raster.
func (t Dihedral) Size(w, h int) (tw, th int) {
	if t.Swap {
		return h, w
	}
	return w, h
}

// Source returns the cell of a w×h source that cell (x, y) of the result
// reads.
func (t Dihedral) Source(w, h, x, y int) (u, v int) {
	tw, th := t.Size(w, h)
	if t.FlipX {
		x = tw - 1 - x
	}
	if t.FlipY {
		y = th - 1 - y
	}
	if t.Swap {
		return y, x
	}
	return x, y
}

// Apply returns t(r) as a new compact raster, with a mask if r has one.
func (t Dihedral) Apply(r raster.Float32Raster) raster.Float32Raster {
	tw, th := t.Size(r.Width, r.Height)
	out := raster.NewFloat32(tw, th, make([]float32, tw*th))
	if r.Valid != nil {
		out.Valid = raster.NewMask(tw * th)
	}
	for y := range th {
		for x := range tw {
			u, v := t.Source(r.Width, r.Height, x, y)
			out.Data[y*tw+x] = r.Data[r.Index(u, v)]
			if out.Valid != nil {
				out.SetValid(x, y, r.IsValid(u, v))
			}
		}
	}
	return out
}

// Gradient maps a gradient (dx, dy) of a raster, derivatives along its
// column and row index, to the gradient of t(r) at the corresponding cell:
// the chain rule for a signed permutation of the axes.
func (t Dihedral) Gradient(dx, dy float64) (tdx, tdy float64) {
	if t.Swap {
		dx, dy = dy, dx
	}
	if t.FlipX {
		dx = -dx
	}
	if t.FlipY {
		dy = -dy
	}
	return dx, dy
}

// Compact returns a deep copy of r's cells as a compact raster, with a
// copy of its validity if it has a mask.
func Compact(r raster.Float32Raster) raster.Float32Raster {
	return Dihedral{}.Apply(r)
}

// Place returns a raster with r's cells and validity laid out as decoded
// from d: compact, with row padding, or a window at an offset of a larger
// root with a mask offset. Cells and mask bits of the root outside the
// result are arbitrary. The result has a mask iff r has one.
func Place(d fuzzdata.Source, r raster.Float32Raster) raster.Float32Raster {
	w, h := r.Width, r.Height
	rootW, rootH, x, y := w, h, 0, 0
	layout := d.IntN(3)
	if layout == 2 {
		x, y = d.IntN(4), d.IntN(3)
		rootW, rootH = w+x+d.IntN(4), h+y+d.IntN(3)
	}
	stride := rootW
	if layout > 0 {
		stride += d.IntN(70)
	}
	n := (rootH-1)*stride + rootW
	root := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
	for i := range root.Data {
		root.Data[i] = d.Float32()
	}
	if r.Valid != nil {
		off := d.IntN(130)
		root.Valid = make([]uint64, raster.MaskWords(off+n)+d.IntN(2))
		root.ValidOffset = off
		for k := range root.Valid {
			root.Valid[k] = d.Uint64()
		}
	}
	win := root.Window(x, y, w, h)
	for yy := range h {
		copy(win.Row(yy), r.Row(yy))
		if r.Valid != nil {
			for xx := range w {
				win.SetValid(xx, yy, r.IsValid(xx, yy))
			}
		}
	}
	return win
}

// Output returns a w×h raster to write results into, placed as by Place,
// with arbitrary stale Data and, if masked, arbitrary stale validity. With
// stale unset its cells are zero and invalid instead.
func Output(d fuzzdata.Source, w, h int, masked, stale bool) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	if masked {
		r.Valid = make([]uint64, raster.MaskWords(w*h))
	}
	if stale {
		for i := range r.Data {
			r.Data[i] = d.Float32()
		}
		for k := range r.Valid {
			r.Valid[k] = d.Uint64()
		}
	}
	return Place(d, r)
}

// ScrambleInvalid overwrites the Data of every invalid cell of r with a
// value from d. Data under an invalid cell is unspecified, so no valid
// result may depend on it.
func ScrambleInvalid(r raster.Float32Raster, d fuzzdata.Source) {
	if r.Valid == nil {
		return
	}
	for y := range r.Height {
		for x := range r.Width {
			if !r.IsValid(x, y) {
				r.Data[r.Index(x, y)] = d.Float32()
			}
		}
	}
}

// SameFloat reports whether a and b have the same bits, or are both NaN.
func SameFloat(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || a != a && b != b
}

// SameValue reports whether a == b (so +0 matches -0) or both are NaN.
func SameValue(a, b float32) bool { return a == b || a != a && b != b }

// Near reports whether cell (x, y) is within Chebyshev distance r of
// (px, py).
func Near(x, y, px, py, r int) bool {
	return abs(x-px) <= r && abs(y-py) <= r
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
