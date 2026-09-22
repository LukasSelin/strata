package focal_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
)

// The metamorphic tests run an operation on related inputs and check
// that the results relate as the mathematics says. Every operation folds
// its terms in a fixed order, so these relations hold bit for bit:
//
//   - A window of the input gives the full result on the window's
//     interior (crop, which is also translation).
//   - Changing one input cell changes only outputs within r of it, and
//     invalidating one cell invalidates exactly the valid outputs within
//     r of it (locality).
//   - Data under invalid cells changes nothing.
//   - Scaling the input or the weights by a power of two scales the
//     result exactly, away from overflow and subnormals.
//   - Negating the input negates Correlate, CorrelateSeparable and Mean
//     (up to the sign of a zero sum), and Min(-x) is -Max(x) exactly.
//     Min(x+c) is Min(x)+c, since rounding is monotone.
//   - Min and Max are invariant under the eight symmetries of the grid.
//   - On exact data (small integers, integer weights, every partial sum
//     below 2²⁴) the order of a sum does not matter, so every operation
//     is invariant under the symmetries with its weights mapped too,
//     CorrelateSeparable is Correlate with the outer product of its taps,
//     and Mean is Correlate with unit weights, divided by (2r+1)². The
//     separable form holds these only up to the sign of a zero: it
//     multiplies a row tap by a column sum, so a negative row tap times a
//     +0 sum is -0 where a transposed or outer-product factorisation of
//     the same zero term gives +0.
//
// On other data, symmetries change the order of the sums and hold only
// to rounding; they are not checked.

// moderateWeights are finite weights of a few significant bits, whose
// products with moderate data are never subnormal.
const moderateWeights weightValues = 2

// relSpec draws an operation with weights of the given kind.
func relSpec(d fuzzdata.Source, kind, r int, wv weightValues) spec {
	if wv != moderateWeights {
		return newSpec(d, kind, r, wv)
	}
	s := newSpec(d, kind, r, intWeights)
	for _, v := range [][]float32{s.w, s.row, s.col} {
		for i := range v {
			v[i] = float32(d.Range(-1000, 1000)) / 64
		}
	}
	return s
}

// focalRelations decodes a relation, an operation and an input from d
// and checks that relation. Each side of it runs through its own
// execution path (plain, Tiled or Chunked, with its own tiles and
// workers) over its own memory layout, so a relation also catches a halo
// or tile that reads the wrong cells.
func focalRelations(t rastertest.TB, d fuzzdata.Source) {
	rel, kind, r := d.IntN(9), d.IntN(numKinds), d.Range(1, 3)
	w, h := d.Range(1, 30), d.Range(1, 2*r+6)
	masked := d.IntN(3) != 0
	id := fmt.Sprintf("relation %d %s r=%d %d×%d masked %v", rel, kindNames[kind], r, w, h, masked)
	run := func(s spec, src raster.Float32Raster) raster.Float32Raster {
		out := rastertest.Output(d, src.Width, src.Height, src.Valid != nil, true)
		eopts := engine.Options{TileWidth: d.Range(0, src.Width+2), TileHeight: d.Range(0, src.Height+2), Workers: d.Range(0, 3)}
		if err := s.run(d.IntN(3), eopts, out, rastertest.Place(d, src)); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		return rastertest.Compact(out)
	}

	switch rel {
	case 0: // crop
		s := relSpec(d, kind, r, floatWeights)
		src := rastertest.Compact(newSrc(d, w, h, anyValues, masked))
		full := run(s, src)
		x0, y0 := d.IntN(w), d.IntN(h)
		cw, ch := d.Range(1, w-x0), d.Range(1, h-y0)
		part := run(s, rastertest.Compact(src.Window(x0, y0, cw, ch)))
		for y := r; y < ch-r; y++ {
			for x := r; x < cw-r; x++ {
				requireCell(t, fmt.Sprintf("%s %v: window (%d, %d) %d×%d", id, s, x0, y0, cw, ch), part, x, y, full, x+x0, y+y0, ident, false)
			}
		}
	case 1: // one changed cell
		s := relSpec(d, kind, r, floatWeights)
		src := rastertest.Compact(newSrc(d, w, h, anyValues, masked))
		px, py := d.IntN(w), d.IntN(h)
		changed := rastertest.Compact(src)
		changed.Data[changed.Index(px, py)] = d.Float32()
		a, b := run(s, src), run(s, changed)
		for y := range h {
			for x := range w {
				if !rastertest.Near(x, y, px, py, r) {
					requireCell(t, fmt.Sprintf("%s %v: changed (%d, %d)", id, s, px, py), b, x, y, a, x, y, ident, false)
				}
			}
		}
	case 2: // one more invalid cell
		s := relSpec(d, kind, r, floatWeights)
		src := rastertest.Compact(newSrc(d, w, h, anyValues, true))
		px, py := d.IntN(w), d.IntN(h)
		less := rastertest.Compact(src)
		less.SetValid(px, py, false)
		a, b := run(s, src), run(s, less)
		for y := range h {
			for x := range w {
				want := a.IsValid(x, y) && !rastertest.Near(x, y, px, py, r)
				if b.IsValid(x, y) != want {
					t.Fatalf("%s %v: invalidated (%d, %d): (%d, %d) valid = %v, want %v", id, s, px, py, x, y, b.IsValid(x, y), want)
				}
				if want {
					requireCell(t, id, b, x, y, a, x, y, ident, false)
				}
			}
		}
	case 3: // data under invalid cells
		s := relSpec(d, kind, r, floatWeights)
		src := rastertest.Compact(newSrc(d, w, h, anyValues, true))
		scrambled := rastertest.Compact(src)
		rastertest.ScrambleInvalid(scrambled, d)
		requireAll(t, id+" scrambled invalid", s, run(s, scrambled), run(s, src), ident, false)
	case 4: // power-of-two scaling of the input or the weights
		s := relSpec(d, kind, r, moderateWeights)
		src := rastertest.Compact(newSrc(d, w, h, smoothValues, masked))
		k := d.Range(-6, 6)
		f := float32(math.Ldexp(1, k))
		scale := func(v float32) float32 { return v * f }
		if d.Bool() || kind >= kMean {
			scaled := rastertest.Compact(src)
			for i := range scaled.Data {
				scaled.Data[i] *= f
			}
			requireAll(t, fmt.Sprintf("%s: input·2^%d", id, k), s, run(s, scaled), run(s, src), scale, false)
			return
		}
		ss := s
		ss.w, ss.row = scaleAll(s.w, f), scaleAll(s.row, f)
		requireAll(t, fmt.Sprintf("%s: weights·2^%d", id, k), s, run(ss, src), run(s, src), scale, false)
	case 5: // negation
		s := relSpec(d, kind, r, floatWeights)
		src := rastertest.Compact(newSrc(d, w, h, anyValues, masked))
		neg := rastertest.Compact(src)
		for i := range neg.Data {
			neg.Data[i] = -neg.Data[i]
		}
		minus := func(v float32) float32 { return -v }
		switch kind {
		case kMin, kMax:
			other := s
			other.kind = kMin + kMax - kind
			requireAll(t, id+": Min(-x) = -Max(x)", s, run(s, neg), run(other, src), minus, false)
		default:
			requireAll(t, id+": negated input", s, run(s, neg), run(s, src), minus, true)
			if kind != kMean {
				ns := s
				ns.w, ns.row = scaleAll(s.w, -1), scaleAll(s.row, -1)
				requireAll(t, id+": negated weights", s, run(ns, src), run(s, src), minus, true)
			}
		}
	case 6: // shifting Min and Max
		kind = kMin + d.IntN(2)
		s := spec{kind: kind, r: r}
		src := rastertest.Compact(newSrc(d, w, h, anyValues, masked))
		c := float32(d.Range(-1000, 1000)) / 8
		moved := rastertest.Compact(src)
		for i := range moved.Data {
			moved.Data[i] += c
		}
		requireAll(t, fmt.Sprintf("%s: %v + %v", id, s, c), s, run(s, moved), run(s, src), func(v float32) float32 { return v + c }, false)
	case 7: // symmetries of the grid
		tr := rastertest.All()[d.IntN(8)]
		vals, wv := integerValues, intWeights
		if (kind == kMin || kind == kMax) && d.Bool() {
			vals, wv = anyValues, floatWeights
		}
		s := relSpec(d, kind, r, wv)
		src := rastertest.Compact(newSrc(d, w, h, vals, masked))
		ts := transformSpec(s, tr)
		got, want := run(ts, tr.Apply(src)), run(s, src)
		tw, th := tr.Size(w, h)
		for y := range th {
			for x := range tw {
				u, v := tr.Source(w, h, x, y)
				requireCell(t, fmt.Sprintf("%s %v under %+v", id, s, tr), got, x, y, want, u, v, ident, kind == kSeparable)
			}
		}
	default: // separable and box forms against Correlate, on exact data
		src := rastertest.Compact(newSrc(d, w, h, integerValues, masked))
		k := 2*r + 1
		if d.Bool() {
			s := relSpec(d, kSeparable, r, intWeights)
			outer := spec{kind: kCorrelate, r: r, w: make([]float32, k*k)}
			for j := range k {
				for c := range k {
					outer.w[j*k+c] = s.col[j] * s.row[c]
				}
			}
			requireAll(t, id+": separable = outer product", s, run(s, src), run(outer, src), ident, true)
			return
		}
		ones := spec{kind: kCorrelate, r: r, w: make([]float32, k*k)}
		for i := range ones.w {
			ones.w[i] = 1
		}
		n := float32(k * k)
		requireAll(t, id+": Mean = Σ/n", spec{kind: kMean, r: r}, run(spec{kind: kMean, r: r}, src), run(ones, src),
			func(v float32) float32 { return v / n }, false)
	}
}

func ident(v float32) float32 { return v }

func scaleAll(v []float32, f float32) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * f
	}
	return out
}

// transformSpec maps s's weights by the symmetry tr, so that s on tr(src)
// is tr of s on src: the weights are a (2r+1)×(2r+1) grid of offsets and
// move as the cells do. For separable taps a transposition swaps the row
// and the column, and a mirror reverses one of them.
func transformSpec(s spec, tr rastertest.Dihedral) spec {
	k := s.size()
	out := s
	if s.w != nil {
		// This holds for Convolve too: it rotates its weights by 180°
		// first, and that rotation commutes with every symmetry of the
		// square.
		out.w = tr.Apply(raster.NewFloat32(k, k, s.w)).Data
	}
	if s.row != nil {
		row, col := s.row, s.col
		if tr.Swap {
			row, col = col, row
		}
		// tr.Source reads the result's x from the source's y when
		// swapped, so the mirrors apply to the result's axes.
		if tr.FlipX {
			row = rot180(row)
		}
		if tr.FlipY {
			col = rot180(col)
		}
		out.row, out.col = row, col
	}
	return out
}

// requireCell checks cell (x, y) of got against f of cell (u, v) of
// want: the same validity, and for valid cells the same bits (or the same
// value, so ±0 match, when sameValue is set).
func requireCell(t rastertest.TB, id string, got raster.Float32Raster, x, y int, want raster.Float32Raster, u, v int, f func(float32) float32, sameValue bool) {
	t.Helper()
	gv, wv := got.Valid == nil || got.IsValid(x, y), want.Valid == nil || want.IsValid(u, v)
	if gv != wv {
		t.Fatalf("%s: (%d, %d) valid = %v, want %v", id, x, y, gv, wv)
	}
	if !wv {
		return // Data under invalid cells is unspecified
	}
	g, w := got.Data[got.Index(x, y)], want.Data[want.Index(u, v)]
	fw := f(w)
	ok := rastertest.SameFloat(g, fw)
	if sameValue {
		ok = rastertest.SameValue(g, fw)
	}
	if !ok {
		t.Fatalf("%s: (%d, %d) = %v (%#x), want %v (%#x)", id, x, y, g, math.Float32bits(g), fw, math.Float32bits(fw))
	}
}

// requireAll is requireCell over every cell, at the same position.
func requireAll(t rastertest.TB, id string, s spec, got, want raster.Float32Raster, f func(float32) float32, sameValue bool) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			requireCell(t, fmt.Sprintf("%s %v", id, s), got, x, y, want, x, y, f, sameValue)
		}
	}
}

// FuzzFocalRelations is focalRelations as a fuzz target.
func FuzzFocalRelations(f *testing.F) {
	for rel := range byte(9) {
		f.Add([]byte{rel, 0, 1, 20, 7})
		f.Add([]byte{rel, 2, 2, 13, 9, 1})
		f.Add([]byte{rel, 4, 0, 30, 3, 2})
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		focalRelations(t, fuzzdata.New(data))
	})
}
