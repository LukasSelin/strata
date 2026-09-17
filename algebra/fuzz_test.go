package algebra_test

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"strata/algebra"
	"strata/engine"
	"strata/internal/fuzzdata"
	"strata/raster"
)

// fuzzRoot is a raster that owns its memory, and the position of the
// operand window in it.
type fuzzRoot struct {
	r    raster.Float32Raster
	x, y int
}

func (o fuzzRoot) clone() fuzzRoot {
	o.r.Data = append([]float32(nil), o.r.Data...)
	if o.r.Valid != nil {
		o.r.Valid = append([]uint64(nil), o.r.Valid...)
	}
	return o
}

// fuzzOperand is one of dst, a and b: a window of one of the roots.
type fuzzOperand struct {
	root int
	x, y int
}

// panicMessage runs f and returns its panic message, or "" if it did not
// panic. A panic that is not a string (a runtime error) fails the test.
func panicMessage(t *testing.T, id string, f func()) (msg string) {
	t.Helper()
	defer func() {
		if v := recover(); v != nil {
			s, ok := v.(string)
			if !ok || s == "" {
				t.Fatalf("%s: panic %v (%T), want a message", id, v, v)
			}
			msg = s
		}
	}()
	f()
	return ""
}

// FuzzOperations runs Add, Sub, Mul, Min, Max and Clamp as plain, Tiled
// and Chunked functions on operands built from the fuzz input: compact
// and strided rasters and windows, with and without masks, in place, and
// sibling windows of one root that may overlap, over arbitrary values and
// clamp bounds, with arbitrary tiles and workers. Each call must either
// panic with its package's message where the documentation says so, or
// write the reference result: each valid cell's value (any NaN matching
// any NaN) and the AND of the inputs' validity, with every bit outside
// dst's cells unchanged. The three functions must agree with each other.
func FuzzOperations(f *testing.F) {
	f.Add([]byte{0, 9, 2, 0, 0, 0, 0, 0})
	f.Add([]byte{3, 65, 3, 1, 1, 2, 1, 1, 1, 0, 1})
	f.Add([]byte{5, 17, 1, 2, 0, 1, 1, 0, 3, 7, 7})
	f.Add([]byte{1, 64, 4, 1, 2, 1, 0, 1, 5, 2, 3, 3, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		type opDef struct {
			name   string
			binary bool
			ref    func(a, b, lo, hi float32) float32
		}
		ops := []opDef{
			{"Add", true, func(a, b, _, _ float32) float32 { return a + b }},
			{"Sub", true, func(a, b, _, _ float32) float32 { return a - b }},
			{"Mul", true, func(a, b, _, _ float32) float32 { return a * b }},
			{"Min", true, func(a, b, _, _ float32) float32 { return min(a, b) }},
			{"Max", true, func(a, b, _, _ float32) float32 { return max(a, b) }},
			{"Clamp", false, func(a, _, lo, hi float32) float32 { return min(max(a, lo), hi) }},
		}
		op := ops[d.IntN(len(ops))]
		w, h := d.Range(1, 70), d.Range(1, 6)

		// Roots: one per operand, then aliasing picks which operands share.
		newRoot := func() fuzzRoot {
			rootW, rootH, x, y := w, h, 0, 0
			layout := d.IntN(3) // compact, strided, window
			if layout == 2 {
				x, y = d.IntN(w+2), d.IntN(4)
				rootW, rootH = w+x+d.IntN(4), h+y+d.IntN(3)
			}
			stride := rootW
			if layout > 0 {
				stride += d.IntN(70)
			}
			n := (rootH-1)*stride + rootW
			r := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
			for i := range r.Data {
				r.Data[i] = d.Float32()
			}
			if d.IntN(3) != 0 {
				off := d.IntN(130)
				r.Valid = make([]uint64, raster.MaskWords(off+n)+d.IntN(2))
				r.ValidOffset = off
				for k := range r.Valid {
					r.Valid[k] = d.Uint64() | d.Uint64() // mostly valid
				}
			}
			return fuzzRoot{r, x, y}
		}
		roots := []fuzzRoot{newRoot(), newRoot(), newRoot()}
		dst, a, b := fuzzOperand{0, roots[0].x, roots[0].y}, fuzzOperand{1, roots[1].x, roots[1].y}, fuzzOperand{2, roots[2].x, roots[2].y}
		sibling := func() fuzzOperand {
			rt := roots[dst.root]
			return fuzzOperand{dst.root, d.IntN(rt.r.Width - w + 1), d.IntN(rt.r.Height - h + 1)}
		}
		alias := d.IntN(8)
		switch alias {
		case 1:
			a = dst
		case 2:
			b = dst
		case 3:
			b = a
		case 4:
			a, b = dst, dst
		case 5:
			a = sibling()
		case 6:
			b = sibling()
		}
		lo, hi := d.Float32(), d.Float32()
		opts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 4)}
		id := fmt.Sprintf("%s %d×%d alias %d opts %+v", op.name, w, h, alias, opts)

		inputs := []fuzzOperand{a}
		if op.binary {
			inputs = append(inputs, b)
		}
		view := func(rs []fuzzRoot, o fuzzOperand) raster.Float32Raster {
			return rs[o.root].r.Window(o.x, o.y, w, h)
		}

		// What the documentation says each call must do.
		inputMasked := false
		for _, in := range inputs {
			inputMasked = inputMasked || roots[in.root].r.Valid != nil
		}
		missingMask := inputMasked && roots[dst.root].r.Valid == nil
		cellIndex := func(o fuzzOperand, x, y int) int { return (o.y+y)*roots[o.root].r.Stride + o.x + x }
		// sharesCells: dst and in share a cell but are not the same
		// window. spansMeet: their Data or mask word spans meet.
		sharesCells := func(in fuzzOperand) bool {
			if in.root != dst.root || in == dst {
				return false
			}
			for y := range h {
				for x := range w {
					i := cellIndex(in, x, y) - cellIndex(dst, 0, 0)
					if i >= 0 && i/roots[dst.root].r.Stride < h && i%roots[dst.root].r.Stride < w {
						return true
					}
				}
			}
			return false
		}
		spansMeet := func(in fuzzOperand) bool {
			if in.root != dst.root {
				return false
			}
			rt := roots[dst.root].r
			span := (h-1)*rt.Stride + w
			p, q := cellIndex(dst, 0, 0), cellIndex(in, 0, 0)
			if p < q+span && q < p+span {
				return true
			}
			wp, wq := rt.ValidOffset+p, rt.ValidOffset+q
			return rt.Valid != nil && wp>>6 <= (wq+span-1)>>6 && wq>>6 <= (wp+span-1)>>6
		}
		partial, chunkedMustPanic, chunkedMayPanic := false, false, false
		for _, in := range inputs {
			partial = partial || sharesCells(in)
			chunkedMustPanic = chunkedMustPanic || in.root == dst.root && (in == dst || sharesCells(in))
			chunkedMayPanic = chunkedMayPanic || spansMeet(in)
		}

		// Reference: the inputs' cells before any call.
		type cell struct {
			v     float32
			valid bool
		}
		want := make([]cell, w*h)
		{
			var av, bv raster.Float32Raster
			av = view(roots, a)
			if op.binary {
				bv = view(roots, b)
			}
			for y := range h {
				for x := range w {
					c := cell{valid: av.IsValid(x, y)}
					va, vb := av.Data[av.Index(x, y)], float32(0)
					if op.binary {
						vb = bv.Data[bv.Index(x, y)]
						c.valid = c.valid && bv.IsValid(x, y)
					}
					c.v = op.ref(va, vb, lo, hi)
					want[y*w+x] = c
				}
			}
		}

		ctx := context.Background()
		type run struct {
			name   string
			prefix string
			call   func(dst, a, b raster.Float32Raster)
			// panics reports whether the call must panic (1), must not
			// (-1), or may (0).
			panics int
		}
		must := func(p bool) int {
			if p {
				return 1
			}
			return -1
		}
		mustErr := func(err error) {
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}
		}
		chunkedPanics := must(missingMask || chunkedMustPanic)
		if chunkedPanics < 0 && chunkedMayPanic {
			chunkedPanics = 0
		}
		runs := []run{
			{"plain", "algebra." + op.name + ": ", func(dv, av, bv raster.Float32Raster) {
				switch op.name {
				case "Add":
					algebra.Add(dv, av, bv)
				case "Sub":
					algebra.Sub(dv, av, bv)
				case "Mul":
					algebra.Mul(dv, av, bv)
				case "Min":
					algebra.Min(dv, av, bv)
				case "Max":
					algebra.Max(dv, av, bv)
				case "Clamp":
					algebra.Clamp(dv, av, lo, hi)
				}
			}, must(missingMask || partial)},
			{"tiled", "engine: ", func(dv, av, bv raster.Float32Raster) {
				switch op.name {
				case "Add":
					mustErr(algebra.AddTiled(ctx, dv, av, bv, opts))
				case "Sub":
					mustErr(algebra.SubTiled(ctx, dv, av, bv, opts))
				case "Mul":
					mustErr(algebra.MulTiled(ctx, dv, av, bv, opts))
				case "Min":
					mustErr(algebra.MinTiled(ctx, dv, av, bv, opts))
				case "Max":
					mustErr(algebra.MaxTiled(ctx, dv, av, bv, opts))
				case "Clamp":
					mustErr(algebra.ClampTiled(ctx, dv, av, lo, hi, opts))
				}
			}, must(missingMask || partial)},
			{"chunked", "engine: ", func(dv, av, bv raster.Float32Raster) {
				sink, sa := engine.NewMemorySink(dv), engine.NewMemorySource(av)
				switch op.name {
				case "Add":
					mustErr(algebra.AddChunked(ctx, sink, sa, engine.NewMemorySource(bv), opts))
				case "Sub":
					mustErr(algebra.SubChunked(ctx, sink, sa, engine.NewMemorySource(bv), opts))
				case "Mul":
					mustErr(algebra.MulChunked(ctx, sink, sa, engine.NewMemorySource(bv), opts))
				case "Min":
					mustErr(algebra.MinChunked(ctx, sink, sa, engine.NewMemorySource(bv), opts))
				case "Max":
					mustErr(algebra.MaxChunked(ctx, sink, sa, engine.NewMemorySource(bv), opts))
				case "Clamp":
					mustErr(algebra.ClampChunked(ctx, sink, sa, lo, hi, opts))
				}
			}, chunkedPanics},
		}

		var first raster.Float32Raster // dst root after the first call that wrote
		firstName := ""
		for _, rn := range runs {
			rs := make([]fuzzRoot, len(roots))
			for i := range roots {
				rs[i] = roots[i].clone()
			}
			dv, av := view(rs, dst), view(rs, a)
			bv := av
			if op.binary {
				bv = view(rs, b)
			}
			rid := id + " " + rn.name
			msg := panicMessage(t, rid, func() { rn.call(dv, av, bv) })
			if msg != "" && !strings.HasPrefix(msg, rn.prefix) {
				t.Fatalf("%s: panic %q, want prefix %q", rid, msg, rn.prefix)
			}
			if (msg != "") != (rn.panics > 0) && rn.panics != 0 {
				t.Fatalf("%s: panic %q, want panic %v (missing mask %v, partial %v)", rid, msg, rn.panics > 0, missingMask, partial)
			}
			orig, got := roots[dst.root].r, rs[dst.root].r
			if msg != "" {
				// A call that panics has changed nothing.
				for i := range orig.Data {
					if math.Float32bits(orig.Data[i]) != math.Float32bits(got.Data[i]) {
						t.Fatalf("%s: panicked with %q after writing root cell %d", rid, msg, i)
					}
				}
				for k := range orig.Valid {
					if orig.Valid[k] != got.Valid[k] {
						t.Fatalf("%s: panicked with %q after writing mask word %d", rid, msg, k)
					}
				}
				continue
			}

			// The reference, and nothing outside dst's cells.
			for y := range h {
				for x := range w {
					c := want[y*w+x]
					if got := dv.IsValid(x, y); got != c.valid {
						t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", rid, x, y, got, c.valid)
					}
					if v := dv.Data[dv.Index(x, y)]; c.valid && !sameFloat(v, c.v) {
						t.Fatalf("%s: cell (%d, %d) = %v (%#08x), want %v (%#08x)", rid, x, y, v, math.Float32bits(v), c.v, math.Float32bits(c.v))
					}
				}
			}
			inDst := func(i int) bool {
				i -= cellIndex(dst, 0, 0)
				return i >= 0 && i/orig.Stride < h && i%orig.Stride < w
			}
			for i := range orig.Data {
				if !inDst(i) && math.Float32bits(orig.Data[i]) != math.Float32bits(got.Data[i]) {
					t.Fatalf("%s: root cell %d outside dst changed", rid, i)
				}
			}
			for bit := range len(orig.Valid) * 64 {
				i := bit - orig.ValidOffset
				if (i < 0 || i >= len(orig.Data) || !inDst(i)) && raster.MaskGet(orig.Valid, bit) != raster.MaskGet(got.Valid, bit) {
					t.Fatalf("%s: mask bit %d outside dst changed", rid, bit)
				}
			}

			// Every function writes the same cells, validity and values,
			// as the first.
			if firstName == "" {
				first, firstName = got, rn.name
				continue
			}
			for i := range got.Data {
				if !sameFloat(got.Data[i], first.Data[i]) {
					t.Fatalf("%s: root cell %d = %v, %s wrote %v", rid, i, got.Data[i], firstName, first.Data[i])
				}
			}
			for k := range got.Valid {
				if got.Valid[k] != first.Valid[k] {
					t.Fatalf("%s: mask word %d = %#x, %s wrote %#x", rid, k, got.Valid[k], firstName, first.Valid[k])
				}
			}
		}
	})
}
