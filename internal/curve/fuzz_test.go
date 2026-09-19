package curve

import (
	"math"
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
)

// refReclass and refLookup are written from the package documentation,
// not from scalar.go, so a kernel that drifts from what it promises is
// caught rather than confirmed. They are deliberately naive: a fresh
// search per cell, no reslicing, no early exit.
func refReclass(v float32, breaks, values []float32) float32 {
	if v != v {
		return v
	}
	k := 0
	for _, b := range breaks {
		if v >= b {
			k++
		}
	}
	return values[k]
}

func refLookup(v float32, xs, ys []float32) float32 {
	if v != v {
		return v
	}
	if v <= xs[0] {
		return ys[0]
	}
	if v >= xs[len(xs)-1] {
		return ys[len(ys)-1]
	}
	for k := 1; k < len(xs); k++ {
		if v < xs[k] {
			x0, y0 := xs[k-1], ys[k-1]
			if v == x0 {
				return y0 // the curve passes through its knots
			}
			t := (v - x0) / (xs[k] - x0)
			return y0 + float32(t*(ys[k]-y0))
		}
	}
	panic("unreachable: v is below the last knot")
}

// increasing builds a strictly increasing table of n finite float32s from
// the fuzz input, the precondition both kernels are documented to expect.
// It starts anywhere and steps by a positive amount, so the table spans
// ordinary values, very small gaps and very large ones.
func increasing(d fuzzdata.Source, n int) []float32 {
	out := make([]float32, n)
	x := float32(d.Range(-1000, 1000))
	for i := range out {
		out[i] = x
		step := float32(d.Range(1, 1000)) / float32(d.Range(1, 64))
		next := x + step
		if !(next > x) { // the step vanished into the rounding
			next = math.Float32frombits(math.Float32bits(x) + 1)
		}
		x = next
	}
	return out
}

func arbitrary(d fuzzdata.Source, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = d.Float32()
	}
	return out
}

// FuzzCurve runs both kernels over arbitrary cells and arbitrary valid
// tables, at lengths around the boundaries a vector backend would split
// on, and requires every cell to match the reference above. It also
// requires that nothing outside dst changes, which is what the engine
// relies on when dst is a view into a larger raster.
func FuzzCurve(f *testing.F) {
	f.Add([]byte{0, 3, 5, 1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{1, 8, 2, 9, 9, 1, 1, 2, 2, 3, 3, 4, 4})
	f.Add([]byte{1, 17, 6, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		lookup := d.Bool()
		n := d.Range(0, 40)
		size := d.Range(1, 8)

		const guard = 5
		backing := make([]float32, n+2*guard)
		for i := range backing {
			backing[i] = d.Float32()
		}
		before := append([]float32(nil), backing...)
		src := arbitrary(d, n)
		dst := backing[guard : guard+n]

		xs := increasing(d, size)
		ys := arbitrary(d, size)
		if lookup {
			Lookup(dst, src, xs, ys)
		} else {
			// Reclass takes one fewer break than it has values.
			Reclass(dst, src, xs[:size-1], ys)
		}

		for i, v := range src {
			var want float32
			if lookup {
				want = refLookup(v, xs, ys)
			} else {
				want = refReclass(v, xs[:size-1], ys)
			}
			if !same(dst[i], want) {
				t.Fatalf("lookup=%v n=%d size=%d: cell %d (%v, %#08x) = %v (%#08x), want %v (%#08x)\nxs %v\nys %v",
					lookup, n, size, i, v, math.Float32bits(v),
					dst[i], math.Float32bits(dst[i]), want, math.Float32bits(want), xs, ys)
			}
		}
		for i := range backing {
			if i >= guard && i < guard+n {
				continue
			}
			if math.Float32bits(backing[i]) != math.Float32bits(before[i]) {
				t.Fatalf("lookup=%v n=%d: wrote backing element %d outside dst", lookup, n, i)
			}
		}
	})
}

// FuzzCurveRelations checks the relations the documentation states, on
// tables and cells it does not choose: every Reclass result is one of
// the values, every Lookup result at a knot is that knot's y, and a
// Lookup whose ys are its xs is exactly a clamp to the table's ends.
func FuzzCurveRelations(f *testing.F) {
	f.Add([]byte{3, 4, 1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{9, 2, 0, 1, 2, 3, 4, 5})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		n := d.Range(1, 30)
		size := d.Range(1, 8)
		src := arbitrary(d, n)
		xs := increasing(d, size)
		ys := arbitrary(d, size)
		dst := make([]float32, n)

		// Every class is one of the values, or the cell's own NaN.
		Reclass(dst, src, xs[:size-1], ys)
		for i, got := range dst {
			if src[i] != src[i] {
				if got == got {
					t.Fatalf("Reclass turned NaN into %v", got)
				}
				continue
			}
			found := false
			for _, v := range ys {
				if same(got, v) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("Reclass cell %d = %v (%#08x), which is not in values %v", i, got, math.Float32bits(got), ys)
			}
		}

		// Every knot comes back exactly, whatever ys holds there or next
		// to it.
		knots := make([]float32, len(xs))
		copy(knots, xs)
		at := make([]float32, len(knots))
		Lookup(at, knots, xs, ys)
		for k, got := range at {
			want := ys[k]
			if !same(got, want) {
				t.Fatalf("Lookup at knot %d (x=%v) = %v (%#08x), want %v (%#08x)",
					k, xs[k], got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}

		// A curve through (0,0) and (1,1) is exactly algebra.Clamp to
		// [0, 1]: t is (v-0)/(1-0) == v, and the segment value is
		// 0 + float32(v*1) == v, with no rounding anywhere. That makes it
		// a bit-exact bridge between this package and algebra.
		//
		// The relation does NOT generalize to ys == xs on an arbitrary
		// table, and FuzzCurveRelations found the counterexample: the
		// segment recovers v as ((v-x0)/(x1-x0))*(y1-y0) + y0, a divide
		// and a multiply that need not round back to v. On
		// xs = {-1000.., -185.16..} a cell of -185.16173 came back as
		// -185.16174, one ulp low. An interpolating curve reproduces its
		// knots exactly (checked above) and nothing else exactly.
		unit := []float32{0, 1}
		Lookup(dst, src, unit, unit)
		for i, v := range src {
			want := min(max(v, 0), 1)
			if !same(dst[i], want) {
				t.Fatalf("Lookup over the unit curve, cell %d (%v) = %v (%#08x), want the clamp %v (%#08x)",
					i, v, dst[i], math.Float32bits(dst[i]), want, math.Float32bits(want))
			}
		}
	})
}
