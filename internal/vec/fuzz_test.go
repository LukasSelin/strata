package vec

import (
	"math"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
)

// sameResult compares bitwise, except that any NaN matches any NaN: the
// SIMD contract (simd_amd64.go).
func sameResult(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// fuzzOps are the exported kernels with a per-element reference written
// from their documentation, not from scalar.go.
var fuzzOps = []struct {
	name string
	run  func(dst, a, b []float32, s, lo, hi float32)
	ref  func(a, b, s, lo, hi float32) float32
}{
	{"Add", func(d, a, b []float32, _, _, _ float32) { Add(d, a, b) }, func(a, b, _, _, _ float32) float32 { return a + b }},
	{"Sub", func(d, a, b []float32, _, _, _ float32) { Sub(d, a, b) }, func(a, b, _, _, _ float32) float32 { return a - b }},
	{"Mul", func(d, a, b []float32, _, _, _ float32) { Mul(d, a, b) }, func(a, b, _, _, _ float32) float32 { return a * b }},
	{"Div", func(d, a, b []float32, _, _, _ float32) { Div(d, a, b) }, func(a, b, _, _, _ float32) float32 { return a / b }},
	{"Min", func(d, a, b []float32, _, _, _ float32) { Min(d, a, b) }, func(a, b, _, _, _ float32) float32 { return min(a, b) }},
	{"Max", func(d, a, b []float32, _, _, _ float32) { Max(d, a, b) }, func(a, b, _, _, _ float32) float32 { return max(a, b) }},
	{"AddScalar", func(d, a, _ []float32, s, _, _ float32) { AddScalar(d, a, s) }, func(a, _, s, _, _ float32) float32 { return a + s }},
	{"MulScalar", func(d, a, _ []float32, s, _, _ float32) { MulScalar(d, a, s) }, func(a, _, s, _, _ float32) float32 { return a * s }},
	{"Clamp", func(d, a, _ []float32, _, lo, hi float32) { Clamp(d, a, lo, hi) }, func(a, _, _, lo, hi float32) float32 { return min(max(a, lo), hi) }},
	{"Abs", func(d, a, _ []float32, _, _, _ float32) { Abs(d, a) }, func(a, _, _, _, _ float32) float32 {
		return math.Float32frombits(math.Float32bits(a) &^ (1 << 31))
	}},
	{"Sqrt", func(d, a, _ []float32, _, _, _ float32) { Sqrt(d, a) }, func(a, _, _, _, _ float32) float32 {
		return float32(math.Sqrt(float64(a)))
	}},
}

// FuzzKernels runs every kernel on both backends over arbitrary float32
// bit patterns, lengths around the 8-lane boundary, slices starting at
// any offset of their backing array, and every aliasing of dst, a and b,
// and requires each element to match the documented operation. It also
// requires that nothing before or after dst in its backing array changes,
// and that unequal lengths panic with a "vec:" message.
func FuzzKernels(f *testing.F) {
	f.Add([]byte{0, 17, 3, 5, 0})
	f.Add([]byte{8, 8, 0, 0, 4})
	f.Add([]byte{10, 33, 7, 1, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		defer UseScalar(false)
		d := fuzzdata.New(data)
		op := fuzzOps[d.IntN(len(fuzzOps))]
		n := d.Range(0, 40)
		const guard = 9
		backing := func() []float32 {
			s := make([]float32, n+2*guard)
			for i := range s {
				s[i] = d.Float32()
			}
			return s
		}
		bd, ba, bb := backing(), backing(), backing()
		// Aliasing: 0 none, 1 dst is a, 2 dst is b, 3 a is b, 4 all one.
		switch d.IntN(5) {
		case 1:
			ba = bd
		case 2:
			bb = bd
		case 3:
			bb = ba
		case 4:
			ba, bb = bd, bd
		}
		// Offsets differ between separate slices, so lanes and tails
		// start at different addresses; aliased slices share one.
		offsets := map[*float32]int{}
		off := func(s []float32) int {
			if o, ok := offsets[&s[0]]; ok {
				return o
			}
			o := d.IntN(guard + 1)
			offsets[&s[0]] = o
			return o
		}
		od, oa, ob := off(bd), off(ba), off(bb)
		s, lo, hi := d.Float32(), d.Float32(), d.Float32()
		mismatch := d.IntN(8) == 0

		for _, scalar := range []bool{true, false} {
			UseScalar(scalar)
			backend := Backend()
			cd, ca, cb := append([]float32(nil), bd...), append([]float32(nil), ba...), append([]float32(nil), bb...)
			// Re-establish the aliasing on the copies.
			switch {
			case &ba[0] == &bd[0] && &bb[0] == &bd[0]:
				ca, cb = cd, cd
			case &ba[0] == &bd[0]:
				ca = cd
			case &bb[0] == &bd[0]:
				cb = cd
			case &bb[0] == &ba[0]:
				cb = ca
			}
			dst, a, b := cd[od:od+n], ca[oa:oa+n], cb[ob:ob+n]
			if mismatch {
				a = ca[oa : oa+n+1]
				panicked := func() (p bool) {
					defer func() {
						if v := recover(); v != nil {
							p = true
							if msg, ok := v.(string); !ok || !strings.HasPrefix(msg, "vec: ") {
								t.Fatalf("%s: panic %v, want a \"vec: \" message", op.name, v)
							}
						}
					}()
					op.run(dst, a, b, s, lo, hi)
					return false
				}()
				if !panicked {
					t.Fatalf("%s on %s: no panic for len(dst) %d, len(a) %d", op.name, backend, len(dst), len(a))
				}
				continue
			}
			want := make([]float32, n)
			for i := range n {
				want[i] = op.ref(a[i], b[i], s, lo, hi)
			}
			op.run(dst, a, b, s, lo, hi)
			for i := range n {
				if !sameResult(dst[i], want[i]) {
					t.Fatalf("%s on %s, n %d, offsets %d %d %d: element %d = %v (%#08x), want %v (%#08x)",
						op.name, backend, n, od, oa, ob, i, dst[i], math.Float32bits(dst[i]), want[i], math.Float32bits(want[i]))
				}
			}
			for i := range cd {
				if (i < od || i >= od+n) && math.Float32bits(cd[i]) != math.Float32bits(bd[i]) {
					t.Fatalf("%s on %s, n %d at offset %d: wrote backing element %d outside dst", op.name, backend, n, od, i)
				}
			}
		}
	})
}
