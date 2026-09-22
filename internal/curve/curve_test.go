package curve

import (
	"math"
	"reflect"
	"testing"
)

var (
	nan     = float32(math.NaN())
	inf     = float32(math.Inf(1))
	ninf    = float32(math.Inf(-1))
	negZero = float32(math.Copysign(0, -1))
)

// same compares bitwise, except that any NaN matches any NaN: the
// backend contract, as in internal/vec.
func same(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

func assertSlicesEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if !same(got[i], want[i]) {
			t.Errorf("%s: index %d: got %v (%#08x), want %v (%#08x)",
				name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestReclass(t *testing.T) {
	cases := []struct {
		name           string
		breaks, values []float32
		src, want      []float32
	}{
		{
			// A cell exactly on a break takes the class above it.
			name:   "breaks are lower bounds",
			breaks: []float32{10, 20, 30}, values: []float32{1, 2, 3, 4},
			src:  []float32{9.999, 10, 10.001, 19.999, 20, 29.999, 30, 30.001},
			want: []float32{1, 2, 2, 2, 3, 3, 4, 4},
		},
		{
			name:   "ends run to the infinities",
			breaks: []float32{0}, values: []float32{-1, 1},
			src:  []float32{ninf, -math.MaxFloat32, -0.001, 0, 0.001, math.MaxFloat32, inf},
			want: []float32{-1, -1, -1, 1, 1, 1, 1},
		},
		{
			// -0 and +0 compare equal, so they share a class.
			name:   "signed zeros share a class",
			breaks: []float32{0}, values: []float32{7, 8},
			src:  []float32{negZero, 0},
			want: []float32{8, 8},
		},
		{
			// The scan alone would count NaN as at or above every break
			// and return the top class.
			name:   "NaN propagates",
			breaks: []float32{10, 20}, values: []float32{1, 2, 3},
			src:  []float32{nan, 15},
			want: []float32{nan, 2},
		},
		{
			name:   "no breaks is a constant",
			breaks: nil, values: []float32{5},
			src:  []float32{-1, 0, 1, inf, ninf},
			want: []float32{5, 5, 5, 5, 5},
		},
		{
			name:   "values may repeat and may be special",
			breaks: []float32{1, 2}, values: []float32{nan, inf, negZero},
			src:  []float32{0, 1.5, 3},
			want: []float32{nan, inf, negZero},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dst := make([]float32, len(c.src))
			Reclass(dst, c.src, c.breaks, c.values)
			assertSlicesEqual(t, "Reclass", dst, c.want)
		})
	}
}

// TestReclassNaNPayload checks that a NaN cell comes back as itself, not
// as some other NaN: unlike a reduction, a per-cell result is fully
// determined, so there is no reason to canonicalize it.
func TestReclassNaNPayload(t *testing.T) {
	src := []float32{math.Float32frombits(0x7fc0_0001), math.Float32frombits(0xffc0_0000)}
	dst := make([]float32, len(src))
	Reclass(dst, src, []float32{0}, []float32{1, 2})
	assertSlicesEqual(t, "Reclass NaN payload", dst, src)
	Lookup(dst, src, []float32{0, 1}, []float32{1, 2})
	assertSlicesEqual(t, "Lookup NaN payload", dst, src)
}

func TestLookup(t *testing.T) {
	cases := []struct {
		name      string
		xs, ys    []float32
		src, want []float32
	}{
		{
			name: "interpolates and clamps",
			xs:   []float32{0, 10, 20}, ys: []float32{0, 1, 0},
			src:  []float32{-5, 0, 2.5, 5, 10, 15, 20, 25},
			want: []float32{0, 0, 0.25, 0.5, 1, 0.5, 0, 0},
		},
		{
			name: "knots are exact",
			xs:   []float32{-3, 0.1, 7.25, 1000}, ys: []float32{5, -2.5, 0.125, 1e30},
			src:  []float32{-3, 0.1, 7.25, 1000},
			want: []float32{5, -2.5, 0.125, 1e30},
		},
		{
			// Found by FuzzCurve. Interpolating from a knot whose
			// neighbour's y is not finite would give 0*NaN = NaN at the
			// knot itself, so a single undefined knot would swallow the
			// defined one below it. A -0 y would come back +0 the same
			// way, through -0 + 0.
			name: "knots are exact next to a NaN, an infinity or a -0",
			xs:   []float32{0, 1, 2, 3, 4}, ys: []float32{10, nan, inf, negZero, 20},
			src:  []float32{0, 1, 2, 3, 4},
			want: []float32{10, nan, inf, negZero, 20},
		},
		{
			// Between such knots the value is still NaN: only the knots
			// themselves are special.
			name: "a NaN knot still poisons its segments",
			xs:   []float32{0, 1, 2}, ys: []float32{10, nan, 20},
			src:  []float32{0.5, 1.5},
			want: []float32{nan, nan},
		},
		{
			name: "one knot is a constant",
			xs:   []float32{4}, ys: []float32{9},
			src:  []float32{-1, 4, 100, inf, ninf},
			want: []float32{9, 9, 9, 9, 9},
		},
		{
			name: "infinities clamp to the ends",
			xs:   []float32{0, 1}, ys: []float32{-2, 3},
			src:  []float32{ninf, inf},
			want: []float32{-2, 3},
		},
		{
			// The scan alone would place NaN above every knot and return
			// the last y.
			name: "NaN propagates",
			xs:   []float32{0, 1}, ys: []float32{-2, 3},
			src:  []float32{nan, 0.5},
			want: []float32{nan, 0.5},
		},
		{
			name: "a flat segment stays flat",
			xs:   []float32{0, 10}, ys: []float32{2, 2},
			src:  []float32{0, 3, 7, 10},
			want: []float32{2, 2, 2, 2},
		},
		{
			// A cell equal to an interior knot is the start of the upper
			// segment, not the end of the lower one; both give that knot's
			// y, which is what makes the two readings agree.
			name: "interior knot belongs to the upper segment",
			xs:   []float32{0, 1, 2}, ys: []float32{0, 10, 20},
			src:  []float32{1},
			want: []float32{10},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dst := make([]float32, len(c.src))
			Lookup(dst, c.src, c.xs, c.ys)
			assertSlicesEqual(t, "Lookup", dst, c.want)
		})
	}
}

// TestLookupIsNotFused pins the float32 conversion in the segment value.
// The knots put ys[k-1] + t·(ys[k]-ys[k-1]) at exactly the case where
// rounding the product before adding and fusing the two disagree, the way
// vec.TestAffineIsNotFused does for a·v + b.
func TestLookupIsNotFused(t *testing.T) {
	// The cell sits just below the upper knot, so t is the float32 below
	// 1 — not a power of two, so the product has bits to round away,
	// which a midpoint cell would not. Against a rise of 1 + 2^-23 the
	// exact product is 1 + 2^-24 - 2^-47, which rounds to 1, and y0
	// cancels that to +0; a fused lane keeps the remainder and returns
	// 2^-24 - 2^-47 instead.
	xs := []float32{0, 1}
	v := math.Float32frombits(math.Float32bits(float32(1)) - 1)
	y0, y1 := float32(-1), float32(math.Ldexp(1, -23))
	// Enough copies of the cell for a vector backend's lanes and its
	// scalar tail both to compute it.
	src := make([]float32, 19)
	for i := range src {
		src[i] = v
	}
	dst := make([]float32, len(src))
	Lookup(dst, src, xs, []float32{y0, y1})

	t32 := v
	want := y0 + float32(t32*(y1-y0))
	fused := float32(math.FMA(float64(t32), float64(y1-y0), float64(y0)))
	if want == fused {
		t.Fatal("the fixture no longer distinguishes a rounded product from a fused one")
	}
	for i, got := range dst {
		if !same(got, want) {
			t.Errorf("Lookup cell %d = %v (%#08x), want %v (%#08x); the multiply-add was fused",
				i, got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
}

func TestPanics(t *testing.T) {
	cases := []struct {
		name string
		want string
		fn   func()
	}{
		{"Reclass/length", "curve: dst and src must have equal length", func() {
			Reclass(make([]float32, 3), make([]float32, 2), []float32{0}, []float32{1, 2})
		}},
		{"Reclass/table", "curve: values must hold one more element than breaks", func() {
			Reclass(make([]float32, 2), make([]float32, 2), []float32{0}, []float32{1})
		}},
		{"Lookup/length", "curve: dst and src must have equal length", func() {
			Lookup(make([]float32, 3), make([]float32, 2), []float32{0}, []float32{1})
		}},
		{"Lookup/table", "curve: xs and ys must have equal length", func() {
			Lookup(make([]float32, 2), make([]float32, 2), []float32{0, 1}, []float32{1})
		}},
		{"Lookup/empty", "curve: the table needs at least one knot", func() {
			Lookup(make([]float32, 2), make([]float32, 2), nil, nil)
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				v := recover()
				msg, ok := v.(string)
				if !ok {
					t.Fatalf("panic %v (%T), want a string", v, v)
				}
				if msg != c.want {
					t.Errorf("panic %q, want %q", msg, c.want)
				}
			}()
			c.fn()
			t.Fatal("no panic")
		})
	}
}

// TestEmptyAndOddLengths runs the kernels over the lengths a vector
// backend splits differently, so the SIMD loop's tail is covered by the
// value tests above as well as its lanes.
func TestEmptyAndOddLengths(t *testing.T) {
	breaks, values := []float32{0, 1}, []float32{10, 20, 30}
	xs, ys := []float32{0, 1}, []float32{0, 1}
	for _, n := range []int{0, 1, 3, 7, 8, 9, 16, 17, 33} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i)/4 - 1
		}
		dst := make([]float32, n)
		Reclass(dst, src, breaks, values)
		for i, v := range src {
			want := values[0]
			switch {
			case v >= breaks[1]:
				want = values[2]
			case v >= breaks[0]:
				want = values[1]
			}
			if dst[i] != want {
				t.Fatalf("Reclass n=%d index %d (%v): got %v, want %v", n, i, v, dst[i], want)
			}
		}
		Lookup(dst, src, xs, ys)
		for i, v := range src {
			want := min(max(v, 0), 1)
			if dst[i] != want {
				t.Fatalf("Lookup n=%d index %d (%v): got %v, want %v", n, i, v, dst[i], want)
			}
		}
	}
}

func kernelsInUse() kernelSet {
	return kernelSet{reclass: reclassFloat32, lookup: lookupFloat32}
}

// assertKernels fails unless every function variable in got is the same
// function as in want. Functions are compared by code pointer.
func assertKernels(t *testing.T, name string, got, want kernelSet) {
	t.Helper()
	g, w := reflect.ValueOf(got), reflect.ValueOf(want)
	for i := range g.NumField() {
		if g.Field(i).Pointer() != w.Field(i).Pointer() {
			t.Errorf("%s: %s kernel is not the expected function", name, g.Type().Field(i).Name)
		}
	}
}

func TestUseScalar(t *testing.T) {
	defer UseScalar(false)
	initial := Backend()
	if initial != "scalar" && initial != "avx2" {
		t.Fatalf("Backend() = %q, want scalar or avx2", initial)
	}
	UseScalar(true)
	if Backend() != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", Backend())
	}
	assertKernels(t, "UseScalar(true)", kernelsInUse(), scalarKernels)
	UseScalar(false)
	if Backend() != initial {
		t.Fatalf("after UseScalar(false), Backend() = %q, want %q", Backend(), initial)
	}
	if simdKernels == nil {
		assertKernels(t, "UseScalar(false) without SIMD", kernelsInUse(), scalarKernels)
	}
}
