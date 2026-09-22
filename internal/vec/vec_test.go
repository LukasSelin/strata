package vec

import (
	"math"
	"reflect"
	"testing"
)

var nan = float32(math.NaN())
var inf = float32(math.Inf(1))
var ninf = float32(math.Inf(-1))
var negZero = float32(math.Copysign(0, -1))

// eqFloat32 treats NaN as equal to NaN, since scalar and future SIMD
// backends must agree on where NaNs occur, not just on ordinary values.
func eqFloat32(a, b float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	return a == b
}

func assertSlicesEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if !eqFloat32(got[i], want[i]) {
			t.Errorf("%s: index %d: got %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestAdd(t *testing.T) {
	a := []float32{1, -2, 0, inf, ninf, nan, 3.5}
	b := []float32{1, 2, 0, 1, 1, 1, -3.5}
	want := []float32{2, 0, 0, inf, ninf, nan, 0}
	dst := make([]float32, len(a))
	Add(dst, a, b)
	assertSlicesEqual(t, "Add", dst, want)
}

func TestSub(t *testing.T) {
	a := []float32{5, 0, inf, ninf, nan}
	b := []float32{3, 0, inf, 1, 1}
	want := []float32{2, 0, nan, ninf, nan}
	dst := make([]float32, len(a))
	Sub(dst, a, b)
	assertSlicesEqual(t, "Sub", dst, want)
}

func TestMul(t *testing.T) {
	a := []float32{2, -2, 0, inf, nan}
	b := []float32{3, 3, ninf, 0, 1}
	want := []float32{6, -6, nan, nan, nan}
	dst := make([]float32, len(a))
	Mul(dst, a, b)
	assertSlicesEqual(t, "Mul", dst, want)
}

func TestDiv(t *testing.T) {
	a := []float32{6, -6, 1, 0, 0}
	b := []float32{3, 3, 0, 0, -1}
	want := []float32{2, -2, inf, nan, 0}
	dst := make([]float32, len(a))
	Div(dst, a, b)
	assertSlicesEqual(t, "Div", dst, want)
}

func TestAddScalar(t *testing.T) {
	src := []float32{1, -1, 0, inf, nan}
	dst := make([]float32, len(src))
	AddScalar(dst, src, 10)
	want := []float32{11, 9, 10, inf, nan}
	assertSlicesEqual(t, "AddScalar", dst, want)
}

func TestMulScalar(t *testing.T) {
	src := []float32{1, -1, 0, inf, nan}
	dst := make([]float32, len(src))
	MulScalar(dst, src, -2)
	want := []float32{-2, 2, 0, ninf, nan}
	assertSlicesEqual(t, "MulScalar", dst, want)
}

func TestAffine(t *testing.T) {
	src := []float32{1, -1, 0, negZero, inf, ninf, nan}
	dst := make([]float32, len(src))
	Affine(dst, src, -2, 10)
	want := []float32{8, 12, 10, 10, ninf, inf, nan}
	assertSlicesEqual(t, "Affine", dst, want)
}

// TestAffineIsNotFused pins the one property scalarAffineFloat32's
// float32 conversion exists for: the product is rounded before the add.
// With a = v = 1 + 2^-23, the float32 above 1, the exact product is
// 1 + 2^-22 + 2^-46, which rounds to 1 + 2^-22; b cancels that exactly,
// so the answer is +0. A fused multiply-add keeps the 2^-46 the rounding
// drops and returns it instead. arm64 emits FMADD for an unwrapped
// a*v + b and amd64 does not, so without the conversion the canonical
// scalar result would depend on the architecture
// (docs/adr/0001-simd-backend.md).
func TestAffineIsNotFused(t *testing.T) {
	const eps = 1.0 / (1 << 23)
	a := float32(1 + eps)
	b := -float32(1 + 2*eps)
	dst := make([]float32, 1)
	Affine(dst, []float32{a}, a, b)
	if got := dst[0]; got != 0 || math.Signbit(float64(got)) {
		t.Errorf("Affine(%v, %v, %v) = %v (%#08x), want +0; the multiply-add was fused",
			a, a, b, got, math.Float32bits(got))
	}
}

func TestMin(t *testing.T) {
	a := []float32{1, 2, -1, inf, ninf}
	b := []float32{2, 1, 1, ninf, inf}
	want := []float32{1, 1, -1, ninf, ninf}
	dst := make([]float32, len(a))
	Min(dst, a, b)
	assertSlicesEqual(t, "Min", dst, want)
}

func TestMax(t *testing.T) {
	a := []float32{1, 2, -1, inf, ninf}
	b := []float32{2, 1, 1, ninf, inf}
	want := []float32{2, 2, 1, inf, inf}
	dst := make([]float32, len(a))
	Max(dst, a, b)
	assertSlicesEqual(t, "Max", dst, want)
}

func TestClamp(t *testing.T) {
	src := []float32{-5, 0, 5, 15, inf, ninf, nan}
	want := []float32{0, 0, 5, 10, 10, 0, nan}
	dst := make([]float32, len(src))
	Clamp(dst, src, 0, 10)
	assertSlicesEqual(t, "Clamp", dst, want)
}

func TestAbs(t *testing.T) {
	src := []float32{-3, 3, 0, negZero, inf, ninf, nan}
	want := []float32{3, 3, 0, 0, inf, inf, nan}
	dst := make([]float32, len(src))
	Abs(dst, src)
	assertSlicesEqual(t, "Abs", dst, want)

	if math.Signbit(float64(dst[3])) {
		t.Errorf("Abs(-0): got sign bit set, want +0")
	}
}

func TestSqrt(t *testing.T) {
	src := []float32{4, 0, -1, inf, nan}
	want := []float32{2, 0, nan, inf, nan}
	dst := make([]float32, len(src))
	Sqrt(dst, src)
	assertSlicesEqual(t, "Sqrt", dst, want)
}

// TestTinyAndOddLengths exercises lengths that won't align to any future
// SIMD vector width (0, 1, and an odd count), so the scalar path's
// behavior on boundary sizes is pinned down before a SIMD backend exists.
func TestTinyAndOddLengths(t *testing.T) {
	for _, n := range []int{0, 1, 3, 7} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(i) + 0.5
			b[i] = float32(i) * 2
		}
		dst := make([]float32, n)
		Add(dst, a, b)
		for i := range dst {
			want := a[i] + b[i]
			if dst[i] != want {
				t.Errorf("n=%d: index %d: got %v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestLengthMismatchPanics(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
	}{
		{"Add", func() { Add(make([]float32, 3), make([]float32, 2), make([]float32, 3)) }},
		{"AddScalar", func() { AddScalar(make([]float32, 3), make([]float32, 2), 1) }},
		{"Clamp", func() { Clamp(make([]float32, 3), make([]float32, 4), 0, 1) }},
		{"Affine", func() { Affine(make([]float32, 3), make([]float32, 4), 1, 0) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected panic on length mismatch", c.name)
				}
			}()
			c.fn()
		})
	}
}

// TestInPlace ensures dst may alias a or src, since callers will often
// want to overwrite the input raster/tile buffer rather than allocate.
func TestInPlace(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20, 30, 40}
	want := []float32{11, 22, 33, 44}
	Add(a, a, b)
	assertSlicesEqual(t, "Add in-place", a, want)
}

// kernelsInUse returns the function variables currently installed.
func kernelsInUse() kernelSet {
	return kernelSet{
		add: addFloat32, sub: subFloat32, mul: mulFloat32, div: divFloat32,
		addScalar: addScalarFloat32, mulScalar: mulScalarFloat32, affine: affineFloat32,
		min: minFloat32, max: maxFloat32, clamp: clampFloat32,
		abs: absFloat32, sqrt: sqrtFloat32,
		reduceMin: reduceMinFloat32, reduceMax: reduceMaxFloat32,
		chain: chainFloat32,
	}
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

func TestReduceMinMax(t *testing.T) {
	cases := []struct {
		name    string
		src     []float32
		wantMin float32
		wantMax float32
	}{
		{"empty keeps acc", nil, inf, ninf},
		{"one", []float32{3}, 3, 3},
		{"ordinary", []float32{3, 1, 4, 1, 5, 9, 2, 6}, 1, 9},
		{"negatives", []float32{-3, -1, -4}, -4, -1},
		{"infinities", []float32{inf, 1, ninf}, ninf, inf},
		{"nan absorbs", []float32{1, nan, 3}, nan, nan},
		{"signed zeros", []float32{0, negZero}, negZero, 0},
		{"signed zeros reversed", []float32{negZero, 0}, negZero, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ReduceMin(inf, tc.src); !sameBits(got, tc.wantMin) {
				t.Errorf("ReduceMin = %v (%#08x), want %v (%#08x)",
					got, math.Float32bits(got), tc.wantMin, math.Float32bits(tc.wantMin))
			}
			if got := ReduceMax(ninf, tc.src); !sameBits(got, tc.wantMax) {
				t.Errorf("ReduceMax = %v (%#08x), want %v (%#08x)",
					got, math.Float32bits(got), tc.wantMax, math.Float32bits(tc.wantMax))
			}
		})
	}
}

// TestReduceSplitsAnywhere is the property the engine leans on: folding a
// slice in two pieces and combining gives the same bits as folding it
// whole, at every split point and for every length around the lane width.
// It is what lets tiles, bands and workers divide a raster however they
// like (DESIGN.md §49).
func TestReduceSplitsAnywhere(t *testing.T) {
	for n := range 40 {
		src := edgeValues(n)
		wholeMin, wholeMax := ReduceMin(inf, src), ReduceMax(ninf, src)
		for k := 0; k <= n; k++ {
			gotMin := ReduceMin(ReduceMin(inf, src[:k]), src[k:])
			gotMax := ReduceMax(ReduceMax(ninf, src[:k]), src[k:])
			if !sameBits(gotMin, wholeMin) {
				t.Fatalf("n=%d split at %d: min %v (%#08x), want %v (%#08x)",
					n, k, gotMin, math.Float32bits(gotMin), wholeMin, math.Float32bits(wholeMin))
			}
			if !sameBits(gotMax, wholeMax) {
				t.Fatalf("n=%d split at %d: max %v (%#08x), want %v (%#08x)",
					n, k, gotMax, math.Float32bits(gotMax), wholeMax, math.Float32bits(wholeMax))
			}
		}
	}
}

// sameBits is bit equality with any NaN matching any NaN, as everywhere
// in this package: min and max do not say whose payload survives.
func sameBits(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// edgeValues returns n values covering the DESIGN.md §39 table, arranged
// so that the extremes fall at different offsets for different n.
func edgeValues(n int) []float32 {
	base := []float32{
		1, -1, 0, negZero, 2.5, -2.5, nan, inf, ninf,
		100, -100, 0.001, -0.001, math.MaxFloat32, -math.MaxFloat32,
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = base[(i*7+i/len(base))%len(base)]
	}
	return out
}
