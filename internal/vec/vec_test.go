package vec

import (
	"math"
	"testing"
)

var nan = float32(math.NaN())
var inf = float32(math.Inf(1))
var ninf = float32(math.Inf(-1))

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
	negZero := float32(math.Copysign(0, -1))
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
