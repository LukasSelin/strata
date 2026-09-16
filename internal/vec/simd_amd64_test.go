//go:build amd64

package vec

import (
	"math"
	"testing"
)

// simdCases returns slice lengths that straddle the AVX2 lane width (8):
// zero and one lane below it, exactly on it, one above it, and several
// lanes plus a partial tail, so the scalar-tail boundary in each
// *Float32AVX2 wrapper is actually exercised.
func simdCases() []int {
	return []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100}
}

// edgeFloat32s returns a fixed pattern of values covering the cases
// section 28 of DESIGN.md calls out: ordinary values, NaN, +/-Inf,
// negatives, and both zeros.
func edgeFloat32s(n int) []float32 {
	base := []float32{
		1, -1, 0, float32(math.Copysign(0, -1)), 2.5, -2.5,
		nan, inf, ninf, 100, -100, 0.001, -0.001,
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = base[i%len(base)]
	}
	return out
}

func requireAVX2(t *testing.T) {
	t.Helper()
	if !hasAVX2() {
		t.Skip("AVX2 not available on this CPU")
	}
}

func TestAVX2MatchesScalar(t *testing.T) {
	requireAVX2(t)

	for _, n := range simdCases() {
		a := edgeFloat32s(n)
		b := make([]float32, n)
		for i := range b {
			b[i] = edgeFloat32s(n + 3)[i]
		}

		t.Run("Add", func(t *testing.T) {
			want := make([]float32, n)
			scalarAddFloat32(want, a, b)
			got := make([]float32, n)
			addFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Add", got, want)
		})
		t.Run("Sub", func(t *testing.T) {
			want := make([]float32, n)
			scalarSubFloat32(want, a, b)
			got := make([]float32, n)
			subFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Sub", got, want)
		})
		t.Run("Mul", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulFloat32(want, a, b)
			got := make([]float32, n)
			mulFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Mul", got, want)
		})
		t.Run("Div", func(t *testing.T) {
			want := make([]float32, n)
			scalarDivFloat32(want, a, b)
			got := make([]float32, n)
			divFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Div", got, want)
		})
		t.Run("Min", func(t *testing.T) {
			want := make([]float32, n)
			scalarMinFloat32(want, a, b)
			got := make([]float32, n)
			minFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Min", got, want)
		})
		t.Run("Max", func(t *testing.T) {
			want := make([]float32, n)
			scalarMaxFloat32(want, a, b)
			got := make([]float32, n)
			maxFloat32AVX2(got, a, b)
			assertSlicesEqual(t, "Max", got, want)
		})
		t.Run("AddScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarAddScalarFloat32(want, a, 3.25)
			got := make([]float32, n)
			addScalarFloat32AVX2(got, a, 3.25)
			assertSlicesEqual(t, "AddScalar", got, want)
		})
		t.Run("MulScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulScalarFloat32(want, a, -2)
			got := make([]float32, n)
			mulScalarFloat32AVX2(got, a, -2)
			assertSlicesEqual(t, "MulScalar", got, want)
		})
		t.Run("Clamp", func(t *testing.T) {
			want := make([]float32, n)
			scalarClampFloat32(want, a, -10, 10)
			got := make([]float32, n)
			clampFloat32AVX2(got, a, -10, 10)
			assertSlicesEqual(t, "Clamp", got, want)
		})
		t.Run("Abs", func(t *testing.T) {
			want := make([]float32, n)
			scalarAbsFloat32(want, a)
			got := make([]float32, n)
			absFloat32AVX2(got, a)
			assertSlicesEqual(t, "Abs", got, want)

			for i, v := range a {
				if v == 0 && math.Signbit(float64(v)) {
					if math.Signbit(float64(got[i])) {
						t.Errorf("Abs(-0) at index %d: got sign bit set, want +0", i)
					}
				}
			}
		})
		t.Run("Sqrt", func(t *testing.T) {
			want := make([]float32, n)
			scalarSqrtFloat32(want, a)
			got := make([]float32, n)
			sqrtFloat32AVX2(got, a)
			assertSlicesEqual(t, "Sqrt", got, want)
		})
	}
}

// TestAVX2InPlace mirrors TestInPlace but forces the AVX2 path directly,
// since dst aliasing a or src must remain safe when the bulk of the
// work runs through YMM loads/stores rather than the scalar loop.
func TestAVX2InPlace(t *testing.T) {
	requireAVX2(t)

	a := make([]float32, 20)
	b := make([]float32, 20)
	want := make([]float32, 20)
	for i := range a {
		a[i] = float32(i) + 0.5
		b[i] = float32(i) * 2
	}
	scalarAddFloat32(want, a, b)
	addFloat32AVX2(a, a, b)
	assertSlicesEqual(t, "Add in-place AVX2", a, want)
}

func TestHasAVX2Deterministic(t *testing.T) {
	first := hasAVX2()
	for i := 0; i < 5; i++ {
		if hasAVX2() != first {
			t.Fatalf("hasAVX2 returned inconsistent results across calls")
		}
	}
}

func benchmarkAdd(b *testing.B, add func(dst, a, x []float32), n int) {
	dst := make([]float32, n)
	a := make([]float32, n)
	x := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		x[i] = float32(n - i)
	}
	b.SetBytes(int64(n) * 4 * 3)
	for i := 0; i < b.N; i++ {
		add(dst, a, x)
	}
}

func BenchmarkAddScalar4096(b *testing.B) { benchmarkAdd(b, scalarAddFloat32, 4096) }

func BenchmarkAddAVX2_4096(b *testing.B) {
	if !hasAVX2() {
		b.Skip("AVX2 not available on this CPU")
	}
	benchmarkAdd(b, addFloat32AVX2, 4096)
}
