//go:build goexperiment.simd && amd64

package vec

import (
	"math"
	"simd/archsimd"
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
// section 39 of DESIGN.md calls out: ordinary values, NaN, +/-Inf,
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

var negZero = float32(math.Copysign(0, -1))

// assertSameBits is stricter than assertSlicesEqual: SIMD lanes must match
// the scalar reference bit-for-bit, so +0 and -0 differ. Any NaN still
// matches any NaN, since hardware and scalar code may pick different
// payloads.
func assertSameBits(t *testing.T, name string, got, want []float32) {
	t.Helper()
	for i := range want {
		g, w := got[i], want[i]
		if g != g && w != w {
			continue
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			t.Errorf("%s: index %d: got %v (%#08x), want %v (%#08x)", name, i, g, math.Float32bits(g), w, math.Float32bits(w))
		}
	}
}

func requireAVX2(t *testing.T) {
	t.Helper()
	if !archsimd.X86.AVX2() {
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
			assertSameBits(t, "Add", got, want)
		})
		t.Run("Sub", func(t *testing.T) {
			want := make([]float32, n)
			scalarSubFloat32(want, a, b)
			got := make([]float32, n)
			subFloat32AVX2(got, a, b)
			assertSameBits(t, "Sub", got, want)
		})
		t.Run("Mul", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulFloat32(want, a, b)
			got := make([]float32, n)
			mulFloat32AVX2(got, a, b)
			assertSameBits(t, "Mul", got, want)
		})
		t.Run("Div", func(t *testing.T) {
			want := make([]float32, n)
			scalarDivFloat32(want, a, b)
			got := make([]float32, n)
			divFloat32AVX2(got, a, b)
			assertSameBits(t, "Div", got, want)
		})
		t.Run("Min", func(t *testing.T) {
			want := make([]float32, n)
			scalarMinFloat32(want, a, b)
			got := make([]float32, n)
			minFloat32AVX2(got, a, b)
			assertSameBits(t, "Min", got, want)
		})
		t.Run("Max", func(t *testing.T) {
			want := make([]float32, n)
			scalarMaxFloat32(want, a, b)
			got := make([]float32, n)
			maxFloat32AVX2(got, a, b)
			assertSameBits(t, "Max", got, want)
		})
		t.Run("AddScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarAddScalarFloat32(want, a, 3.25)
			got := make([]float32, n)
			addScalarFloat32AVX2(got, a, 3.25)
			assertSameBits(t, "AddScalar", got, want)
		})
		t.Run("MulScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulScalarFloat32(want, a, -2)
			got := make([]float32, n)
			mulScalarFloat32AVX2(got, a, -2)
			assertSameBits(t, "MulScalar", got, want)
		})
		t.Run("Clamp", func(t *testing.T) {
			want := make([]float32, n)
			scalarClampFloat32(want, a, -10, 10)
			got := make([]float32, n)
			clampFloat32AVX2(got, a, -10, 10)
			assertSameBits(t, "Clamp", got, want)
		})
		t.Run("ClampSignedZeroBounds", func(t *testing.T) {
			for _, bounds := range [][2]float32{{negZero, 0}, {0, negZero}, {0, 0}, {negZero, negZero}, {nan, 1}, {-1, nan}} {
				want := make([]float32, n)
				scalarClampFloat32(want, a, bounds[0], bounds[1])
				got := make([]float32, n)
				clampFloat32AVX2(got, a, bounds[0], bounds[1])
				assertSameBits(t, "Clamp", got, want)
			}
		})
		t.Run("Abs", func(t *testing.T) {
			want := make([]float32, n)
			scalarAbsFloat32(want, a)
			got := make([]float32, n)
			absFloat32AVX2(got, a)
			assertSameBits(t, "Abs", got, want)

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
			assertSameBits(t, "Sqrt", got, want)
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

// TestBackendSelection checks that the SIMD set is installed at init,
// that UseScalar(true) replaces all of it, and that UseScalar(false)
// puts every kernel back.
func TestBackendSelection(t *testing.T) {
	requireAVX2(t)
	defer UseScalar(false)
	if Backend() != "avx2" {
		t.Fatalf("Backend() = %q, want avx2", Backend())
	}
	assertKernels(t, "init", kernelsInUse(), *simdKernels)
	assertKernels(t, "SIMD set", *simdKernels, kernelSet{
		add: addFloat32AVX2, sub: subFloat32AVX2, mul: mulFloat32AVX2, div: divFloat32AVX2,
		addScalar: addScalarFloat32AVX2, mulScalar: mulScalarFloat32AVX2,
		min: minFloat32AVX2, max: maxFloat32AVX2, clamp: clampFloat32AVX2,
		abs: absFloat32AVX2, sqrt: sqrtFloat32AVX2,
	})
	UseScalar(true)
	if Backend() != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", Backend())
	}
	assertKernels(t, "UseScalar(true)", kernelsInUse(), scalarKernels)
	UseScalar(false)
	if Backend() != "avx2" {
		t.Fatalf("after UseScalar(false), Backend() = %q, want avx2", Backend())
	}
	assertKernels(t, "UseScalar(false)", kernelsInUse(), *simdKernels)
}

const benchN = 4096

func benchInputs(n int) (dst, a, x []float32) {
	dst = make([]float32, n)
	a = make([]float32, n)
	x = make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		x[i] = float32(n - i)
	}
	return dst, a, x
}

func benchmarkBinary(b *testing.B, fn func(dst, a, x []float32)) {
	dst, a, x := benchInputs(benchN)
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		fn(dst, a, x)
	}
}

func requireAVX2Bench(b *testing.B) {
	b.Helper()
	if !archsimd.X86.AVX2() {
		b.Skip("AVX2 not available on this CPU")
	}
}

func BenchmarkAddScalar4096(b *testing.B) { benchmarkBinary(b, scalarAddFloat32) }

func BenchmarkAddAVX2_4096(b *testing.B) {
	requireAVX2Bench(b)
	benchmarkBinary(b, addFloat32AVX2)
}

func BenchmarkMinScalar4096(b *testing.B) { benchmarkBinary(b, scalarMinFloat32) }

func BenchmarkMinAVX2_4096(b *testing.B) {
	requireAVX2Bench(b)
	benchmarkBinary(b, minFloat32AVX2)
}

func benchmarkClamp(b *testing.B, fn func(dst, src []float32, lo, hi float32)) {
	dst, src, _ := benchInputs(benchN)
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		fn(dst, src, 100, 3000)
	}
}

func BenchmarkClampScalar4096(b *testing.B) { benchmarkClamp(b, scalarClampFloat32) }

func BenchmarkClampAVX2_4096(b *testing.B) {
	requireAVX2Bench(b)
	benchmarkClamp(b, clampFloat32AVX2)
}
