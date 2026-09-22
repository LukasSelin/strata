//go:build goexperiment.simd && (amd64 || arm64)

package vec

import (
	"math"
	"testing"
)

// simdCases returns slice lengths that straddle both vector lane widths
// (8 for AVX2, 4 for NEON): zero and one lane below it, exactly on it, one
// above it, and several lanes plus a partial tail, so the scalar-tail
// boundary in each SIMD kernel is actually exercised.
func simdCases() []int {
	return []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100}
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

func TestSIMDMatchesScalar(t *testing.T) {
	requireSIMD(t)

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
			simdTestKernels.add(got, a, b)
			assertSameBits(t, "Add", got, want)
		})
		t.Run("Sub", func(t *testing.T) {
			want := make([]float32, n)
			scalarSubFloat32(want, a, b)
			got := make([]float32, n)
			simdTestKernels.sub(got, a, b)
			assertSameBits(t, "Sub", got, want)
		})
		t.Run("Mul", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulFloat32(want, a, b)
			got := make([]float32, n)
			simdTestKernels.mul(got, a, b)
			assertSameBits(t, "Mul", got, want)
		})
		t.Run("Div", func(t *testing.T) {
			want := make([]float32, n)
			scalarDivFloat32(want, a, b)
			got := make([]float32, n)
			simdTestKernels.div(got, a, b)
			assertSameBits(t, "Div", got, want)
		})
		t.Run("Min", func(t *testing.T) {
			want := make([]float32, n)
			scalarMinFloat32(want, a, b)
			got := make([]float32, n)
			simdTestKernels.min(got, a, b)
			assertSameBits(t, "Min", got, want)
		})
		t.Run("Max", func(t *testing.T) {
			want := make([]float32, n)
			scalarMaxFloat32(want, a, b)
			got := make([]float32, n)
			simdTestKernels.max(got, a, b)
			assertSameBits(t, "Max", got, want)
		})
		t.Run("AddScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarAddScalarFloat32(want, a, 3.25)
			got := make([]float32, n)
			simdTestKernels.addScalar(got, a, 3.25)
			assertSameBits(t, "AddScalar", got, want)
		})
		t.Run("MulScalar", func(t *testing.T) {
			want := make([]float32, n)
			scalarMulScalarFloat32(want, a, -2)
			got := make([]float32, n)
			simdTestKernels.mulScalar(got, a, -2)
			assertSameBits(t, "MulScalar", got, want)
		})
		t.Run("Affine", func(t *testing.T) {
			for _, ab := range [][2]float32{{2, 0.5}, {-0.25, -3}, {0, 7}, {1, 0}, {nan, 1}, {1, nan}, {inf, 1}} {
				want := make([]float32, n)
				scalarAffineFloat32(want, a, ab[0], ab[1])
				got := make([]float32, n)
				simdTestKernels.affine(got, a, ab[0], ab[1])
				assertSameBits(t, "Affine", got, want)
			}
		})
		t.Run("SubDiv", func(t *testing.T) {
			for _, ls := range [][2]float32{{2, 0.5}, {-0.25, -3}, {0, 0}, {1, inf}, {nan, 1}, {1, nan}, {inf, 1}} {
				want := make([]float32, n)
				scalarSubDivFloat32(want, a, ls[0], ls[1])
				got := make([]float32, n)
				simdTestKernels.subDiv(got, a, ls[0], ls[1])
				assertSameBits(t, "SubDiv", got, want)
			}
		})
		t.Run("Clamp", func(t *testing.T) {
			want := make([]float32, n)
			scalarClampFloat32(want, a, -10, 10)
			got := make([]float32, n)
			simdTestKernels.clamp(got, a, -10, 10)
			assertSameBits(t, "Clamp", got, want)
		})
		t.Run("ClampSignedZeroBounds", func(t *testing.T) {
			for _, bounds := range [][2]float32{{negZero, 0}, {0, negZero}, {0, 0}, {negZero, negZero}, {nan, 1}, {-1, nan}} {
				want := make([]float32, n)
				scalarClampFloat32(want, a, bounds[0], bounds[1])
				got := make([]float32, n)
				simdTestKernels.clamp(got, a, bounds[0], bounds[1])
				assertSameBits(t, "Clamp", got, want)
			}
		})
		t.Run("Abs", func(t *testing.T) {
			want := make([]float32, n)
			scalarAbsFloat32(want, a)
			got := make([]float32, n)
			simdTestKernels.abs(got, a)
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
			simdTestKernels.sqrt(got, a)
			assertSameBits(t, "Sqrt", got, want)
		})
	}
}

// TestSIMDInPlace mirrors TestInPlace but forces the SIMD path directly,
// since dst aliasing a or src must remain safe when the bulk of the
// work runs through vector loads/stores rather than the scalar loop.
func TestSIMDInPlace(t *testing.T) {
	requireSIMD(t)

	a := make([]float32, 20)
	b := make([]float32, 20)
	want := make([]float32, 20)
	for i := range a {
		a[i] = float32(i) + 0.5
		b[i] = float32(i) * 2
	}
	scalarAddFloat32(want, a, b)
	simdTestKernels.add(a, a, b)
	assertSlicesEqual(t, "Add in-place SIMD", a, want)
}

// TestSIMDReduceMatchesScalar is DESIGN.md §15 for the folds: the vector
// kernels keep one accumulator vector and combine its lanes in a different
// order from the scalar loop, so this checks the answer is the same bits
// anyway — which it is because min and max are associative and
// commutative over NaN and signed zeros alike. Lengths straddle the lane
// width so the scalar tail and the shorter-than-a-lane path both run, and
// the special values are rotated through every lane position.
func TestSIMDReduceMatchesScalar(t *testing.T) {
	requireSIMD(t)
	for _, n := range simdCases() {
		for rot := range 9 {
			src := rotated(edgeValues(n), rot)
			for _, acc := range []float32{inf, ninf, 0, negZero, nan, 1} {
				gotMin, wantMin := simdTestKernels.reduceMin(acc, src), scalarReduceMinFloat32(acc, src)
				if !sameBits(gotMin, wantMin) {
					t.Fatalf("ReduceMin n=%d rot=%d acc=%v: %v (%#08x), want %v (%#08x)",
						n, rot, acc, gotMin, math.Float32bits(gotMin), wantMin, math.Float32bits(wantMin))
				}
				gotMax, wantMax := simdTestKernels.reduceMax(acc, src), scalarReduceMaxFloat32(acc, src)
				if !sameBits(gotMax, wantMax) {
					t.Fatalf("ReduceMax n=%d rot=%d acc=%v: %v (%#08x), want %v (%#08x)",
						n, rot, acc, gotMax, math.Float32bits(gotMax), wantMax, math.Float32bits(wantMax))
				}
			}
		}
	}
}

// TestSIMDReduceSplitsAnywhere checks the property the engine leans on
// against the vector kernels specifically: any split of a slice folds to
// the same bits as the whole, so tiles, bands and workers are free to
// divide a raster however they like.
func TestSIMDReduceSplitsAnywhere(t *testing.T) {
	requireSIMD(t)
	for n := range 40 {
		src := edgeValues(n)
		wholeMin := simdTestKernels.reduceMin(inf, src)
		wholeMax := simdTestKernels.reduceMax(ninf, src)
		for k := 0; k <= n; k++ {
			gotMin := simdTestKernels.reduceMin(simdTestKernels.reduceMin(inf, src[:k]), src[k:])
			if !sameBits(gotMin, wholeMin) {
				t.Fatalf("n=%d split at %d: min %v, want %v", n, k, gotMin, wholeMin)
			}
			gotMax := simdTestKernels.reduceMax(simdTestKernels.reduceMax(ninf, src[:k]), src[k:])
			if !sameBits(gotMax, wholeMax) {
				t.Fatalf("n=%d split at %d: max %v, want %v", n, k, gotMax, wholeMax)
			}
		}
	}
}

// TestSIMDMinMaxEdgePairs runs every ordered pair of special values
// through the kernels built on lanewise min and max. AVX2 has to repair
// VMINPS/VMAXPS for a NaN first operand and for signed zeros; NEON's
// FMIN/FMAX are meant to match Go's builtins as they are. Either way the
// bits must be the scalar ones, in every lane position.
func TestSIMDMinMaxEdgePairs(t *testing.T) {
	requireSIMD(t)
	edges := []float32{nan, inf, ninf, 0, negZero, 1, -1, math.MaxFloat32, -math.MaxFloat32, 0x1p-149}
	var a, b []float32
	for _, x := range edges {
		for _, y := range edges {
			a, b = append(a, x), append(b, y)
		}
	}
	for rot := range 8 {
		ra, rb := rotated(a, rot), rotated(b, rot)
		want, got := make([]float32, len(ra)), make([]float32, len(ra))
		scalarMinFloat32(want, ra, rb)
		simdTestKernels.min(got, ra, rb)
		assertSameBits(t, "Min", got, want)
		scalarMaxFloat32(want, ra, rb)
		simdTestKernels.max(got, ra, rb)
		assertSameBits(t, "Max", got, want)
		for _, lo := range edges {
			for _, hi := range edges {
				scalarClampFloat32(want, ra, lo, hi)
				simdTestKernels.clamp(got, ra, lo, hi)
				assertSameBits(t, "Clamp", got, want)
			}
		}
		for _, acc := range edges {
			if g, w := simdTestKernels.reduceMin(acc, ra), scalarReduceMinFloat32(acc, ra); !sameBits(g, w) {
				t.Fatalf("ReduceMin rot=%d acc=%v: %v, want %v", rot, acc, g, w)
			}
			if g, w := simdTestKernels.reduceMax(acc, rb), scalarReduceMaxFloat32(acc, rb); !sameBits(g, w) {
				t.Fatalf("ReduceMax rot=%d acc=%v: %v, want %v", rot, acc, g, w)
			}
		}
	}
}

// TestBackendSelection checks that the SIMD set is installed at init,
// that UseScalar(true) replaces all of it, and that UseScalar(false)
// puts every kernel back.
func TestBackendSelection(t *testing.T) {
	requireSIMD(t)
	defer UseScalar(false)
	if Backend() != simdTestName {
		t.Fatalf("Backend() = %q, want %s", Backend(), simdTestName)
	}
	assertKernels(t, "init", kernelsInUse(), *simdKernels)
	assertKernels(t, "SIMD set", *simdKernels, simdTestKernels)
	UseScalar(true)
	if Backend() != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", Backend())
	}
	assertKernels(t, "UseScalar(true)", kernelsInUse(), scalarKernels)
	UseScalar(false)
	if Backend() != simdTestName {
		t.Fatalf("after UseScalar(false), Backend() = %q, want %s", Backend(), simdTestName)
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

func BenchmarkAddScalar4096(b *testing.B) { benchmarkBinary(b, scalarAddFloat32) }

func BenchmarkAddSIMD4096(b *testing.B) {
	requireSIMD(b)
	benchmarkBinary(b, simdTestKernels.add)
}

func BenchmarkMinScalar4096(b *testing.B) { benchmarkBinary(b, scalarMinFloat32) }

func BenchmarkMinSIMD4096(b *testing.B) {
	requireSIMD(b)
	benchmarkBinary(b, simdTestKernels.min)
}

func benchmarkClamp(b *testing.B, fn func(dst, src []float32, lo, hi float32)) {
	dst, src, _ := benchInputs(benchN)
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		fn(dst, src, 100, 3000)
	}
}

func BenchmarkClampScalar4096(b *testing.B) { benchmarkClamp(b, scalarClampFloat32) }

func BenchmarkClampSIMD4096(b *testing.B) {
	requireSIMD(b)
	benchmarkClamp(b, simdTestKernels.clamp)
}

func rotated(src []float32, by int) []float32 {
	if len(src) == 0 {
		return src
	}
	out := make([]float32, len(src))
	for i := range out {
		out[i] = src[(i+by)%len(src)]
	}
	return out
}
