//go:build goexperiment.simd && amd64

package stencil

import (
	"math"
	"math/rand/v2"
	"simd/archsimd"
	"testing"
)

func requireAVX2(t testing.TB) {
	t.Helper()
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not available on this CPU")
	}
}

// sameBits compares bit patterns, except that any NaN matches any NaN.
func sameBits(a, b float32) bool {
	if a != a && b != b {
		return true
	}
	return math.Float32bits(a) == math.Float32bits(b)
}

// rowValues fills s with smooth terrain plus, with probability special,
// values from a list of hazards: NaN, ±Inf, ±0, huge and tiny numbers.
func rowValues(rng *rand.Rand, s []float32, special float64) {
	hazards := []float32{
		float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		0, float32(math.Copysign(0, -1)), math.MaxFloat32, -math.MaxFloat32,
		1e30, -1e30, math.SmallestNonzeroFloat32, 1e-30,
	}
	for i := range s {
		if rng.Float64() < special {
			s[i] = hazards[rng.IntN(len(hazards))]
		} else {
			s[i] = float32(rng.NormFloat64()*50 + 500)
		}
	}
}

func TestAVX2RowsMatchScalar(t *testing.T) {
	requireAVX2(t)
	rng := rand.New(rand.NewPCG(5, 6))
	type params struct{ kx, ky, scale float32 }
	ps := []params{{0.0125, 0.0125, 1}, {0.05, 0.025, float32(180 / math.Pi)}, {1, 1, 100}, {-3, 7e-5, 1}}
	for n := 0; n <= 120; n++ {
		for _, special := range []float64{0, 0.1, 0.5} {
			for rep := range 3 {
				r := make([][]float32, 3)
				for k := range r {
					// Extra cells past n+2 must not be read.
					r[k] = make([]float32, n+2+rep)
					rowValues(rng, r[k], special)
				}
				for _, p := range ps {
					for _, atan := range []bool{false, true} {
						want := make([]float32, n)
						got := make([]float32, n)
						scalarHornSlopeRow(want, r[0], r[1], r[2], p.kx, p.ky, p.scale, atan)
						hornSlopeRowAVX2(got, r[0], r[1], r[2], p.kx, p.ky, p.scale, atan)
						for i := range want {
							if !sameBits(got[i], want[i]) {
								t.Fatalf("slope n=%d atan=%v %+v cell %d: got %g (%#x), want %g (%#x)",
									n, atan, p, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
							}
						}
					}
					wx, wy := make([]float32, n), make([]float32, n)
					gx, gy := make([]float32, n), make([]float32, n)
					scalarHornGradientRow(wx, wy, r[0], r[1], r[2], p.kx, p.ky)
					hornGradientRowAVX2(gx, gy, r[0], r[1], r[2], p.kx, p.ky)
					for i := range wx {
						if !sameBits(gx[i], wx[i]) || !sameBits(gy[i], wy[i]) {
							t.Fatalf("gradient n=%d %+v cell %d: got (%g, %g), want (%g, %g)",
								n, p, i, gx[i], gy[i], wx[i], wy[i])
						}
					}
				}
			}
		}
	}
}

// TestAtan8MatchesAtan32 checks the vector arctangent lane by lane over
// a stride through every non-negative float32, the reduction thresholds
// and the special values.
func TestAtan8MatchesAtan32(t *testing.T) {
	requireAVX2(t)
	var xs []float32
	for b := uint32(0); b <= 0x7F800000; b += 7919 {
		xs = append(xs, math.Float32frombits(b))
	}
	for _, th := range []float32{atanTanPi8, atanTan3Pi8, 1} {
		x := th
		for range 100 {
			x = math.Nextafter32(x, 0)
		}
		for range 200 {
			xs = append(xs, x)
			x = math.Nextafter32(x, 10)
		}
	}
	xs = append(xs, 0, float32(math.Inf(1)), float32(math.NaN()), math.MaxFloat32)
	for len(xs)%lane != 0 {
		xs = append(xs, 1)
	}
	c := newAtanConsts()
	got := make([]float32, lane)
	for i := 0; i < len(xs); i += lane {
		store8(atan8(load8(xs[i:]), &c), got)
		for j, g := range got {
			if w := Atan32(xs[i+j]); !sameBits(g, w) {
				t.Fatalf("atan8(%g) = %g (%#x), Atan32 = %g (%#x)",
					xs[i+j], g, math.Float32bits(g), w, math.Float32bits(w))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

func TestBackendSelection(t *testing.T) {
	requireAVX2(t)
	defer UseScalar(false)
	if Backend() != "avx2" {
		t.Fatalf("Backend() = %q, want avx2", Backend())
	}
	UseScalar(true)
	if Backend() != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", Backend())
	}
}
