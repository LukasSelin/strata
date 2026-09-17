//go:build goexperiment.simd && amd64

package simdbackend

import (
	"fmt"
	"math"
	"math/rand/v2"
	"simd"
	"simd/archsimd"
	"testing"
)

var (
	nan  = float32(math.NaN())
	inf  = float32(math.Inf(1))
	ninf = float32(math.Inf(-1))
)

func edge(n, offset int) []float32 {
	base := []float32{1, -1, 0, float32(math.Copysign(0, -1)), 2.5, -2.5,
		nan, inf, ninf, 100, -100, 0.001, -0.001, 17.25, -3e7}
	out := make([]float32, n)
	for i := range out {
		out[i] = base[(i+offset)%len(base)]
	}
	return out
}

// sameBits requires identical bits, except that any NaN matches any NaN.
func sameBits(t *testing.T, name string, got, want []float32) {
	t.Helper()
	for i := range want {
		g, w := got[i], want[i]
		if g != g && w != w {
			continue
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			t.Fatalf("%s: index %d: got %v (%#x), want %v (%#x)", name, i, g, math.Float32bits(g), w, math.Float32bits(w))
		}
	}
}

var sizes = []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100}

func TestMatchesScalar(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("no AVX2")
	}
	var zero simd.Float32s
	t.Logf("portable simd.Float32s width on this CPU: %d lanes", zero.Len())
	for _, n := range sizes {
		a, b := edge(n, 0), edge(n, 4)
		want, got := make([]float32, n), make([]float32, n)

		scalarAdd(want, a, b)
		for name, fn := range map[string]func(dst, a, b []float32){"arch": addArch, "archBCE": addArchBCE, "portable": addPortable} {
			clear(got)
			fn(got, a, b)
			sameBits(t, fmt.Sprintf("Add/%s/n=%d", name, n), got, want)
		}

		scalarClamp(want, a, -10, 10)
		for name, fn := range map[string]func(dst, src []float32, lo, hi float32){"arch": clampArch, "archBCE": clampArchBCE, "portable": clampPortable} {
			clear(got)
			fn(got, a, -10, 10)
			sameBits(t, fmt.Sprintf("Clamp/%s/n=%d", name, n), got, want)
		}

		up, mid, down := edge(n+2, 0), edge(n+2, 5), edge(n+2, 11)
		scalarSlopeRow(want, up, mid, down, 0.0125, 0.0125)
		for name, fn := range map[string]func(dst, up, mid, down []float32, kx, ky float32){"asm": slopeRowAsm, "arch": slopeRowArch, "archBCE": slopeRowArchBCE, "portable": slopeRowPortable} {
			clear(got)
			fn(got, up, mid, down, 0.0125, 0.0125)
			sameBits(t, fmt.Sprintf("SlopeRow/%s/n=%d", name, n), got, want)
		}
	}
}

// benchN is one 4096-cell row, matching the existing internal/vec benchmark
// and small enough to stay in L1/L2 so compute (not memory) dominates.
const benchN = 4096

func randRow(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 1))
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2000 - 100
	}
	return s
}

func BenchmarkAdd(b *testing.B) {
	a, x, dst := randRow(benchN, 1), randRow(benchN, 2), make([]float32, benchN)
	for _, v := range []struct {
		name string
		fn   func(dst, a, b []float32)
	}{{"scalar", scalarAdd}, {"archsimd", addArch}, {"archsimd-bce", addArchBCE}, {"portable", addPortable}} {
		b.Run(v.name, func(b *testing.B) {
			b.SetBytes(benchN * 4)
			for b.Loop() {
				v.fn(dst, a, x)
			}
		})
	}
}

func BenchmarkClamp(b *testing.B) {
	src, dst := randRow(benchN, 3), make([]float32, benchN)
	for _, v := range []struct {
		name string
		fn   func(dst, src []float32, lo, hi float32)
	}{{"scalar", scalarClamp}, {"archsimd", clampArch}, {"archsimd-bce", clampArchBCE}, {"portable", clampPortable}} {
		b.Run(v.name, func(b *testing.B) {
			b.SetBytes(benchN * 4)
			for b.Loop() {
				v.fn(dst, src, 0, 1000)
			}
		})
	}
}

func BenchmarkSlopeRow(b *testing.B) {
	up, mid, down := randRow(benchN+2, 4), randRow(benchN+2, 5), randRow(benchN+2, 6)
	dst := make([]float32, benchN)
	for _, v := range []struct {
		name string
		fn   func(dst, up, mid, down []float32, kx, ky float32)
	}{{"scalar", scalarSlopeRow}, {"asm", slopeRowAsm}, {"archsimd", slopeRowArch}, {"archsimd-bce", slopeRowArchBCE}, {"portable", slopeRowPortable}} {
		b.Run(v.name, func(b *testing.B) {
			b.SetBytes(benchN * 4)
			for b.Loop() {
				v.fn(dst, up, mid, down, 0.0125, 0.0125)
			}
		})
	}
}
