//go:build goexperiment.simd && (amd64 || arm64)

package resamprow

import (
	"math/rand/v2"
	"testing"
)

// withBackend runs f on the scalar kernels, then on the SIMD ones.
func withBackend(t *testing.T, f func(simd bool)) {
	t.Helper()
	if simdKernels == nil {
		t.Skip("no SIMD backend on this CPU")
	}
	defer UseScalar(false)
	UseScalar(true)
	f(false)
	UseScalar(false)
	f(true)
}

// TestSIMDMatchesScalar holds both passes to the scalar kernels bit for
// bit (any NaN matches any NaN, +0 ≠ -0) over random plans of every
// method and scale, edge values, every row count 1…19 so the lane
// blocks and scalar tails both run, column subranges, and footprints
// offset within wider source rows.
func TestSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	for iter := range 400 {
		m := []Method{Bilinear, Cubic, Lanczos, Average}[iter%4]
		p, sw, sh := randomAxes(rng, m)
		if p.X.Lo >= p.X.Hi {
			continue
		}
		pad := rng.IntN(9)
		stride := sw + pad + rng.IntN(5)
		src := randomData(rng, (sh-1)*stride+sw+pad, true)
		c0 := p.X.Lo + rng.IntN(p.X.Hi-p.X.Lo)
		c1 := c0 + 1 + rng.IntN(p.X.Hi-c0)
		fx0, fx1 := p.X.Footprint(c0, c1, false)
		rows := 1 + rng.IntN(min(19, sh))
		cw := c1 - c0
		tStride := cw + rng.IntN(3)
		var out [2][]float32
		withBackend(t, func(simd bool) {
			tt := make([]float32, (rows-1)*tStride+cw)
			for i := range tt {
				tt[i] = -7 // stale
			}
			HRows(tt, tStride, src[pad:], stride, rows, &p.X, c0, c1, fx0, make([]float32, HScratch(fx1-fx0)))
			out[b2i(simd)] = tt
		})
		for i := range out[0] {
			if !sameFloat(out[0][i], out[1][i]) {
				t.Fatalf("HRows %v iter %d: cell %d scalar %v, SIMD %v", m, iter, i, out[0][i], out[1][i])
			}
		}

		// VRow over the intermediate just computed, any tap count.
		n := 1 + rng.IntN(max(1, rows))
		w := make([]float32, n)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		if rows < n {
			continue
		}
		var v [2][]float32
		withBackend(t, func(simd bool) {
			d := make([]float32, cw)
			VRow(d, out[0], tStride, w)
			v[b2i(simd)] = d
		})
		for i := range v[0] {
			if !sameFloat(v[0][i], v[1][i]) {
				t.Fatalf("VRow iter %d: column %d scalar %v, SIMD %v", iter, i, v[0][i], v[1][i])
			}
		}
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
