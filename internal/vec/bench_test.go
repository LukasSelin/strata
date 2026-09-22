package vec

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkWidth times one call of a few kernels at widths from a lane to
// a 4096-cell row, to separate the per-call cost from the per-cell cost.
// The engine calls them once per row of a tile when views are not
// compact, so a large per-call cost would make narrow tiles slower.
//
//	GOEXPERIMENT=simd go test -run - -bench Width ./internal/vec
func BenchmarkWidth(b *testing.B) {
	const maxN = 4096
	rng := rand.New(rand.NewPCG(11, 12))
	a, c, dst := make([]float32, maxN), make([]float32, maxN), make([]float32, maxN)
	for i := range a {
		a[i] = float32(rng.NormFloat64() * 100)
		c[i] = float32(rng.NormFloat64() * 100)
	}
	ops := []struct {
		name string
		call func(n int)
	}{
		{"clamp", func(n int) { Clamp(dst[:n], a[:n], -50, 50) }},
		{"add", func(n int) { Add(dst[:n], a[:n], c[:n]) }},
		{"max", func(n int) { Max(dst[:n], a[:n], c[:n]) }},
		{"affine", func(n int) { Affine(dst[:n], a[:n], 0.25, -3) }},
		{"subdiv", func(n int) { SubDiv(dst[:n], a[:n], -3, 7) }},
	}
	b.Logf("backend: %s", Backend())
	for _, op := range ops {
		for _, n := range []int{8, 12, 16, 64, 254, 256, 1024, 4096} {
			b.Run(fmt.Sprintf("op=%s/width=%d", op.name, n), func(b *testing.B) {
				for b.Loop() {
					op.call(n)
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n)/float64(b.N), "ns/cell")
			})
		}
	}
}
