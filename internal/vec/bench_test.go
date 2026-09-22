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

// BenchmarkChain times an n-input product run as one fused Chain against
// the same product run one kernel at a time through a buffer, which is
// what an unfused pipeline does (DESIGN.md §29, §52). The widths are the
// run lengths the engine actually calls with: a row of a narrow tile at
// one end, a whole band at the other. The buffer the staged form carries
// its value in is the width, so it leaves L1 where the fused form never
// touches memory at all.
//
//	GOEXPERIMENT=simd go test -run - -bench Chain ./internal/vec
func BenchmarkChain(b *testing.B) {
	const maxN = 1 << 16
	rng := rand.New(rand.NewPCG(21, 22))
	b.Logf("backend: %s", Backend())

	for _, steps := range []int{1, 2, 5} {
		inputs := steps + 1
		srcs := make([][]float32, inputs)
		for i := range srcs {
			srcs[i] = make([]float32, maxN)
			for j := range srcs[i] {
				srcs[i][j] = float32(rng.NormFloat64())
			}
		}
		dst := make([]float32, maxN)
		acc := make([]float32, maxN)

		chainSteps := make([]Step, steps)
		for i := range chainSteps {
			chainSteps[i] = Step{Op: OpMul, Src: i + 1}
		}
		c := NewChain(inputs, 0, chainSteps)
		last := steps - 1

		for _, n := range []int{256, 4096, 1 << 16} {
			run := make([][]float32, inputs)
			for i := range run {
				run[i] = srcs[i][:n]
			}
			for _, form := range []struct {
				name string
				call func()
			}{
				{"fused", func() { c.Run(dst[:n], run) }},
				// The staged arm is the best an unfused chain can do:
				// the first step reads the inputs and the last writes
				// dst, so it copies nothing, and it reuses one buffer
				// where a Pipeline carries one per intermediate.
				{"staged", func() {
					out := acc[:n]
					for i, s := range chainSteps {
						in := acc[:n]
						if i == 0 {
							in = run[0]
						}
						if i == last {
							out = dst[:n]
						}
						Mul(out, in, run[s.Src])
					}
				}},
			} {
				b.Run(fmt.Sprintf("steps=%d/width=%d/form=%s", steps, n, form.name), func(b *testing.B) {
					for b.Loop() {
						form.call()
					}
					b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n)/float64(b.N), "ns/cell")
				})
			}
		}
	}
}
