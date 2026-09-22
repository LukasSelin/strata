package accum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/LukasSelin/strata/internal/vec"
)

// BenchmarkAccumulators is the accumulator decision of DESIGN.md §49: the
// candidates summing the same cells, beside a max fold, which reads every
// cell once and does almost nothing with it and so is the bandwidth bound
// a sum is measured against. Record results in
// benchmarks/reduce/RESULTS.md.
//
//	go test ./internal/accum -run '^$' -bench Accumulators -count 5
//	GOEXPERIMENT=simd go test ./internal/accum -run '^$' -bench Accumulators -count 5
func BenchmarkAccumulators(b *testing.B) {
	b.Logf("vec backend: %s", vec.Backend())
	var sink float64
	cands := []struct {
		name string
		run  func(xs []float32) float64
	}{
		{"max", func(xs []float32) float64 { return float64(vec.ReduceMax(float32(math.Inf(-1)), xs)) }},
		{"float64", func(xs []float32) float64 {
			var s float64
			for _, v := range xs {
				s += float64(v)
			}
			return s
		}},
		{"neumaier", neumaier},
		{"binned1", func(xs []float32) float64 {
			// One bin set: what the four interleaved sets buy.
			var s Sum
			s.n, s.pending = int64(len(xs)), int64(len(xs))
			for _, v := range xs {
				s.notNegZero |= addSum1(&s.bins[0], &s.specials, v)
			}
			return float64(s.bins[0][140])
		}},
		{"binned", func(xs []float32) float64 {
			var s Sum
			s.Add(xs)
			return float64(s.bins[0][140])
		}},
		{"moments", func(xs []float32) float64 {
			var m Moments
			m.Add(xs)
			return float64(m.sq[0][300])
		}},
	}
	for _, dist := range []string{"dem", "normal"} {
		for _, n := range []int{4096, 1 << 20, 1 << 26} {
			if n > 1<<20 && testing.Short() {
				continue
			}
			xs := make([]float32, n)
			r := rand.New(rand.NewPCG(5, 6))
			for i := range xs {
				xs[i] = generators[dist](r)
			}
			for _, c := range cands {
				b.Run(fmt.Sprintf("dist=%s/n=%d/acc=%s", dist, n, c.name), func(b *testing.B) {
					b.SetBytes(int64(4 * n))
					for b.Loop() {
						sink += c.run(xs)
					}
					b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n)/float64(b.N), "ns/cell")
				})
			}
		}
	}
	_ = sink
}

// BenchmarkResult times reading a result, which happens once per call.
func BenchmarkResult(b *testing.B) {
	var m Moments
	r := rand.New(rand.NewPCG(7, 8))
	xs := make([]float32, 4096)
	for i := range xs {
		xs[i] = generators["dem"](r)
	}
	m.Add(xs)
	for b.Loop() {
		_ = m.Mean()
		_ = m.StdDev()
	}
}

// neumaier is Neumaier's compensated sum in float64: near-exact, but its
// result still depends on the order of the cells.
func neumaier(xs []float32) float64 {
	var s, c float64
	for _, v := range xs {
		x := float64(v)
		t := s + x
		if math.Abs(s) >= math.Abs(x) {
			c += (s - t) + x
		} else {
			c += (x - t) + s
		}
		s = t
	}
	return s + c
}

// BenchmarkWorkers sums a 256 MiB raster from memory split across
// goroutines, beside a max fold, to show how close the exact
// accumulators get to the machine's read bandwidth once enough cores
// share the work, as the engine's workers do.
//
//	GOEXPERIMENT=simd go test ./internal/accum -run '^$' -bench Workers -count 3
func BenchmarkWorkers(b *testing.B) {
	if testing.Short() {
		b.Skip("256 MiB fixture")
	}
	const n = 1 << 26
	xs := make([]float32, n)
	r := rand.New(rand.NewPCG(5, 6))
	for i := range xs {
		xs[i] = generators["dem"](r)
	}
	runs := []struct {
		name string
		run  func(xs []float32)
	}{
		{"max", func(xs []float32) { vec.ReduceMax(0, xs) }},
		{"binned", func(xs []float32) { new(Sum).Add(xs) }},
		{"moments", func(xs []float32) { new(Moments).Add(xs) }},
	}
	for _, c := range runs {
		for _, w := range []int{1, 2, 4, 8, 12} {
			b.Run(fmt.Sprintf("acc=%s/workers=%d", c.name, w), func(b *testing.B) {
				b.SetBytes(4 * n)
				for b.Loop() {
					var wg sync.WaitGroup
					chunk := n / w
					for k := range w {
						wg.Go(func() { c.run(xs[k*chunk : (k+1)*chunk]) })
					}
					wg.Wait()
				}
			})
		}
	}
}
