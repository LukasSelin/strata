package curve

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// The scan is the whole cost of these kernels, so the benchmarks measure
// the three things that could change which scan to use: how big the
// table is, how the cells are distributed over it, and how wide a row is.
//
//	go test -run - -bench . ./internal/curve

// table builds n strictly increasing breaks spanning [0, 100] and the
// n+1 values they select.
func table(n int) (breaks, values []float32) {
	breaks = make([]float32, n)
	for i := range breaks {
		breaks[i] = float32(i+1) * (100 / float32(n+1))
	}
	values = make([]float32, n+1)
	for i := range values {
		values[i] = float32(i)
	}
	return breaks, values
}

// distributions are how cells fall across the table. Spread is the
// honest case; clustered keeps every cell in one class, which flatters
// any search whose branches then predict perfectly; edges puts every cell
// outside the table, the shortest and the longest path through the scan.
var distributions = []struct {
	name string
	fill func(dst []float32, rng *rand.Rand)
}{
	{"spread", func(dst []float32, rng *rand.Rand) {
		for i := range dst {
			dst[i] = rng.Float32() * 100
		}
	}},
	{"clustered", func(dst []float32, rng *rand.Rand) {
		for i := range dst {
			dst[i] = 50 + rng.Float32()
		}
	}},
	{"edges", func(dst []float32, rng *rand.Rand) {
		for i := range dst {
			if rng.IntN(2) == 0 {
				dst[i] = -1
			} else {
				dst[i] = 101
			}
		}
	}},
}

func benchSrc(n int, fill func([]float32, *rand.Rand)) []float32 {
	src := make([]float32, n)
	fill(src, rand.New(rand.NewPCG(21, 22)))
	return src
}

// BenchmarkTableSize is the measurement DESIGN.md §50 leaves the choice
// of scan to: where, if anywhere, the forward scan stops beating a
// binary search over unsorted cells. scanBreak is the shipped kernel;
// scanCount and scanBinary live in this file only, so the comparison
// never costs the kernels anything.
func BenchmarkTableSize(b *testing.B) {
	const n = 1 << 16
	dst := make([]float32, n)
	forms := []struct {
		name string
		run  func(dst, src, breaks, values []float32)
	}{
		{"break", scalarReclassFloat32},
		{"count", reclassCount},
		{"binary", reclassBinary},
	}
	for _, d := range distributions {
		src := benchSrc(n, d.fill)
		for _, size := range []int{1, 2, 4, 8, 16, 32, 64} {
			breaks, values := table(size)
			for _, f := range forms {
				b.Run(fmt.Sprintf("dist=%s/breaks=%d/scan=%s", d.name, size, f.name), func(b *testing.B) {
					for b.Loop() {
						f.run(dst, src, breaks, values)
					}
					b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mcells/s")
				})
			}
		}
	}
}

// BenchmarkWidth separates the per-call cost from the per-cell cost, as
// internal/vec's does: the engine calls these once per row of a tile
// when the views are not compact, so a large per-call cost would make
// narrow tiles slower.
func BenchmarkWidth(b *testing.B) {
	const maxN = 4096
	src := benchSrc(maxN, distributions[0].fill)
	dst := make([]float32, maxN)
	breaks, values := table(4)
	xs, ys := table(6)
	for _, width := range []int{8, 16, 64, 256, 1024, 4096} {
		b.Run(fmt.Sprintf("op=Reclass/width=%d", width), func(b *testing.B) {
			for b.Loop() {
				Reclass(dst[:width], src[:width], breaks, values)
			}
			b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(width)/float64(b.N), "ns/cell")
		})
		b.Run(fmt.Sprintf("op=Lookup/width=%d", width), func(b *testing.B) {
			for b.Loop() {
				Lookup(dst[:width], src[:width], xs, ys[:len(xs)])
			}
			b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(width)/float64(b.N), "ns/cell")
		})
	}
}

// reclassCount is the branchless alternative to the shipped forward
// scan: it compares against every break and counts, so its cost does not
// depend on the data and a misprediction cannot happen, but it never
// stops early.
func reclassCount(dst, src, breaks, values []float32) {
	dst = dst[:len(src)]
	values = values[:len(breaks)+1]
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		k := 0
		for _, b := range breaks {
			if v >= b {
				k++
			}
		}
		dst[i] = values[k]
	}
}

// reclassBinary is the textbook alternative, whose branches are a coin
// flip at every level on cells that are not sorted.
func reclassBinary(dst, src, breaks, values []float32) {
	dst = dst[:len(src)]
	values = values[:len(breaks)+1]
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		lo, hi := 0, len(breaks)
		for lo < hi {
			mid := int(uint(lo+hi) >> 1)
			if v < breaks[mid] {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		dst[i] = values[lo]
	}
}

// TestBenchmarkScansAgree keeps the alternatives honest: a benchmark
// comparing kernels that compute different things measures nothing.
func TestBenchmarkScansAgree(t *testing.T) {
	src := benchSrc(1000, distributions[0].fill)
	src = append(src, nan, inf, ninf, negZero, 0)
	want, got := make([]float32, len(src)), make([]float32, len(src))
	for _, size := range []int{0, 1, 2, 4, 8, 16, 32, 64} {
		breaks, values := table(size)
		scalarReclassFloat32(want, src, breaks, values)
		for _, f := range []struct {
			name string
			run  func(dst, src, breaks, values []float32)
		}{{"count", reclassCount}, {"binary", reclassBinary}} {
			f.run(got, src, breaks, values)
			assertSlicesEqual(t, fmt.Sprintf("%s/breaks=%d", f.name, size), got, want)
		}
	}
}
