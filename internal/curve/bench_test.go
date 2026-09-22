package curve

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// The search is the whole cost of these kernels, so the benchmarks
// measure the three things that could change which search to use: how
// big the table is, how the cells are distributed over it, and how wide
// a row is.
//
//	go test -run - -bench . ./internal/curve
//	GOEXPERIMENT=simd go test -run - -bench . ./internal/curve

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
// honest case for a search that ignores its neighbours; clustered keeps
// every cell in one class, which flatters any search whose branches then
// predict perfectly; edges puts every cell outside the table, the
// shortest and the longest path through the scan; smooth is a correlated
// walk, the way a row of a computed surface moves, which is the only case
// a search that starts from the previous cell's entry could win.
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
	{"smooth", func(dst []float32, rng *rand.Rand) {
		// Steps of about half a percent of the table's span, reflected
		// at its ends, so a cell usually shares its neighbour's entry and
		// the walk still visits every entry many times over a row.
		x := 50.0
		for i := range dst {
			x += rng.NormFloat64() * 0.5
			if x < 0 {
				x = -x
			}
			if x > 100 {
				x = 200 - x
			}
			dst[i] = float32(x)
		}
	}},
}

// form is one way of computing a kernel, for the benchmarks to compare:
// breaks and values for Reclass, xs and ys for Lookup.
type form struct {
	name string
	run  func(dst, src, a, b []float32)
}

// reclassForms and lookupForms are the forms BenchmarkTableSize runs.
// The first of each is the scalar kernel; a build with a vector backend
// appends its kernels and the candidates they were chosen from
// (bench_amd64_test.go), so the comparison runs in one binary.
var (
	reclassForms = []form{
		{"break", scalarReclassFloat32},
		{"count", reclassCount},
		{"binary", reclassBinary},
		{"prev", reclassPrev},
	}
	lookupForms = []form{
		{"scan", scalarLookupFloat32},
		{"prev", lookupPrev},
	}
)

// tableSizes spans the one-entry table to twice the longest a model
// uses, far enough to find where each form stops paying.
var tableSizes = []int{1, 2, 4, 8, 16, 32, 64}

func benchSrc(n int, fill func([]float32, *rand.Rand)) []float32 {
	src := make([]float32, n)
	fill(src, rand.New(rand.NewPCG(21, 22)))
	return src
}

// BenchmarkTableSize is the measurement DESIGN.md §50 leaves the choice
// of search to: where, if anywhere, the forward scan stops beating a
// binary search over unsorted cells, whether starting from the previous
// cell's entry pays on correlated cells, and — in a build with a vector
// backend — where the lanes' walk over the whole table stops beating the
// scan, which sets the vector kernels' dispatch limits. The alternatives
// live in test files only, so the comparison never costs the kernels
// anything.
//
// A Reclass table of n breaks and a Lookup table of n knots are the same
// n entries, so the two operations' rows line up.
func BenchmarkTableSize(b *testing.B) {
	const n = 1 << 16
	dst := make([]float32, n)
	for _, op := range []struct {
		name, entries string
		forms         []form
		table         func(size int) (a, b []float32)
	}{
		{"Reclass", "breaks", reclassForms, table},
		{"Lookup", "knots", lookupForms, func(size int) (xs, ys []float32) {
			xs, ys = table(size)
			return xs, ys[:size]
		}},
	} {
		for _, d := range distributions {
			src := benchSrc(n, d.fill)
			for _, size := range tableSizes {
				ta, tb := op.table(size)
				for _, f := range op.forms {
					b.Run(fmt.Sprintf("op=%s/dist=%s/%s=%d/scan=%s", op.name, d.name, op.entries, size, f.name), func(b *testing.B) {
						for b.Loop() {
							f.run(dst, src, ta, tb)
						}
						b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mcells/s")
					})
				}
			}
		}
	}
}

// BenchmarkWidth separates the per-call cost from the per-cell cost, as
// internal/vec's does: the engine calls these once per row of a tile
// when the views are not compact, so a large per-call cost — the vector
// kernels expand their table on every call — would make narrow tiles
// slower. It runs each backend the build has.
func BenchmarkWidth(b *testing.B) {
	const maxN = 4096
	src := benchSrc(maxN, distributions[0].fill)
	dst := make([]float32, maxN)
	breaks, values := table(4)
	xs, ys := table(6)
	defer UseScalar(false)
	backends := []string{"scalar"}
	if simdKernels != nil {
		backends = append(backends, "simd")
	}
	for _, backend := range backends {
		UseScalar(backend == "scalar")
		for _, width := range []int{8, 16, 64, 256, 1024, 4096} {
			b.Run(fmt.Sprintf("op=Reclass/backend=%s/width=%d", backend, width), func(b *testing.B) {
				for b.Loop() {
					Reclass(dst[:width], src[:width], breaks, values)
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(width)/float64(b.N), "ns/cell")
			})
			b.Run(fmt.Sprintf("op=Lookup/backend=%s/width=%d", backend, width), func(b *testing.B) {
				for b.Loop() {
					Lookup(dst[:width], src[:width], xs, ys[:len(xs)])
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(width)/float64(b.N), "ns/cell")
			})
		}
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

// reclassPrev tries the previous cell's class before scanning: the
// idea DESIGN.md §50 records as safe, because a class is a pure function
// of the cell, but unmeasured. It costs two compares that hit on a
// correlated row, and a loop-carried dependency and one more branch
// everywhere else.
func reclassPrev(dst, src, breaks, values []float32) {
	dst = dst[:len(src)]
	values = values[:len(breaks)+1]
	k := 0
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		if (k == 0 || v >= breaks[k-1]) && (k == len(breaks) || v < breaks[k]) {
			dst[i] = values[k]
			continue
		}
		k = len(breaks)
		for j, b := range breaks {
			if v < b {
				k = j
				break
			}
		}
		dst[i] = values[k]
	}
}

// lookupPrev is scalarLookupFloat32 with reclassPrev's first guess.
func lookupPrev(dst, src, xs, ys []float32) {
	dst = dst[:len(src)]
	ys = ys[:len(xs)]
	k := 0
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		if !((k == 0 || v >= xs[k-1]) && (k == len(xs) || v < xs[k])) {
			k = len(xs)
			for j, x := range xs {
				if v < x {
					k = j
					break
				}
			}
		}
		switch {
		case k == 0:
			dst[i] = ys[0]
		case k == len(xs):
			dst[i] = ys[len(ys)-1]
		default:
			x0, y0 := xs[k-1], ys[k-1]
			if v == x0 {
				dst[i] = y0
				continue
			}
			t := (v - x0) / (xs[k] - x0)
			dst[i] = y0 + float32(t*(ys[k]-y0))
		}
	}
}

// TestBenchmarkScansAgree keeps the alternatives honest: a benchmark
// comparing kernels that compute different things measures nothing. It
// covers every form BenchmarkTableSize runs, the vector candidates
// included when the build appends them.
func TestBenchmarkScansAgree(t *testing.T) {
	var src []float32
	for _, d := range distributions {
		src = append(src, benchSrc(1000, d.fill)...)
	}
	src = append(src, nan, inf, ninf, negZero, 0, 100, -1, 101)
	want, got := make([]float32, len(src)), make([]float32, len(src))
	for _, size := range append([]int{0}, tableSizes...) {
		breaks, values := table(size)
		scalarReclassFloat32(want, src, breaks, values)
		for _, f := range reclassForms[1:] {
			f.run(got, src, breaks, values)
			assertSlicesEqual(t, fmt.Sprintf("Reclass %s/breaks=%d", f.name, size), got, want)
		}
		if size == 0 {
			continue
		}
		// Every knot's own x, so the knot short circuit is exercised.
		src := append(src, breaks...)
		want, got := make([]float32, len(src)), make([]float32, len(src))
		xs, ys := breaks, values[:size]
		scalarLookupFloat32(want, src, xs, ys)
		for _, f := range lookupForms[1:] {
			f.run(got, src, xs, ys)
			assertSlicesEqual(t, fmt.Sprintf("Lookup %s/knots=%d", f.name, size), got, want)
		}
	}
}
