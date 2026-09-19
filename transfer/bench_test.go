package transfer_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/transfer"
)

// Benchmarks for the four operations at 1024² and 4096², with and
// without a validity mask, compact and strided, and for the two
// table-driven ones over the table sizes a model actually uses.
//
// The split that matters here is Rescale against the rest: it runs
// internal/vec.Affine, which has an AVX2 backend, while Reclass and
// Lookup are a per-cell search in every build. These numbers are what a
// decision to vectorize the search would have to beat
// (internal/curve/bench_test.go compares the search forms themselves).
//
//	go test ./transfer -run '^$' -bench . -count 5
//	GOEXPERIMENT=simd go test ./transfer -run '^$' -bench . -count 5

var benchSizes = []int{1024, 4096}

func benchRaster(rng *rand.Rand, size int, strided, masked bool) raster.Float32Raster {
	stride := size
	if strided {
		stride = (size/64 + 1) * 64
	}
	r := raster.NewFloat32Stride(size, size, stride, make([]float32, (size-1)*stride+size))
	for i := range r.Data {
		// A slope-like spread over the tables below, which span 0 to 40.
		r.Data[i] = rng.Float32() * 45
	}
	if masked {
		r.Valid = raster.NewMask(len(r.Data))
		for i := range r.Data {
			if rng.IntN(10) == 0 {
				raster.MaskSet(r.Valid, i, false)
			}
		}
	}
	return r
}

// benchTable is a factor curve of n knots over 0 to 40 degrees, and the
// breaks and values of the matching classification.
func benchTable(n int) (xs, ys, breaks, values []float32) {
	xs = make([]float32, n)
	ys = make([]float32, n)
	for i := range xs {
		xs[i] = float32(i) * (40 / float32(n-1))
		ys[i] = 1 + float32(i)*0.35
	}
	breaks = xs[1:]
	values = ys
	return xs, ys, breaks, values
}

func reportCells(b *testing.B, size int) {
	b.ReportMetric(float64(size)*float64(size)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mcells/s")
}

// BenchmarkOps times each operation over the layouts and mask settings
// algebra's benchmarks use, with a five-knot table: the size a factor
// curve or a five-class scale actually has.
func BenchmarkOps(b *testing.B) {
	rng := rand.New(rand.NewPCG(31, 32))
	xs, ys, breaks, values := benchTable(5)
	b.Logf("backend: %s", vec.Backend())
	for _, size := range benchSizes {
		for _, strided := range []bool{false, true} {
			for _, masked := range []bool{false, true} {
				src := benchRaster(rng, size, strided, masked)
				dst := benchRaster(rng, size, strided, masked)
				ops := []struct {
					name string
					call func()
				}{
					{"Reclass", func() { transfer.Reclass(dst, src, breaks, values) }},
					{"Lookup", func() { transfer.Lookup(dst, src, xs, ys) }},
					{"Rescale", func() { transfer.Rescale(dst, src, 0.025, -1) }},
					{"RescaleRange", func() { transfer.RescaleRange(dst, src, 0, 40, 0, 1) }},
				}
				for _, op := range ops {
					name := fmt.Sprintf("op=%s/size=%d/strided=%v/mask=%v", op.name, size, strided, masked)
					b.Run(name, func(b *testing.B) {
						for b.Loop() {
							op.call()
						}
						reportCells(b, size)
					})
				}
			}
		}
	}
}

// BenchmarkTableSize is how the cost of the two table-driven operations
// grows with the table, at the raster scale the engine sees rather than
// the flat-slice scale internal/curve measures.
func BenchmarkTableSize(b *testing.B) {
	const size = 1024
	rng := rand.New(rand.NewPCG(33, 34))
	src := benchRaster(rng, size, false, false)
	dst := benchRaster(rng, size, false, false)
	for _, n := range []int{2, 5, 9, 17, 33} {
		xs, ys, breaks, values := benchTable(n)
		b.Run(fmt.Sprintf("op=Reclass/knots=%d", n), func(b *testing.B) {
			for b.Loop() {
				transfer.Reclass(dst, src, breaks, values)
			}
			reportCells(b, size)
		})
		b.Run(fmt.Sprintf("op=Lookup/knots=%d", n), func(b *testing.B) {
			for b.Loop() {
				transfer.Lookup(dst, src, xs, ys)
			}
			reportCells(b, size)
		})
	}
}
