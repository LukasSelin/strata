package focal_test

import (
	"testing"

	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/raster"
)

// Run the suite with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/focal -run '^$' -bench . -count 5 -timeout 6h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var focalKernels = suite.Kernels{Name: "focalrow", Backend: focalrow.Backend, UseScalar: focalrow.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, focalKernels) }

// fixture holds a DEM, one output and their masks for one size.
type fixture struct {
	size     int
	src, out []float32
	srcValid []uint64
	outValid []uint64
}

func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, src: make([]float32, n), out: make([]float32, n)}
	suite.FillDEM(f.src, size, 1)
	suite.FillUniform(f.out, 3, 0, 1) // fault the output's pages in before timing
	f.srcValid = suite.RandomMask(n, 4, 0.1)
	f.outValid = raster.NewMask(n)
	return f
}

func (f *fixture) rasters(masked bool) (dst, src raster.Float32Raster) {
	dst, src = raster.NewFloat32(f.size, f.size, f.out), raster.NewFloat32(f.size, f.size, f.src)
	if masked {
		dst.Valid, src.Valid = f.outValid, f.srcValid
	}
	return dst, src
}

// op is one focal operation at one radius.
type op func(dst, src raster.Float32Raster)

// weights are (2r+1)² weights summing to 1, not all equal, so no
// kernel can take a shortcut for a box.
func weights(r int) []float32 {
	k := 2*r + 1
	w := make([]float32, k*k)
	var sum float32
	for i := range w {
		w[i] = float32(1 + i%3)
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	return w
}

func correlate(r int) op {
	o := focal.WeightsOptions{Radius: r, Weights: weights(r)}
	return func(dst, src raster.Float32Raster) { focal.Correlate(dst, src, o) }
}

func gaussian(r int) op {
	g := focal.Gaussian(r, float64(r)/2)
	o := focal.SeparableOptions{Radius: r, Row: g, Col: g}
	return func(dst, src raster.Float32Raster) { focal.CorrelateSeparable(dst, src, o) }
}

func mean(r int) op {
	return func(dst, src raster.Float32Raster) { focal.Mean(dst, src, focal.BoxOptions{Radius: r}) }
}

func minimum(r int) op {
	return func(dst, src raster.Float32Raster) { focal.Min(dst, src, focal.BoxOptions{Radius: r}) }
}

func maximum(r int) op {
	return func(dst, src raster.Float32Raster) { focal.Max(dst, src, focal.BoxOptions{Radius: r}) }
}

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	dst, src := f.rasters(c.Masked)
	// The input and the output, whatever the radius: the neighbourhood's
	// other rows come from cache.
	bytes := 8.0
	if c.Masked {
		bytes += 2.0 / 8
	}
	return suite.Workload{Run: func() { o(dst, src) }, BytesPerCell: bytes}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  focalKernels,
		Fixture:  newFixture,
		Workload: o.workload,
	})
}

func BenchmarkCorrelateR1(b *testing.B) { correlate(1).bench(b) }
func BenchmarkCorrelateR2(b *testing.B) { correlate(2).bench(b) }
func BenchmarkCorrelateR3(b *testing.B) { correlate(3).bench(b) }
func BenchmarkCorrelateR5(b *testing.B) { correlate(5).bench(b) }

func BenchmarkGaussianR1(b *testing.B) { gaussian(1).bench(b) }
func BenchmarkGaussianR2(b *testing.B) { gaussian(2).bench(b) }
func BenchmarkGaussianR3(b *testing.B) { gaussian(3).bench(b) }
func BenchmarkGaussianR5(b *testing.B) { gaussian(5).bench(b) }

func BenchmarkMeanR1(b *testing.B) { mean(1).bench(b) }
func BenchmarkMeanR2(b *testing.B) { mean(2).bench(b) }
func BenchmarkMeanR3(b *testing.B) { mean(3).bench(b) }
func BenchmarkMeanR5(b *testing.B) { mean(5).bench(b) }

func BenchmarkMinR1(b *testing.B) { minimum(1).bench(b) }
func BenchmarkMinR2(b *testing.B) { minimum(2).bench(b) }
func BenchmarkMinR3(b *testing.B) { minimum(3).bench(b) }
func BenchmarkMinR5(b *testing.B) { minimum(5).bench(b) }

func BenchmarkMaxR3(b *testing.B) { maximum(3).bench(b) }

// TestAllocs checks every operation on a small raster against a bound
// that does not grow with the raster: the plain functions run through
// the engine's one-worker path, which allocates a few slices per call,
// and the separable kernels' row of scratch comes from a pool. It runs
// without -bench, so a regression fails go test.
func TestAllocs(t *testing.T) {
	defer focalKernels.UseScalar(false)
	const limit = 16
	ops := map[string]op{
		"CorrelateR3": correlate(3), "GaussianR3": gaussian(3), "MeanR3": mean(3),
		"MinR3": minimum(3), "MaxR3": maximum(3),
	}
	f := newFixture(64)
	for name, o := range ops {
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: 1}
				focalKernels.UseScalar(backend == suite.Scalar)
				w := o.workload(f, c)
				w.Run() // fill the scratch pool
				if allocs := testing.AllocsPerRun(10, w.Run); allocs > limit {
					t.Errorf("%s/%s: %v allocs/op, want at most %d", name, c.Name(), allocs, limit)
				}
			}
		}
	}
}
