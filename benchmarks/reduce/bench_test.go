package reduce_test

import (
	"context"
	"testing"

	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/accum"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// Run the suite with, e.g.:
//
//	go test ./benchmarks/reduce -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	GOEXPERIMENT=simd go test ./benchmarks/reduce -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var (
	vecKernels   = suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar}
	accumKernels = suite.Kernels{Name: "accum", Backend: accum.Backend, UseScalar: accum.UseScalar}
	// kernels switches both packages at once: a reduction's backend is
	// whichever the two of them are running.
	kernels = suite.Kernels{
		Name:    "vec+accum",
		Backend: vec.Backend,
		UseScalar: func(scalar bool) {
			vec.UseScalar(scalar)
			accum.UseScalar(scalar)
		},
	}
)

func TestMain(m *testing.M) { suite.Main(m, vecKernels, accumKernels) }

// fixture is one size's DEM and its mask, attached per case.
type fixture struct {
	size  int
	dem   []float32
	valid []uint64
}

func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, dem: make([]float32, n)}
	suite.FillDEM(f.dem, size, 1)
	f.valid = suite.RandomMask(n, 4, 0.1)
	return f
}

func (f *fixture) raster(masked bool) raster.Float32Raster {
	r := raster.NewFloat32(f.size, f.size, f.dem)
	if masked {
		r.Valid = f.valid
	}
	return r
}

// op runs one reduction through the engine.
type op func(ctx context.Context, r raster.Float32Raster, opts engine.Options) error

var (
	opCount = op(func(ctx context.Context, r raster.Float32Raster, opts engine.Options) error {
		_, err := reduce.CountTiled(ctx, r, opts)
		return err
	})
	opMinMax = op(func(ctx context.Context, r raster.Float32Raster, opts engine.Options) error {
		_, _, _, err := reduce.MinMaxTiled(ctx, r, opts)
		return err
	})
	opSum = op(func(ctx context.Context, r raster.Float32Raster, opts engine.Options) error {
		_, _, err := reduce.SumTiled(ctx, r, opts)
		return err
	})
	opStats = op(func(ctx context.Context, r raster.Float32Raster, opts engine.Options) error {
		_, err := reduce.StatsTiled(ctx, r, opts)
		return err
	})
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	r := f.raster(c.Masked)
	opts := engine.Options{Workers: c.Workers}
	bytes := 4.0
	if c.Masked {
		bytes += 1.0 / 8
	}
	return suite.Workload{
		Run: func() {
			if err := o(context.Background(), r, opts); err != nil {
				panic(err)
			}
		},
		BytesPerCell: bytes,
	}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  kernels,
		Fixture:  newFixture,
		Workload: o.workload,
		Workers:  suite.Workers(),
	})
}

func BenchmarkCount(b *testing.B)  { opCount.bench(b) }
func BenchmarkMinMax(b *testing.B) { opMinMax.bench(b) }
func BenchmarkSum(b *testing.B)    { opSum.bench(b) }
func BenchmarkStats(b *testing.B)  { opStats.bench(b) }

// TestRuns runs every operation and case once on a small raster, so a
// broken workload fails go test without running benchmarks. Reductions
// allocate a few small slices per call, by design (package reduce's
// documentation), so there is no zero-allocation check here.
func TestRuns(t *testing.T) {
	defer kernels.UseScalar(false)
	f := newFixture(64)
	for _, o := range []op{opCount, opMinMax, opSum, opStats} {
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				kernels.UseScalar(backend == suite.Scalar)
				for _, w := range suite.Workers() {
					o.workload(f, suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: w}).Run()
				}
			}
		}
	}
}
