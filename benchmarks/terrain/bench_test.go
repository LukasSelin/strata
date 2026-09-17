package terrain_test

import (
	"testing"

	"strata/benchmarks/internal/suite"
	"strata/internal/stencil"
	"strata/raster"
	"strata/terrain"
)

// Run the suite with, e.g.:
//
//	go test ./benchmarks/terrain -run '^$' -bench . -count 5 -timeout 3h > bench.txt
//	GOEXPERIMENT=simd go test ./benchmarks/terrain -run '^$' -bench . -count 5 -timeout 3h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var stencilKernels = suite.Kernels{Name: "stencil", Backend: stencil.Backend, UseScalar: stencil.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, stencilKernels) }

// fixture holds a DEM and one output per result for one size. Gradient
// writes two, the other operations one, and only what the operation
// writes is allocated: at 16384² the second output is another 1 GiB.
// Masks are attached per case.
type fixture struct {
	size     int
	dem      []float32
	outs     [][]float32
	demValid []uint64
	outValid [][]uint64
}

func newFixture(size, outputs int) *fixture {
	n := size * size
	f := &fixture{size: size, dem: make([]float32, n)}
	suite.FillDEM(f.dem, size, 1)
	f.demValid = suite.RandomMask(n, 4, 0.1)
	for range outputs {
		out := make([]float32, n)
		suite.FillUniform(out, 3, 0, 1) // fault the output's pages in before timing
		f.outs = append(f.outs, out)
		f.outValid = append(f.outValid, raster.NewMask(n))
	}
	return f
}

// rasters returns the DEM and the outputs, masked or not.
func (f *fixture) rasters(masked bool) (dem raster.Float32Raster, outs []raster.Float32Raster) {
	dem = raster.NewFloat32(f.size, f.size, f.dem)
	if masked {
		dem.Valid = f.demValid
	}
	for i, data := range f.outs {
		r := raster.NewFloat32(f.size, f.size, data)
		if masked {
			r.Valid = f.outValid[i]
		}
		outs = append(outs, r)
	}
	return dem, outs
}

// op is one terrain operation over a DEM and its outputs.
type op struct {
	outputs int
	run     func(dem raster.Float32Raster, outs []raster.Float32Raster)
}

const cellSize = 10

var (
	opGradient = op{2, func(dem raster.Float32Raster, outs []raster.Float32Raster) {
		terrain.Gradient(outs[0], outs[1], dem, terrain.GradientOptions{CellSize: cellSize})
	}}
	opSlope = op{1, func(dem raster.Float32Raster, outs []raster.Float32Raster) {
		terrain.Slope(outs[0], dem, terrain.SlopeOptions{CellSize: cellSize})
	}}
	opAspect = op{1, func(dem raster.Float32Raster, outs []raster.Float32Raster) {
		terrain.Aspect(outs[0], dem, terrain.AspectOptions{CellSize: cellSize})
	}}
	opHillshade = op{1, func(dem raster.Float32Raster, outs []raster.Float32Raster) {
		terrain.Hillshade(outs[0], dem, terrain.HillshadeOptions{CellSize: cellSize})
	}}
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	dem, outs := f.rasters(c.Masked)
	operands := float64(o.outputs + 1) // the DEM and every output
	bytes := 4 * operands
	if c.Masked {
		bytes += operands / 8
	}
	return suite.Workload{
		Run:          func() { o.run(dem, outs) },
		BytesPerCell: bytes,
	}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  stencilKernels,
		Fixture:  func(size int) *fixture { return newFixture(size, o.outputs) },
		Workload: o.workload,
	})
}

func BenchmarkGradient(b *testing.B)  { opGradient.bench(b) }
func BenchmarkSlope(b *testing.B)     { opSlope.bench(b) }
func BenchmarkAspect(b *testing.B)    { opAspect.bench(b) }
func BenchmarkHillshade(b *testing.B) { opHillshade.bench(b) }

// TestAllocs checks every case on a small raster against a bound that
// does not grow with the raster: the plain functions run through the
// engine's one-worker path, which allocates a few slices per call. It
// runs without -bench, so a regression fails go test.
func TestAllocs(t *testing.T) {
	defer stencilKernels.UseScalar(false)
	const limit = 16
	ops := map[string]op{"Gradient": opGradient, "Slope": opSlope, "Aspect": opAspect, "Hillshade": opHillshade}
	for name, o := range ops {
		f := newFixture(64, o.outputs)
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: 1}
				stencilKernels.UseScalar(backend == suite.Scalar)
				w := o.workload(f, c)
				if allocs := testing.AllocsPerRun(10, w.Run); allocs > limit {
					t.Errorf("%s/%s: %v allocs/op, want at most %d", name, c.Name(), allocs, limit)
				}
			}
		}
	}
}
