package algebra_test

import (
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Run the suite with, e.g.:
//
//	go test ./benchmarks/algebra -run '^$' -bench . -count 5 -timeout 2h | tee bench.txt
//	GOEXPERIMENT=simd go test ./benchmarks/algebra -run '^$' -bench . -count 5 -timeout 2h | tee bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var vecKernels = suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, vecKernels) }

// fixture holds the operands for one size. Only what the operation reads
// is allocated: b and its mask are nil for single-input operations, which
// saves a third of the memory at 16384². Masks are attached per case.
type fixture struct {
	size           int
	a, b, dst      []float32
	aValid, bValid []uint64
	dstValid       []uint64
}

func newFixture(size, inputs int) *fixture {
	n := size * size
	f := &fixture{
		size: size,
		a:    make([]float32, n),
		dst:  make([]float32, n),
	}
	suite.FillUniform(f.a, 1, -100, 100)
	suite.FillUniform(f.dst, 3, -100, 100) // fault dst's pages in before timing
	f.aValid = suite.RandomMask(n, 4, 0.1)
	f.dstValid = raster.NewMask(n)
	if inputs == 2 {
		f.b = make([]float32, n)
		suite.FillUniform(f.b, 2, -100, 100)
		f.bValid = suite.RandomMask(n, 5, 0.1)
	}
	return f
}

// raster wraps data, or returns the zero raster when the operation does not
// use this operand.
func (f *fixture) raster(data []float32, valid []uint64, masked bool) raster.Float32Raster {
	if data == nil {
		return raster.Float32Raster{}
	}
	r := raster.NewFloat32(f.size, f.size, data)
	if masked {
		r.Valid = valid
	}
	return r
}

// op is one algebra operation over dst and up to two inputs.
type op struct {
	inputs int
	run    func(dst, a, b raster.Float32Raster)
}

var (
	opAdd   = op{2, algebra.Add}
	opSub   = op{2, algebra.Sub}
	opMul   = op{2, algebra.Mul}
	opMin   = op{2, algebra.Min}
	opMax   = op{2, algebra.Max}
	opClamp = op{1, func(dst, src, _ raster.Float32Raster) { algebra.Clamp(dst, src, -50, 50) }}
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	dst := f.raster(f.dst, f.dstValid, c.Masked)
	a := f.raster(f.a, f.aValid, c.Masked)
	b := f.raster(f.b, f.bValid, c.Masked)
	operands := float64(o.inputs + 1)
	bytes := 4 * operands
	if c.Masked {
		bytes += operands / 8
	}
	return suite.Workload{
		Run:          func() { o.run(dst, a, b) },
		BytesPerCell: bytes,
	}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  vecKernels,
		Fixture:  func(size int) *fixture { return newFixture(size, o.inputs) },
		Workload: o.workload,
	})
}

func BenchmarkAdd(b *testing.B)   { opAdd.bench(b) }
func BenchmarkSub(b *testing.B)   { opSub.bench(b) }
func BenchmarkMul(b *testing.B)   { opMul.bench(b) }
func BenchmarkMin(b *testing.B)   { opMin.bench(b) }
func BenchmarkMax(b *testing.B)   { opMax.bench(b) }
func BenchmarkClamp(b *testing.B) { opClamp.bench(b) }

// TestZeroAllocs checks the suite's 0 allocs/op on a small raster, for
// every operation and case, so a regression fails go test without running
// benchmarks.
func TestZeroAllocs(t *testing.T) {
	defer vec.UseScalar(false)
	ops := map[string]op{"Add": opAdd, "Sub": opSub, "Mul": opMul, "Min": opMin, "Max": opMax, "Clamp": opClamp}
	for name, o := range ops {
		f := newFixture(64, o.inputs)
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: 1}
				vec.UseScalar(backend == suite.Scalar)
				w := o.workload(f, c)
				if allocs := testing.AllocsPerRun(10, w.Run); allocs != 0 {
					t.Errorf("%s/%s: %v allocs/op, want 0", name, c.Name(), allocs)
				}
			}
		}
	}
}
