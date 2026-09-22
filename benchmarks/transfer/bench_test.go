package transfer_test

import (
	"testing"

	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/internal/curve"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/transfer"
)

// Run the suite with, e.g.:
//
//	go test ./benchmarks/transfer -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	GOEXPERIMENT=simd go test ./benchmarks/transfer -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var (
	vecKernels   = suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar}
	curveKernels = suite.Kernels{Name: "curve", Backend: curve.Backend, UseScalar: curve.UseScalar}
	// kernels switches both packages at once: the table-driven operations
	// run curve's kernels and the Rescales vec's.
	kernels = suite.Kernels{
		Name:    "curve+vec",
		Backend: curve.Backend,
		UseScalar: func(scalar bool) {
			curve.UseScalar(scalar)
			vec.UseScalar(scalar)
		},
	}
)

func TestMain(m *testing.M) { suite.Main(m, curveKernels, vecKernels) }

// The tables: a five-class slope scale and a five-knot slope factor, in
// degrees, fitted to the operand's range (about 0–25°) so every class
// and segment is used. Class intervals are half-open upward (DESIGN.md
// §50).
var (
	breaks = []float32{4, 8, 12, 16}
	values = []float32{1, 2, 3, 4, 5}
	xs     = []float32{0, 4, 8, 16, 30}
	ys     = []float32{1, 1.1, 1.4, 1.8, 2.5}
)

// fixture is one size's slope raster and an output, and their masks,
// attached per case.
type fixture struct {
	size     int
	slope    []float32
	out      []float32
	valid    []uint64
	outValid []uint64
}

func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, slope: make([]float32, n), out: make([]float32, n)}
	dem := make([]float32, n)
	suite.FillDEM(dem, size, 1)
	terrain.Slope(raster.NewFloat32(size, size, f.slope), raster.NewFloat32(size, size, dem), terrain.SlopeOptions{CellSize: 10})
	suite.FillUniform(f.out, 3, 0, 1) // fault the output's pages in before timing
	f.valid = suite.RandomMask(n, 4, 0.1)
	f.outValid = raster.NewMask(n)
	return f
}

func (f *fixture) rasters(masked bool) (src, dst raster.Float32Raster) {
	src = raster.NewFloat32(f.size, f.size, f.slope)
	dst = raster.NewFloat32(f.size, f.size, f.out)
	if masked {
		src.Valid, dst.Valid = f.valid, f.outValid
	}
	return src, dst
}

// op is one transfer operation from src to dst.
type op func(dst, src raster.Float32Raster)

var (
	opReclass = op(func(dst, src raster.Float32Raster) { transfer.Reclass(dst, src, breaks, values) })
	opLookup  = op(func(dst, src raster.Float32Raster) { transfer.Lookup(dst, src, xs, ys) })
	opRescale = op(func(dst, src raster.Float32Raster) { transfer.Rescale(dst, src, 0.025, -1) })
	opRange   = op(func(dst, src raster.Float32Raster) { transfer.RescaleRange(dst, src, 0, 40, 0, 1) })
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	src, dst := f.rasters(c.Masked)
	bytes := 8.0
	if c.Masked {
		bytes += 2.0 / 8
	}
	return suite.Workload{
		Run:          func() { o(dst, src) },
		BytesPerCell: bytes,
	}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  kernels,
		Fixture:  newFixture,
		Workload: o.workload,
	})
}

func BenchmarkReclass(b *testing.B)      { opReclass.bench(b) }
func BenchmarkLookup(b *testing.B)       { opLookup.bench(b) }
func BenchmarkRescale(b *testing.B)      { opRescale.bench(b) }
func BenchmarkRescaleRange(b *testing.B) { opRange.bench(b) }

// TestRuns runs every operation and case once on a small raster, so a
// broken workload fails go test without running benchmarks, and checks
// the plain functions allocate a bound that does not grow with the
// raster.
func TestRuns(t *testing.T) {
	defer kernels.UseScalar(false)
	const limit = 16
	f := newFixture(64)
	ops := map[string]op{"Reclass": opReclass, "Lookup": opLookup, "Rescale": opRescale, "RescaleRange": opRange}
	for name, o := range ops {
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: 1}
				kernels.UseScalar(backend == suite.Scalar)
				w := o.workload(f, c)
				if allocs := testing.AllocsPerRun(10, w.Run); allocs > limit {
					t.Errorf("%s/%s: %v allocs/op, want at most %d", name, c.Name(), allocs, limit)
				}
			}
		}
	}
}

// TestOperandSpansTheTables checks the fixture is what doc.go says: a
// slope field that uses every class of the scale above, in which
// neighbouring cells usually but not always share a class, so the
// benchmarks time neither a one-class raster nor white noise.
func TestOperandSpansTheTables(t *testing.T) {
	f := newFixture(256)
	dst := make([]float32, len(f.slope))
	curve.Reclass(dst, f.slope, breaks, values)
	seen := map[float32]int{}
	same, pairs := 0, 0
	for i, v := range dst {
		if v != v {
			continue // Slope's one-cell border
		}
		seen[v]++
		if i > 0 && dst[i-1] == dst[i-1] {
			pairs++
			if dst[i-1] == v {
				same++
			}
		}
	}
	if len(seen) != len(values) {
		t.Errorf("the slope operand reaches classes %v, want all %d", seen, len(values))
	}
	if frac := float64(same) / float64(pairs); frac < 0.5 || frac > 0.95 {
		t.Errorf("neighbouring cells share a class %.2f of the time, want between 0.5 and 0.95", frac)
	}
}
