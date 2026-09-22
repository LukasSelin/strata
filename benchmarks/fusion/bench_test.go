package fusion_test

import (
	"context"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/benchmarks/fusion"
	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Run the category with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/fusion -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var kernels = suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, kernels) }

// Tile shapes, the last level of every benchmark name, as in the engine
// category: default strips, and 256×256 tiles.
const (
	tilesStrips = "strips"
	tiles256    = "256x256"
)

var tileShapes = []string{tilesStrips, tiles256}

// sizes: 1024² keeps the six inputs (24 MiB) near L3, 4096² (384 MiB of
// operands) and 8192² are out of cache. 16384² is not run: the chained
// form's nine rasters would be 9 GiB.
var sizes = []int{1024, 4096, 8192}

// fixture holds the six factors, the output and the chained form's two
// intermediates, with a mask for each so that one fixture serves both
// mask levels.
type fixture struct {
	size    int
	in      [fusion.Inputs][]float32
	inValid [fusion.Inputs][]uint64
	dst     []float32
	tmp     [2][]float32
	valid   [3][]uint64 // dst's, then the intermediates'
}

// newFixture builds factors uniform in [0.5, 2): the product of six stays
// within [1/64, 64), far from overflow and from subnormals, which would
// time the FPU's slow path rather than the memory system. Each input has
// its own mask with 2% of cells invalid.
func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, dst: make([]float32, n)}
	for i := range f.in {
		f.in[i] = make([]float32, n)
		suite.FillUniform(f.in[i], uint64(10+i), 0.5, 2)
		f.inValid[i] = suite.RandomMask(n, uint64(20+i), 0.02)
	}
	// Fault the outputs' pages in before timing.
	suite.FillUniform(f.dst, 3, 0, 1)
	for i := range f.tmp {
		f.tmp[i] = make([]float32, n)
		suite.FillUniform(f.tmp[i], uint64(4+i), 0, 1)
	}
	for i := range f.valid {
		f.valid[i] = raster.NewMask(n)
	}
	return f
}

// operands returns the factors and the output, masked or not.
func (f *fixture) operands(masked bool) (dst raster.Float32Raster, src []raster.Float32Raster) {
	src = make([]raster.Float32Raster, fusion.Inputs)
	for i := range src {
		src[i] = raster.NewFloat32(f.size, f.size, f.in[i])
		if masked {
			src[i].Valid = f.inValid[i]
		}
	}
	dst = raster.NewFloat32(f.size, f.size, f.dst)
	if masked {
		dst.Valid = f.valid[0]
	}
	return dst, src
}

func (f *fixture) temps(masked bool) [2]raster.Float32Raster {
	var t [2]raster.Float32Raster
	for i := range t {
		t[i] = raster.NewFloat32(f.size, f.size, f.tmp[i])
		if masked {
			t[i].Valid = f.valid[1+i]
		}
	}
	return t
}

// A form is one way of computing the product through the engine.
type form func(ctx context.Context, f *fixture, masked bool, o engine.Options) error

// chained is five algebra.MulTiled calls, as a caller writes the product
// today: t0 = a0·a1, t1 = t0·a2, t0 = t1·a3, t1 = t0·a4, dst = t1·a5.
// Every call reads its two operands from memory and writes one, and the
// intermediates ping-pong between two whole rasters.
func chained(ctx context.Context, f *fixture, masked bool, o engine.Options) error {
	dst, src := f.operands(masked)
	t := f.temps(masked)
	acc := src[0]
	for i := 1; i < fusion.Inputs; i++ {
		out := t[i%2]
		if i == fusion.Inputs-1 {
			out = dst
		}
		if err := algebra.MulTiled(ctx, out, acc, src[i], o); err != nil {
			return err
		}
		acc = out
	}
	return nil
}

// pipeline is built once: a Pipeline is immutable, and building it is
// planning a caller would do once, not per call.
var pipeline = fusion.NewPipeline()

// pipelined is the same five multiplies as one tile-level fused kernel
// (DESIGN.md §52): each tile is read once, its intermediates go through
// the worker's scratch.
func pipelined(ctx context.Context, f *fixture, masked bool, o engine.Options) error {
	dst, src := f.operands(masked)
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, src, pipeline, o)
}

// fused is the hand-written register-level kernel (DESIGN.md §29): one
// loop, six loads, five multiplies and one store per cell.
func fused(ctx context.Context, f *fixture, masked bool, o engine.Options) error {
	dst, src := f.operands(masked)
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, src, fusion.NewFused(), o)
}

// bytesPerCell is what a form moves per output cell, as engine.Stats
// counts it plus the masks it reads and writes: the chained form's five
// calls each move three operands, 60 B/cell; the pipeline and the fused
// kernel read six and write one, 28 B/cell. TestCounter checks the
// float32 part against the engine's own counter. For the pipeline this
// leaves out the scratch traffic between stages, which the counter does
// not see and which is what the fused form removes; GB/s is therefore the
// same measure for the two, and differences between them are that
// traffic.
func bytesPerCell(name string, masked bool) float64 {
	operands := float64(fusion.Inputs + 1)
	if name == "chained" {
		operands = 3 * (fusion.Inputs - 1)
	}
	b := 4 * operands
	if masked {
		b += operands / 8
	}
	return b
}

func workload(name string, run form) func(f *fixture, c suite.Case) suite.Workload {
	return func(f *fixture, c suite.Case) suite.Workload {
		w := suite.Workload{BytesPerCell: bytesPerCell(name, c.Masked)}
		if c.Backend == suite.Scalar && c.Workers > 1 {
			return w // as in the engine category: scalar runs one worker only
		}
		opts := engine.Options{Workers: c.Workers}
		if c.Tiles == tiles256 {
			opts.TileWidth, opts.TileHeight = 256, 256
		}
		ctx := context.Background()
		w.Run = func() { _ = run(ctx, f, c.Masked, opts) }
		return w
	}
}

func bench(b *testing.B, name string, run form) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  kernels,
		Fixture:  newFixture,
		Workload: workload(name, run),
		Sizes:    sizes,
		Workers:  suite.Workers(),
		Tiles:    tileShapes,
	})
}

func BenchmarkMulChained(b *testing.B)  { bench(b, "chained", chained) }
func BenchmarkMulPipeline(b *testing.B) { bench(b, "pipeline", pipelined) }
func BenchmarkMulFused(b *testing.B)    { bench(b, "fused", fused) }
