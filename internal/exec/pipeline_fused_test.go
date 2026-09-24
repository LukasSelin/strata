package exec_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Register-level fusion (DESIGN.md §29) is an optimisation of §52's
// Pipeline, so the staged pipeline is its reference and these tests are
// all the same comparison: build one pipeline with fusion on and one
// with it off, run both, and require the same bits. The staged form is
// already checked against separate whole-raster calls by
// TestPipelineMatchesUnfused, so equalling it is equalling those too.

// staged returns the pipeline the stages would make with fusion turned
// off, and fails the test if it lowered anyway.
func staged(t *testing.T, inputs int, stages []exec.Stage, out int) *exec.Pipeline {
	t.Helper()
	defer exec.SetFusion(false)()
	p := exec.NewPipeline(inputs, stages, []int{out})
	if p.Fused() {
		t.Fatal("a pipeline built with fusion off is fused")
	}
	return p
}

// fused returns the pipeline the stages make, and fails the test unless
// it lowered to a chain.
func fused(t *testing.T, inputs int, stages []exec.Stage, out int) *exec.Pipeline {
	t.Helper()
	p := exec.NewPipeline(inputs, stages, []int{out})
	if !p.Fused() {
		t.Fatal("the pipeline did not lower to a fused chain")
	}
	return p
}

// eachBackend runs f on every vector backend this build has, so that a
// SIMD build checks the fused lane loop and the blocked scalar
// evaluator in one run.
func eachBackend(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	backends := []bool{true} // scalar
	if vec.Backend() != "scalar" {
		backends = append(backends, false)
	}
	for _, scalar := range backends {
		vec.UseScalar(scalar)
		name := vec.Backend()
		t.Run("backend="+name, func(t *testing.T) { f(t) })
	}
	vec.UseScalar(false)
}

// TestPipelineFusedMatchesStaged is the load-bearing test of §29: over
// the §23 matrix, on every backend, the fused chain must write the bits
// the staged pipeline writes — Data and validity, cell for cell.
func TestPipelineFusedMatchesStaged(t *testing.T) {
	const w, h = 61, 43
	const inputs = 5
	stages, out := chain(inputs)

	eachBackend(t, func(t *testing.T) {
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(13, 21))
				src := make([]raster.Float32Raster, inputs)
				for i := range src {
					src[i] = newOperand(rng, w, h, windowed, masked).r
				}

				want := raster.NewFloat32Like(src[0])
				run(t, "staged", staged(t, inputs, stages, out), want, src, engine.Options{})

				for _, workers := range []int{1, 2, 5} {
					for _, tile := range [][2]int{{0, 0}, {16, 16}, {0, 8}, {7, 3}} {
						id := fmt.Sprintf("masked=%v windowed=%v workers=%d tile=%v",
							masked, windowed, workers, tile)
						got := raster.NewFloat32Like(src[0])
						run(t, id, fused(t, inputs, stages, out), got, src,
							engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers})
						sameRaster(t, id, got, want)
					}
				}
			}
		}
	})
}

// TestPipelineFusedChunkedMatchesStaged is the same comparison through
// the chunked driver, where the pipeline runs over a tile that has been
// copied into a worker's buffer rather than over the raster itself.
func TestPipelineFusedChunkedMatchesStaged(t *testing.T) {
	const w, h = 53, 37
	const inputs = 4
	stages, out := chain(inputs)

	eachBackend(t, func(t *testing.T) {
		for _, masked := range []bool{false, true} {
			rng := rand.New(rand.NewPCG(3, 17))
			src := make([]raster.Float32Raster, inputs)
			for i := range src {
				src[i] = newOperand(rng, w, h, false, masked).r
			}
			want := raster.NewFloat32Like(src[0])
			run(t, "staged", staged(t, inputs, stages, out), want, src, engine.Options{})

			for _, workers := range []int{1, 3} {
				for _, tile := range [][2]int{{0, 0}, {16, 16}, {0, 8}, {9, 5}} {
					id := fmt.Sprintf("masked=%v workers=%d tile=%v", masked, workers, tile)
					got := raster.NewFloat32Like(src[0])
					sources := make([]engine.RasterSource, inputs)
					for i := range sources {
						sources[i] = engine.NewMemorySource(src[i])
					}
					err := exec.ProcessChunked(context.Background(),
						[]engine.RasterSink{engine.NewMemorySink(got)}, sources,
						fused(t, inputs, stages, out),
						engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers})
					if err != nil {
						t.Fatalf("%s: %v", id, err)
					}
					sameRaster(t, id, got, want)
				}
			}
		}
	})
}

// run is one ProcessN call, failing the test on error.
func run(t *testing.T, id string, p *exec.Pipeline, dst raster.Float32Raster, src []raster.Float32Raster, opts engine.Options) {
	t.Helper()
	if err := exec.ProcessN(context.Background(), []raster.Float32Raster{dst}, src, p, opts); err != nil {
		t.Fatalf("%s: %v", id, err)
	}
}

// TestPipelineFusedEveryOp runs a chain that uses every operation a
// chain can hold, against the same stages staged. The unary and
// immediate steps are where the fused lane loop and the blocked scalar
// evaluator differ most from each other, and from a chain of multiplies.
func TestPipelineFusedEveryOp(t *testing.T) {
	const w, h = 37, 29
	const inputs = 3

	// 3 = 0 op 1, 4 = 3 op 2, then a tail of unary and immediate steps.
	stages := []exec.Stage{
		{Kernel: opKernel{op: vec.OpAdd, inputs: 2}, In: []int{0, 1}},
		{Kernel: opKernel{op: vec.OpSub, inputs: 2}, In: []int{3, 2}},
		{Kernel: opKernel{op: vec.OpMul, inputs: 2}, In: []int{4, 1}},
		{Kernel: opKernel{op: vec.OpDiv, inputs: 2}, In: []int{5, 2}},
		{Kernel: opKernel{op: vec.OpMin, inputs: 2}, In: []int{6, 0}},
		{Kernel: opKernel{op: vec.OpMax, inputs: 2}, In: []int{7, 1}},
		{Kernel: opKernel{op: vec.OpAbs, inputs: 1}, In: []int{8}},
		{Kernel: opKernel{op: vec.OpSqrt, inputs: 1}, In: []int{9}},
		{Kernel: opKernel{op: vec.OpMulScalar, inputs: 1, k: [2]float32{-3}}, In: []int{10}},
		{Kernel: opKernel{op: vec.OpAddScalar, inputs: 1, k: [2]float32{0.25}}, In: []int{11}},
		{Kernel: opKernel{op: vec.OpAffine, inputs: 1, k: [2]float32{2, -1}}, In: []int{12}},
		{Kernel: opKernel{op: vec.OpClamp, inputs: 1, k: [2]float32{-8, 8}}, In: []int{13}},
	}
	const out = 14

	eachBackend(t, func(t *testing.T) {
		for _, windowed := range []bool{false, true} {
			rng := rand.New(rand.NewPCG(31, 5))
			src := make([]raster.Float32Raster, inputs)
			for i := range src {
				src[i] = newOperand(rng, w, h, windowed, true).r
			}
			want := raster.NewFloat32Like(src[0])
			run(t, "staged", staged(t, inputs, stages, out), want, src, engine.Options{})

			for _, tile := range [][2]int{{0, 0}, {8, 8}, {0, 3}} {
				id := fmt.Sprintf("windowed=%v tile=%v", windowed, tile)
				got := raster.NewFloat32Like(src[0])
				run(t, id, fused(t, inputs, stages, out), got, src,
					engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: 3})
				sameRaster(t, id, got, want)
			}
		}
	})
}

// TestPipelineFusedInPlace writes the result over one of the operands.
// The fused pass reads every input for a run of cells before it stores
// any of them, so an output that is also an input is safe — but it is
// safe by construction, not by accident, and this is what says so.
func TestPipelineFusedInPlace(t *testing.T) {
	const w, h = 45, 31
	const inputs = 4
	stages, out := chain(inputs)
	rng := rand.New(rand.NewPCG(19, 23))
	src := make([]raster.Float32Raster, inputs)
	for i := range src {
		src[i] = newOperand(rng, w, h, false, true).r
	}

	want := raster.NewFloat32Like(src[0])
	run(t, "staged", staged(t, inputs, stages, out), want, src, engine.Options{})

	// dst is src[0], the value the chain starts from.
	inPlace := make([]raster.Float32Raster, inputs)
	copy(inPlace, src)
	inPlace[0] = raster.NewFloat32Like(src[0])
	copyRaster(inPlace[0], src[0])
	run(t, "in place", fused(t, inputs, stages, out), inPlace[0], inPlace,
		engine.Options{TileHeight: 4, Workers: 3})
	sameRaster(t, "in place", inPlace[0], want)
}

func copyRaster(dst, src raster.Float32Raster) {
	for y := range src.Height {
		copy(dst.Row(y), src.Row(y))
		for x := range src.Width {
			dst.SetValid(x, y, src.IsValid(x, y))
		}
	}
}

// TestPipelineFallsBackOffTheCut names the shapes register-level fusion
// does not take. Each must still run, and still equal its unfused form:
// falling back is the design, not a failure.
func TestPipelineFallsBackOffTheCut(t *testing.T) {
	const w, h = 29, 17
	rng := rand.New(rand.NewPCG(41, 6))
	src := make([]raster.Float32Raster, 4)
	for i := range src {
		src[i] = newOperand(rng, w, h, false, true).r
	}

	cases := []struct {
		name   string
		inputs int
		stages []exec.Stage
		out    int
	}{{
		// (0*1) * (2*3): two live intermediates, so not left-deep.
		name: "diamond", inputs: 4, out: 6,
		stages: []exec.Stage{
			{Kernel: mulKernel{inputs: 2}, In: []int{0, 1}},
			{Kernel: mulKernel{inputs: 2}, In: []int{2, 3}},
			{Kernel: mulKernel{inputs: 2}, In: []int{4, 5}},
		},
	}, {
		// The running value is the second operand, not the first.
		name: "right-deep", inputs: 3, out: 4,
		stages: []exec.Stage{
			{Kernel: mulKernel{inputs: 2}, In: []int{0, 1}},
			{Kernel: mulKernel{inputs: 2}, In: []int{2, 3}},
		},
	}, {
		// A stage that is not a FusableKernel at all.
		name: "unfusable stage", inputs: 3, out: 4,
		stages: []exec.Stage{
			{Kernel: boxKernel{inputs: 2, outputs: 1}, In: []int{0, 1}},
			{Kernel: mulKernel{inputs: 2}, In: []int{3, 2}},
		},
	}, {
		// A kernel that is fusable in principle but not at this arity.
		name: "wider than a step", inputs: 3, out: 4,
		stages: []exec.Stage{
			{Kernel: mulKernel{inputs: 3}, In: []int{0, 1, 2}},
			{Kernel: mulKernel{inputs: 2}, In: []int{3, 0}},
		},
	}, {
		// The output is the first stage's, not the last's.
		name: "output is not the last stage", inputs: 2, out: 2,
		stages: []exec.Stage{
			{Kernel: mulKernel{inputs: 2}, In: []int{0, 1}},
			{Kernel: mulKernel{inputs: 2}, In: []int{2, 1}},
		},
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := exec.NewPipeline(tc.inputs, tc.stages, []int{tc.out})
			if p.Fused() {
				t.Fatal("lowered to a fused chain; this shape is off the cut")
			}
			want := unfused(t, src[:tc.inputs], tc.stages, tc.out)
			dst := raster.NewFloat32Like(src[0])
			run(t, tc.name, p, dst, src[:tc.inputs],
				engine.Options{TileWidth: 8, TileHeight: 5, Workers: 3})
			sameRaster(t, tc.name, dst, want)
		})
	}
}

// TestPipelineFusedAsksNoCells is the claim §29 makes, as an assertion:
// a fused pipeline needs no buffer between its stages, only the slice
// headers it hands its operands to vec.Chain in. The staged form of the
// same pipeline asks for a buffer per intermediate per cell, which is
// what the fused one stops moving.
func TestPipelineFusedAsksNoCells(t *testing.T) {
	const inputs = 6
	const w, h = 256, 64
	stages, out := chain(inputs)

	got := fused(t, inputs, stages, out).Scratch(w, h)
	want := exec.ScratchSize{Runs: inputs}
	if got != want {
		t.Errorf("fused Scratch(%d, %d) = %+v, want %+v", w, h, got, want)
	}

	was := staged(t, inputs, stages, out).Scratch(w, h)
	if was.Cells == 0 {
		t.Fatalf("staged Scratch(%d, %d) = %+v, want cells to compare against", w, h, was)
	}
	t.Logf("scratch cells per span: staged %d, fused %d", was.Cells, got.Cells)
}

// TestPipelineFuseArityMismatchPanics covers the one contradiction the
// lowering refuses to paper over: a kernel that says it is fusable to an
// operation whose operand count is not its own.
func TestPipelineFuseArityMismatchPanics(t *testing.T) {
	defer func() {
		v := recover()
		s, isString := v.(string)
		if !isString || !strings.Contains(s, "fuses to Mul, which takes 2 inputs") {
			t.Fatalf("panic %v, want one naming the arity mismatch", v)
		}
	}()
	exec.NewPipeline(1, []exec.Stage{
		{Kernel: liarKernel{}, In: []int{0}},
	}, []int{1})
}

// opKernel is one vec operation as a Kernel, so that a test can build a
// pipeline stage for any step a chain can hold. Its Process runs the
// operation through the same exported vec functions the chain's staged
// reference uses, a row at a time.
type opKernel struct {
	op     vec.Op
	inputs int
	k      [2]float32
}

func (opKernel) Radius() int                    { return 0 }
func (k opKernel) Arity() (inputs, outputs int) { return k.inputs, 1 }

func (k opKernel) Fuse() (vec.Step, bool) { return vec.Step{Op: k.op, K: k.k}, true }

func (k opKernel) Process(dst exec.Span, src exec.Window) {
	out := dst.Dst[0]
	for y := range dst.Height {
		d, a := out.Row(y), src.Src[0].Row(y)
		switch k.op {
		case vec.OpAdd:
			vec.Add(d, a, src.Src[1].Row(y))
		case vec.OpSub:
			vec.Sub(d, a, src.Src[1].Row(y))
		case vec.OpMul:
			vec.Mul(d, a, src.Src[1].Row(y))
		case vec.OpDiv:
			vec.Div(d, a, src.Src[1].Row(y))
		case vec.OpMin:
			vec.Min(d, a, src.Src[1].Row(y))
		case vec.OpMax:
			vec.Max(d, a, src.Src[1].Row(y))
		case vec.OpAddScalar:
			vec.AddScalar(d, a, k.k[0])
		case vec.OpMulScalar:
			vec.MulScalar(d, a, k.k[0])
		case vec.OpAffine:
			vec.Affine(d, a, k.k[0], k.k[1])
		case vec.OpClamp:
			vec.Clamp(d, a, k.k[0], k.k[1])
		case vec.OpAbs:
			vec.Abs(d, a)
		case vec.OpSqrt:
			vec.Sqrt(d, a)
		default:
			panic("opKernel: " + k.op.String())
		}
	}
}

// liarKernel takes one input and claims to fuse to a binary operation.
type liarKernel struct{}

func (liarKernel) Radius() int                  { return 0 }
func (liarKernel) Arity() (inputs, outputs int) { return 1, 1 }
func (liarKernel) Fuse() (vec.Step, bool)       { return vec.Step{Op: vec.OpMul}, true }
func (liarKernel) Process(dst exec.Span, src exec.Window) {
	copy(dst.Dst[0].Data, src.Src[0].Data)
}
