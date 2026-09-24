package exec_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// radiusShape is a pipeline with stages of radius > 0, and the radius
// and reaches NewPipeline must work out for it.
type radiusShape struct {
	name   string
	inputs int
	stages []exec.Stage
	out    int
	radius int
	reach  []int
}

func radiusShapes() []radiusShape {
	box := func(r int) exec.Kernel { return boxKernel{r: r, inputs: 1} }
	mul := mulKernel{inputs: 2}
	st := func(k exec.Kernel, in ...int) exec.Stage { return exec.Stage{Kernel: k, In: in} }
	return []radiusShape{
		{
			// Slope(a)·b in miniature: the case where eroding every input
			// by the pipeline's radius would be wrong. 2 = box1(a), 3 = 2·b.
			name: "stencil times pointwise", inputs: 2,
			stages: []exec.Stage{st(box(1), 0), st(mul, 2, 1)}, out: 3,
			radius: 1, reach: []int{1, 0},
		},
		{
			// Radii add along a chain. 1 = box1(a), 2 = box2(1).
			name: "chain", inputs: 1,
			stages: []exec.Stage{st(box(1), 0), st(box(2), 1)}, out: 2,
			radius: 3, reach: []int{3},
		},
		{
			// A stencil after a pointwise stage runs it grown. 2 = a·b,
			// 3 = box2(2).
			name: "pointwise first", inputs: 2,
			stages: []exec.Stage{st(mul, 0, 1), st(box(2), 2)}, out: 3,
			radius: 2, reach: []int{2, 2},
		},
		{
			// Two paths from a of different lengths, and b on the
			// shorter: a's reach is the longer path's. 2 = a·b,
			// 3 = box1(2), 4 = box2(a), 5 = 3·4.
			name: "paths of different lengths", inputs: 2,
			stages: []exec.Stage{st(mul, 0, 1), st(box(1), 2), st(box(2), 0), st(mul, 3, 4)}, out: 5,
			radius: 2, reach: []int{2, 1},
		},
		{
			// An intermediate held beyond the span and read from its
			// middle, and an input read from the middle of its window.
			// 2 = box1(a), 3 = 2·b, 4 = box1(3).
			name: "grown intermediate", inputs: 2,
			stages: []exec.Stage{st(box(1), 0), st(mul, 2, 1), st(box(1), 3)}, out: 4,
			radius: 2, reach: []int{2, 1},
		},
		{
			// A two-output stage, one output read on, and a stage of
			// radius 3 whose output nothing reads: it must neither run
			// nor widen the pipeline. 1, 2 = box1(a), 3 = box1(2),
			// 4 = box3(1).
			name: "two outputs and a dead stage", inputs: 1,
			stages: []exec.Stage{
				st(boxKernel{r: 1, inputs: 1, outputs: 2}, 0), st(box(1), 2), st(box(3), 1),
			}, out: 3,
			radius: 2, reach: []int{2},
		},
	}
}

// TestPipelineRadiusShape checks the radius and reaches NewPipeline
// works out, which the engine trusts for the halo and for validity.
func TestPipelineRadiusShape(t *testing.T) {
	for _, s := range radiusShapes() {
		p := exec.NewPipeline(s.inputs, s.stages, s.out)
		if got := p.Radius(); got != s.radius {
			t.Errorf("%s: Radius() = %d, want %d", s.name, got, s.radius)
		}
		var rk exec.ReachKernel = p
		for in, want := range s.reach {
			if got := rk.Reach(0, in); got != want {
				t.Errorf("%s: Reach(0, %d) = %d, want %d", s.name, in, got, want)
			}
		}
		if p.Fused() {
			t.Errorf("%s: a pipeline with a radius lowered to a fused chain", s.name)
		}
	}
}

// TestPipelineRadiusMatchesUnfused runs every shape through the §23
// matrix — engineRuns' tiles, bands and workers, windowed and compact,
// masked and not — against the same stages as separate whole-raster
// calls, bit for bit, validity included.
func TestPipelineRadiusMatchesUnfused(t *testing.T) {
	const w, h = 29, 23
	for _, s := range radiusShapes() {
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(17, uint64(len(s.name))))
				src := make([]raster.Float32Raster, s.inputs)
				for i := range src {
					src[i] = newOperand(rng, w, h, windowed, masked).r
				}
				want := unfused(t, src, s.stages, s.out)
				for _, run := range engineRuns {
					id := fmt.Sprintf("%s masked=%v windowed=%v %v", s.name, masked, windowed, run)
					dst := raster.NewFloat32Like(src[0])
					process(t, run, []raster.Float32Raster{dst}, src, exec.NewPipeline(s.inputs, s.stages, s.out))
					sameRaster(t, id, dst, want)
				}
			}
		}
	}
}

// TestPipelineRadiusChunkedMatchesUnfused is the same reference through
// the chunked driver, where each tile's halo is the pipeline's whole
// radius, read from the sources.
func TestPipelineRadiusChunkedMatchesUnfused(t *testing.T) {
	const w, h = 31, 26
	for _, s := range radiusShapes() {
		for _, masked := range []bool{false, true} {
			rng := rand.New(rand.NewPCG(19, uint64(len(s.name))))
			src := make([]raster.Float32Raster, s.inputs)
			for i := range src {
				src[i] = newOperand(rng, w, h, false, masked).r
			}
			want := unfused(t, src, s.stages, s.out)
			for _, workers := range []int{1, 3} {
				for _, tile := range [][2]int{{0, 0}, {16, 16}, {0, 8}, {9, 5}, {3, 2}} {
					id := fmt.Sprintf("%s masked=%v workers=%d tile=%v", s.name, masked, workers, tile)
					dst := raster.NewFloat32Like(src[0])
					sources := make([]engine.RasterSource, s.inputs)
					for i := range sources {
						sources[i] = engine.NewMemorySource(src[i])
					}
					err := exec.ProcessChunked(context.Background(),
						[]engine.RasterSink{engine.NewMemorySink(dst)}, sources,
						exec.NewPipeline(s.inputs, s.stages, s.out),
						engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers})
					if err != nil {
						t.Fatalf("%s: %v", id, err)
					}
					sameRaster(t, id, dst, want)
				}
			}
		}
	}
}

// TestPipelineRadiusPermuted checks that the order of independent
// stages changes nothing: "paths of different lengths" with its two
// branches declared the other way round.
func TestPipelineRadiusPermuted(t *testing.T) {
	const w, h = 27, 19
	box := func(r int) exec.Kernel { return boxKernel{r: r, inputs: 1} }
	mul := mulKernel{inputs: 2}
	st := func(k exec.Kernel, in ...int) exec.Stage { return exec.Stage{Kernel: k, In: in} }
	// 2 = box2(a), 3 = a·b, 4 = box1(3), 5 = 4·2: the same product, in
	// the same operand order, as 5 = box1(a·b)·box2(a).
	permuted := []exec.Stage{st(box(2), 0), st(mul, 0, 1), st(box(1), 3), st(mul, 4, 2)}
	var original radiusShape
	for _, s := range radiusShapes() {
		if s.name == "paths of different lengths" {
			original = s
		}
	}
	rng := rand.New(rand.NewPCG(23, 5))
	src := []raster.Float32Raster{
		newOperand(rng, w, h, true, true).r,
		newOperand(rng, w, h, true, true).r,
	}
	for _, run := range engineRuns {
		a, b := raster.NewFloat32Like(src[0]), raster.NewFloat32Like(src[0])
		process(t, run, []raster.Float32Raster{a}, src, exec.NewPipeline(2, original.stages, original.out))
		process(t, run, []raster.Float32Raster{b}, src, exec.NewPipeline(2, permuted, 5))
		sameRaster(t, fmt.Sprint(run), b, a)
	}
}

// TestPipelineRadiusSingleStage checks that a pipeline of one stencil
// stage is that stage: its halo, its edges and its validity.
func TestPipelineRadiusSingleStage(t *testing.T) {
	const w, h = 25, 18
	for _, r := range []int{1, 2, 3} {
		k := boxKernel{r: r, inputs: 2}
		rng := rand.New(rand.NewPCG(29, uint64(r)))
		src := []raster.Float32Raster{
			newOperand(rng, w, h, true, true).r,
			newOperand(rng, w, h, false, true).r,
		}
		want := raster.NewFloat32Like(src[0])
		if err := exec.ProcessN(context.Background(), []raster.Float32Raster{want}, src, k, engine.Options{}); err != nil {
			t.Fatal(err)
		}
		for _, run := range engineRuns {
			dst := raster.NewFloat32Like(src[0])
			p := exec.NewPipeline(2, []exec.Stage{{Kernel: k, In: []int{0, 1}}}, 2)
			process(t, run, []raster.Float32Raster{dst}, src, p)
			sameRaster(t, fmt.Sprintf("r=%d %v", r, run), dst, want)
		}
	}
}

// TestPipelineRadiusEdgeValue checks the one stage allowed an edge value
// other than NaN — the output's, with the pipeline's radius — and that
// the engine writes it in the whole ring.
func TestPipelineRadiusEdgeValue(t *testing.T) {
	const w, h = 21, 17
	k := edgeBox{boxKernel{r: 2, inputs: 1}, -9}
	rng := rand.New(rand.NewPCG(31, 1))
	src := []raster.Float32Raster{newOperand(rng, w, h, false, true).r}
	// 1 = a·a, pointwise; 2 = box2(1) with edge -9.
	stages := []exec.Stage{{Kernel: mulKernel{inputs: 2}, In: []int{0, 0}}, {Kernel: k, In: []int{1}}}
	want := unfused(t, src, stages, 2)
	for _, run := range engineRuns {
		dst := raster.NewFloat32Like(src[0])
		process(t, run, []raster.Float32Raster{dst}, src, exec.NewPipeline(1, stages, 2))
		sameRaster(t, fmt.Sprint(run), dst, want)
	}
}
