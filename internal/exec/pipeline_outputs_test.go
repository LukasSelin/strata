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

// reachBox is a ReachKernel whose output o is boxKernel of radius
// outs[o].r over input outs[o].in alone, so its outputs read different
// inputs over different distances. The kernel's radius is the largest.
// Each output must equal boxKernel run on its own, which is what makes
// it a test of the engine's per-output validity, edge rings and padded
// windows (DESIGN.md §52) with nothing else in the way.
type reachBox struct {
	inputs int
	outs   []struct{ in, r int }
}

func (k reachBox) Radius() int {
	r := 0
	for _, o := range k.outs {
		r = max(r, o.r)
	}
	return r
}

func (k reachBox) Arity() (inputs, outputs int) { return k.inputs, len(k.outs) }

func (k reachBox) Reach(out, in int) int {
	if k.outs[out].in != in {
		return -1
	}
	return k.outs[out].r
}

func (k reachBox) Process(dst exec.Span, src exec.Window) {
	for o, spec := range k.outs {
		in, r := src.Src[spec.in], spec.r
		off := src.Radius - r
		n := float32((2*r + 1) * (2*r + 1))
		out := dst.Dst[o]
		for y := range dst.Height {
			for x := range dst.Width {
				var sum float32
				for j := range 2*r + 1 {
					for i := range 2*r + 1 {
						sum += in.Data[in.Index(x+off+i, y+off+j)]
					}
				}
				out.Row(y)[x] = sum / n
			}
		}
	}
}

// TestReachKernelOutputsMatchAlone runs reachBox through the §23 matrix,
// both drivers, and requires each output to be boxKernel of its own
// radius over its own input, run alone: Data, validity and edge ring.
func TestReachKernelOutputsMatchAlone(t *testing.T) {
	const w, h = 27, 21
	kernels := []reachBox{
		// Rings of 1 and 3 over one input: the narrow output's cells one
		// and two from the edge read padding in the wide one's window.
		{1, []struct{ in, r int }{{0, 1}, {0, 3}}},
		// An output of ring 0 beside one of ring 2, over different
		// inputs: the first is computed on every cell, and its validity
		// is its own input's alone.
		{2, []struct{ in, r int }{{0, 2}, {1, 0}}},
		// Three outputs, the widest in the middle.
		{2, []struct{ in, r int }{{1, 1}, {0, 2}, {1, 0}}},
	}
	for ki, k := range kernels {
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(37, uint64(ki)))
				src := make([]raster.Float32Raster, k.inputs)
				for i := range src {
					src[i] = newOperand(rng, w, h, windowed, masked).r
				}
				want := make([]raster.Float32Raster, len(k.outs))
				for o, spec := range k.outs {
					want[o] = raster.NewFloat32Like(src[0])
					err := exec.Process(context.Background(), want[o], src[spec.in],
						boxKernel{r: spec.r, inputs: 1}, engine.Options{})
					if err != nil {
						t.Fatal(err)
					}
				}
				newDst := func() []raster.Float32Raster {
					d := make([]raster.Float32Raster, len(k.outs))
					for o := range d {
						d[o] = raster.NewFloat32Like(src[0])
					}
					return d
				}
				for _, run := range engineRuns {
					dst := newDst()
					process(t, run, dst, src, k)
					for o := range dst {
						sameRaster(t, fmt.Sprintf("kernel %d output %d masked=%v windowed=%v %v",
							ki, o, masked, windowed, run), dst[o], want[o])
					}
				}
				if windowed {
					continue // memory sources are read whole, windows or not
				}
				for _, tile := range [][2]int{{0, 0}, {8, 8}, {5, 3}} {
					dst := newDst()
					sinks := make([]engine.RasterSink, len(dst))
					for o := range dst {
						sinks[o] = engine.NewMemorySink(dst[o])
					}
					sources := make([]engine.RasterSource, len(src))
					for i := range src {
						sources[i] = engine.NewMemorySource(src[i])
					}
					err := exec.ProcessChunked(context.Background(), sinks, sources, k,
						engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: 2})
					if err != nil {
						t.Fatal(err)
					}
					for o := range dst {
						sameRaster(t, fmt.Sprintf("chunked kernel %d output %d masked=%v tile=%v",
							ki, o, masked, tile), dst[o], want[o])
					}
				}
			}
		}
	}
}

// outputsShape is a pipeline with several outputs, and the rings
// (largest reach) NewPipeline must give them.
type outputsShape struct {
	name   string
	inputs int
	stages []exec.Stage
	outs   []int
	rings  []int
}

func outputsShapes() []outputsShape {
	box := func(r int) exec.Kernel { return boxKernel{r: r, inputs: 1} }
	mul := mulKernel{inputs: 2}
	st := func(k exec.Kernel, in ...int) exec.Stage { return exec.Stage{Kernel: k, In: in} }
	return []outputsShape{
		{
			// Two stencils of the input: rings 1 and 3. 1 = box1(a),
			// 2 = box3(a).
			name: "unequal radii", inputs: 1,
			stages: []exec.Stage{st(box(1), 0), st(box(3), 0)}, outs: []int{1, 2},
			rings: []int{1, 3},
		},
		{
			// One intermediate feeding two stencils, held 2 beyond the
			// span for the wider one. 2 = a·b, 3 = box1(2), 4 = box2(2).
			name: "shared intermediate", inputs: 2,
			stages: []exec.Stage{st(mul, 0, 1), st(box(1), 2), st(box(2), 2)}, outs: []int{3, 4},
			rings: []int{1, 2},
		},
		{
			// Outputs that read disjoint inputs: the pointwise one has no
			// ring and a's NoData must not reach it. 2 = box1(a), 3 = b·b.
			name: "disjoint inputs", inputs: 2,
			stages: []exec.Stage{st(box(1), 0), st(mul, 1, 1)}, outs: []int{2, 3},
			rings: []int{1, 0},
		},
		{
			// An output that a later pointwise stage reads for another
			// output, from the Span's view. 2 = box1(a), 3 = 2·b.
			name: "output read on", inputs: 2,
			stages: []exec.Stage{st(box(1), 0), st(mul, 2, 1)}, outs: []int{2, 3},
			rings: []int{1, 1},
		},
		{
			// Both outputs of one stage, in the other order, as Gradient's
			// dx and dy. 1, 2 = box1(a) twice.
			name: "a two-output stage", inputs: 1,
			stages: []exec.Stage{st(boxKernel{r: 1, inputs: 1, outputs: 2}, 0)}, outs: []int{2, 1},
			rings: []int{1, 1},
		},
		{
			// Three outputs from a chain: every value is an output, and
			// the middle one is read on at radius 0. 2 = a·b, 3 = 2·a,
			// 4 = box2(a).
			name: "three outputs", inputs: 2,
			stages: []exec.Stage{st(mul, 0, 1), st(mul, 2, 0), st(box(2), 0)}, outs: []int{2, 3, 4},
			rings: []int{0, 0, 2},
		},
	}
}

// unfusedValues is unfused returning every value, for pipelines with
// several outputs.
func unfusedValues(t *testing.T, src []raster.Float32Raster, stages []exec.Stage) []raster.Float32Raster {
	t.Helper()
	values := append([]raster.Float32Raster(nil), src...)
	for i, st := range stages {
		_, nout := st.Kernel.Arity()
		dst := make([]raster.Float32Raster, nout)
		for j := range dst {
			dst[j] = raster.NewFloat32Like(src[0])
		}
		in := make([]raster.Float32Raster, len(st.In))
		for j, id := range st.In {
			in[j] = values[id]
		}
		if err := exec.ProcessN(context.Background(), dst, in, st.Kernel, engine.Options{}); err != nil {
			t.Fatalf("unfused stage %d: %v", i, err)
		}
		values = append(values, dst...)
	}
	return values
}

func TestPipelineOutputsRings(t *testing.T) {
	for _, s := range outputsShapes() {
		p := exec.NewPipeline(s.inputs, s.stages, s.outs)
		if _, n := p.Arity(); n != len(s.outs) {
			t.Fatalf("%s: %d outputs, want %d", s.name, n, len(s.outs))
		}
		for o, want := range s.rings {
			ring := -1
			for in := range s.inputs {
				ring = max(ring, p.Reach(o, in))
			}
			if ring != want {
				t.Errorf("%s: output %d reaches %d, want %d", s.name, o, ring, want)
			}
		}
	}
}

// TestPipelineOutputsMatchUnfused runs every shape through the §23
// matrix, both drivers, against its stages run as separate whole-raster
// calls: every output bit for bit, Data and validity, edge ring
// included.
func TestPipelineOutputsMatchUnfused(t *testing.T) {
	const w, h = 26, 22
	for _, s := range outputsShapes() {
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(41, uint64(len(s.name))))
				src := make([]raster.Float32Raster, s.inputs)
				for i := range src {
					src[i] = newOperand(rng, w, h, windowed, masked).r
				}
				values := unfusedValues(t, src, s.stages)
				newDst := func() []raster.Float32Raster {
					d := make([]raster.Float32Raster, len(s.outs))
					for o := range d {
						d[o] = raster.NewFloat32Like(src[0])
					}
					return d
				}
				for _, run := range engineRuns {
					dst := newDst()
					process(t, run, dst, src, exec.NewPipeline(s.inputs, s.stages, s.outs))
					for o, id := range s.outs {
						sameRaster(t, fmt.Sprintf("%s output %d masked=%v windowed=%v %v",
							s.name, o, masked, windowed, run), dst[o], values[id])
					}
				}
				if windowed {
					continue
				}
				for _, tile := range [][2]int{{0, 0}, {16, 16}, {9, 5}, {3, 2}} {
					dst := newDst()
					sinks := make([]engine.RasterSink, len(dst))
					for o := range dst {
						sinks[o] = engine.NewMemorySink(dst[o])
					}
					sources := make([]engine.RasterSource, len(src))
					for i := range src {
						sources[i] = engine.NewMemorySource(src[i])
					}
					err := exec.ProcessChunked(context.Background(), sinks, sources,
						exec.NewPipeline(s.inputs, s.stages, s.outs),
						engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: 3})
					if err != nil {
						t.Fatal(err)
					}
					for o, id := range s.outs {
						sameRaster(t, fmt.Sprintf("chunked %s output %d masked=%v tile=%v",
							s.name, o, masked, tile), dst[o], values[id])
					}
				}
			}
		}
	}
}

// TestPipelineOutputsSharedEdgeValue checks outputs that each declare
// the same edge value other than NaN, which the one pipeline edge value
// can keep.
func TestPipelineOutputsSharedEdgeValue(t *testing.T) {
	const w, h = 19, 15
	e := func(r int) exec.Kernel { return edgeBox{boxKernel{r: r, inputs: 1}, -3} }
	stages := []exec.Stage{{Kernel: e(1), In: []int{0}}, {Kernel: e(2), In: []int{0}}}
	rng := rand.New(rand.NewPCG(43, 1))
	src := []raster.Float32Raster{newOperand(rng, w, h, true, true).r}
	values := unfusedValues(t, src, stages)
	for _, run := range engineRuns {
		dst := []raster.Float32Raster{raster.NewFloat32Like(src[0]), raster.NewFloat32Like(src[0])}
		process(t, run, dst, src, exec.NewPipeline(1, stages, []int{1, 2}))
		sameRaster(t, fmt.Sprint(run, " ring 1"), dst[0], values[1])
		sameRaster(t, fmt.Sprint(run, " ring 2"), dst[1], values[2])
	}
}
