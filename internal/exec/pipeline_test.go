package exec_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// mulKernel writes the product of its inputs. Pointwise, and with no
// reduction over a neighbourhood, so a pipeline of them is exactly the
// weighted factor product §52 is written for.
type mulKernel struct{ inputs int }

func (mulKernel) Radius() int                    { return 0 }
func (k mulKernel) Arity() (inputs, outputs int) { return k.inputs, 1 }

func (k mulKernel) Process(dst exec.Span, src exec.Window) {
	out := dst.Dst[0]
	for y := range dst.Height {
		row := out.Row(y)
		for x := range dst.Width {
			v := src.Src[0].Data[src.Src[0].Index(x, y)]
			for _, in := range src.Src[1:] {
				v *= in.Data[in.Index(x, y)]
			}
			row[x] = v
		}
	}
}

// Fuse makes a two-input mulKernel a step of a fused chain, so the
// pipeline tests below run the fused path as well as the staged one. A
// wider one is not a chain step — a step combines the running value with
// one operand — and says so, which is what the ok result is for.
func (k mulKernel) Fuse() (vec.Step, bool) {
	return vec.Step{Op: vec.OpMul}, k.inputs == 2
}

// chain returns the stages of an n-input product: n-1 binary multiplies,
// each folding the running value with the next input. Value ids are the
// n inputs, then one per stage.
func chain(n int) (stages []exec.Stage, out int) {
	acc := 0
	for i := 1; i < n; i++ {
		stages = append(stages, exec.Stage{Kernel: mulKernel{inputs: 2}, In: []int{acc, i}})
		acc = n + len(stages) - 1
	}
	return stages, acc
}

// unfused runs stages one at a time over whole rasters, the way a caller
// would today, and returns the value the pipeline would write. It is the
// reference every pipeline test compares against: a fused chain that
// does not equal its unfused form is a bug, and the unfused form has a
// reference of its own already.
func unfused(t *testing.T, src []raster.Float32Raster, stages []exec.Stage, out int) raster.Float32Raster {
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
	return values[out]
}

// requireFused fails unless p lowered to a fused chain, so that a test
// written for the pipeline keeps covering the fused path rather than
// quietly falling back to the staged one (DESIGN.md §29).
func requireFused(t *testing.T, id string, p *exec.Pipeline) {
	t.Helper()
	if !p.Fused() {
		t.Fatalf("%s: the pipeline did not lower to a fused chain", id)
	}
}

func sameRaster(t *testing.T, id string, got, want raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			g, w := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]
			if !sameFloat(g, w) {
				t.Fatalf("%s: Data at (%d, %d) = %v, want %v", id, x, y, g, w)
			}
			if want.Valid == nil {
				continue
			}
			if got.IsValid(x, y) != want.IsValid(x, y) {
				t.Fatalf("%s: validity at (%d, %d) = %v, want %v",
					id, x, y, got.IsValid(x, y), want.IsValid(x, y))
			}
		}
	}
}

// TestPipelineMatchesUnfused runs the §23 matrix: every tile size and
// worker count, windowed and compact operands, with and without masks,
// against the same stages run as separate calls over whole rasters.
func TestPipelineMatchesUnfused(t *testing.T) {
	const w, h = 61, 43
	const inputs = 4
	stages, out := chain(inputs)

	for _, masked := range []bool{false, true} {
		for _, windowed := range []bool{false, true} {
			rng := rand.New(rand.NewPCG(7, 99))
			src := make([]raster.Float32Raster, inputs)
			for i := range src {
				src[i] = newOperand(rng, w, h, windowed, masked).r
			}
			want := unfused(t, src, stages, out)

			for _, workers := range []int{1, 2, 5} {
				for _, tile := range [][2]int{{0, 0}, {16, 16}, {0, 8}, {7, 3}} {
					id := fmt.Sprintf("masked=%v windowed=%v workers=%d tile=%v",
						masked, windowed, workers, tile)
					dst := raster.NewFloat32Like(src[0])
					p := exec.NewPipeline(inputs, stages, out)
					requireFused(t, id, p)
					err := exec.ProcessN(context.Background(),
						[]raster.Float32Raster{dst}, src, p,
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

// TestPipelineChunkedMatchesUnfused is the same reference through the
// chunked driver, where the pipeline runs inside a tile that has been
// copied into a worker's buffer.
func TestPipelineChunkedMatchesUnfused(t *testing.T) {
	const w, h = 53, 37
	const inputs = 4
	stages, out := chain(inputs)

	for _, masked := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(11, 3))
		src := make([]raster.Float32Raster, inputs)
		for i := range src {
			src[i] = newOperand(rng, w, h, false, masked).r
		}
		want := unfused(t, src, stages, out)

		for _, workers := range []int{1, 3} {
			for _, tile := range [][2]int{{0, 0}, {16, 16}, {0, 8}, {9, 5}} {
				id := fmt.Sprintf("masked=%v workers=%d tile=%v", masked, workers, tile)
				dst := raster.NewFloat32Like(src[0])
				sources := make([]engine.RasterSource, inputs)
				for i := range sources {
					sources[i] = engine.NewMemorySource(src[i])
				}
				p := exec.NewPipeline(inputs, stages, out)
				requireFused(t, id, p)
				err := exec.ProcessChunked(context.Background(),
					[]engine.RasterSink{engine.NewMemorySink(dst)}, sources, p,
					engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers})
				if err != nil {
					t.Fatalf("%s: %v", id, err)
				}
				sameRaster(t, id, dst, want)
			}
		}
	}
}

// TestPipelineSingleStage checks the identity case, which is where
// wiring mistakes show up first: a pipeline of one stage must be that
// stage.
func TestPipelineSingleStage(t *testing.T) {
	const w, h = 33, 21
	rng := rand.New(rand.NewPCG(5, 5))
	src := []raster.Float32Raster{
		newOperand(rng, w, h, true, true).r,
		newOperand(rng, w, h, true, true).r,
	}
	k := mulKernel{inputs: 2}

	want := raster.NewFloat32Like(src[0])
	if err := exec.ProcessN(context.Background(), []raster.Float32Raster{want}, src, k,
		engine.Options{}); err != nil {
		t.Fatalf("plain: %v", err)
	}

	got := raster.NewFloat32Like(src[0])
	p := exec.NewPipeline(2, []exec.Stage{{Kernel: k, In: []int{0, 1}}}, 2)
	requireFused(t, "one stage", p)
	if err := exec.ProcessN(context.Background(), []raster.Float32Raster{got}, src, p,
		engine.Options{TileHeight: 4, Workers: 3}); err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	sameRaster(t, "one stage", got, want)
}

// TestPipelineReusesAValue checks the wiring a pure chain never
// exercises: one input feeding two stages, and one stage's output read
// twice.
func TestPipelineReusesAValue(t *testing.T) {
	const w, h = 29, 17
	rng := rand.New(rand.NewPCG(2, 8))
	src := make([]raster.Float32Raster, 3)
	for i := range src {
		src[i] = newOperand(rng, w, h, false, true).r
	}
	// 3 = 0*1, 4 = 3*3, 5 = 4*2
	stages := []exec.Stage{
		{Kernel: mulKernel{inputs: 2}, In: []int{0, 1}},
		{Kernel: mulKernel{inputs: 2}, In: []int{3, 3}},
		{Kernel: mulKernel{inputs: 2}, In: []int{4, 2}},
	}
	want := unfused(t, src, stages, 5)

	dst := raster.NewFloat32Like(src[0])
	p := exec.NewPipeline(3, stages, 5)
	// A value read twice is two live intermediates, which is off §29's
	// left-deep cut: this pipeline must run staged.
	if p.Fused() {
		t.Fatal("a pipeline that reuses a value lowered to a fused chain")
	}
	if err := exec.ProcessN(context.Background(), []raster.Float32Raster{dst}, src, p,
		engine.Options{TileWidth: 8, TileHeight: 5, Workers: 4}); err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	sameRaster(t, "reused value", dst, want)
}

// TestPipelineTraffic is §52's acceptance test. A six-input product runs
// as five chained calls or as one pipeline; the pipeline must move what
// one pass over six inputs and one output moves, and no more.
func TestPipelineTraffic(t *testing.T) {
	const w, h = 256, 256
	const inputs = 6
	stages, out := chain(inputs)
	cells := int64(w * h)

	src := make([]raster.Float32Raster, inputs)
	for i := range src {
		src[i] = ramp(w, h)
	}

	// Five chained binary multiplies, one Stats totalling them.
	var chained engine.Stats
	acc := ramp(w, h)
	for i := 1; i < inputs; i++ {
		err := exec.ProcessN(context.Background(), []raster.Float32Raster{acc},
			[]raster.Float32Raster{acc, src[i]}, mulKernel{inputs: 2},
			engine.Options{Workers: 4, Stats: &chained})
		if err != nil {
			t.Fatalf("chained %d: %v", i, err)
		}
	}
	if got := float64(chained.Total()) / float64(cells); got != 60 {
		t.Errorf("chained tiled = %v B/cell, want 60 (%v)", got, chained)
	}

	var fusedStats engine.Stats
	dst := ramp(w, h)
	p := exec.NewPipeline(inputs, stages, out)
	requireFused(t, "traffic", p)
	if err := exec.ProcessN(context.Background(), []raster.Float32Raster{dst}, src, p,
		engine.Options{Workers: 4, Stats: &fusedStats}); err != nil {
		t.Fatalf("fused: %v", err)
	}
	if got := float64(fusedStats.Total()) / float64(cells); got != 28 {
		t.Errorf("fused tiled = %v B/cell, want 28 (%v)", got, fusedStats)
	}
	if got := fusedStats.Amplification(); got != 1 {
		t.Errorf("fused tiled Amplification = %v, want 1 (%v)", got, fusedStats)
	}

	var fusedChunk engine.Stats
	out2 := ramp(w, h)
	sources := make([]engine.RasterSource, inputs)
	for i := range sources {
		sources[i] = engine.NewMemorySource(src[i])
	}
	err := exec.ProcessChunked(context.Background(),
		[]engine.RasterSink{engine.NewMemorySink(out2)}, sources,
		exec.NewPipeline(inputs, stages, out),
		engine.Options{TileHeight: 64, Workers: 4, Stats: &fusedChunk})
	if err != nil {
		t.Fatalf("fused chunked: %v", err)
	}
	if got := float64(fusedChunk.Total()) / float64(cells); got != 56 {
		t.Errorf("fused chunked = %v B/cell, want 56 (%v)", got, fusedChunk)
	}
	if got := fusedChunk.Amplification(); got != 2 {
		t.Errorf("fused chunked Amplification = %v, want 2 (%v)", got, fusedChunk)
	}
}

// spyKernel records the largest span it is asked to process and the size
// the engine sized its scratch for. It exists to check that second
// number bounds the first, which is the whole of plan.spanSize and
// chunkJob.spanSize — the arithmetic that says a clipped tile takes
// taller bands and may therefore cover more cells than a full one.
type spyKernel struct {
	mu       sync.Mutex
	maxSpan  int
	asked    int
	askedW   int
	askedH   int
	askCount int
}

func (*spyKernel) Radius() int                  { return 0 }
func (*spyKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k *spyKernel) Scratch(w, h int) exec.ScratchSize {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.asked, k.askedW, k.askedH = w*h, w, h
	k.askCount++
	return exec.ScratchSize{Cells: w * h, Views: 2}
}

func (k *spyKernel) Process(dst exec.Span, src exec.Window) {
	k.mu.Lock()
	if n := dst.Width * dst.Height; n > k.maxSpan {
		k.maxSpan = n
	}
	k.mu.Unlock()
	if len(dst.Scratch.Cells) < dst.Width*dst.Height {
		panic(fmt.Sprintf("scratch %d cells for a %d×%d span",
			len(dst.Scratch.Cells), dst.Width, dst.Height))
	}
	// Touch every cell of the scratch this span is entitled to, so an
	// undersized allocation trips the bounds check rather than passing.
	sc := dst.Scratch.Cells[:dst.Width*dst.Height]
	for y := range dst.Height {
		row := dst.Dst[0].Row(y)
		for x := range dst.Width {
			sc[y*dst.Width+x] = src.Src[0].Data[src.Src[0].Index(x, y)]
			row[x] = sc[y*dst.Width+x]
		}
	}
}

// TestScratchBoundsEverySpan checks the engine's span arithmetic over
// raster and tile sizes that do not divide each other, including tiles
// one cell wide, where a clipped tile's own plan gives it the tallest
// bands in the call.
func TestScratchBoundsEverySpan(t *testing.T) {
	defer exec.SetBandCells(64)() // small enough that band splitting bites
	for _, size := range [][2]int{{37, 23}, {64, 64}, {129, 7}} {
		for _, tile := range [][2]int{{0, 0}, {1, 9}, {16, 16}, {13, 5}, {64, 1}} {
			for _, chunked := range []bool{false, true} {
				k := &spyKernel{}
				opts := engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: 3}
				src, dst := ramp(size[0], size[1]), ramp(size[0], size[1])
				var err error
				if chunked {
					err = exec.ProcessChunked(context.Background(),
						[]engine.RasterSink{engine.NewMemorySink(dst)},
						[]engine.RasterSource{engine.NewMemorySource(src)}, k, opts)
				} else {
					err = exec.ProcessN(context.Background(),
						[]raster.Float32Raster{dst}, []raster.Float32Raster{src}, k, opts)
				}
				id := fmt.Sprintf("size=%v tile=%v chunked=%v", size, tile, chunked)
				if err != nil {
					t.Fatalf("%s: %v", id, err)
				}
				if k.askCount == 0 {
					t.Fatalf("%s: Scratch was never called", id)
				}
				if k.maxSpan > k.asked {
					t.Errorf("%s: largest span %d cells, scratch sized for %d (%d×%d)",
						id, k.maxSpan, k.asked, k.askedW, k.askedH)
				}
			}
		}
	}
}

func TestPipelineChecks(t *testing.T) {
	ok := mulKernel{inputs: 2}
	stage := func(k exec.Kernel, in ...int) exec.Stage { return exec.Stage{Kernel: k, In: in} }

	cases := []struct {
		name string
		want string
		fn   func()
	}{
		{"no stages", "no stages", func() { exec.NewPipeline(1, nil, 1) }},
		{"no inputs", "at least one", func() { exec.NewPipeline(0, []exec.Stage{stage(ok, 0, 0)}, 1) }},
		{"nil kernel", "nil kernel", func() { exec.NewPipeline(2, []exec.Stage{stage(nil, 0, 1)}, 2) }},
		{"radius", "radius 1", func() {
			exec.NewPipeline(1, []exec.Stage{stage(boxKernel{r: 1, inputs: 1, outputs: 1}, 0)}, 1)
		}},
		{"arity", "names 1 inputs", func() { exec.NewPipeline(2, []exec.Stage{stage(ok, 0)}, 2) }},
		{"forward reference", "not defined before it", func() {
			exec.NewPipeline(2, []exec.Stage{stage(ok, 0, 2)}, 2)
		}},
		{"output is an input", "not produced by a stage", func() {
			exec.NewPipeline(2, []exec.Stage{stage(ok, 0, 1)}, 1)
		}},
		{"output out of range", "not produced by a stage", func() {
			exec.NewPipeline(2, []exec.Stage{stage(ok, 0, 1)}, 9)
		}},
		{"unreachable input", "cannot be reached", func() {
			// 3 = 0*1, 4 = 3*3; input 2 is never read.
			exec.NewPipeline(3, []exec.Stage{stage(ok, 0, 1), stage(ok, 3, 3)}, 4)
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				v := recover()
				if v == nil {
					t.Fatalf("no panic")
				}
				if msg, _ := v.(string); !contains(msg, c.want) {
					t.Fatalf("panic %q does not mention %q", v, c.want)
				}
			}()
			c.fn()
		})
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
