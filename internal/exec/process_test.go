package exec_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// boxKernel writes the mean of the (2r+1)×(2r+1) neighbourhood of every
// input, summed in row-major order per input, to every output. It is
// deliberately naive (scalar, per cell) so the engine's radius handling
// can be checked against a reference that shares no code with it.
type boxKernel struct{ r, inputs, outputs int }

func (k boxKernel) Radius() int                  { return k.r }
func (k boxKernel) Arity() (inputs, outputs int) { return k.inputs, max(k.outputs, 1) }

func (k boxKernel) Process(dst exec.Span, src exec.Window) {
	n := float32(k.inputs * (2*k.r + 1) * (2*k.r + 1))
	for y := range dst.Height {
		for x := range dst.Width {
			var sum float32
			for _, in := range src.Src {
				for j := range 2*k.r + 1 {
					for i := range 2*k.r + 1 {
						sum += in.Data[in.Index(x+i, y+j)]
					}
				}
			}
			for _, out := range dst.Dst {
				out.Row(y)[x] = sum / n
			}
		}
	}
}

// edgeBox is boxKernel with a declared edge value.
type edgeBox struct {
	boxKernel
	edge float32
}

func (k edgeBox) Edge() float32 { return k.edge }

// naiveBox is the per-cell reference for boxKernel over whole rasters.
func naiveBox(dst raster.Float32Raster, src []raster.Float32Raster, r int, edge float32) {
	w, h := dst.Width, dst.Height
	n := float32(len(src) * (2*r + 1) * (2*r + 1))
	for y := range h {
		for x := range w {
			i := dst.Index(x, y)
			if x < r || y < r || x >= w-r || y >= h-r {
				dst.Data[i] = edge
				if dst.Valid != nil {
					dst.SetValid(x, y, false)
				}
				continue
			}
			var sum float32
			valid := true
			for _, in := range src {
				for j := -r; j <= r; j++ {
					for k := -r; k <= r; k++ {
						sum += in.Data[in.Index(x+k, y+j)]
						valid = valid && in.IsValid(x+k, y+j)
					}
				}
			}
			dst.Data[i] = sum / n
			if dst.Valid != nil {
				dst.SetValid(x, y, valid)
			}
		}
	}
}

// TestRadiusMatchesNaive runs box kernels of radius 0 to 3 with one or
// two inputs and outputs through every engine run and compares them with
// the naive reference: edges of width r get the edge value and are
// invalid, and validity is every input's neighbourhood eroded by r.
func TestRadiusMatchesNaive(t *testing.T) {
	sizes := [][2]int{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 3}, {7, 9}, {13, 6}, {66, 8}, {131, 7}}
	for _, r := range []int{0, 1, 2, 3} {
		for _, arity := range [][2]int{{1, 1}, {2, 1}, {1, 2}} {
			for _, sz := range sizes {
				for _, lay := range layouts {
					for _, masks := range []struct{ in, out bool }{{false, false}, {false, true}, {true, true}} {
						for _, custom := range []bool{false, true} {
							testBox(t, r, arity[0], arity[1], sz[0], sz[1], lay, masks.in, masks.out, custom)
						}
					}
				}
			}
		}
	}
}

func testBox(t *testing.T, r, inputs, outputs, w, h int, lay layout, inMask, outMask, customEdge bool) {
	rng := rand.New(rand.NewPCG(uint64(r*100+inputs*10+outputs), uint64(w*1000+h)))
	var ins, outs []operand
	for i := range inputs {
		ins = append(ins, newOperand(rng, w, h, lay.in, inMask && (i == 0 || rng.IntN(2) == 0)))
	}
	for range outputs {
		outs = append(outs, newOperand(rng, w, h, lay.out, outMask))
	}
	box := boxKernel{r: r, inputs: inputs, outputs: outputs}
	var k exec.Kernel = box
	edge := float32(math.NaN())
	if customEdge {
		edge = -7
		k = edgeBox{box, edge}
	}
	for _, run := range engineRuns {
		id := fmt.Sprintf("r=%d inputs=%d outputs=%d %dx%d %v inMask=%v outMask=%v edge=%v %v",
			r, inputs, outputs, w, h, lay, inMask, outMask, edge, run)
		want, got := cloneAll(ins, outs, false), cloneAll(ins, outs, false)
		for _, out := range want.outs {
			naiveBox(out.r, want.src(), r, edge)
		}
		process(t, run, got.dst(), got.src(), k)
		for i := range outs {
			requireSameRoots(t, fmt.Sprintf("%s dst[%d]", id, i), got.outs[i].root, want.outs[i].root)
			// Data under invalid cells is unspecified in general, but
			// edges must carry the edge value whatever the mask says.
			o := got.outs[i].r
			for y := range h {
				for x := range w {
					if (x < r || y < r || x >= w-r || y >= h-r) && !sameFloat(o.Data[o.Index(x, y)], edge) {
						t.Fatalf("%s: dst[%d] edge cell (%d, %d) = %g, want %g", id, i, x, y, o.Data[o.Index(x, y)], edge)
					}
				}
			}
		}
	}
}

// posKernel writes 1000·Y + X for every cell, from the span's position.
type posKernel struct{ r int }

func (k posKernel) Radius() int                { return k.r }
func (posKernel) Arity() (inputs, outputs int) { return 1, 1 }
func (posKernel) Process(dst exec.Span, src exec.Window) {
	out := dst.Dst[0]
	for y := range dst.Height {
		for x := range dst.Width {
			out.Row(y)[x] = float32(1000*(dst.Y+y) + dst.X + x)
		}
	}
}

func TestSpanPosition(t *testing.T) {
	src := newOperand(rand.New(rand.NewPCG(1, 1)), 23, 17, true, false)
	for _, r := range []int{0, 2} {
		for _, run := range engineRuns {
			dst := raster.NewFloat32Like(src.r)
			process(t, run, []raster.Float32Raster{dst}, []raster.Float32Raster{src.r}, posKernel{r})
			for y := r; y < dst.Height-r; y++ {
				for x := r; x < dst.Width-r; x++ {
					if got, want := dst.Data[dst.Index(x, y)], float32(1000*y+x); got != want {
						t.Fatalf("r=%d %v: cell (%d, %d) = %g, want %g", r, run, x, y, got, want)
					}
				}
			}
		}
	}
}

// TestCancellation cancels a radius-1 kernel after some bands with one
// worker. ProcessN must return context.Canceled without calling the
// kernel again, the bands it finished must hold final results (Data and
// validity), and every other cell must be untouched.
// TestCancellationWorkers covers several workers.
func TestCancellation(t *testing.T) {
	defer exec.SetBandCells(1)() // one-row bands
	const w, h = 40, 30
	rng := rand.New(rand.NewPCG(4, 4))
	dem := newOperand(rng, w, h, true, true)
	out := newOperand(rng, w, h, true, true)
	box := boxKernel{r: 1, inputs: 1}

	final := out.clone()
	naiveBox(final.r, []raster.Float32Raster{dem.r}, 1, float32(math.NaN()))

	for _, tiles := range []engine.Options{{Workers: 1}, {TileWidth: 7, TileHeight: 4, Workers: 1}} {
		for _, after := range []int64{1, 5, 17} {
			id := fmt.Sprintf("tiles=%+v after=%d", tiles, after)
			got := out.clone()
			ctx, cancel := context.WithCancel(context.Background())
			var calls atomic.Int64
			k := countCalls{box, after, &calls, cancel}
			err := exec.Process(ctx, got.r, dem.r, k, tiles)
			cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("%s: err = %v, want context.Canceled", id, err)
			}
			if calls.Load() != after {
				t.Fatalf("%s: kernel called %d times, want %d", id, calls.Load(), after)
			}
			if tiles.TileWidth == 0 {
				// One tile: bands are rows top-down. Row 0 is all edge (no
				// kernel call), so after n calls rows 0..n are done.
				requireRows(t, id, got, final, out, int(after)+1)
				continue
			}
			requireFinalOrUntouched(t, id, got, final, out)
		}
	}

	// A context that is already done writes nothing and calls nothing,
	// with any number of workers.
	for _, workers := range []int{1, 4} {
		id := fmt.Sprintf("expired context, workers=%d", workers)
		got := out.clone()
		ctx, cancel := context.WithTimeout(context.Background(), -time.Second)
		var calls atomic.Int64
		k := countCalls{box, -1, &calls, cancel}
		err := exec.Process(ctx, got.r, dem.r, k, engine.Options{Workers: workers})
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("%s: err = %v", id, err)
		}
		if calls.Load() != 0 {
			t.Fatalf("%s: %d kernel calls", id, calls.Load())
		}
		requireSameRoots(t, id, got.root, out.root)
		for i := range got.root.Data {
			if math.Float32bits(got.root.Data[i]) != math.Float32bits(out.root.Data[i]) {
				t.Fatalf("%s: root cell %d changed", id, i)
			}
		}
	}
}

// requireRows checks that output rows [0, done) match final and every
// other bit of the root matches orig, bit for bit.
func requireRows(t *testing.T, id string, got, final, orig operand, done int) {
	t.Helper()
	for y := range got.r.Height {
		for x := range got.r.Width {
			want := orig
			if y < done {
				want = final
			}
			requireCell(t, id, got, want, x, y)
		}
	}
	requireOutsideUntouched(t, id, got, orig)
}

// requireFinalOrUntouched checks that every output cell matches either
// final or orig, Data and validity together.
func requireFinalOrUntouched(t *testing.T, id string, got, final, orig operand) {
	t.Helper()
	g, f, o := got.r, final.r, orig.r
	for y := range g.Height {
		for x := range g.Width {
			i := g.Index(x, y)
			isOrig := math.Float32bits(g.Data[i]) == math.Float32bits(o.Data[i]) && g.IsValid(x, y) == o.IsValid(x, y)
			isFinal := g.IsValid(x, y) == f.IsValid(x, y) && (!g.IsValid(x, y) || sameFloat(g.Data[i], f.Data[i]))
			if !isOrig && !isFinal {
				t.Fatalf("%s: cell (%d, %d) is neither final nor untouched", id, x, y)
			}
		}
	}
	requireOutsideUntouched(t, id, got, orig)
}

func requireCell(t *testing.T, id string, got, want operand, x, y int) {
	t.Helper()
	g, w := got.r, want.r
	i := g.Index(x, y)
	if g.IsValid(x, y) != w.IsValid(x, y) {
		t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", id, x, y, g.IsValid(x, y), w.IsValid(x, y))
	}
	if w.IsValid(x, y) && !sameFloat(g.Data[i], w.Data[i]) {
		t.Fatalf("%s: cell (%d, %d) = %g, want %g", id, x, y, g.Data[i], w.Data[i])
	}
}

// requireOutsideUntouched checks the root's Data and bits outside the
// output window.
func requireOutsideUntouched(t *testing.T, id string, got, orig operand) {
	t.Helper()
	in := make(map[int]bool)
	for y := range got.r.Height {
		for x := range got.r.Width {
			in[got.y*got.root.Stride+got.x+got.r.Index(x, y)] = true
		}
	}
	for i := range got.root.Data {
		if in[i] {
			continue
		}
		if math.Float32bits(got.root.Data[i]) != math.Float32bits(orig.root.Data[i]) ||
			raster.MaskGet(got.root.Valid, got.root.ValidOffset+i) != raster.MaskGet(orig.root.Valid, orig.root.ValidOffset+i) {
			t.Fatalf("%s: root cell %d outside the output changed", id, i)
		}
	}
}

// TestNoAllocsPerBand checks that a call's allocations do not depend on
// the number of tiles or bands, and grow by at most two per worker.
// testing.AllocsPerRun runs with GOMAXPROCS 1, so worker counts are
// explicit.
func TestNoAllocsPerBand(t *testing.T) {
	const w, h = 64, 48
	rng := rand.New(rand.NewPCG(9, 9))
	dem := newOperand(rng, w, h, true, true)
	dst := newOperand(rng, w, h, true, true)
	two := newOperand(rng, w, h, false, true)
	slope := boxKernel{r: 1, inputs: 1}
	add := boxKernel{r: 0, inputs: 2}
	box := boxKernel{r: 2, inputs: 2}
	ctx := context.Background()
	cases := map[string]func(engine.Options){
		"radius 1": func(o engine.Options) { _ = exec.Process(ctx, dst.r, dem.r, slope, o) },
		"radius 0": func(o engine.Options) {
			_ = exec.ProcessN(ctx, []raster.Float32Raster{dst.r}, []raster.Float32Raster{dem.r, two.r}, add, o)
		},
		"radius 2": func(o engine.Options) {
			_ = exec.ProcessN(ctx, []raster.Float32Raster{dst.r}, []raster.Float32Raster{dem.r, two.r}, box, o)
		},
	}
	count := func(bandCells int, opts engine.Options, f func(engine.Options)) float64 {
		defer exec.SetBandCells(bandCells)()
		return testing.AllocsPerRun(20, func() { f(opts) })
	}
	for name, f := range cases {
		serial := count(1<<16, engine.Options{Workers: 1}, f) // one band
		for _, workers := range []int{1, 4} {
			// 48 one-row bands, then 528 bands of 3×2 tiles.
			rows := count(1, engine.Options{Workers: workers}, f)
			tiled := count(1, engine.Options{TileWidth: 3, TileHeight: 2, Workers: workers}, f)
			if tiled != rows {
				t.Errorf("%s workers=%d: %v allocs with %d bands, %v with %d", name, workers, rows, h, tiled, w*h/3)
			}
			if limit := serial + float64(2*(workers-1)); rows > limit {
				t.Errorf("%s workers=%d: %v allocs, want at most %v (%v with one worker and one band)",
					name, workers, rows, limit, serial)
			}
		}
	}
}

func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", want)
		}
		msg, _ := r.(string)
		if !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

func TestPanics(t *testing.T) {
	ctx := context.Background()
	r := func(w, h int) raster.Float32Raster { return raster.NewFloat32(w, h, make([]float32, w*h)) }
	one := func(rs ...raster.Float32Raster) []raster.Float32Raster { return rs }
	a, b := r(6, 5), r(6, 5)
	masked := r(6, 5)
	masked.Valid = raster.NewMask(30)
	slope := boxKernel{r: 1, inputs: 1}
	clamp := boxKernel{r: 0, inputs: 1}
	gradient := boxKernel{r: 1, inputs: 1, outputs: 2}
	add := boxKernel{r: 0, inputs: 2}

	mustPanic(t, "engine: nil kernel", func() { _ = exec.Process(ctx, a, b, nil, engine.Options{}) })
	mustPanic(t, "radius -1 is negative", func() { _ = exec.Process(ctx, a, b, boxKernel{r: -1, inputs: 1}, engine.Options{}) })
	mustPanic(t, "needs at least one output", func() {
		_ = exec.ProcessN(ctx, nil, one(a), zeroOut{}, engine.Options{})
	})
	mustPanic(t, "takes 2 inputs and 1 outputs, got 1 and 1", func() { _ = exec.Process(ctx, a, b, add, engine.Options{}) })
	mustPanic(t, "takes 1 inputs and 2 outputs, got 1 and 1", func() { _ = exec.Process(ctx, a, b, gradient, engine.Options{}) })
	mustPanic(t, "negative Options", func() { _ = exec.Process(ctx, a, b, clamp, engine.Options{TileWidth: -1}) })
	mustPanic(t, "negative Options", func() { _ = exec.Process(ctx, a, b, clamp, engine.Options{Workers: -2}) })

	mustPanic(t, "engine: src[0] is 5×6, dst[0] is 6×5", func() { _ = exec.Process(ctx, a, r(5, 6), clamp, engine.Options{}) })
	mustPanic(t, "engine: dst[1] is 6×4, dst[0] is 6×5", func() {
		_ = exec.ProcessN(ctx, one(a, r(6, 4)), one(b), gradient, engine.Options{})
	})
	mustPanic(t, "engine: src[0]: raster: data has", func() {
		short := b
		short.Data = short.Data[:5]
		_ = exec.Process(ctx, a, short, clamp, engine.Options{})
	})
	mustPanic(t, "an input has a validity mask but dst[0].Valid is nil", func() {
		_ = exec.Process(ctx, a, masked, slope, engine.Options{})
	})
	mustPanic(t, "an input has a validity mask but dst[1].Valid is nil", func() {
		_ = exec.ProcessN(ctx, one(raster.NewFloat32Like(masked), b), one(masked), gradient, engine.Options{})
	})

	// Radius > 0: outputs must not share any memory with inputs or each
	// other, judged by span.
	big := r(6, 12)
	mustPanic(t, "dst[0] and src[0] share Data", func() { _ = exec.Process(ctx, a, a, slope, engine.Options{}) })
	mustPanic(t, "dst[0] and src[0] share Data", func() {
		_ = exec.Process(ctx, big.Window(0, 2, 6, 5), big.Window(0, 0, 6, 5), slope, engine.Options{})
	})
	mustPanic(t, "dst[0] and src[0] share validity bits", func() {
		dst := raster.NewFloat32Like(masked)
		dst.Valid = masked.Valid
		_ = exec.Process(ctx, dst, masked, slope, engine.Options{})
	})
	mustPanic(t, "dst[0] and dst[1] share Data", func() {
		_ = exec.ProcessN(ctx, one(a, a), one(b), gradient, engine.Options{})
	})
	// Disjoint windows of one parent are fine, including a shared mask.
	big.Valid = raster.NewMask(72)
	_ = exec.Process(ctx, big.Window(0, 6, 6, 5), big.Window(0, 0, 6, 5), slope, engine.Options{})

	// Radius 0: the same cells are fine (in place), partial overlap is not.
	_ = exec.Process(ctx, a, a, clamp, engine.Options{})
	_ = exec.Process(ctx, masked, masked, clamp, engine.Options{})
	wide := r(10, 10)
	w1, w2 := wide.Window(0, 0, 6, 5), wide.Window(1, 0, 6, 5)
	mustPanic(t, "dst[0] overlaps src[0] at a different offset or stride", func() {
		_ = exec.Process(ctx, w1, w2, clamp, engine.Options{})
	})
	mustPanic(t, "dst[0] overlaps src[1] at a different offset or stride", func() {
		_ = exec.ProcessN(ctx, one(w2), one(b, w1), add, engine.Options{})
	})
	mustPanic(t, "dst[0] validity bits overlap src[0]'s", func() {
		src := r(6, 5)
		src.Valid = raster.NewMask(40)
		dst := raster.NewFloat32Like(src)
		dst.Valid, dst.ValidOffset = src.Valid, 3
		_ = exec.Process(ctx, dst, src, clamp, engine.Options{})
	})
	mustPanic(t, "dst[1] is src[0]; only a kernel with one output runs in place", func() {
		_ = exec.ProcessN(ctx, one(b, a), one(a), splitKernel{}, engine.Options{})
	})
	mustPanic(t, "dst[0] and dst[1] share Data", func() {
		_ = exec.ProcessN(ctx, one(b, b), one(a), splitKernel{}, engine.Options{})
	})
}

// splitKernel copies its input to two outputs.
type splitKernel struct{}

func (splitKernel) Radius() int                  { return 0 }
func (splitKernel) Arity() (inputs, outputs int) { return 1, 2 }
func (splitKernel) Process(dst exec.Span, src exec.Window) {
	for y := range dst.Height {
		copy(dst.Dst[0].Row(y), src.Src[0].Row(y))
		copy(dst.Dst[1].Row(y), src.Src[0].Row(y))
	}
}

// zeroOut is a kernel without outputs, which the engine rejects.
type zeroOut struct{}

func (zeroOut) Radius() int                            { return 0 }
func (zeroOut) Arity() (inputs, outputs int)           { return 1, 0 }
func (zeroOut) Process(dst exec.Span, src exec.Window) {}
