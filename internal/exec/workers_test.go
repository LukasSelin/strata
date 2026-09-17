package exec_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"strata/engine"
	"strata/internal/exec"
	"strata/raster"
)

// TestTilesAndWorkers is the §23 contract: for every tile width and height
// in {1, 7, 64, 256, full, larger than the raster} and every worker count
// in {1, 2, 3, GOMAXPROCS}, Process gives the whole-raster call's bits in
// Data (on valid cells) and in every validity word. It runs Clamp, Add,
// Slope, Aspect, Hillshade, Gradient and a radius-2 box kernel, without
// masks, with masks on the inputs and outputs, and with masks only on the
// outputs, on an odd-sized compact raster and on two windows whose Stride
// exceeds Width and is not a multiple of 64 (one wider than 256, one
// taller). Bands are one row or 97 cells, so that workers share many
// bands even on small rasters.
func TestTilesAndWorkers(t *testing.T) {
	type shape struct {
		name     string
		w, h     int
		windowed bool
		pad      int
	}
	shapes := []shape{
		{"odd-compact", 61, 37, false, 0},
		{"wide-window", 300, 9, true, 5},
		{"tall-window", 21, 131, true, 37},
	}
	dims := []int{1, 7, 64, 256, 0, 1 << 20}
	workers := []int{1, 2, 3, runtime.GOMAXPROCS(0)}

	ops := []adapter{box2Adapter}
	for _, name := range []string{"clamp", "add", "slope-degrees", "aspect", "hillshade", "gradient"} {
		ops = append(ops, adapterNamed(t, name))
	}
	masks := []struct{ in, out bool }{{false, false}, {true, true}, {false, true}}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			for _, sh := range shapes {
				for _, m := range masks {
					rng := rand.New(rand.NewPCG(uint64(sh.w*1000+sh.h), 11))
					var ins, outs []operand
					for i := range op.inputs {
						ins = append(ins, newOperandPad(rng, sh.w, sh.h, sh.windowed, m.in && (i == 0 || rng.IntN(2) == 0), sh.pad))
					}
					for range op.outputs {
						outs = append(outs, newOperandPad(rng, sh.w, sh.h, sh.windowed, m.out, sh.pad))
					}
					want := cloneAll(ins, outs, false)
					op.direct(want.dst(), want.src())
					n := 0
					for _, tw := range dims {
						for _, th := range dims {
							for _, wk := range workers {
								run := engineRun{
									opts:      engine.Options{TileWidth: tw, TileHeight: th, Workers: wk},
									bandCells: []int{1, 97}[n%2],
								}
								n++
								id := fmt.Sprintf("%s %dx%d inMask=%v outMask=%v %v", sh.name, sh.w, sh.h, m.in, m.out, run)
								got := cloneAll(ins, outs, false)
								processWith(t, run, func(ctx context.Context, o engine.Options) error {
									return op.tiled(ctx, got.dst(), got.src(), o)
								})
								for i := range outs {
									requireSameRoots(t, fmt.Sprintf("%s dst[%d]", id, i), got.outs[i].root, want.outs[i].root)
								}
								for i := range ins {
									requireSameRoots(t, fmt.Sprintf("%s src[%d]", id, i), got.ins[i].root, want.ins[i].root)
								}
							}
						}
					}
				}
			}
		})
	}
}

// box2Adapter is the radius-2 box kernel with its naive reference.
var box2Adapter = adapter{
	name: "box-r2", inputs: 1, outputs: 1,
	direct: func(dst, src []raster.Float32Raster) { naiveBox(dst[0], src, 2, float32(math.NaN())) },
	tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
		return exec.ProcessN(ctx, dst, src, boxKernel{r: 2, inputs: 1, outputs: 1}, o)
	},
}

func adapterNamed(t *testing.T, name string) adapter {
	for _, a := range adapters {
		if a.name == name {
			return a
		}
	}
	t.Fatalf("no adapter %q", name)
	return adapter{}
}

// countCalls wraps a kernel, counts Process calls atomically and cancels
// a context when the count reaches after, before running that call. late,
// if not nil, counts the calls that started after cancel returned.
type countCalls struct {
	exec.Kernel
	after     int64
	calls     *atomic.Int64
	cancel    context.CancelFunc
	cancelled *atomic.Bool
	late      *atomic.Int64
}

func (k countCalls) Process(dst exec.Span, src exec.Window) {
	if k.late != nil && k.cancelled.Load() {
		k.late.Add(1)
	}
	if k.calls.Add(1) == k.after {
		k.cancel()
		if k.cancelled != nil {
			k.cancelled.Store(true)
		}
	}
	k.Kernel.Process(dst, src)
}

// TestCancellationWorkers cancels a radius-1 kernel from inside a kernel call with
// several workers. ProcessN must return context.Canceled once every
// worker has exited, having started no kernel call after the cancellation
// except in bands already claimed (at most one per other worker). Every
// output cell must be final (Data and validity) or untouched, the
// finished bands must be a prefix of the plan, and no goroutine may
// outlive the call.
func TestCancellationWorkers(t *testing.T) {
	defer exec.SetBandCells(1)() // one-row bands
	const w, h = 40, 30
	rng := rand.New(rand.NewPCG(4, 5))
	dem := newOperand(rng, w, h, true, true)
	out := newOperand(rng, w, h, true, true)
	box := boxKernel{r: 1, inputs: 1}
	final := out.clone()
	naiveBox(final.r, []raster.Float32Raster{dem.r}, 1, float32(math.NaN()))

	for _, workers := range []int{1, 2, 3, 8, runtime.GOMAXPROCS(0)} {
		for _, tiles := range [][2]int{{0, 0}, {7, 4}, {1, 1}} {
			for _, after := range []int64{1, 5, 17} {
				o := engine.Options{TileWidth: tiles[0], TileHeight: tiles[1], Workers: workers}
				id := fmt.Sprintf("%+v after=%d", o, after)
				got := out.clone()
				ctx, cancel := context.WithCancel(context.Background())
				var calls, late atomic.Int64
				var cancelled atomic.Bool
				k := countCalls{box, after, &calls, cancel, &cancelled, &late}
				err := exec.Process(ctx, got.r, dem.r, k, o)
				cancel()
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("%s: err = %v, want context.Canceled", id, err)
				}
				if calls.Load() < after {
					t.Fatalf("%s: kernel called %d times, want at least %d", id, calls.Load(), after)
				}
				if n := late.Load(); n > int64(workers-1) {
					t.Fatalf("%s: %d kernel calls started after cancellation, want at most %d", id, n, workers-1)
				}
				requireFinalOrUntouched(t, id, got, final, out)
				requirePrefix(t, id, got, final, exec.Bands(w, h, tiles[0], tiles[1]))
				requireNoLeaks(t, id)
			}
		}
	}
}

// requirePrefix checks that the bands whose cells all hold final results
// come before every band that has an untouched cell, in plan order.
// requireFinalOrUntouched has already checked each cell is one or the
// other; bands whose final bits equal the original ones cannot occur with
// these random fixtures.
func requirePrefix(t *testing.T, id string, got, final operand, bands [][4]int) {
	t.Helper()
	g, f := got.r, final.r
	isFinal := func(b [4]int) bool {
		for y := b[1]; y < b[3]; y++ {
			for x := b[0]; x < b[2]; x++ {
				i := g.Index(x, y)
				if g.IsValid(x, y) != f.IsValid(x, y) || !sameFloat(g.Data[i], f.Data[i]) {
					return false
				}
			}
		}
		return true
	}
	done := 0
	for done < len(bands) && isFinal(bands[done]) {
		done++
	}
	for i := done; i < len(bands); i++ {
		if isFinal(bands[i]) {
			t.Fatalf("%s: band %d %v is finished but band %d %v is not", id, i, bands[i], done, bands[done])
		}
	}
}

// panicAt panics with value in the band starting at row y, or in every
// band when y < 0.
type panicAt struct {
	exec.Kernel
	y     int
	value any
	calls *atomic.Int64
}

func (k panicAt) Process(dst exec.Span, src exec.Window) {
	k.calls.Add(1)
	if k.y < 0 || dst.Y == k.y {
		panic(k.value)
	}
	k.Kernel.Process(dst, src)
}

// TestKernelPanic checks that a kernel panic in any worker reaches the
// caller of Process as the same value, after every worker has stopped,
// and that the workers stop claiming bands.
func TestKernelPanic(t *testing.T) {
	defer exec.SetBandCells(1)()
	const w, h = 40, 60
	rng := rand.New(rand.NewPCG(6, 7))
	dem := newOperand(rng, w, h, false, true)
	errBoom := errors.New("boom")
	for _, workers := range []int{1, 2, 4, runtime.GOMAXPROCS(0)} {
		for _, y := range []int{1, 30, h - 2, -1} {
			for _, value := range []any{"kernel failed", errBoom} {
				id := fmt.Sprintf("workers=%d y=%d value=%v", workers, y, value)
				dst := raster.NewFloat32Like(dem.r)
				var calls atomic.Int64
				k := panicAt{boxKernel{r: 1, inputs: 1}, y, value, &calls}
				got := func() (r any) {
					defer func() { r = recover() }()
					_ = exec.Process(context.Background(), dst, dem.r, k, engine.Options{Workers: workers})
					return nil
				}()
				if got != value {
					t.Fatalf("%s: recovered %v, want %v", id, got, value)
				}
				if y < 0 && calls.Load() > int64(workers) {
					t.Fatalf("%s: %d kernel calls after panics in every band, want at most %d", id, calls.Load(), workers)
				}
				requireNoLeaks(t, id)
			}
		}
	}
}

// concurrency records how many Process calls run at once. Each call
// sleeps, so calls overlap whenever the engine runs them concurrently.
type concurrency struct {
	exec.Kernel
	now, peak *atomic.Int64
}

func (k concurrency) Process(dst exec.Span, src exec.Window) {
	n := k.now.Add(1)
	for {
		p := k.peak.Load()
		if n <= p || k.peak.CompareAndSwap(p, n) {
			break
		}
	}
	time.Sleep(2 * time.Millisecond)
	k.Kernel.Process(dst, src)
	k.now.Add(-1)
}

// TestWorkersRunConcurrently checks that Workers bounds the number of
// concurrent kernel calls, that more than one worker actually runs, and
// that Workers == 1 runs one call at a time.
func TestWorkersRunConcurrently(t *testing.T) {
	defer exec.SetBandCells(1)()
	rng := rand.New(rand.NewPCG(8, 9))
	src := newOperand(rng, 16, 34, false, false)
	for _, tc := range []struct{ workers, rows, maxPeak, minPeak int64 }{
		{1, 34, 1, 1},
		{3, 34, 3, 2},
		{8, 34, 8, 2},
		{8, 4, 2, 1}, // two interior rows, so at most two kernel calls
	} {
		s := src.r.Window(0, 0, 16, int(tc.rows))
		dst := raster.NewFloat32Like(s)
		var now, peak atomic.Int64
		k := concurrency{boxKernel{r: 1, inputs: 1}, &now, &peak}
		if err := exec.Process(context.Background(), dst, s, k, engine.Options{Workers: int(tc.workers)}); err != nil {
			t.Fatal(err)
		}
		if p := peak.Load(); p > tc.maxPeak || p < tc.minPeak {
			t.Errorf("workers=%d, %d rows: %d concurrent kernel calls, want %d to %d",
				tc.workers, s.Height, p, tc.minPeak, tc.maxPeak)
		}
	}
}
