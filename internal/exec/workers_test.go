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

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
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
									minBandW:  []int{0, 8, 3}[n%3],
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
// a context when the count reaches after, before running that call.
type countCalls struct {
	exec.Kernel
	after  int64
	calls  *atomic.Int64
	cancel context.CancelFunc
}

func (k countCalls) Process(dst exec.Span, src exec.Window) {
	if k.calls.Add(1) == k.after {
		k.cancel()
	}
	k.Kernel.Process(dst, src)
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
