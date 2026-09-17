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
	"testing/synctest"
	"time"

	"strata/engine"
	"strata/internal/exec"
	"strata/raster"
)

// The cancellation tests run in synctest bubbles, where time is virtual:
// it advances only when every goroutine of the bubble is blocked. Each
// band or tile costs one tick of that time (a kernel call or a read that
// sleeps), and a cancellation, deadline or failure lands half a tick
// after a round of claims, while every worker is asleep inside its unit
// of work. So which units are claimed is exact, not bounded by the
// scheduler: with W workers, a stop in round k leaves exactly W·(k+1)
// units claimed and finished, no unit starts after the stop, and the
// call returns only once every worker has woken, finished and exited
// (synctest.Test fails if a goroutine of the bubble is left blocked).

const tick = time.Second

// sleepKernel spends a tick of virtual time in every call, then runs its
// kernel.
type sleepKernel struct {
	exec.Kernel
	calls *atomic.Int64
}

func (k sleepKernel) Process(dst exec.Span, src exec.Window) {
	k.calls.Add(1)
	time.Sleep(tick)
	k.Kernel.Process(dst, src)
}

// stopAfter arranges for ctx's call to stop after round rounds of claims,
// half a tick into the next one, by cancelling ctx or by its deadline,
// and returns the context and a flag set at that moment.
func stopAfter(parent context.Context, round int, deadline bool) (context.Context, *atomic.Bool, context.CancelFunc) {
	at := time.Duration(round)*tick + tick/2
	stopped := new(atomic.Bool)
	if deadline {
		ctx, cancel := context.WithTimeout(parent, at)
		context.AfterFunc(ctx, func() { stopped.Store(true) })
		return ctx, stopped, cancel
	}
	ctx, cancel := context.WithCancel(parent)
	time.AfterFunc(at, func() { stopped.Store(true); cancel() })
	return ctx, stopped, cancel
}

func stopName(deadline bool) string {
	if deadline {
		return "deadline"
	}
	return "cancel"
}

// claimed is the number of units of n that workers claim before a stop
// half a tick into round round (from 0), and the number of workers a call
// uses.
func claimed(workers, n, round int) (units, used int) {
	used = max(1, min(workers, n))
	return min(n, used*(round+1)), used
}

// requireBands checks that the first done bands of the plan hold final
// results, Data and validity, and every other cell is untouched.
func requireBands(t *testing.T, id string, got, final, orig operand, bands [][4]int, done int) {
	t.Helper()
	for i, b := range bands {
		want := orig
		if i < done {
			want = final
		}
		for y := b[1]; y < b[3]; y++ {
			for x := b[0]; x < b[2]; x++ {
				requireCell(t, fmt.Sprintf("%s band %d %v", id, i, b), got, want, x, y)
			}
		}
	}
	requireOutsideUntouched(t, id, got, orig)
}

// TestCancellationWorkers stops ProcessN, by cancellation or by a
// deadline, while every worker is inside a kernel call. It must return
// the context's error once every worker has finished its band, having
// written exactly the bands claimed before the stop, a prefix of the
// plan, and started no kernel call after it.
func TestCancellationWorkers(t *testing.T) {
	defer exec.SetBandCells(1)() // one-row bands
	// Radius 0, so that every band makes exactly one kernel call; 200
	// rows, so that a stop leaves bands unclaimed with any worker count.
	const w, h = 40, 200
	rng := rand.New(rand.NewPCG(4, 5))
	dem := newOperand(rng, w, h, true, true)
	out := newOperand(rng, w, h, true, true)
	box := boxKernel{r: 0, inputs: 1}
	final := out.clone()
	naiveBox(final.r, []raster.Float32Raster{dem.r}, 0, float32(math.NaN()))

	for _, workers := range []int{1, 2, 3, 8, runtime.GOMAXPROCS(0)} {
		for _, tiles := range [][2]int{{0, 0}, {7, 4}, {1, 1}} {
			for _, round := range []int{0, 1, 4} {
				for _, deadline := range []bool{false, true} {
					o := engine.Options{TileWidth: tiles[0], TileHeight: tiles[1], Workers: workers}
					id := fmt.Sprintf("%+v %s in round %d", o, stopName(deadline), round)
					bands := exec.Bands(w, h, tiles[0], tiles[1])
					want, _ := claimed(workers, len(bands), round)
					synctest.Test(t, func(t *testing.T) {
						got := out.clone()
						ctx, stopped, cancel := stopAfter(context.Background(), round, deadline)
						defer cancel()
						var calls, late atomic.Int64
						k := lateKernel{sleepKernel{box, &calls}, stopped, &late}
						err := exec.Process(ctx, got.r, dem.r, k, o)
						wantErr := context.Canceled
						if deadline {
							wantErr = context.DeadlineExceeded
						}
						if !errors.Is(err, wantErr) {
							t.Fatalf("%s: err = %v, want %v", id, err, wantErr)
						}
						if n := calls.Load(); n != int64(want) {
							t.Fatalf("%s: %d kernel calls, want exactly %d", id, n, want)
						}
						if n := late.Load(); n != 0 {
							t.Fatalf("%s: %d kernel calls started after the stop", id, n)
						}
						requireBands(t, id, got, final, out, bands, want)
					})
				}
			}
		}
	}
}

// lateKernel counts the calls that start after stopped is set.
type lateKernel struct {
	exec.Kernel
	stopped *atomic.Bool
	late    *atomic.Int64
}

func (k lateKernel) Process(dst exec.Span, src exec.Window) {
	if k.stopped.Load() {
		k.late.Add(1)
	}
	k.Kernel.Process(dst, src)
}

// tickSource reads a tile in a tick of virtual time. Its nth read (from
// 1), if n > 0, fails with err half a tick after it starts, and sets
// failed. It counts reads, and reads that start once failed is set.
type tickSource struct {
	engine.RasterSource
	n           int64
	err         error
	failed      *atomic.Bool
	reads, late atomic.Int64
}

func (s *tickSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	if s.failed.Load() {
		s.late.Add(1)
	}
	if s.reads.Add(1) == s.n {
		time.Sleep(tick / 2)
		s.failed.Store(true)
		return s.err
	}
	time.Sleep(tick)
	return s.RasterSource.ReadWindow(ctx, dst, x, y)
}

// tickSink fails its nth write, if n > 0, half a tick after it starts
// and after writing the tile's first rows, and sets failed. Other writes
// take no time.
type tickSink struct {
	engine.RasterSink
	n            int64
	err          error
	failed       *atomic.Bool
	writes       atomic.Int64
	failX, failY int
}

func (s *tickSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	if s.writes.Add(1) != s.n {
		return s.RasterSink.WriteWindow(ctx, src, x, y)
	}
	time.Sleep(tick / 2)
	s.failX, s.failY = x, y
	if src.Height > 1 {
		if err := s.RasterSink.WriteWindow(ctx, src.Window(0, 0, src.Width, src.Height/2), x, y); err != nil {
			return err
		}
	}
	s.failed.Store(true)
	return s.err
}

// TestChunkedCancellation stops ProcessChunked, by cancellation or by a
// deadline, while every worker is reading a tile. It must return the
// context's error once every worker has written its tile, having read and
// written exactly the tiles claimed before the stop, whole and in plan
// order, and started no read after it. A stop after every tile has been
// claimed returns nil.
func TestChunkedCancellation(t *testing.T) {
	const w, h = 40, 30
	rng := rand.New(rand.NewPCG(4, 5))
	dem := newOperand(rng, w, h, true, true)
	out := newOperand(rng, w, h, true, true)
	box := boxKernel{r: 1, inputs: 1}
	final := out.clone()
	naiveBox(final.r, []raster.Float32Raster{dem.r}, 1, float32(math.NaN()))

	for _, workers := range []int{1, 2, 3, 8, runtime.GOMAXPROCS(0)} {
		for _, tiles := range [][2]int{{0, 0}, {7, 4}, {1, 1}, {40, 3}} {
			for _, round := range []int{0, 1, 4} {
				for _, deadline := range []bool{false, true} {
					o := engine.Options{TileWidth: tiles[0], TileHeight: tiles[1], Workers: workers}
					id := fmt.Sprintf("%+v %s in round %d", o, stopName(deadline), round)
					plan := chunkTiles(w, h, orDim(tiles[0], w), orDim(tiles[1], h))
					want, _ := claimed(workers, len(plan), round)
					synctest.Test(t, func(t *testing.T) {
						got := out.clone()
						ctx, stopped, cancel := stopAfter(context.Background(), round, deadline)
						defer cancel()
						src := &tickSource{RasterSource: engine.NewMemorySource(dem.r), failed: stopped}
						err := exec.ProcessChunked(ctx, []engine.RasterSink{engine.NewMemorySink(got.r)},
							[]engine.RasterSource{src}, box, o)
						wantErr := context.Canceled
						switch {
						case want == len(plan):
							wantErr = nil
						case deadline:
							wantErr = context.DeadlineExceeded
						}
						if !errors.Is(err, wantErr) || (wantErr == nil) != (err == nil) {
							t.Fatalf("%s: err = %v, want %v", id, err, wantErr)
						}
						if n := src.reads.Load(); n != int64(want) {
							t.Fatalf("%s: %d tiles read, want exactly %d", id, n, want)
						}
						if n := src.late.Load(); n != 0 {
							t.Fatalf("%s: %d tiles started after the stop", id, n)
						}
						if done := requireTiles(t, id, got, final, out, plan, 0, -1); done != want {
							t.Fatalf("%s: %d tiles written, want exactly %d", id, done, want)
						}
					})
				}
			}
		}
	}
}

// TestChunkedIOErrors fails a source read or a sink write in one tile,
// half a tick into it, while every other worker is reading a tile. The
// call must return that error (wrapped), finish exactly the tiles claimed
// before the failure and start none after it, and leave whole tiles
// except the failed one: untouched if its read failed, anything if its
// write did.
func TestChunkedIOErrors(t *testing.T) {
	const w, h = 36, 28
	rng := rand.New(rand.NewPCG(7, 8))
	dem := newOperand(rng, w, h, true, true)
	out := newOperand(rng, w, h, true, true)
	final := out.clone()
	naiveBox(final.r, []raster.Float32Raster{dem.r}, 1, float32(math.NaN()))
	errIO := errors.New("disk on fire")
	for _, workers := range []int{1, 2, 5, runtime.GOMAXPROCS(0)} {
		for _, tiles := range [][2]int{{6, 5}, {0, 4}, {1, 1}} {
			tw, th := orDim(tiles[0], w), orDim(tiles[1], h)
			plan := chunkTiles(w, h, tw, th)
			for _, n := range []int{1, 3, len(plan)} {
				for _, inSink := range []bool{false, true} {
					o := engine.Options{TileWidth: tiles[0], TileHeight: tiles[1], Workers: workers}
					id := fmt.Sprintf("%+v n=%d inSink=%v", o, n, inSink)
					// The nth read, or write, belongs to round q = (n-1)/used.
					// A failed read stops the workers during round q, which
					// all of them claimed. A failed write happens at the end
					// of round q, when the other workers claim round q+1
					// while the failing one is still writing.
					_, used := claimed(workers, len(plan), 0)
					q := (n - 1) / used
					want := min(len(plan), used*(q+1))
					if inSink {
						want = min(len(plan), used*(q+2)-1)
					}
					synctest.Test(t, func(t *testing.T) {
						got := out.clone()
						failed := new(atomic.Bool)
						src := &tickSource{RasterSource: engine.NewMemorySource(dem.r), err: errIO, failed: failed}
						sink := &tickSink{RasterSink: engine.NewMemorySink(got.r), err: errIO, failed: failed}
						if inSink {
							sink.n = int64(n)
						} else {
							src.n = int64(n)
						}
						err := exec.ProcessChunked(context.Background(), []engine.RasterSink{sink},
							[]engine.RasterSource{src}, boxKernel{r: 1, inputs: 1}, o)
						if !errors.Is(err, errIO) {
							t.Fatalf("%s: err = %v, want %v", id, err, errIO)
						}
						if r := src.reads.Load(); r != int64(want) {
							t.Fatalf("%s: %d tiles started, want exactly %d", id, r, want)
						}
						if late := src.late.Load(); late != 0 {
							t.Fatalf("%s: %d tiles started after the failure", id, late)
						}
						holes, anyTile := 1, -1
						if inSink {
							holes, anyTile = 0, (sink.failY/th)*((w+tw-1)/tw)+sink.failX/tw
						}
						if done := requireTiles(t, id, got, final, out, plan, holes, anyTile); done != want-1 {
							t.Fatalf("%s: %d tiles written, want exactly %d (all claimed but the failed one)", id, done, want-1)
						}
					})
				}
			}
		}
	}
}
