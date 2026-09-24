package exec

import (
	"context"
	"errors"
	"sync"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// Chunked calls write behind: each worker hands its finished tile to a
// writer goroutine of its own and computes the next tile while the last
// one is written. A worker owns two sets of output buffers and
// alternates between them (Behind.Acquire and Behind.Submit), so at most
// one tile is being written while the next is computed, and a call's
// memory grows by one output tile per worker.
//
// A write blocked in a system call does not hold a P, so the writer runs
// on another core while the worker computes. That is where write-behind
// pays: with one worker and GOMAXPROCS ≥ 2 it took a third off a raw
// file to raw file Slope (benchmarks/rawio). Under GOMAXPROCS=1 the
// writer still needs the one P between its system calls, and waits up
// to a scheduler time slice for it each time, so the overlap is partial;
// a write that copies memory, such as MemorySink's or a mapped
// engine.RawFile's, needs a P throughout and overlaps only when one is
// free. Either way a Workers=1 call now keeps up to two cores busy.
//
// Failures keep the contract of calls without write-behind. A write
// error or panic stops the workers claiming tiles at once, through the
// context runWorkers checks, as a failed tile does; the tile a worker is
// computing is still written, so every claimed tile is written but the
// failed ones, and the call returns once every write has finished.

// RunUnitsBehind is RunUnits with write-behind to dst: do(w, i, b) runs
// unit i on worker w and hands its tile to b, the worker's writer, with
// b.Acquire and b.Submit. wrap turns sink j's error at (x, y) into the
// error to return. Sinks are written with context.WithoutCancel(ctx).
//
// It returns as RunUnits does, once every submitted tile is written,
// with a write's error counting as a unit's: nil if every unit ran and
// every write succeeded, otherwise the first error of a unit, or else
// of a write, or else ctx.Err(). A panic of a unit or of a write is
// re-raised on the calling goroutine once every worker and writer has
// stopped, a unit's first.
func RunUnitsBehind(ctx context.Context, workers, n int, dst []engine.RasterSink,
	wrap func(j, x, y int, err error) error, do func(w, i int, b *Behind) error) error {
	if workers < 1 || n < 0 {
		panic("exec: RunUnitsBehind needs workers >= 1 and n >= 0")
	}
	return runBehind(ctx, workers, n, dst, wrap, do)
}

func runBehind(ctx context.Context, workers, n int, dst []engine.RasterSink,
	wrap func(j, x, y int, err error) error, do func(w, i int, b *Behind) error) (err error) {
	// wctx is what runWorkers checks before every claim: the caller's
	// ctx, or a failed write.
	wctx, stop := context.WithCancel(ctx)
	defer stop()
	sh := &behindShared{stop: stop}
	ioCtx := context.WithoutCancel(ctx)
	bs := make([]*Behind, workers)
	for i := range bs {
		bs[i] = newBehind(ioCtx, dst, wrap, sh)
	}
	defer func() {
		p := recover()
		for _, b := range bs {
			b.close()
		}
		// Every writer has stopped: sh is settled.
		switch {
		case p != nil:
			panic(p)
		case sh.panicked:
			panic(sh.panicVal)
		case sh.err != nil && (err == nil || isContextErr(err)):
			err = sh.err
		}
	}()
	return runWorkers(wctx, workers, n, func(w, i int) error { return do(w, i, bs[w]) })
}

func isContextErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// behindShared is the state a call's writers share: the first write
// error and panic, and stop, which ends the workers' claims.
type behindShared struct {
	stop     context.CancelFunc
	mu       sync.Mutex
	err      error
	panicked bool
	panicVal any
}

// Behind is one worker's writer; see RunUnitsBehind.
type Behind struct {
	ctx  context.Context
	dst  []engine.RasterSink
	wrap func(j, x, y int, err error) error
	sh   *behindShared

	jobs chan behindJob
	free chan int
	done chan struct{}
	// views holds the views each set's tile is written from, one per
	// sink, copied at Submit: the worker reuses its own view slice.
	views [2][]raster.Float32Raster
}

type behindJob struct {
	set  int
	x, y int
}

func newBehind(ctx context.Context, dst []engine.RasterSink, wrap func(j, x, y int, err error) error, sh *behindShared) *Behind {
	b := &Behind{
		ctx: ctx, dst: dst, wrap: wrap, sh: sh,
		jobs: make(chan behindJob, 2),
		free: make(chan int, 2),
		done: make(chan struct{}),
	}
	b.views[0] = make([]raster.Float32Raster, len(dst))
	b.views[1] = make([]raster.Float32Raster, len(dst))
	b.free <- 0
	b.free <- 1
	go b.loop()
	return b
}

func (b *Behind) loop() {
	defer close(b.done)
	for j := range b.jobs {
		b.write(j)
		b.free <- j.set
	}
}

func (b *Behind) write(j behindJob) {
	sh := b.sh
	sh.mu.Lock()
	skip := sh.panicked
	sh.mu.Unlock()
	if skip {
		// A sink panicked: sink contents are unspecified, and the panic
		// is on its way to the caller.
		return
	}
	defer func() {
		if v := recover(); v != nil {
			sh.mu.Lock()
			if !sh.panicked {
				sh.panicked, sh.panicVal = true, v
			}
			sh.mu.Unlock()
			sh.stop()
		}
	}()
	for k, d := range b.dst {
		if err := d.WriteWindow(b.ctx, b.views[j.set][k], j.x, j.y); err != nil {
			sh.mu.Lock()
			if sh.err == nil {
				sh.err = b.wrap(k, j.x, j.y, err)
			}
			sh.mu.Unlock()
			sh.stop()
			return
		}
	}
}

// Acquire returns the buffer set, 0 or 1, to compute the worker's next
// tile into: one whose last write has finished, waiting for it if both
// sets are in use.
func (b *Behind) Acquire() int { return <-b.free }

// Submit queues the tile in set, whose views are views, one per sink,
// for writing at (x, y). views is copied; the buffers are the writer's
// until Acquire returns set again.
func (b *Behind) Submit(set int, views []raster.Float32Raster, x, y int) {
	copy(b.views[set], views)
	b.jobs <- behindJob{set, x, y}
}

// close waits for every submitted tile to be written and stops the
// goroutine.
func (b *Behind) close() {
	close(b.jobs)
	<-b.done
}
