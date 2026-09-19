package exec

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// worker is the state one goroutine uses for its bands. Nothing in it is
// shared with another worker.
type worker struct {
	// dstViews and srcViews are the span and window views handed to the
	// kernel, overwritten for every band.
	dstViews, srcViews []raster.Float32Raster
	// regions and scratch are ErodeBox's arguments for radius > 0.
	regions []stencil.MaskRegion
	scratch []uint64
	// stats counts the bytes this worker's bands moved. It is per worker
	// and summed once the workers have stopped, so the hot path takes no
	// atomic and no lock. It is written once per band and padded for the
	// same reason reduceSlot is: workers lie next to each other in one
	// slice, and a counter they share a cache line for would show up as
	// scaling that flattens for no visible reason — in a measurement of
	// exactly that, which is worse than not measuring.
	stats engine.Stats
	_     [64]byte
}

// run processes every band of the plan on the job's workers.
func (e *job) run(ctx context.Context) error {
	err := runWorkers(ctx, len(e.workers), e.plan.bands, func(w, i int) error {
		e.band(&e.workers[w], i)
		return nil
	})
	e.report()
	return err
}

// report totals the workers' counters into the caller's Stats. It runs
// after runWorkers, so every worker has stopped and nothing else is
// writing them; a cancelled or failed call reports the bands that ran.
func (e *job) report() {
	if e.out == nil {
		return
	}
	for i := range e.workers {
		s := e.workers[i].stats
		s.Ideal = s.Cells * bytesPerCell * int64(len(e.src)+len(e.dst))
		e.out.Add(s)
	}
}

// runWorkers runs do(w, i) for every unit of work i in [0, n) on workers
// w in [0, workers). Workers check ctx, then claim the next unit in order
// and always finish a unit they have claimed, so the claimed units are a
// prefix of [0, n) whenever runWorkers returns. A unit that returns an
// error, like a done ctx, stops the workers claiming more.
//
// It returns nil if every unit ran and returned nil; otherwise the first
// error a unit returned, or ctx.Err() if units were left unclaimed. A
// panic in do is re-raised on the calling goroutine once every worker
// has stopped. The calling goroutine is worker 0; the others are
// goroutines started here and joined before runWorkers returns, whatever
// happens. With one worker nothing is started and a panic unwinds as
// usual.
func runWorkers(ctx context.Context, workers, n int, do func(w, i int) error) error {
	if workers == 1 {
		return serial(ctx, n, do)
	}
	var s schedule
	var wg sync.WaitGroup
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.work(ctx, n, w, do)
		}()
	}
	s.work(ctx, n, 0, do)
	wg.Wait()
	if s.panicked {
		panic(s.panicVal)
	}
	if s.err != nil {
		return s.err
	}
	if int(s.next.Load()) < n {
		return ctx.Err()
	}
	return nil
}

// serial is runWorkers with one worker, on the calling goroutine: no
// goroutines and no locks.
func serial(ctx context.Context, n int, do func(w, i int) error) error {
	done := ctx.Done()
	for i := range n {
		if done != nil {
			select {
			case <-done:
				return ctx.Err()
			default:
			}
		}
		if err := do(0, i); err != nil {
			return err
		}
	}
	return nil
}

// schedule is the state workers share.
type schedule struct {
	// next is the number of units claimed; units [0, next) are claimed.
	next atomic.Int64
	// stop tells workers to claim nothing more, after a panic or error.
	stop atomic.Bool
	// mu guards the first panic (panicked and panicVal) and the first
	// error, recorded separately so that an error never hides a panic.
	// They are read only after every worker has stopped.
	mu       sync.Mutex
	panicked bool
	panicVal any
	err      error
}

func (s *schedule) work(ctx context.Context, n, w int, do func(w, i int) error) {
	defer func() {
		if v := recover(); v != nil {
			s.mu.Lock()
			if !s.panicked {
				s.panicked, s.panicVal = true, v
			}
			s.mu.Unlock()
			s.stop.Store(true)
		}
	}()
	done := ctx.Done()
	for !s.stop.Load() {
		if done != nil {
			select {
			case <-done:
				return
			default:
			}
		}
		i := int(s.next.Add(1)) - 1
		if i >= n {
			// Undo the claim so next stays n, which runWorkers compares
			// with n.
			s.next.Add(-1)
			return
		}
		if err := do(w, i); err != nil {
			s.stop.Store(true)
			s.mu.Lock()
			if s.err == nil {
				s.err = err
			}
			s.mu.Unlock()
			return
		}
	}
}

// maskLock serialises validity work between workers. Masks are processed
// in words, and cells of different bands can share a word: side by side
// in a row, across rows whose stride is not a multiple of 64, and between
// an input's bits and an output's when they live in the same mask. Data
// never needs a lock, since bands own disjoint output cells and inputs
// are only read.
func (e *job) maskLock() {
	if len(e.workers) > 1 {
		e.maskMu.Lock()
	}
}

func (e *job) maskUnlock() {
	if len(e.workers) > 1 {
		e.maskMu.Unlock()
	}
}
