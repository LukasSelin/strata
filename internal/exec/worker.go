package exec

import (
	"context"
	"sync"
	"sync/atomic"

	"strata/internal/stencil"
	"strata/raster"
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
}

// run processes every band of the plan. Workers check ctx, then claim the
// next band in plan order and always finish a band they have claimed, so
// the finished bands are a prefix of the plan whenever run returns. The
// calling goroutine is worker 0; the others are goroutines started here
// and joined before run returns, whatever happens.
func (e *job) run(ctx context.Context) error {
	if len(e.workers) == 1 {
		return e.serial(ctx)
	}
	var s schedule
	var wg sync.WaitGroup
	for i := 1; i < len(e.workers); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e.work(ctx, &s, &e.workers[i])
		}()
	}
	e.work(ctx, &s, &e.workers[0])
	wg.Wait()
	if s.panicked.Load() {
		panic(s.panicVal)
	}
	if int(s.next.Load()) < e.plan.bands {
		return ctx.Err()
	}
	return nil
}

// serial is run with one worker, on the calling goroutine: no goroutines,
// no locks, and a kernel panic unwinds through Process as usual.
func (e *job) serial(ctx context.Context) error {
	done := ctx.Done()
	wk := &e.workers[0]
	for i := range e.plan.bands {
		if done != nil {
			select {
			case <-done:
				return ctx.Err()
			default:
			}
		}
		e.band(wk, i)
	}
	return nil
}

// schedule is the state workers share.
type schedule struct {
	// next is the number of bands claimed; bands [0, next) are claimed.
	next atomic.Int64
	// stop tells workers to claim nothing more, after a panic.
	stop atomic.Bool
	// panicked and panicVal hold the first panic of any worker.
	panicked  atomic.Bool
	panicOnce sync.Once
	panicVal  any
}

func (e *job) work(ctx context.Context, s *schedule, wk *worker) {
	defer func() {
		if v := recover(); v != nil {
			s.panicOnce.Do(func() {
				s.panicVal = v
				s.panicked.Store(true)
			})
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
		if i >= e.plan.bands {
			// Undo the claim so next stays the number of bands, which
			// run compares with plan.bands.
			s.next.Add(-1)
			return
		}
		e.band(wk, i)
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
