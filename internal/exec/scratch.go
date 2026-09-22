package exec

import (
	"fmt"
	"math"
	"math/bits"
	"sync"

	"github.com/LukasSelin/strata/raster"
)

// A ScratchKernel's working memory is lent, not allocated, per call.
// Every call used to make and zero it afresh: for a Pipeline of four
// intermediates in default strips that is 1 MiB per worker per call, and
// on 12 workers it cost the 1024² pipeline more than its compute
// (DESIGN.md §52). The memory has no owner outside the call — the
// contract says what a call leaves in it is unspecified and that a
// kernel must not keep it — so the engine keeps it between calls in
// pools, one per kind and size class, and hands each worker a block on
// the way in and takes it back on the way out.
//
// sync.Pool is the right store for that: it is safe for concurrent
// calls, which a cache on the kernel would have to build for itself; it
// serves every ScratchKernel and both drivers without an API change; and
// the collector empties it, so memory a caller stopped using is not
// held forever. Nothing is zeroed on reuse, which is the other half of
// the saving and what "unspecified" permits.
var (
	cellPool slab[float32]
	bitPool  slab[uint64]
	viewPool slab[raster.Float32Raster]
	runPool  slab[[]float32]
)

// poisonScratch makes every block handed out be filled with values a
// correct kernel never reads before writing: NaN cells, alternating
// bits, garbage views and nil runs. Tests set it, so that a kernel
// relying on fresh or zeroed scratch fails there rather than on a pool
// miss in production.
var poisonScratch bool

// slab is a set of pools of []T by size class. A block's capacity is
// exactly its class's size, so a block goes back to the pool it came
// from.
type slab[T any] struct {
	pools [classes]sync.Pool
}

// classes is enough for every n up to 1<<62, far past anything make
// could satisfy: sizeClass(n) is then at most 16 + 8·57 + 8, for s = 58.
const classes = 16 + 8*57 + 8 + 1

// get returns a block of at least n values, with length n. Its contents
// are whatever an earlier call left. It is a pointer so that putting it
// back does not allocate an interface value.
func (s *slab[T]) get(n int) *[]T {
	c, size := sizeClass(n)
	if p, _ := s.pools[c].Get().(*[]T); p != nil {
		*p = (*p)[:n]
		return p
	}
	b := make([]T, n, size)
	return &b
}

// put returns a block to its pool. The caller must not use it again.
func (s *slab[T]) put(p *[]T) {
	c, _ := sizeClass(cap(*p))
	s.pools[c].Put(p)
}

// sizeClass rounds n in (0, 1<<62] up to a size class, eight to an octave above 16,
// so a block is at most 1/8 larger than asked for. A class's size maps
// to that class, which is what lets put find a block's pool from its
// capacity. Powers of two are classes, so the common spans of 1<<k cells
// waste nothing.
func sizeClass(n int) (class, size int) {
	if n <= 16 {
		return n, n
	}
	s := bits.Len(uint(n-1)) - 4 // (n-1)>>s is in [8, 16)
	m := (n-1)>>s + 1            // in [9, 16]
	return 16 + 8*(s-1) + (m - 8), m << s
}

// allocScratch gives each worker the working memory a ScratchKernel asks
// for, for a span of at most w×h, from the pools. releaseScratch gives it
// back; the job must call it once every worker has stopped.
//
// A band still allocates nothing (DESIGN.md §26): this runs once per
// call, before any band. Each worker gets a block of its own rather
// than a slice of one per call, so that a block fits a call with any
// number of workers and a chunked call's per-tile jobs as well. The
// slices handed to the kernel are exactly the lengths it asked for, with
// capacity to match, however large the block behind them.
func (e *job) allocScratch(w, h int) {
	sk, ok := e.k.(ScratchKernel)
	if !ok || w <= 0 || h <= 0 {
		return
	}
	need := sk.Scratch(w, h)
	if need.Cells < 0 || need.Words < 0 || need.Views < 0 || need.Runs < 0 {
		panic(fmt.Sprintf("engine: kernel asked for negative scratch %+v", need))
	}
	for i := range e.workers {
		wk := &e.workers[i]
		s := &wk.kscratch
		if need.Cells > 0 {
			wk.cells = cellPool.get(need.Cells)
			s.Cells = (*wk.cells)[:need.Cells:need.Cells]
		}
		if need.Words > 0 {
			wk.bits = bitPool.get(need.Words)
			s.Bits = (*wk.bits)[:need.Words:need.Words]
		}
		if need.Views > 0 {
			wk.views = viewPool.get(need.Views)
			s.Views = (*wk.views)[:need.Views:need.Views]
		}
		if need.Runs > 0 {
			wk.runs = runPool.get(need.Runs)
			s.Runs = (*wk.runs)[:need.Runs:need.Runs]
		}
		if poisonScratch {
			poison(s)
		}
	}
}

// releaseScratch returns every worker's scratch to the pools. Views and
// runs are cleared first: a Pipeline leaves views of, and slices into,
// the caller's rasters in them, and a pooled block must not keep those
// rasters alive.
func (e *job) releaseScratch() {
	for i := range e.workers {
		wk := &e.workers[i]
		if wk.cells != nil {
			cellPool.put(wk.cells)
		}
		if wk.bits != nil {
			bitPool.put(wk.bits)
		}
		if wk.views != nil {
			clear((*wk.views)[:cap(*wk.views)])
			viewPool.put(wk.views)
		}
		if wk.runs != nil {
			clear((*wk.runs)[:cap(*wk.runs)])
			runPool.put(wk.runs)
		}
		wk.cells, wk.bits, wk.views, wk.runs = nil, nil, nil, nil
		wk.kscratch = Scratch{}
	}
}

// poison fills s with values a kernel must write before it reads.
func poison(s *Scratch) {
	nan := float32(math.NaN())
	for i := range s.Cells {
		s.Cells[i] = nan
	}
	for i := range s.Bits {
		s.Bits[i] = 0xaaaa_aaaa_aaaa_aaaa
	}
	junk := raster.Float32Raster{Data: s.Cells, Width: -1, Height: -1, Stride: -1}
	for i := range s.Views {
		s.Views[i] = junk
	}
	clear(s.Runs)
}
