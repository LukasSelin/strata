package exec

import (
	"math"
	"runtime"
	"sync"

	"strata/engine"
	"strata/internal/overlap"
	"strata/internal/stencil"
	"strata/raster"
)

// bandCells is the target number of cells in a band, the unit of work
// between cancellation checks and of scheduling across workers: bands are
// whole rows of a tile, at least one. 1<<16 cells keeps a check within a
// millisecond or so of work for the slowest kernels while amortising the
// per-call cost of pointwise ones, and a band of a few operands fits in
// L2 cache. It is a variable so tests can force one-row bands.
var bandCells = 1 << 16

// job is one ProcessN call after its checks. Everything a band needs is
// allocated here, once per call, so bands allocate nothing.
type job struct {
	k    Kernel
	r    int
	edge float32
	w, h int

	plan plan

	dst, src []raster.Float32Raster

	// masked lists the inputs that have a validity mask, dstMasked
	// whether any output has one.
	masked    []int
	dstMasked bool
	// sameBits[i] is the masked input whose bits are dst[i]'s own (radius
	// 0 in place), or -1.
	sameBits []int

	// workers holds each worker's views and scratch; workers[0] runs on
	// the calling goroutine.
	workers []worker
	// maskMu serialises all validity work (reads and writes) when there
	// is more than one worker. See maskLock.
	maskMu sync.Mutex
}

func newJob(dst, src []raster.Float32Raster, k Kernel, r int, opts engine.Options) *job {
	e := &job{
		k:    k,
		r:    r,
		edge: float32(math.NaN()),
		w:    dst[0].Width,
		h:    dst[0].Height,
		dst:  dst,
		src:  src,
	}
	if ek, ok := k.(EdgeKernel); ok {
		e.edge = ek.Edge()
	}
	e.plan = newPlan(e.w, e.h, opts.TileWidth, opts.TileHeight)

	for _, d := range dst {
		e.dstMasked = e.dstMasked || d.Valid != nil
	}
	for j, s := range src {
		if s.Valid != nil {
			e.masked = append(e.masked, j)
		}
	}
	erode := e.dstMasked && len(e.masked) > 0 && r > 0
	if e.dstMasked && len(e.masked) > 0 && r == 0 {
		e.sameBits = make([]int, len(dst))
		for i, d := range dst {
			e.sameBits[i] = -1
			for _, j := range e.masked {
				if overlap.Bits(d, src[j]) == overlap.Same {
					e.sameBits[i] = j
					break
				}
			}
		}
	}

	n := opts.Workers
	if n == 0 {
		n = runtime.GOMAXPROCS(0)
	}
	n = max(1, min(n, e.plan.bands))
	// One backing array per kind for all workers, so a call's allocations
	// do not grow with the worker count.
	e.workers = make([]worker, n)
	dstViews := make([]raster.Float32Raster, n*len(dst))
	srcViews := make([]raster.Float32Raster, n*len(src))
	var regions []stencil.MaskRegion
	var scratch []uint64
	sw := 0
	if erode {
		regions = make([]stencil.MaskRegion, n*len(e.masked))
		sw = stencil.ErodeScratch(e.plan.tileW, r)
		scratch = make([]uint64, n*sw)
	}
	for i := range e.workers {
		wk := &e.workers[i]
		wk.dstViews = dstViews[i*len(dst) : (i+1)*len(dst)]
		wk.srcViews = srcViews[i*len(src) : (i+1)*len(src)]
		if erode {
			wk.regions = regions[i*len(e.masked) : (i+1)*len(e.masked)]
			wk.scratch = scratch[i*sw : (i+1)*sw]
		}
	}
	return e
}

// plan divides a w×h raster into tiles in row-major order and each tile
// into bands of whole rows, and numbers the bands in that order. Band i
// is found by arithmetic, so a plan of millions of bands costs nothing.
type plan struct {
	w, h         int
	tileW, tileH int
	bandRows     int
	tilesX       int // tiles per tile row
	tileRows     int // rows of tiles
	perTile      int // bands in a tile of full height
	perLastTile  int // bands in a tile of the last tile row
	bands        int // total
}

func newPlan(w, h, tileW, tileH int) plan {
	p := plan{w: w, h: h, tileW: w, tileH: h}
	if tileW > 0 {
		p.tileW = min(tileW, w)
	}
	if tileH > 0 {
		p.tileH = min(tileH, h)
	}
	if w == 0 || h == 0 {
		return p // no bands
	}
	p.bandRows = max(1, bandCells/p.tileW)
	p.tilesX = ceilDiv(w, p.tileW)
	p.tileRows = ceilDiv(h, p.tileH)
	p.perTile = ceilDiv(p.tileH, p.bandRows)
	p.perLastTile = ceilDiv(h-(p.tileRows-1)*p.tileH, p.bandRows)
	p.bands = p.tilesX * ((p.tileRows-1)*p.perTile + p.perLastTile)
	return p
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// band returns band i's cells [x0, x1) × [y0, y1).
func (p *plan) band(i int) (x0, y0, x1, y1 int) {
	perRow := p.tilesX * p.perTile // bands per full tile row
	tr, j, per := i/perRow, i%perRow, p.perTile
	if tr >= p.tileRows-1 {
		tr, j, per = p.tileRows-1, i-(p.tileRows-1)*perRow, p.perLastTile
	}
	tx, b := j/per, j%per
	x0 = tx * p.tileW
	x1 = min(x0+p.tileW, p.w)
	ty := tr * p.tileH
	y0 = ty + b*p.bandRows
	y1 = min(y0+p.bandRows, ty+p.tileH, p.h)
	return x0, y0, x1, y1
}
