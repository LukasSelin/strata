package exec

import (
	"math"
	"runtime"
	"sync"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// bytesPerCell is the size of one float32 cell. engine.Stats counts the
// same unit; the two constants are separate only because engine cannot
// import this package.
const bytesPerCell = 4

// bandCells is the target number of cells in a band, the unit of work
// between cancellation checks and of scheduling across workers: bands are
// whole rows of a tile, at least one. 1<<16 cells keeps a check within a
// millisecond or so of work for the slowest kernels while amortising the
// per-call cost of pointwise ones, and a band of a few operands fits in
// L2 cache. It is a variable so tests can force one-row bands.
var bandCells = 1 << 16

// job is one ProcessN call after its checks, or one worker's tile of a
// ProcessChunked call. Everything a band needs is allocated here, once per
// call, so bands allocate nothing.
type job struct {
	k    Kernel
	r    int
	edge float32
	// w and h are the size of the whole raster, whose edge gets the edge
	// policy.
	w, h int

	plan plan

	// dst and src are the operands. In a ProcessN call they are the whole
	// rasters. In a ProcessChunked tile they are buffers: dst covers the
	// tile and src the tile grown by the radius, clipped to the raster.
	dst, src []raster.Float32Raster
	// dx, dy and sx, sy are the raster positions of the first cell of
	// dst and src; the plan's bands are relative to (dx, dy). All are 0
	// in a ProcessN call.
	dx, dy, sx, sy int

	// masked lists the inputs that have a validity mask, dstMasked
	// whether any output has one.
	masked    []int
	dstMasked bool
	// sameBits[i] is the masked input whose bits are dst[i]'s own (radius
	// 0 in place), or -1. It is set whenever pointwise validity runs.
	sameBits []int

	// workers holds each worker's views and scratch; workers[0] runs on
	// the calling goroutine.
	workers []worker
	// out is the caller's Options.Stats, or nil. A chunked call's
	// per-tile job leaves it nil and is totalled by its chunkJob instead,
	// so the caller's Stats is written once per call rather than once per
	// tile.
	out *engine.Stats
	// maskMu serialises all validity work (reads and writes) when there
	// is more than one worker. See maskLock.
	maskMu sync.Mutex
}

func newJob(dst, src []raster.Float32Raster, k Kernel, r int, opts engine.Options) *job {
	e := &job{dst: dst, src: src, out: opts.Stats}
	var masked []int
	for j, s := range src {
		if s.Valid != nil {
			masked = append(masked, j)
		}
	}
	dstMasked := false
	for _, d := range dst {
		dstMasked = dstMasked || d.Valid != nil
	}
	e.setup(k, r, dst[0].Width, dst[0].Height, masked, dstMasked)
	e.plan = newPlan(e.w, e.h, opts.TileWidth, opts.TileHeight)
	for i, d := range dst {
		if e.sameBits == nil {
			break
		}
		for _, j := range masked {
			if overlap.Bits(d, src[j]) == overlap.Same {
				e.sameBits[i] = j
				break
			}
		}
	}
	e.allocWorkers(workerCount(opts.Workers, e.plan.bands), e.plan.tileW)
	e.allocScratch(e.plan.spanSize())
	return e
}

// setup sets what a job takes from its kernel and the masks of its
// operands: the kernel, its radius and edge value, the raster size, the
// masked inputs and whether any output has a mask.
func (e *job) setup(k Kernel, r, w, h int, masked []int, dstMasked bool) {
	e.k, e.r, e.w, e.h = k, r, w, h
	e.edge = float32(math.NaN())
	if ek, ok := k.(EdgeKernel); ok {
		e.edge = ek.Edge()
	}
	e.masked, e.dstMasked = masked, dstMasked
	if dstMasked && len(masked) > 0 && r == 0 {
		e.sameBits = make([]int, len(e.dst))
		for i := range e.sameBits {
			e.sameBits[i] = -1
		}
	}
}

// workerCount resolves Options.Workers for a plan of n units of work.
func workerCount(workers, n int) int {
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return max(1, min(workers, n))
}

// allocWorkers gives the job n workers, with views for its operands and,
// when validity is eroded, scratch for tiles up to tileW cells wide.
func (e *job) allocWorkers(n, tileW int) {
	erode := e.dstMasked && len(e.masked) > 0 && e.r > 0
	// One backing array per kind for all workers, so a call's allocations
	// do not grow with the worker count.
	e.workers = make([]worker, n)
	nd, ns := len(e.dst), len(e.src)
	dstViews := make([]raster.Float32Raster, n*nd)
	srcViews := make([]raster.Float32Raster, n*ns)
	var regions []stencil.MaskRegion
	var scratch []uint64
	sw := 0
	if erode {
		regions = make([]stencil.MaskRegion, n*len(e.masked))
		sw = stencil.ErodeScratch(tileW, e.r)
		scratch = make([]uint64, n*sw)
	}
	for i := range e.workers {
		wk := &e.workers[i]
		wk.dstViews = dstViews[i*nd : (i+1)*nd]
		wk.srcViews = srcViews[i*ns : (i+1)*ns]
		if erode {
			wk.regions = regions[i*len(e.masked) : (i+1)*len(e.masked)]
			wk.scratch = scratch[i*sw : (i+1)*sw]
		}
	}
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

// spanSize is the largest span one Process call can cover under p: a
// band is the tile's width by at most bandRows rows, and a tile clipped
// at the raster's edge is only ever smaller. It returns 0, 0 for a plan
// with no bands.
func (p *plan) spanSize() (w, h int) {
	if p.bands == 0 {
		return 0, 0
	}
	return p.tileW, min(p.bandRows, p.tileH)
}

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
