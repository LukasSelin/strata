package exec

import (
	"fmt"
	"math"
	"runtime"
	"slices"
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
	// valid[i] is how dst[i]'s validity is derived, when an output has a
	// mask: see outValidity. valid1 holds it for a kernel of one output,
	// so that the common case allocates nothing for it.
	valid  []outValidity
	valid1 [1]outValidity
	// edgeW[i] is the width of dst[i]'s edge ring, when the outputs'
	// rings differ (see edgeWidths); nil when every ring is r wide. rmin
	// is the narrowest ring: the kernel runs over every cell at least
	// rmin from the edge, with its window padded where it leaves the
	// rasters.
	edgeW []int
	rmin  int

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
	dstMasked := false
	for _, d := range dst {
		dstMasked = dstMasked || d.Valid != nil
	}
	var masked, rules []int
	for _, s := range src {
		if s.Valid != nil {
			// The masked inputs and the validity rules' ints in one
			// allocation: see validityInts.
			n := len(src)
			ints := make([]int, n+validityInts(n, len(dst)))
			masked, rules = ints[:0:n], ints[n:]
			break
		}
	}
	for j, s := range src {
		if s.Valid != nil {
			masked = append(masked, j)
		}
	}
	valid := e.valid1[:]
	if len(dst) > 1 {
		valid = make([]outValidity, len(dst))
	}
	e.setup(k, r, dst[0].Width, dst[0].Height, masked, dstMasked,
		outValidities(k, r, masked, dstMasked, valid, rules), edgeWidths(k, r, len(src), len(dst)))
	e.plan = newPlan(e.w, e.h, opts.TileWidth, opts.TileHeight)
	for i, d := range dst {
		if e.sameBits == nil {
			break
		}
		for _, j := range e.valid[i].ins {
			if overlap.Bits(d, src[j]) == overlap.Same {
				e.sameBits[i] = j
				break
			}
		}
	}
	e.allocWorkers(workerCount(opts.Workers, e.plan.bands), e.plan.tileW)
	e.allocScratch(e.plan.spanSize())
	e.allocPad(e.plan.spanSize())
	return e
}

// setup sets what a job takes from its kernel and the masks of its
// operands: the kernel, its radius and edge value, the raster size, the
// masked inputs, whether any output has a mask and, from outValidities,
// how each output's validity is derived, and from edgeWidths the width
// of each output's edge ring. valid and edgeW are only read, so a
// chunked call's tiles share them.
func (e *job) setup(k Kernel, r, w, h int, masked []int, dstMasked bool, valid []outValidity, edgeW []int) {
	e.k, e.r, e.w, e.h = k, r, w, h
	e.edgeW, e.rmin = edgeW, r
	for _, b := range edgeW {
		e.rmin = min(e.rmin, b)
	}
	e.edge = float32(math.NaN())
	if ek, ok := k.(EdgeKernel); ok {
		e.edge = ek.Edge()
	}
	e.masked, e.dstMasked, e.valid = masked, dstMasked, valid
	if dstMasked && len(masked) > 0 && r == 0 {
		e.sameBits = make([]int, len(e.dst))
		for i := range e.sameBits {
			e.sameBits[i] = -1
		}
	}
}

// edgeWidths returns the width of each output's edge ring, or nil when
// every ring is r wide, which is the case for every kernel but a
// ReachKernel whose outputs read less far than its radius. An output's
// ring is the largest distance at which it reads any input: its cells
// beyond that read only cells inside the rasters, so they get real
// values, as they would from a kernel computing that output alone
// (DESIGN.md §52). An output that reads no input has a ring of r.
func edgeWidths(k Kernel, r, nin, nout int) []int {
	rk, ok := k.(ReachKernel)
	if !ok || r == 0 {
		return nil
	}
	width := func(o int) int {
		b := -1
		for in := range nin {
			b = max(b, rk.Reach(o, in))
		}
		if b < 0 {
			return r
		}
		return min(b, r)
	}
	for o := range nout {
		if width(o) != r {
			w := make([]int, nout)
			for o := range w {
				w[o] = width(o)
			}
			return w
		}
	}
	return nil
}

// outValidity is how one output's validity is derived: the AND of the
// masked inputs it reads, ins, each eroded by its reach in reach. Equal
// reaches are adjacent, so stencil.ErodeReach erodes each group once.
// from is an earlier output with the same ins and reaches, whose bits
// this one copies, or -1.
type outValidity struct {
	ins, reach []int
	from       int
}

// validityInts is how many ints outValidities needs for a kernel of nin
// inputs and nout outputs, at most: an input list and a reach list per
// output.
func validityInts(nin, nout int) int { return 2 * nin * nout }

// outValidities fills v, one rule per output of k, from the masked
// inputs, and returns it, or nil when no output has a mask: every masked
// input over radius r, unless k is a ReachKernel. The rules' lists are
// carved from ints, which holds at least validityInts(len(masked),
// len(v)). It panics if a ReachKernel reports a reach outside [-1, r].
func outValidities(k Kernel, r int, masked []int, dstMasked bool, v []outValidity, ints []int) []outValidity {
	if !dstMasked {
		return nil
	}
	carve := func() []int {
		s := ints[:0:len(masked)]
		ints = ints[len(masked):]
		return s
	}
	rk, ok := k.(ReachKernel)
	if !ok {
		// Every output reads every masked input over r: erode once into
		// the first output and copy it to the others, as the engine
		// always has.
		reach := carve()
		for range masked {
			reach = append(reach, r)
		}
		for o := range v {
			v[o] = outValidity{ins: masked, reach: reach, from: min(o, 1) - 1}
		}
		return v
	}
	for o := range v {
		v[o] = outValidity{ins: carve(), reach: carve(), from: -1}
		for _, j := range masked {
			reach := rk.Reach(o, j)
			if reach < -1 || reach > r {
				panic(fmt.Sprintf("engine: kernel reports reach %d from input %d to output %d; "+
					"a reach is -1 or 0 to its radius %d", reach, j, o, r))
			}
			if reach < 0 {
				continue
			}
			// Insert keeping equal reaches together, largest first.
			at := len(v[o].ins)
			for at > 0 && v[o].reach[at-1] < reach {
				at--
			}
			v[o].ins = slices.Insert(v[o].ins, at, j)
			v[o].reach = slices.Insert(v[o].reach, at, reach)
		}
		for p := range o {
			if len(v[o].ins) > 0 && slices.Equal(v[p].ins, v[o].ins) && slices.Equal(v[p].reach, v[o].reach) {
				v[o].from = p
				break
			}
		}
	}
	return v
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
		sw = stencil.ErodeReachScratch(tileW, e.r)
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
