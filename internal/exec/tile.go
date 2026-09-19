package exec

import (
	"fmt"
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
// between cancellation checks and of scheduling across workers. It sets a
// band's area; bandShape sets its shape. 1<<16 cells keeps a check within a
// millisecond or so of work for the slowest kernels while amortising the
// per-call cost of pointwise ones, and a band of a few operands fits in
// L2 cache. It is a variable so tests can force one-row bands.
var bandCells = 1 << 16

// minBandWidth is the shortest row a band is given when a tile is split
// across its width, and so the knob that turns band shaping on: a band is
// the squarest rectangle of bandCells whose rows are still this long, not
// the squarest rectangle, because a row kernel pays a fixed cost per row
// and short rows read memory in streams the prefetcher has not seen.
// BenchmarkBandWidth measures rows below 1024 cells costing 4-24%,
// monotonically, whatever halo the shape reads.
//
// It sits above every real tile width, so the rule is inert and bands are
// whole tile rows, as they were before §53 separated the compute tile
// from the IO tile. That is a measurement, not a preference. Shaping the
// band does cut the halo exactly as the arithmetic says — 12.4% to 3.2%
// at 4096 wide, 50.0% to 3.3% at 16384, where it is a fifth of all the
// traffic a tiled call moves — and buys no time for it: at 16384² the two
// shapes run within the run-to-run spread of each other while moving
// 10.00 and 8.13 bytes per cell. That is §51's "logical traffic is not
// DRAM traffic" holding for the very change §51 proposed.
//
// 1024 is the value to restore to turn it back on. The measurement is one
// machine's — an Apple M4 in the scalar build, where the rest of the
// project's figures are from a Zen 2 with AVX2 and a quarter of the L1d —
// so it is worth rerunning before the default is treated as settled;
// benchmarks/engine/RESULTS-bandshape.md says what it can and cannot be
// compared with, and §53 what result would earn the default.
var minBandWidth = 1 << 30

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
	e.plan = newPlan(e.w, e.h, tilingOf(opts, r))
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
	e.allocWorkers(workerCount(opts.Workers, e.plan.bands), e.plan.bandW)
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
// when validity is eroded, scratch for bands up to bandW cells wide.
func (e *job) allocWorkers(n, bandW int) {
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
		sw = stencil.ErodeScratch(bandW, e.r)
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

// tiling is the geometry a plan is built from: the IO tile the caller
// asked for, the compute tile it forced, and the kernel's radius. A 0
// size means the engine's choice.
type tiling struct {
	tileW, tileH       int
	computeW, computeH int
	r                  int
}

func tilingOf(opts engine.Options, r int) tiling {
	return tiling{opts.TileWidth, opts.TileHeight, opts.ComputeWidth, opts.ComputeHeight, r}
}

// inner is the tiling for a plan over a tile a Chunked call has already
// read: that tile is the whole raster the plan sees, so only the compute
// tile and the radius carry over.
func (t tiling) inner() tiling {
	return tiling{computeW: t.computeW, computeH: t.computeH, r: t.r}
}

// bandShape is the size of the bands a tileW×tileH tile is split into:
// the compute tile, which the caller's TileWidth and TileHeight no longer
// decide (DESIGN.md §53).
//
// At radius 0 the band is whole tile rows, exactly as it always was: there
// is no halo to save, and a band as wide as its tile keeps a pointwise
// kernel's views compact (Stride == Width), which is what lets it and
// pointwiseValidity take their whole-span paths. Above radius 0 the band
// is made squarer, because the halo it reads is its perimeter.
//
// Two bounds keep that from going too far. A tile no taller than one
// whole-row band is already a single band, so splitting it across its
// width only adds perimeter: a 4096×16 tile in bands of 1024×16 reads a
// 12.7% halo where the whole tile reads 12.5%, with shorter rows as well.
// And minBandWidth stops the rows getting short. The tile is divided into
// equal columns by rounding to nearest rather than up, so that no band is
// left narrow — ceilDiv(1500, 1024) would give two columns of 750, where
// this gives one of 1500 — which bounds a band at 3/4 of minBandWidth once
// there is more than one column, not at minBandWidth itself.
//
// Band width is deliberately not rounded to a multiple of 64 for mask
// alignment. It would fight the equal division in almost every case (a
// 4000-wide tile divides into four clean columns of 1000, which rounding
// would make 1024, 1024, 1024, 928), and §23's stride-64 reasoning is
// about where a row's bits start, which is set by Stride, not by a band's
// x0. A band's horizontal offset only reaches partial words inside
// ErodeBox, whose cost is dominated by the AND of 2r+1 source rows. If a
// benchmark ever shows the shifts there mattering, that is the
// measurement that would justify the uneven columns.
func bandShape(tileW, tileH int, t tiling) (bandW, bandH int) {
	if t.computeW > 0 || t.computeH > 0 {
		bandW = tileW
		if t.computeW > 0 {
			bandW = min(t.computeW, tileW)
		}
		bandH = max(1, bandCells/bandW)
		if t.computeH > 0 {
			bandH = min(t.computeH, tileH)
		}
		return bandW, bandH
	}
	rows := max(1, bandCells/tileW)
	if t.r == 0 || tileW*tileH <= bandCells {
		return tileW, rows
	}
	want := max(minBandWidth, isqrt(bandCells))
	cols := max(1, (tileW+want/2)/want)
	bandW = ceilDiv(tileW, cols)
	return bandW, max(1, bandCells/bandW)
}

// plan divides a w×h raster into tiles in row-major order and each tile
// into bands, and numbers the bands in that order: tile by tile, then row
// of bands, then band across. Band i is found by arithmetic, so a plan of
// millions of bands costs nothing.
//
// A band is bandW×bandH, clipped by its tile and by the raster, so the
// last tile column and the last tile row each hold a different number of
// bands from a full one — the four counts below. Every other tile holds
// colsFull×rowsFull, because ceilDiv leaves the last column of a full tile
// non-empty.
type plan struct {
	w, h         int
	tileW, tileH int
	bandW, bandH int
	tilesX       int // tiles per tile row
	tileRows     int // rows of tiles
	colsFull     int // bands across a tile of full width
	colsLast     int // bands across a tile of the last tile column
	rowsFull     int // rows of bands in a tile of full height
	rowsLast     int // rows of bands in a tile of the last tile row
	perTileRow   int // bands across a whole row of tiles
	bands        int // total
}

func newPlan(w, h int, t tiling) plan {
	p := plan{w: w, h: h, tileW: w, tileH: h}
	if t.tileW > 0 {
		p.tileW = min(t.tileW, w)
	}
	if t.tileH > 0 {
		p.tileH = min(t.tileH, h)
	}
	if w == 0 || h == 0 {
		return p // no bands
	}
	p.bandW, p.bandH = bandShape(p.tileW, p.tileH, t)
	p.tilesX = ceilDiv(w, p.tileW)
	p.tileRows = ceilDiv(h, p.tileH)
	p.colsFull = ceilDiv(p.tileW, p.bandW)
	p.colsLast = ceilDiv(w-(p.tilesX-1)*p.tileW, p.bandW)
	p.rowsFull = ceilDiv(p.tileH, p.bandH)
	p.rowsLast = ceilDiv(h-(p.tileRows-1)*p.tileH, p.bandH)
	p.perTileRow = (p.tilesX-1)*p.colsFull + p.colsLast
	p.bands = ((p.tileRows-1)*p.rowsFull + p.rowsLast) * p.perTileRow
	return p
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// isqrt is the integer square root of n, 0 for n <= 0.
func isqrt(n int) int {
	if n <= 0 {
		return 0
	}
	x := int(math.Sqrt(float64(n)))
	for x > 0 && x*x > n {
		x--
	}
	for (x+1)*(x+1) <= n {
		x++
	}
	return x
}

// spanSize is the largest span one Process call can cover under p: a
// band is bandW by at most bandH rows, and a band clipped by its tile or
// by the raster's edge is only ever smaller. It returns 0, 0 for a plan
// with no bands.
func (p *plan) spanSize() (w, h int) {
	if p.bands == 0 {
		return 0, 0
	}
	return p.bandW, min(p.bandH, p.tileH)
}

// allocScratch gives each worker the working memory a ScratchKernel asks
// for, for a span of at most w×h. It is one allocation of each kind for
// the whole call, like allocWorkers, so the worker count does not
// multiply the number of allocations — only their size, which is what
// DESIGN.md §27's bound says it does.
func (e *job) allocScratch(w, h int) {
	sk, ok := e.k.(ScratchKernel)
	if !ok || w <= 0 || h <= 0 {
		return
	}
	need := sk.Scratch(w, h)
	if need.Cells < 0 || need.Words < 0 || need.Views < 0 {
		panic(fmt.Sprintf("engine: kernel asked for negative scratch %+v", need))
	}
	n := len(e.workers)
	var cells []float32
	var bits []uint64
	var views []raster.Float32Raster
	if need.Cells > 0 {
		cells = make([]float32, n*need.Cells)
	}
	if need.Words > 0 {
		bits = make([]uint64, n*need.Words)
	}
	if need.Views > 0 {
		views = make([]raster.Float32Raster, n*need.Views)
	}
	for i := range e.workers {
		s := &e.workers[i].kscratch
		if need.Cells > 0 {
			s.Cells = cells[i*need.Cells : (i+1)*need.Cells : (i+1)*need.Cells]
		}
		if need.Words > 0 {
			s.Bits = bits[i*need.Words : (i+1)*need.Words : (i+1)*need.Words]
		}
		if need.Views > 0 {
			s.Views = views[i*need.Views : (i+1)*need.Views : (i+1)*need.Views]
		}
	}
}

// band returns band i's cells [x0, x1) × [y0, y1). It undoes the
// numbering one level at a time — tile row, tile column, then the band
// within the tile — each level a division and one comparison against the
// clipped last row or column. With whole-row bands colsFull, colsLast and
// the band column are all 1 and this is the tile-and-rows arithmetic it
// replaced.
func (p *plan) band(i int) (x0, y0, x1, y1 int) {
	perFullRow := p.rowsFull * p.perTileRow // bands in a full row of tiles
	tr, j, rows := i/perFullRow, i%perFullRow, p.rowsFull
	if tr >= p.tileRows-1 {
		tr, j, rows = p.tileRows-1, i-(p.tileRows-1)*perFullRow, p.rowsLast
	}
	perFullTile := rows * p.colsFull
	tx, cols := p.tilesX-1, p.colsLast
	if j < (p.tilesX-1)*perFullTile {
		tx, cols, j = j/perFullTile, p.colsFull, j%perFullTile
	} else {
		j -= (p.tilesX - 1) * perFullTile
	}
	br, bc := j/cols, j%cols
	tx0, ty0 := tx*p.tileW, tr*p.tileH
	x0 = tx0 + bc*p.bandW
	x1 = min(x0+p.bandW, tx0+p.tileW, p.w)
	y0 = ty0 + br*p.bandH
	y1 = min(y0+p.bandH, ty0+p.tileH, p.h)
	return x0, y0, x1, y1
}
