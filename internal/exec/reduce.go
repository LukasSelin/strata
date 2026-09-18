package exec

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// Reducer folds raster cells into a partial result of type P. It is the
// fold counterpart of Kernel — inputs and no outputs — and the engine
// runs it over the same tiles, bands and workers (DESIGN.md §49).
//
// The engine calls Fold once per band, concurrently on disjoint bands and
// each on a partial of its own, then combines the partials on the calling
// goroutine once every worker has stopped. So a Reducer holds no mutable
// state, starts no goroutines, and keeps no reference to the Cells it is
// given after Fold returns, exactly as a Kernel does.
//
// A reduction always has radius 0. A neighbourhood reduction is a map
// pass followed by a fold, not a reducer with a halo.
//
// The partial is a type parameter rather than an interface value because
// it is folded once per band, not once per cell: an interface would cost
// nothing in time (DESIGN.md §19) but would box every partial and force a
// type assertion in every Fold. P is the accumulator, not the cell type,
// so it says nothing about the generic element type of DESIGN.md §9.
type Reducer[P any] interface {
	// Inputs is the number of input rasters Fold expects, at least one.
	Inputs() int
	// Fold folds the valid cells of src into p.
	Fold(p *P, src Cells)
	// Combine folds b into a. It must be associative and commutative,
	// and the zero P must be its identity: the engine starts every
	// worker from a zero partial and combines them in an order that
	// depends on the tiling and the worker count, and the result must
	// not (DESIGN.md §49).
	Combine(a *P, b P)
}

// Cells is the rectangle of input cells one Fold call reads: the fold
// counterpart of Span and Window, with no outputs and no halo.
type Cells struct {
	// X and Y locate the rectangle's top-left cell in the whole raster,
	// for reducers whose result depends on position.
	X, Y int
	// Width and Height are the rectangle's size in cells.
	Width, Height int
	// Src holds one Width×Height view per input, in the order given to
	// Reduce. The views share the inputs' memory and keep their strides,
	// so Row(y) and Stride == Width (a compact view, one contiguous run
	// of cells) work as on any raster.
	Src []raster.Float32Raster
	// Masked reports whether any input has a validity mask. When it is
	// false every cell takes part and a reducer folds whole rows, or the
	// whole rectangle when the views are compact, without consulting
	// ValidBits at all.
	Masked bool

	// masked lists the inputs of Src that have a mask. ValidBits uses it.
	masked []int
}

// ValidBits returns the validity of the k cells starting at cell (x, y)
// of the rectangle — the AND of every masked input over them — as the low
// k bits of the result, LSB first, cell (x, y) in bit 0. k must be in
// [1, 64] and the cells must lie in the rectangle's row y.
//
// It returns all ones when no input has a mask, so a reducer that ignores
// Masked is still correct, only slower.
func (c Cells) ValidBits(x, y, k int) uint64 {
	all := ^uint64(0) >> uint(64-k)
	if len(c.masked) == 0 {
		return all
	}
	v := all
	for _, j := range c.masked {
		s := c.Src[j]
		v &= raster.MaskBits(s.Valid, s.ValidOffset+y*s.Stride+x, k)
	}
	return v
}

// Reduce folds every cell of the inputs src, which must have r's arity
// and all the same Width and Height, into one result. It returns the
// combined partials once every band has been folded, or ctx.Err() and the
// zero P if ctx is done before then. ctx must not be nil.
//
// Unlike a cancelled map, a cancelled reduction returns no value at all:
// a fold over an unknown subset of a raster is not a useful answer, where
// the prefix of tiles a cancelled ProcessN leaves in its outputs is
// (DESIGN.md §49). A call whose every band finishes before ctx is done
// returns its value, even if ctx is done by the time it returns, as
// ProcessN does.
//
// It panics on programming errors: a nil reducer, a reducer with no
// inputs, a wrong number of rasters, rasters that fail Validate or differ
// in size, and negative Options.
//
// Inputs are only read, so unlike ProcessN they may overlap each other,
// and each other's validity words, freely.
func Reduce[P any](ctx context.Context, src []raster.Float32Raster, r Reducer[P], opts engine.Options) (P, error) {
	var zero P
	if r == nil {
		panic("engine: nil reducer")
	}
	checkReduce(src, r.Inputs(), opts)
	e := newReduceJob(src, 0, 0, opts)
	slots := make([]reduceSlot[P], len(e.workers))
	err := runWorkers(ctx, len(e.workers), e.plan.bands, func(w, i int) error {
		foldBand(e, &e.workers[w], r, &slots[w].p, i)
		return nil
	})
	if err != nil {
		return zero, err
	}
	return combineSlots(r, slots), nil
}

func checkReduce(src []raster.Float32Raster, nin int, opts engine.Options) {
	checkReducerArity(len(src), nin, opts)
	for i, s := range src {
		requireRaster("src", i, s, src[0], "src[0]")
	}
}

// checkReducerArity checks a reducer's arity against the operand count,
// and the Options, for both drivers.
func checkReducerArity(nsrc, nin int, opts engine.Options) {
	if nin < 1 {
		panic(fmt.Sprintf("engine: reducer arity (%d inputs, 0 outputs) needs at least one input", nin))
	}
	if nsrc != nin {
		panic(fmt.Sprintf("engine: reducer takes %d inputs, got %d", nin, nsrc))
	}
	if opts.TileWidth < 0 || opts.TileHeight < 0 || opts.Workers < 0 {
		panic(fmt.Sprintf("engine: negative Options %+v", opts))
	}
}

// reduceSlot is one worker's partial, padded so that workers folding into
// neighbouring slots do not share a cache line. A partial is written once
// per band, which is often enough for false sharing to show as scaling
// that flattens for no visible reason.
type reduceSlot[P any] struct {
	p P
	_ [64]byte
}

// combineSlots folds every worker's partial into the first, on the
// calling goroutine. runWorkers has joined every worker by now, so each
// partial's writes are visible here.
func combineSlots[P any](r Reducer[P], slots []reduceSlot[P]) P {
	out := slots[0].p
	for k := 1; k < len(slots); k++ {
		r.Combine(&out, slots[k].p)
	}
	return out
}

// reduceJob is one Reduce call after its checks, or one worker's tile of
// a ReduceChunked call. Everything a band needs is allocated here, once
// per call, so bands allocate nothing.
type reduceJob struct {
	plan plan
	// src holds the operands: the whole rasters in a Reduce call, and one
	// buffer per input covering the tile in a ReduceChunked one. A
	// reduction has radius 0, so a tile buffer is exactly its tile and
	// carries no halo.
	src []raster.Float32Raster
	// ox and oy are the raster position of the plan's origin: (0, 0) in a
	// Reduce call and the tile's corner in a chunked one. They reach the
	// reducer as Cells.X and Cells.Y.
	ox, oy int
	// masked lists the inputs that have a validity mask.
	masked []int
	// workers holds each worker's views; workers[0] runs on the calling
	// goroutine.
	workers []reduceWorker
}

// reduceWorker is the state one goroutine uses for its bands. Nothing in
// it is shared with another worker, and no band takes a lock: a reduction
// writes no validity bits, and the input words that neighbouring bands
// share are only read (compare job.maskLock, which exists for writes).
type reduceWorker struct {
	// views are the Cells views handed to the reducer, overwritten for
	// every band.
	views []raster.Float32Raster
}

func newReduceJob(src []raster.Float32Raster, ox, oy int, opts engine.Options) *reduceJob {
	e := &reduceJob{src: src, ox: ox, oy: oy}
	for j, s := range src {
		if s.Valid != nil {
			e.masked = append(e.masked, j)
		}
	}
	e.plan = newPlan(src[0].Width, src[0].Height, opts.TileWidth, opts.TileHeight)
	e.allocReduceWorkers(workerCount(opts.Workers, e.plan.bands))
	return e
}

// allocReduceWorkers gives the job n workers, each with views for its
// inputs, from one backing array, so a call's allocations do not grow
// with the worker count.
func (e *reduceJob) allocReduceWorkers(n int) {
	e.workers = make([]reduceWorker, n)
	ns := len(e.src)
	views := make([]raster.Float32Raster, n*ns)
	for i := range e.workers {
		e.workers[i].views = views[i*ns : (i+1)*ns]
	}
}

// foldBand folds band i of the plan into p on worker wk.
func foldBand[P any](e *reduceJob, wk *reduceWorker, r Reducer[P], p *P, i int) {
	x0, y0, x1, y1 := e.plan.band(i)
	w, h := x1-x0, y1-y0
	for j, s := range e.src {
		wk.views[j] = s.Window(x0, y0, w, h)
	}
	r.Fold(p, Cells{
		X: x0 + e.ox, Y: y0 + e.oy,
		Width: w, Height: h,
		Src:    wk.views,
		Masked: len(e.masked) > 0,
		masked: e.masked,
	})
}
