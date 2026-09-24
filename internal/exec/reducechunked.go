package exec

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// ReduceChunked folds every cell of the sources src into one result with
// bounded memory (DESIGN.md §24, §27, §49). Each worker owns one buffer
// per input, the size of a tile, allocated once per call; a reduction has
// radius 0, so a buffer needs no halo and a tile is read exactly once.
// For every tile a worker reads the tile from each source and folds its
// bands exactly as Reduce would, and the partials combine on the calling
// goroutine.
//
// It returns the combined partials once every tile has been folded, the
// first error a source returned (wrapped with the operand and the tile's
// position), or ctx.Err() if ctx is done first — and on any error the
// zero P, never a partial fold (DESIGN.md §49). ctx must not be nil.
//
// It panics on programming errors: a nil reducer or source, a reducer
// with no inputs, a wrong number of sources, sources of different sizes,
// negative Options, and panics of the reducer or the sources.
//
// Unlike ProcessChunked there is nothing to write, so no source may be
// invalidated by the call and memory sources need no aliasing checks:
// they are only read, and may overlap each other freely.
func ReduceChunked[P any](ctx context.Context, src []engine.RasterSource, r Reducer[P], opts engine.Options) (P, error) {
	var zero P
	if r == nil {
		panic("engine: nil reducer")
	}
	checkReduceChunked(src, r.Inputs(), opts)
	c := newReduceChunkJob(src, opts)
	slots := make([]reduceSlot[P], len(c.workers))
	ioCtx := context.WithoutCancel(ctx)
	err := runWorkers(ctx, len(c.workers), c.tiles, func(w, i int) error {
		return foldTile(ioCtx, c, &c.workers[w], r, &slots[w].p, i)
	})
	c.report(opts.Stats)
	if err != nil {
		return zero, err
	}
	return combineSlots(r, slots), nil
}

func checkReduceChunked(src []engine.RasterSource, nin int, opts engine.Options) {
	checkReducerArity(len(src), nin, opts)
	for i, s := range src {
		if s == nil {
			panic(fmt.Sprintf("engine: src[%d] is a nil source", i))
		}
	}
	w, h := src[0].Size()
	if w <= 0 || h <= 0 {
		panic(fmt.Sprintf("engine: src[0] is %d×%d; sizes must be positive", w, h))
	}
	for i, s := range src {
		sw, sh := s.Size()
		if sw != w || sh != h {
			panic(fmt.Sprintf("engine: src[%d] is %d×%d, src[0] is %d×%d", i, sw, sh, w, h))
		}
	}
}

// reduceChunkJob is one ReduceChunked call after its checks.
type reduceChunkJob struct {
	w, h int
	// tileW and tileH are the tile size, tilesX the tiles per tile row and
	// tiles their number. Tiles are numbered in row-major order.
	tileW, tileH  int
	tilesX, tiles int

	src []engine.RasterSource
	// masked lists the Masked sources.
	masked []int

	workers []reduceChunkWorker
}

// reduceChunkWorker is one worker's buffers and the job that folds its
// tiles. Nothing in it is shared with another worker.
type reduceChunkWorker struct {
	// in holds one whole buffer per input, the size of a tile. A buffer
	// with a mask has a Stride that is a multiple of 64 and its mask at
	// bit 0, so every row's validity starts on a word boundary and
	// Cells.ValidBits reads a plain word (DESIGN.md §23); one without is
	// compact, so a source moves consecutive full-width rows in one call
	// and a reducer keeps its whole-rectangle path.
	in []raster.Float32Raster
	// t folds one tile at a time, with views of in as its operands and one
	// worker. Its own worker holds this worker's fold counters; stats
	// here holds what crossed the source interface.
	t reduceJob
	// live holds the masked sources whose current tile has an invalid
	// cell, with room for all of them (unmaskAllValid).
	live []int
	// stats counts the bytes this worker's tiles read from sources.
	// Padded like reduceSlot.
	stats engine.Stats
	_     [64]byte
}

func newReduceChunkJob(src []engine.RasterSource, opts engine.Options) *reduceChunkJob {
	c := &reduceChunkJob{src: src}
	c.w, c.h = src[0].Size()
	c.tileW, c.tileH = c.w, c.h
	if opts.TileWidth > 0 {
		c.tileW = min(opts.TileWidth, c.w)
	}
	if opts.TileHeight > 0 {
		c.tileH = min(opts.TileHeight, c.h)
	}
	c.tilesX = ceilDiv(c.w, c.tileW)
	c.tiles = c.tilesX * ceilDiv(c.h, c.tileH)

	var masked []int
	for j, s := range src {
		if s.Masked() {
			masked = append(masked, j)
		}
	}
	c.masked = masked

	c.workers = make([]reduceChunkWorker, workerCount(opts.Workers, c.tiles))
	for i := range c.workers {
		wk := &c.workers[i]
		// One Data and one mask allocation per worker for all its buffers.
		cellsTotal, maskWords := 0, 0
		for _, s := range src {
			_, cells, words := reduceBuffer(c.tileW, c.tileH, s.Masked())
			cellsTotal, maskWords = cellsTotal+cells, maskWords+words
		}
		data := make([]float32, cellsTotal)
		var bits []uint64
		if maskWords > 0 {
			bits = make([]uint64, maskWords)
		}
		nin := len(src)
		views := make([]raster.Float32Raster, 2*nin)
		wk.in = views[:nin:nin]
		for j, s := range src {
			stride, cells, words := reduceBuffer(c.tileW, c.tileH, s.Masked())
			d := data[:cells:cells]
			data = data[cells:]
			var m []uint64
			if words > 0 {
				m = bits[:words:words]
				bits = bits[words:]
			}
			wk.in[j] = raster.Float32Raster{Data: d, Width: c.tileW, Height: c.tileH, Stride: stride, Valid: m}
		}

		t := &wk.t
		t.src = views[nin:]
		t.masked = masked
		wk.live = make([]int, 0, len(masked))
		t.allocReduceWorkers(1)
	}
	return c
}

// reduceBuffer returns the stride, cell count and mask words of one
// worker's buffer for a w×h tile. A buffer with validity has its Stride
// rounded up to a multiple of 64, so each row's bits start a word; one
// without is compact.
func reduceBuffer(w, h int, masked bool) (stride, cells, words int) {
	stride = w
	if masked {
		stride = roundUp64(w)
	}
	cells = (h-1)*stride + w
	if masked {
		words = raster.MaskWords(cells)
	}
	return stride, cells, words
}

// report totals the workers' counters into out, if it is not nil. It
// takes the Stats rather than holding one because ReduceChunked is
// generic and builds its job before its slots; there is no other reason.
func (c *reduceChunkJob) report(out *engine.Stats) {
	if out == nil {
		return
	}
	for i := range c.workers {
		s := c.workers[i].stats
		s.Add(c.workers[i].t.workers[0].stats)
		s.Ideal = s.Cells * bytesPerCell * int64(len(c.src))
		out.Add(s)
	}
}

// foldTile reads tile i on worker wk and folds its bands into p.
func foldTile[P any](ctx context.Context, c *reduceChunkJob, wk *reduceChunkWorker, r Reducer[P], p *P, i int) error {
	x0, y0 := (i%c.tilesX)*c.tileW, (i/c.tilesX)*c.tileH
	x1, y1 := min(x0+c.tileW, c.w), min(y0+c.tileH, c.h)

	t := &wk.t
	wk.stats.Tiles++
	// A reduction has radius 0, so a tile is read exactly once: no halo
	// here, unlike ProcessChunked.
	wk.stats.SourceRead += int64(x1-x0) * int64(y1-y0) * bytesPerCell * int64(len(c.src))
	for j, s := range c.src {
		v := bufferView(wk.in[j], x1-x0, y1-y0)
		if err := s.ReadWindow(ctx, v, x0, y0); err != nil {
			return &ioError{"reading src", j, x0, y0, err}
		}
		t.src[j] = v
	}
	// A tile whose masked sources are all valid folds as an unmasked one.
	t.masked = unmaskAllValid(t.src, c.masked, wk.live)
	t.ox, t.oy = x0, y0
	t.plan = newPlan(x1-x0, y1-y0, 0, 0)
	for b := range t.plan.bands {
		foldBand(t, &t.workers[0], r, p, b)
	}
	return nil
}
