package exec

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/raster"
)

// ProcessChunked runs k over sources and sinks with bounded memory
// (DESIGN.md §24, §27). Each worker owns one buffer per input, the size
// of a tile grown by the kernel's radius on every side, and one per
// output, the size of a tile, all allocated once per call. For every
// tile, a worker reads the tile and its halo from each source, runs the
// kernel over the tile's bands exactly as ProcessN would, and writes the
// finished tile to each sink. Only the raster's own edge gets the edge
// policy, so the sinks receive the bits ProcessN would write into
// in-memory outputs, for every Options value.
//
// It returns nil when every tile is written, the first error a source or
// sink returned (wrapped with the operand and the tile's position), or
// ctx.Err() if ctx is done first. See the package documentation for what
// a failed or cancelled call leaves in the sinks. ctx must not be nil.
//
// It panics on programming errors: a nil kernel, source or sink, a
// negative radius, a wrong number of operands, operands of different
// sizes, a Masked source with a sink that is not Masked, memory sources
// and sinks that share memory (see the package documentation), negative
// Options, and panics of the kernel, sources or sinks.
func ProcessChunked(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, k Kernel, opts engine.Options) error {
	if k == nil {
		panic("engine: nil kernel")
	}
	r := k.Radius()
	checkChunked(dst, src, k, r, opts)
	return newChunkJob(dst, src, k, r, opts).run(ctx)
}

func checkChunked(dst []engine.RasterSink, src []engine.RasterSource, k Kernel, r int, opts engine.Options) {
	checkKernel(len(dst), len(src), k, r, opts)
	for i, d := range dst {
		if d == nil {
			panic(fmt.Sprintf("engine: dst[%d] is a nil sink", i))
		}
	}
	for i, s := range src {
		if s == nil {
			panic(fmt.Sprintf("engine: src[%d] is a nil source", i))
		}
	}
	w, h := dst[0].Size()
	if w <= 0 || h <= 0 {
		panic(fmt.Sprintf("engine: dst[0] is %d×%d; sizes must be positive", w, h))
	}
	requireSize := func(name string, i, sw, sh int) {
		if sw != w || sh != h {
			panic(fmt.Sprintf("engine: %s[%d] is %d×%d, dst[0] is %d×%d", name, i, sw, sh, w, h))
		}
	}
	masked := false
	for i, s := range src {
		sw, sh := s.Size()
		requireSize("src", i, sw, sh)
		masked = masked || s.Masked()
	}
	for i, d := range dst {
		sw, sh := d.Size()
		requireSize("dst", i, sw, sh)
		if masked && !d.Masked() {
			panic(fmt.Sprintf("engine: a source is Masked but dst[%d] is not; its validity would be lost", i))
		}
	}

	// Memory sinks are written while memory sources are read and other
	// sinks written, a tile at a time and concurrently, so they must share
	// neither cells nor mask words with another operand. Other storage
	// cannot be checked here.
	for i, d := range dst {
		ms, ok := d.(*engine.MemorySink)
		if !ok {
			continue
		}
		a := ms.Raster()
		for j, other := range dst[i+1:] {
			if mo, ok := other.(*engine.MemorySink); ok && memoryMeet(a, mo.Raster()) {
				panic(fmt.Sprintf("engine: memory sinks dst[%d] and dst[%d] share memory", i, i+1+j))
			}
		}
		for j, s := range src {
			if mo, ok := s.(*engine.MemorySource); ok && memoryMeet(a, mo.Raster()) {
				panic(fmt.Sprintf("engine: memory sink dst[%d] shares memory with memory source src[%d]; "+
					"tiles would read cells other tiles write (use a Tiled entry point on the rasters instead)", i, j))
			}
		}
	}
}

// memoryMeet reports whether two rasters share Data or validity words.
func memoryMeet(a, b raster.Float32Raster) bool {
	return overlap.DataSpans(a, b) || overlap.Words(a, b)
}

// chunkJob is one ProcessChunked call after its checks.
type chunkJob struct {
	r    int
	w, h int
	// band is the tiling a tile's own plan is built from: the compute
	// tile the caller forced, if any, and the radius.
	band tiling
	// tileW and tileH are the tile size, tilesX the tiles per tile row
	// and tiles their number. Tiles are numbered in row-major order.
	tileW, tileH  int
	tilesX, tiles int

	dst []engine.RasterSink
	src []engine.RasterSource

	workers []chunkWorker
	// out is the caller's Options.Stats, or nil.
	out *engine.Stats
}

// chunkWorker is one worker's buffers and the job that runs its tiles.
// Nothing in it is shared with another worker.
type chunkWorker struct {
	// in and out are whole buffers, one per input and per output: in
	// holds a tile grown by the radius and clipped to the raster, out a
	// tile. Buffers with a mask have a Stride that is a multiple of 64
	// and the mask at bit 0, so every row's validity starts on a word
	// boundary (DESIGN.md §23); buffers without one are compact.
	in, out []raster.Float32Raster
	// t runs the kernel over one tile at a time, with views of in and out
	// as its operands and one worker, so it takes no locks. Its own
	// worker holds this worker's kernel counters; stats here holds what
	// crossed the source and sink interfaces, which t cannot see.
	t job
	// stats counts the bytes this worker's tiles read from sources and
	// wrote to sinks. Padded like worker.stats.
	stats engine.Stats
	_     [64]byte
}

func newChunkJob(dst []engine.RasterSink, src []engine.RasterSource, k Kernel, r int, opts engine.Options) *chunkJob {
	c := &chunkJob{r: r, band: tilingOf(opts, r).inner(), dst: dst, src: src, out: opts.Stats}
	c.w, c.h = dst[0].Size()
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
	dstMasked := false
	for _, d := range dst {
		dstMasked = dstMasked || d.Masked()
	}

	// Input buffers hold at most tileW+2r by tileH+2r cells, and never
	// more than the raster. A buffer with a mask has Stride rounded up to
	// a multiple of 64, so each row's bits start a word; one without is
	// compact, which lets sources read and sinks write consecutive rows
	// in one call and pointwise kernels take their whole-span path.
	inW, inH := min(c.w, c.tileW+2*r), min(c.h, c.tileH+2*r)
	buffer := func(w, h int, masked bool) (stride, cells, words int) {
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

	c.workers = make([]chunkWorker, workerCount(opts.Workers, c.tiles))
	for i := range c.workers {
		wk := &c.workers[i]
		// One Data and one mask allocation per worker for all its
		// buffers.
		cellsTotal, maskWords := 0, 0
		for _, s := range src {
			_, cells, words := buffer(inW, inH, s.Masked())
			cellsTotal, maskWords = cellsTotal+cells, maskWords+words
		}
		for _, d := range dst {
			_, cells, words := buffer(c.tileW, c.tileH, d.Masked())
			cellsTotal, maskWords = cellsTotal+cells, maskWords+words
		}
		data := make([]float32, cellsTotal)
		var bits []uint64
		if maskWords > 0 {
			bits = make([]uint64, maskWords)
		}
		nin, nout := len(src), len(dst)
		views := make([]raster.Float32Raster, 2*(nin+nout))
		wk.in, wk.out = views[:nin:nin], views[nin:nin+nout:nin+nout]
		carve := func(cells, words int) ([]float32, []uint64) {
			d := data[:cells:cells]
			data = data[cells:]
			if words == 0 {
				return d, nil
			}
			m := bits[:words:words]
			bits = bits[words:]
			return d, m
		}
		for j, s := range src {
			stride, cells, words := buffer(inW, inH, s.Masked())
			d, m := carve(cells, words)
			wk.in[j] = raster.Float32Raster{Data: d, Width: inW, Height: inH, Stride: stride, Valid: m}
		}
		for j, s := range dst {
			stride, cells, words := buffer(c.tileW, c.tileH, s.Masked())
			d, m := carve(cells, words)
			wk.out[j] = raster.Float32Raster{Data: d, Width: c.tileW, Height: c.tileH, Stride: stride, Valid: m}
		}

		t := &wk.t
		t.src, t.dst = views[nin+nout:2*nin+nout:2*nin+nout], views[2*nin+nout:]
		t.setup(k, r, c.w, c.h, masked, dstMasked)
		spanW, spanH, bandW := c.bandBounds()
		t.allocWorkers(1, bandW)
		t.allocScratch(spanW, spanH)
	}
	return c
}

// bandBounds is the largest span one Process call can cover in any of
// this job's tiles, and the widest band any of them plans. A tile gets
// its own plan, so a tile clipped at the raster's edge has its own band
// shape, and the largest is not always the full tile's: a narrower tile
// takes taller bands, and — since bandShape leaves a tile no wider than
// minBandWidth whole — a *wider* band as well, so a clipped tile of 1500
// bands at 1500 where the full 4096 tile bands at 1024. The widest band
// is what ErodeScratch must cover and it is not the largest span's width,
// so the two are tracked separately. There are only ever two widths and
// two heights, so all four are checked rather than bounded.
func (c *chunkJob) bandBounds() (spanW, spanH, bandW int) {
	lastW := c.w - (c.tilesX-1)*c.tileW
	lastH := c.h - (ceilDiv(c.h, c.tileH)-1)*c.tileH
	best := 0
	for _, tw := range [2]int{c.tileW, lastW} {
		for _, th := range [2]int{c.tileH, lastH} {
			p := newPlan(tw, th, c.band)
			sw, sh := p.spanSize()
			if sw*sh > best {
				best, spanW, spanH = sw*sh, sw, sh
			}
			bandW = max(bandW, p.bandW)
		}
	}
	return spanW, spanH, bandW
}

func roundUp64(n int) int { return (n + 63) &^ 63 }

// run processes every tile on the workers. Reads and writes get a
// context without ctx's cancellation, so that a tile a worker has claimed
// is always read, computed and written completely: cancellation stops
// workers claiming tiles, and never leaves a tile half written.
func (c *chunkJob) run(ctx context.Context) error {
	ioCtx := context.WithoutCancel(ctx)
	err := runWorkers(ctx, len(c.workers), c.tiles, func(w, i int) error {
		return c.tile(ioCtx, &c.workers[w], i)
	})
	c.report()
	return err
}

// report totals the workers' counters into the caller's Stats. Each
// worker's traffic is in two places — the source and sink bytes it moved
// itself, and the kernel bytes its per-tile job recorded — because the
// inner job has no idea it is running inside a tile. Adding them here is
// what makes a chunked call's Amplification comparable with a tiled
// call's: same denominator, four stages instead of two.
func (c *chunkJob) report() {
	if c.out == nil {
		return
	}
	for i := range c.workers {
		s := c.workers[i].stats
		s.Add(c.workers[i].t.workers[0].stats)
		s.Ideal = s.Cells * bytesPerCell * int64(len(c.src)+len(c.dst))
		c.out.Add(s)
	}
}

// tile reads, computes and writes tile i on worker wk.
func (c *chunkJob) tile(ctx context.Context, wk *chunkWorker, i int) error {
	r := c.r
	x0, y0 := (i%c.tilesX)*c.tileW, (i/c.tilesX)*c.tileH
	x1, y1 := min(x0+c.tileW, c.w), min(y0+c.tileH, c.h)
	rx0, ry0 := max(0, x0-r), max(0, y0-r)
	rx1, ry1 := min(c.w, x1+r), min(c.h, y1+r)

	t := &wk.t
	wk.stats.Tiles++
	// A tile's halo is read from the sources as well as by the kernel, so
	// the window, not the tile, is what crossed the interface.
	wk.stats.SourceRead += int64(rx1-rx0) * int64(ry1-ry0) * bytesPerCell * int64(len(c.src))
	for j, s := range c.src {
		v := bufferView(wk.in[j], rx1-rx0, ry1-ry0)
		if err := s.ReadWindow(ctx, v, rx0, ry0); err != nil {
			return &ioError{"reading src", j, rx0, ry0, err}
		}
		t.src[j] = v
	}
	for j := range c.dst {
		t.dst[j] = bufferView(wk.out[j], x1-x0, y1-y0)
	}
	t.dx, t.dy, t.sx, t.sy = x0, y0, rx0, ry0
	t.plan = newPlan(x1-x0, y1-y0, c.band)
	for b := range t.plan.bands {
		t.band(&t.workers[0], b)
	}
	wk.stats.SinkWritten += int64(x1-x0) * int64(y1-y0) * bytesPerCell * int64(len(c.dst))
	for j, d := range c.dst {
		if err := d.WriteWindow(ctx, t.dst[j], x0, y0); err != nil {
			return &ioError{"writing dst", j, x0, y0, err}
		}
	}
	return nil
}

// ioError is an error of a source or sink, with the operand and window
// position it came from. It is formatted only when printed, so that
// workers stop claiming tiles as soon as a tile fails.
type ioError struct {
	op   string // "reading src" or "writing dst"
	j    int
	x, y int
	err  error
}

func (e *ioError) Error() string {
	return fmt.Sprintf("engine: %s[%d] at (%d, %d): %v", e.op, e.j, e.x, e.y, e.err)
}

func (e *ioError) Unwrap() error { return e.err }

// bufferView returns the w×h raster at the start of buf, with buf's
// stride and mask.
func bufferView(buf raster.Float32Raster, w, h int) raster.Float32Raster {
	n := (h-1)*buf.Stride + w
	return raster.Float32Raster{Data: buf.Data[:n:n], Width: w, Height: h, Stride: buf.Stride, Valid: buf.Valid}
}
