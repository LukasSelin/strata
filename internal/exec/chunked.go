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
// of a tile grown by the kernel's radius on every side, and two per
// output, the size of a tile, all allocated once per call. For every
// tile, a worker reads the tile and its halo from each source, runs the
// kernel over the tile's bands exactly as ProcessN would, and hands the
// finished tile to its writer, which writes it to each sink while the
// worker goes on to the next (write-behind; see RunUnitsBehind). Only the raster's own edge gets the edge
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
	// tileW and tileH are the tile size, tilesX the tiles per tile row
	// and tiles their number. Tiles are numbered in row-major order.
	tileW, tileH  int
	tilesX, tiles int

	dst []engine.RasterSink
	src []engine.RasterSource
	// masked lists the Masked sources.
	masked []int
	// valid is how each output's validity is derived from all of them
	// (outValidities), shared by every worker for tiles that keep every
	// mask; nil when no output has a mask.
	valid []outValidity

	workers []chunkWorker
	// out is the caller's Options.Stats, or nil.
	out *engine.Stats
}

// chunkWorker is one worker's buffers and the job that runs its tiles.
// Nothing in it is shared with another worker.
type chunkWorker struct {
	// in and out are whole buffers, one per input and two sets of one
	// per output, for write-behind: in holds a tile grown by the radius
	// and clipped to the raster, out a tile. Buffers with a mask have a
	// Stride that is a multiple of 64 and the mask at bit 0, so every
	// row's validity starts on a word boundary (DESIGN.md §23); buffers
	// without one are compact.
	in  []raster.Float32Raster
	out [2][]raster.Float32Raster
	// t runs the kernel over one tile at a time, with views of in and out
	// as its operands and one worker, so it takes no locks. Its own
	// worker holds this worker's kernel counters; stats here holds what
	// crossed the source and sink interfaces, which t cannot see.
	t job
	// live holds the masked sources whose current tile has an invalid
	// cell, with room for all of them. See unmaskAllValid.
	live []int
	// valid and validInts hold the validity rules of a tile that dropped
	// a mask, derived from live alone; nil when no output or no source
	// has a mask.
	valid     []outValidity
	validInts []int
	// stats counts the bytes this worker's tiles read from sources and
	// wrote to sinks. Padded like worker.stats.
	stats engine.Stats
	_     [64]byte
}

func newChunkJob(dst []engine.RasterSink, src []engine.RasterSource, k Kernel, r int, opts engine.Options) *chunkJob {
	c := &chunkJob{r: r, dst: dst, src: src, out: opts.Stats}
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
	c.masked = masked
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

	valid := outValidities(k, r, masked, dstMasked,
		make([]outValidity, len(dst)), make([]int, validityInts(len(masked), len(dst))))
	c.valid = valid
	edgeW := edgeWidths(k, r, len(src), len(dst))
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
			cellsTotal, maskWords = cellsTotal+2*cells, maskWords+2*words
		}
		data := make([]float32, cellsTotal)
		var bits []uint64
		if maskWords > 0 {
			bits = make([]uint64, maskWords)
		}
		nin, nout := len(src), len(dst)
		views := make([]raster.Float32Raster, 2*nin+3*nout)
		wk.in = views[:nin:nin]
		wk.out[0], wk.out[1] = views[nin:nin+nout:nin+nout], views[nin+nout:nin+2*nout:nin+2*nout]
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
		for set := range wk.out {
			for j, s := range dst {
				stride, cells, words := buffer(c.tileW, c.tileH, s.Masked())
				d, m := carve(cells, words)
				wk.out[set][j] = raster.Float32Raster{Data: d, Width: c.tileW, Height: c.tileH, Stride: stride, Valid: m}
			}
		}

		// live and a tile's validity rules share one allocation.
		ints := make([]int, len(masked)+validityInts(len(masked), len(dst)))
		wk.live = ints[:0:len(masked)]
		if valid != nil && len(masked) > 0 {
			wk.valid = make([]outValidity, len(dst))
			wk.validInts = ints[len(masked):]
		}
		t := &wk.t
		t.src, t.dst = views[nin+2*nout:2*nin+2*nout:2*nin+2*nout], views[2*nin+2*nout:]
		t.setup(k, r, c.w, c.h, masked, dstMasked, valid, edgeW)
		t.allocWorkers(1, c.tileW)
		t.allocScratch(c.spanSize())
		t.allocPad(c.spanSize())
	}
	return c
}

// spanSize is the largest span one Process call can cover in any of this
// job's tiles. A tile gets its own plan, so a tile clipped at the
// raster's edge has its own band height — a narrower tile takes taller
// bands — and the largest span is not always the full tile's. There are
// only ever two widths and two heights, so all four are checked rather
// than bounded.
//
// The result bounds each side separately, the widest span's width and
// the tallest span's height, not the pair with the largest area:
// ScratchKernel.Scratch may size its memory from w alone (a row, as
// focal's separable kernels do), and the widest span, from the full
// tile, can have less area than a clipped tile's taller band
// (DESIGN.md §53).
func (c *chunkJob) spanSize() (w, h int) {
	lastW := c.w - (c.tilesX-1)*c.tileW
	lastH := c.h - (ceilDiv(c.h, c.tileH)-1)*c.tileH
	for _, tw := range [2]int{c.tileW, lastW} {
		for _, th := range [2]int{c.tileH, lastH} {
			p := newPlan(tw, th, 0, 0)
			sw, sh := p.spanSize()
			w, h = max(w, sw), max(h, sh)
		}
	}
	return w, h
}

func roundUp64(n int) int { return (n + 63) &^ 63 }

// run processes every tile on the workers, writing behind (see
// RunUnitsBehind). Reads and writes get a context without ctx's
// cancellation, so that a tile a worker has claimed is always read,
// computed and written completely: cancellation stops workers claiming
// tiles, and never leaves a tile half written.
func (c *chunkJob) run(ctx context.Context) error {
	defer func() {
		for i := range c.workers {
			c.workers[i].t.releaseScratch()
		}
	}()
	ioCtx := context.WithoutCancel(ctx)
	defer c.report()
	return runBehind(ctx, len(c.workers), c.tiles, c.dst,
		func(j, x, y int, err error) error { return &ioError{"writing dst", j, x, y, err} },
		func(w, i int, b *Behind) error { return c.tile(ioCtx, &c.workers[w], b, i) })
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

// tile reads and computes tile i on worker wk, and hands it to b for
// writing.
func (c *chunkJob) tile(ctx context.Context, wk *chunkWorker, b *Behind, i int) error {
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
	t.masked = unmaskAllValid(t.src, c.masked, wk.live)
	t.valid = c.valid
	if c.valid != nil && len(t.masked) < len(c.masked) {
		// The shared rules name inputs whose masks this tile dropped, and
		// would erode a nil mask: derive the tile's from the inputs that
		// kept theirs. Dropping all-valid inputs from an AND changes no
		// bit (DESIGN.md §31).
		t.valid = outValidities(t.k, c.r, t.masked, true, wk.valid, wk.validInts)
	}
	// The output buffers are taken after the reads, so that the last
	// tile's write overlaps them too.
	set := b.Acquire()
	for j := range c.dst {
		t.dst[j] = bufferView(wk.out[set][j], x1-x0, y1-y0)
	}
	t.dx, t.dy, t.sx, t.sy = x0, y0, rx0, ry0
	t.plan = newPlan(x1-x0, y1-y0, 0, 0)
	for b := range t.plan.bands {
		t.band(&t.workers[0], b)
	}
	wk.stats.SinkWritten += int64(x1-x0) * int64(y1-y0) * bytesPerCell * int64(len(c.dst))
	b.Submit(set, t.dst, x0, y0)
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

// unmaskAllValid drops the mask of each tile buffer in bufs, of the
// sources masked lists, whose cells are all valid, and returns the rest
// of masked in live's memory: the inputs the tile's validity still has to
// be worked out from.
//
// A nil mask means every cell valid (DESIGN.md §31), so this changes no
// output: an eroded or ANDed mask of all-valid inputs is all valid, and
// the reductions' results do not depend on the path that folds them
// (§49). What it changes is the work. A tile read from a source with a
// fill value that holds none, the usual case away from a raster's
// NoData border, then gets the unmasked paths: interior validity filled
// rather than eroded, and a fold that never consults ValidBits. The test
// costs one pass over the tile's validity words, 1/32 of its cells'
// bytes, and stops at the first invalid cell.
func unmaskAllValid(bufs []raster.Float32Raster, masked, live []int) []int {
	live = live[:0]
	for _, j := range masked {
		if allValid(bufs[j]) {
			bufs[j].Valid = nil
		} else {
			live = append(live, j)
		}
	}
	return live
}

// allValid reports whether every cell of r, which has a mask, is valid.
func allValid(r raster.Float32Raster) bool {
	if compact(r) {
		return bitsAllSet(r.Valid, r.ValidOffset, r.Width*r.Height)
	}
	for y := range r.Height {
		if !bitsAllSet(r.Valid, r.ValidOffset+y*r.Stride, r.Width) {
			return false
		}
	}
	return true
}

// bitsAllSet reports whether bits [off, off+n) of m are all set, a word
// at a time where they are word-aligned, as a tile buffer's rows are.
func bitsAllSet(m []uint64, off, n int) bool {
	for n > 0 {
		if off&63 == 0 && n >= 64 {
			words := m[off>>6 : off>>6+n>>6]
			for _, w := range words {
				if w != ^uint64(0) {
					return false
				}
			}
			off += len(words) * 64
			n -= len(words) * 64
			continue
		}
		k := min(n, 64-off&63)
		if raster.MaskBits(m, off, k) != ^uint64(0)>>(64-uint(k)) {
			return false
		}
		off += k
		n -= k
	}
	return true
}

// bufferView returns the w×h raster at the start of buf, with buf's
// stride and mask.
func bufferView(buf raster.Float32Raster, w, h int) raster.Float32Raster {
	n := (h-1)*buf.Stride + w
	return raster.Float32Raster{Data: buf.Data[:n:n], Width: w, Height: h, Stride: buf.Stride, Valid: buf.Valid}
}
