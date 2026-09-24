package resample

import (
	"context"
	"fmt"
	"math/bits"
	"slices"
	"sync"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
)

// Mosaic resamples every source onto dst's grid and lays them over one
// another in order: each cell of dst takes its value from the last source
// whose resampling gives it a valid one, and is invalid if none does.
// That is gdalwarp with several sources, and the bits of Resample of each
// source onto dst's grid followed by that overlay. See the package
// documentation.
//
// It panics as Resample does, for dst against each source, and also if
// two sources' CRSs differ, or if dst has no validity mask and a cell
// can be invalid: when a source has a mask, or when the sources do not
// cover every cell of dst.
func Mosaic(dst raster.Dataset, srcs []raster.Dataset, opts Options) {
	_ = MosaicTiled(context.Background(), dst, srcs, opts, engine.Options{Workers: 1})
}

// MosaicTiled is Mosaic run in tiles on engine.Options.Workers
// goroutines: it applies the same checks and writes the same bits for
// every engine.Options, and returns ctx.Err() if ctx is done before every
// cell is written.
func MosaicTiled(ctx context.Context, dst raster.Dataset, srcs []raster.Dataset, opts Options, eopts engine.Options) error {
	checkDataset("dst", dst)
	grids := make([]raster.Grid, len(srcs))
	masked := make([]bool, len(srcs))
	d := dst.Raster
	for i, s := range srcs {
		checkDataset(fmt.Sprintf("src %d", i), s)
		if overlap.DataSpans(d, s.Raster) || overlap.Words(d, s.Raster) {
			panic(fmt.Sprintf("resample: dst and src %d share memory", i))
		}
		grids[i], masked[i] = s.Grid, s.Raster.Valid != nil
	}
	m := newMosaic(dst.Grid, grids, masked, opts, eopts)
	m.requireMask(d.Valid != nil)

	tiles := newTiling(d.Width, d.Height, eopts, nil)
	workers := workerCount(eopts.Workers, tiles.units)
	var mu *sync.Mutex
	if workers > 1 && d.Valid != nil {
		mu = new(sync.Mutex)
	}
	wks := make([]mosaicWorker, workers)
	for i := range wks {
		wks[i].scratch = buffer(tiles.tileW, tiles.bandRows, true)
	}
	stats := make([]engine.Stats, workers)
	err := exec.RunUnits(ctx, workers, tiles.units, func(w, i int) error {
		x0, y0, x1, y1 := tiles.unit(i)
		if y0 == y1 {
			return nil
		}
		read := func(l *layer, _, _, _, _ int) (source, error) {
			return source{R: srcs[l.src].Raster}, nil
		}
		n, _ := m.compose(&wks[w], d.Window(x0, y0, x1-x0, y1-y0), x0, y0, read, mu)
		st := &stats[w]
		st.Bands++
		st.Cells += int64(x1-x0) * int64(y1-y0)
		st.KernelRead += int64(n) * 4
		return nil
	})
	report(eopts.Stats, stats)
	return err
}

// MosaicChunked is Mosaic over sources and a sink a tile of dst at a
// time, in bounded memory (DESIGN.md §27, §54). dstGrid and srcGrids[i]
// are the grids the sink and srcs[i] are laid out on. Each tile reads, of
// each source, only the window its cells reach, and only while some of
// its cells are still invalid: a source hidden under the ones after it is
// not read there. A worker holds one output tile, one source window at a
// time, a resampled layer and the intermediate between the passes. It
// applies Mosaic's checks, writes the same bits for every engine.Options,
// and returns ctx.Err() or the first source or sink error. A claimed tile
// is always read, computed and written in full, so after an error or
// cancellation the sink holds a prefix of whole tiles.
func MosaicChunked(ctx context.Context, dst engine.RasterSink, dstGrid raster.Grid, srcs []engine.RasterSource,
	srcGrids []raster.Grid, opts Options, eopts engine.Options) error {
	if dst == nil {
		panic("resample: nil sink")
	}
	if len(srcs) != len(srcGrids) {
		panic(fmt.Sprintf("resample: %d sources but %d source grids", len(srcs), len(srcGrids)))
	}
	checkSize("dst", dstGrid, dst)
	masked := make([]bool, len(srcs))
	ms, _ := dst.(*engine.MemorySink)
	anyMasked := false
	for i, s := range srcs {
		if s == nil {
			panic(fmt.Sprintf("resample: nil source %d", i))
		}
		checkSize(fmt.Sprintf("src %d", i), srcGrids[i], s)
		if mo, ok := s.(*engine.MemorySource); ok && ms != nil && (overlap.DataSpans(ms.Raster(), mo.Raster()) || overlap.Words(ms.Raster(), mo.Raster())) {
			panic(fmt.Sprintf("resample: memory sink shares memory with memory source %d", i))
		}
		masked[i] = s.Masked()
		anyMasked = anyMasked || masked[i]
	}
	m := newMosaic(dstGrid, srcGrids, masked, opts, eopts)
	m.requireMask(dst.Masked())

	w, h := dst.Size()
	tw, th := eopts.TileWidth, eopts.TileHeight
	if tw == 0 || tw > w {
		tw = w
	}
	if th == 0 || th > h {
		th = h
	}
	tilesX, tilesY := ceilDiv(w, tw), ceilDiv(h, th)
	n := tilesX * tilesY
	// The input buffer holds the largest window any layer reads for any
	// tile, found one axis at a time as runChunked does.
	fpW, fpH := 1, 1
	for i := range m.layers {
		l := &m.layers[i]
		for t := range tilesX {
			f, e := l.plan.X.Footprint(t*tw-l.x0, min((t+1)*tw, w)-l.x0, l.plan.Window)
			fpW = max(fpW, e-f)
		}
		for t := range tilesY {
			f, e := l.plan.Y.Footprint(t*th-l.y0, min((t+1)*th, h)-l.y0, l.plan.Window)
			fpH = max(fpH, e-f)
		}
	}

	workers := workerCount(eopts.Workers, n)
	wks := make([]chunkedMosaicWorker, workers)
	for i := range wks {
		wks[i].in = buffer(fpW, fpH, anyMasked)
		wks[i].scratch = buffer(tw, th, true)
		wks[i].out[0] = buffer(tw, th, dst.Masked())
		wks[i].out[1] = buffer(tw, th, dst.Masked())
	}
	ioCtx := context.WithoutCancel(ctx)
	sinks := []engine.RasterSink{dst}
	wrap := func(_, x, y int, err error) error {
		return fmt.Errorf("resample: writing dst at (%d, %d): %w", x, y, err)
	}
	err := exec.RunUnitsBehind(ctx, workers, n, sinks, wrap, func(wi, i int, b *exec.Behind) error {
		wk := &wks[wi]
		x0, y0 := (i%tilesX)*tw, (i/tilesX)*th
		x1, y1 := min(x0+tw, w), min(y0+th, h)
		read := func(l *layer, px0, py0, px1, py1 int) (source, error) {
			fx0, fy0, fx1, fy1 := l.plan.Footprint(px0, py0, px1, py1)
			in := view(wk.in, fx1-fx0, fy1-fy0)
			if !masked[l.src] {
				in.Valid = nil
			}
			if err := srcs[l.src].ReadWindow(ioCtx, in, fx0, fy0); err != nil {
				return source{}, fmt.Errorf("resample: reading src %d at (%d, %d): %w", l.src, fx0, fy0, err)
			}
			wk.stats.SourceRead += int64(fx1-fx0) * int64(fy1-fy0) * 4
			return source{R: in, X0: fx0, Y0: fy0}, nil
		}
		set := b.Acquire()
		out := view(wk.out[set], x1-x0, y1-y0)
		cells, err := m.compose(&wk.mosaicWorker, out, x0, y0, read, nil)
		if err != nil {
			return err
		}
		views := [1]raster.Float32Raster{out}
		b.Submit(set, views[:], x0, y0)
		st := &wk.stats
		st.Tiles++
		st.Bands++
		st.Cells += int64(x1-x0) * int64(y1-y0)
		st.KernelRead += int64(cells) * 4
		st.SinkWritten += int64(x1-x0) * int64(y1-y0) * 4
		return nil
	})
	stats := make([]engine.Stats, len(wks))
	for i := range wks {
		stats[i] = wks[i].stats
	}
	report(eopts.Stats, stats)
	return err
}

// MosaicCovers reports whether a mosaic of sources laid out on srcGrids
// onto dst, none of them with a mask, makes every cell of dst valid: the
// condition under which Mosaic accepts a dst without a mask. It panics
// on the grids and options Mosaic panics on.
func MosaicCovers(dst raster.Grid, srcGrids []raster.Grid, opts Options) bool {
	return newMosaic(dst, srcGrids, make([]bool, len(srcGrids)), opts, engine.Options{}).covers()
}

// mosaic is one call's plan: a layer for every source that reaches dst,
// in source order.
type mosaic struct {
	w, h   int
	layers []layer
	masked []bool
}

// layer is one source's resampling, planned over the window of dst it
// reaches (resamp.Reach), so that its tables cost that window rather than
// dst's width and height.
type layer struct {
	src int
	// x0, y0 is the window's top-left cell in dst: the plan's output
	// cell (c, r) is dst cell (x0+c, y0+r).
	x0, y0 int
	plan   *resamp.Plan
	// cx0, cy0, cx1, cy1 are the dst cells the plan covers.
	cx0, cy0, cx1, cy1 int
}

func newMosaic(dg raster.Grid, sgs []raster.Grid, masked []bool, opts Options, eopts engine.Options) *mosaic {
	if opts.Method > Average {
		panic(fmt.Sprintf("resample: unknown %v", opts.Method))
	}
	if eopts.TileWidth < 0 || eopts.TileHeight < 0 || eopts.Workers < 0 {
		panic(fmt.Sprintf("resample: negative Options %+v", eopts))
	}
	checkGrid("dst", dg)
	// Every grid must match every other (DESIGN.md §36). Matching is
	// equality among the codes that are not empty, so one reference
	// code, the first known, is enough to compare against.
	ref, refName := dg.CRS, "dst"
	for i, sg := range sgs {
		checkGrid(fmt.Sprintf("src %d", i), sg)
		if !ref.Matches(sg.CRS) {
			panic(fmt.Sprintf("resample: %s CRS %s differs from src %d CRS %s; reprojection is not supported",
				refName, ref.Describe(), i, sg.CRS.Describe()))
		}
		if ref.Code == "" && sg.CRS.Code != "" {
			ref, refName = sg.CRS, fmt.Sprintf("src %d", i)
		}
	}
	m := &mosaic{w: dg.Width, h: dg.Height, masked: masked}
	for i, sg := range sgs {
		xs := resamp.Spec{N: dg.Width, Origin: dg.OriginX, Res: dg.ResolutionX, SrcN: sg.Width, SrcOrigin: sg.OriginX, SrcRes: sg.ResolutionX}
		ys := resamp.Spec{N: dg.Height, Origin: dg.OriginY, Res: dg.ResolutionY, SrcN: sg.Height, SrcOrigin: sg.OriginY, SrcRes: sg.ResolutionY}
		xlo, xhi := resamp.Reach(xs)
		ylo, yhi := resamp.Reach(ys)
		if xlo >= xhi || ylo >= yhi {
			continue
		}
		xs.N, xs.Offset = xhi-xlo, xlo
		ys.N, ys.Offset = yhi-ylo, ylo
		p := resamp.NewPlan(resamp.Method(opts.Method), xs, ys)
		if p.X.Lo >= p.X.Hi || p.Y.Lo >= p.Y.Hi {
			continue
		}
		m.layers = append(m.layers, layer{
			src: i, x0: xlo, y0: ylo, plan: p,
			cx0: xlo + p.X.Lo, cy0: ylo + p.Y.Lo, cx1: xlo + p.X.Hi, cy1: ylo + p.Y.Hi,
		})
	}
	return m
}

// requireMask panics if dst cannot record a cell's validity that the
// mosaic may clear: when a source has a mask, or when some cell of dst is
// covered by no source.
func (m *mosaic) requireMask(dstMasked bool) {
	if dstMasked {
		return
	}
	for i, ms := range m.masked {
		if ms {
			panic(fmt.Sprintf("resample: src %d has a validity mask but dst has none; its validity would be lost", i))
		}
	}
	if !m.covers() {
		panic("resample: the sources do not cover dst, so some cells are invalid, but dst has no validity mask")
	}
}

// covers reports whether the layers' covered rectangles together cover
// every cell of dst. It sweeps the columns where a rectangle starts or
// ends, and in each strip between them requires the rectangles spanning
// it to cover every row.
func (m *mosaic) covers() bool {
	xs := []int{0}
	for _, l := range m.layers {
		xs = append(xs, l.cx0, l.cx1)
	}
	slices.Sort(xs)
	xs = slices.Compact(xs)
	var spans [][2]int
	for _, x := range xs {
		if x >= m.w {
			break
		}
		spans = spans[:0]
		for _, l := range m.layers {
			if l.cx0 <= x && x < l.cx1 {
				spans = append(spans, [2]int{l.cy0, l.cy1})
			}
		}
		slices.SortFunc(spans, func(a, b [2]int) int { return a[0] - b[0] })
		reach := 0
		for _, s := range spans {
			if s[0] > reach {
				return false
			}
			reach = max(reach, s[1])
		}
		if reach < m.h {
			return false
		}
	}
	return true
}

// mosaicWorker is one worker's scratch for compose.
type mosaicWorker struct {
	ws workspace
	// scratch holds one layer resampled over a tile, with its validity.
	scratch raster.Float32Raster
	// set marks the cells of the tile a layer has already written, one
	// row of words per tile row, as band's bits are laid out.
	set []uint64
}

// chunkedMosaicWorker is one MosaicChunked worker's buffers.
type chunkedMosaicWorker struct {
	mosaicWorker
	in    raster.Float32Raster
	out   [2]raster.Float32Raster
	stats engine.Stats
	_     [64]byte
}

// compose computes the cells [x0, x0+out.Width) × [y0, y0+out.Height) of
// dst into out, a view whose cell (0, 0) is dst cell (x0, y0). It visits
// the layers from the last, resamples each over the tile's cells it
// covers into the worker's scratch, reading its source through read with
// the plan's output rectangle, and keeps the valid cells that no later
// layer has set. It stops once every cell is set, so a layer hidden under
// later ones is neither read nor computed. Cells no layer sets get NaN.
// Since band gives each cell the value and validity of its plan alone,
// every tiling gives the bits of the per-source Resample calls. If out
// has a mask, compose writes its bits under mu, which may be nil when no
// other goroutine writes out's mask words. It returns the number of
// source cells read, for engine.Stats.
func (m *mosaic) compose(wk *mosaicWorker, out raster.Float32Raster, x0, y0 int,
	read func(l *layer, px0, py0, px1, py1 int) (source, error), mu *sync.Mutex) (n int, err error) {
	w, h := out.Width, out.Height
	words := raster.MaskWords(w)
	wk.set = grow(wk.set, h*words)
	clear(wk.set)
	left := w * h
	for li := len(m.layers) - 1; li >= 0 && left > 0; li-- {
		l := &m.layers[li]
		ix0, iy0 := max(x0, l.cx0), max(y0, l.cy0)
		ix1, iy1 := min(x0+w, l.cx1), min(y0+h, l.cy1)
		if ix0 >= ix1 || iy0 >= iy1 {
			continue
		}
		px0, py0 := ix0-l.x0, iy0-l.y0
		src, err := read(l, px0, py0, ix1-l.x0, iy1-l.y0)
		if err != nil {
			return n, err
		}
		sc := view(wk.scratch, ix1-ix0, iy1-iy0)
		n += band(l.plan, &wk.ws, sc, px0, py0, src, nil)
		left -= overlay(out, wk.set, words, ix0-x0, iy0-y0, sc)
	}
	for r := range h {
		row := out.Row(r)
		for c := range w {
			if !raster.MaskGet(wk.set, r*words*64+c) {
				row[c] = nan
			}
		}
	}
	if out.Valid != nil {
		if mu != nil {
			mu.Lock()
			defer mu.Unlock()
		}
		for r := range h {
			raster.MaskCopyRange(out.Valid, out.ValidOffset+r*out.Stride, wk.set[r*words:(r+1)*words], 0, w)
		}
	}
	return n, nil
}

// overlay copies the valid cells of sc that set does not yet mark into
// out at (ox, oy), marks them, and returns how many it copied.
func overlay(out raster.Float32Raster, set []uint64, words, ox, oy int, sc raster.Float32Raster) (copied int) {
	for r := range sc.Height {
		row := out.Row(oy + r)[ox : ox+sc.Width]
		srow := sc.Row(r)
		base := (oy+r)*words*64 + ox
		for i := 0; i < sc.Width; i += 64 {
			k := min(64, sc.Width-i)
			b := raster.MaskBits(sc.Valid, sc.ValidOffset+r*sc.Stride+i, k) &^ raster.MaskBits(set, base+i, k)
			copied += bits.OnesCount64(b)
			for ; b != 0; b &= b - 1 {
				q := bits.TrailingZeros64(b)
				row[i+q] = srow[i+q]
				raster.MaskSet(set, base+i+q, true)
			}
		}
	}
	return copied
}
