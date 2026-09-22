package resample

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
)

// bandCells is the target number of output cells in one band of a Tiled
// call, and minBandRows the fewest rows a band has when its tile is that
// tall. Each band runs the horizontal pass over every source row its
// output rows reach, so neighbouring bands both compute the rows their
// footprints share; tall bands keep that repeated work to a few per cent
// even for Lanczos, whose six-row reach would otherwise dominate short
// bands. A variable so tests can make bands small.
var (
	bandCells   = 1 << 17
	minBandRows = 64
)

// tiling is the output tiles of a Tiled call, each split into bands of
// whole rows; a band is the unit of work.
type tiling struct {
	w, h           int
	tileW, tileH   int
	tilesX, tilesY int
	bandRows       int
	bandsPerTile   int
	units          int
}

func newTiling(w, h int, eopts engine.Options, _ *resamp.Plan) tiling {
	t := tiling{w: w, h: h, tileW: eopts.TileWidth, tileH: eopts.TileHeight}
	if t.tileW == 0 || t.tileW > w {
		t.tileW = w
	}
	if t.tileH == 0 || t.tileH > h {
		t.tileH = h
	}
	t.tilesX, t.tilesY = ceilDiv(w, t.tileW), ceilDiv(h, t.tileH)
	t.bandRows = min(t.tileH, max(minBandRows, ceilDiv(bandCells, t.tileW)))
	t.bandsPerTile = ceilDiv(t.tileH, t.bandRows)
	t.units = t.tilesX * t.tilesY * t.bandsPerTile
	return t
}

// unit returns band i's output rectangle [x0, x1) × [y0, y1). Bands are
// numbered tile by tile, tiles in row-major order.
func (t tiling) unit(i int) (x0, y0, x1, y1 int) {
	tile, b := i/t.bandsPerTile, i%t.bandsPerTile
	x0, ty0 := (tile%t.tilesX)*t.tileW, (tile/t.tilesX)*t.tileH
	x1, ty1 := min(x0+t.tileW, t.w), min(ty0+t.tileH, t.h)
	y0 = ty0 + b*t.bandRows
	y1 = min(y0+t.bandRows, ty1)
	if y0 >= y1 {
		// The last tile row is shorter than the others: its surplus bands
		// are empty.
		return x0, ty1, x1, ty1
	}
	return x0, y0, x1, y1
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// chunkWorker is one Chunked worker's buffers.
type chunkWorker struct {
	ws    workspace
	in    raster.Float32Raster
	out   raster.Float32Raster
	stats engine.Stats
	_     [64]byte
}

// runChunked runs a Chunked call: output tiles on workers, each reading
// its source footprint into the worker's input buffer, resampling it into
// the worker's output buffer and writing that to dst.
func runChunked(ctx context.Context, p plan, dst engine.RasterSink, src engine.RasterSource, eopts engine.Options) error {
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
	pl := p.plan
	// The input buffer holds the largest footprint of any tile. Tiles
	// share a column range per tile column and a row range per tile row,
	// so the widest and tallest are found one axis at a time.
	fpW, fpH := 0, 0
	for i := range tilesX {
		f, e := pl.X.Footprint(i*tw, min((i+1)*tw, w), pl.HalfValid)
		fpW = max(fpW, e-f)
	}
	for i := range tilesY {
		f, e := pl.Y.Footprint(i*th, min((i+1)*th, h), pl.HalfValid)
		fpH = max(fpH, e-f)
	}
	fpW, fpH = max(fpW, 1), max(fpH, 1)

	workers := workerCount(eopts.Workers, n)
	wks := make([]chunkWorker, workers)
	for i := range wks {
		wks[i].in = buffer(fpW, fpH, src.Masked())
		wks[i].out = buffer(tw, th, dst.Masked())
	}
	ioCtx := context.WithoutCancel(ctx)
	err := exec.RunUnits(ctx, workers, n, func(wi, i int) error {
		wk := &wks[wi]
		x0, y0 := (i%tilesX)*tw, (i/tilesX)*th
		x1, y1 := min(x0+tw, w), min(y0+th, h)
		out := view(wk.out, x1-x0, y1-y0)
		var s source
		fx0, fy0, fx1, fy1 := pl.Footprint(x0, y0, x1, y1)
		if fx1 > fx0 {
			in := view(wk.in, fx1-fx0, fy1-fy0)
			if err := src.ReadWindow(ioCtx, in, fx0, fy0); err != nil {
				return fmt.Errorf("resample: reading src at (%d, %d): %w", fx0, fy0, err)
			}
			s = source{R: in, X0: fx0, Y0: fy0}
			wk.stats.SourceRead += int64(fx1-fx0) * int64(fy1-fy0) * 4
		}
		read := band(pl, &wk.ws, out, x0, y0, s, nil)
		if err := dst.WriteWindow(ioCtx, out, x0, y0); err != nil {
			return fmt.Errorf("resample: writing dst at (%d, %d): %w", x0, y0, err)
		}
		st := &wk.stats
		st.Tiles++
		st.Bands++
		st.Cells += int64(x1-x0) * int64(y1-y0)
		st.KernelRead += int64(read) * 4
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

// buffer allocates a w×h raster, with a mask and a Stride rounded up to
// a multiple of 64 when masked, so every row's bits start a word
// (DESIGN.md §23), and compact otherwise.
func buffer(w, h int, masked bool) raster.Float32Raster {
	stride := w
	if masked {
		stride = (w + 63) &^ 63
	}
	r := raster.NewFloat32Stride(w, h, stride, make([]float32, (h-1)*stride+w))
	if masked {
		r.Valid = raster.NewMask(h * stride)
	}
	return r
}

// view is the w×h raster at the start of buf, with buf's stride and mask.
func view(buf raster.Float32Raster, w, h int) raster.Float32Raster {
	n := (h-1)*buf.Stride + w
	return raster.Float32Raster{Data: buf.Data[:n:n], Width: w, Height: h, Stride: buf.Stride, Valid: buf.Valid}
}
