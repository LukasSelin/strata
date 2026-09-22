package resample

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
)

// Method selects how Resample computes an output cell from the source
// cells around it. The zero value is Nearest.
type Method uint8

const (
	// Nearest copies the source cell under the output cell's centre,
	// gdalwarp's -r near.
	Nearest Method = iota
	// Bilinear interpolates linearly between the four nearest source
	// centres, gdalwarp's -r bilinear.
	Bilinear
	// Cubic is Keys' cubic convolution with a = -0.5, gdalwarp's -r cubic.
	Cubic
	// Lanczos is the Lanczos-windowed sinc with a = 3, gdalwarp's
	// -r lanczos.
	Lanczos
	// Average is the mean of the source cells the output cell covers,
	// weighted by the area each overlaps, gdalwarp's -r average.
	Average
)

func (m Method) String() string { return resamp.Method(m).String() }

// Options configures Resample.
type Options struct {
	// Method is the resampling method. The zero value is Nearest.
	Method Method
}

// Resample resamples src onto dst's grid: every cell of dst.Raster gets
// the value src.Raster takes at that cell under opts.Method. See the
// package documentation for the geometry, edges and validity.
//
// It panics on programming errors (see the package documentation): grids
// that do not match their rasters, differ in CRS or have a zero or
// non-finite resolution or origin, operands that share memory, or a dst
// without a validity mask where a cell can be invalid.
func Resample(dst, src raster.Dataset, opts Options) {
	_ = ResampleTiled(context.Background(), dst, src, opts, engine.Options{Workers: 1})
}

// ResampleTiled is Resample run in tiles on engine.Options.Workers
// goroutines: it applies the same checks and writes the same bits for
// every engine.Options, and returns ctx.Err() if ctx is done before
// every cell is written. See package engine for tiling and cancellation.
func ResampleTiled(ctx context.Context, dst, src raster.Dataset, opts Options, eopts engine.Options) error {
	checkDataset("dst", dst)
	checkDataset("src", src)
	p := newPlan(dst.Grid, src.Grid, opts, eopts)
	d, s := dst.Raster, src.Raster
	if overlap.DataSpans(d, s) || overlap.Words(d, s) {
		panic("resample: dst and src share memory")
	}
	requireMask(p, d.Valid != nil, s.Valid != nil)

	tiles := newTiling(d.Width, d.Height, eopts, p.plan)
	workers := workerCount(eopts.Workers, tiles.units)
	var mu *sync.Mutex
	if workers > 1 && d.Valid != nil {
		mu = new(sync.Mutex)
	}
	ws := getWorkspaces(workers)
	defer putWorkspaces(ws)
	stats := make([]engine.Stats, workers)
	source := resamp.Source{R: s}
	err := exec.RunUnits(ctx, workers, tiles.units, func(w, i int) error {
		x0, y0, x1, y1 := tiles.unit(i)
		if y0 == y1 {
			return nil
		}
		read := p.plan.Band(ws[w], d.Window(x0, y0, x1-x0, y1-y0), x0, y0, source, mu)
		st := &stats[w]
		st.Bands++
		st.Cells += int64(x1-x0) * int64(y1-y0)
		st.KernelRead += int64(read) * 4
		return nil
	})
	report(eopts.Stats, stats)
	return err
}

// ResampleChunked is Resample over a source and sink a tile of dst at a
// time, in bounded memory (DESIGN.md §27, §54). dstGrid and srcGrid are
// the grids the sink and source are laid out on. Each tile reads only the
// source window its cells reach, so a worker holds one output tile, one
// such window and the intermediate between the passes; see the package
// documentation for the bound. It applies Resample's checks, writes the
// same bits for every engine.Options, and returns ctx.Err() or the first
// source or sink error. A claimed tile is always read, computed and
// written in full, so after an error or cancellation the sink holds a
// prefix of whole tiles.
func ResampleChunked(ctx context.Context, dst engine.RasterSink, dstGrid raster.Grid, src engine.RasterSource,
	srcGrid raster.Grid, opts Options, eopts engine.Options) error {
	if dst == nil || src == nil {
		panic("resample: nil source or sink")
	}
	checkSize("dst", dstGrid, dst)
	checkSize("src", srcGrid, src)
	p := newPlan(dstGrid, srcGrid, opts, eopts)
	if ms, ok := dst.(*engine.MemorySink); ok {
		if mo, ok := src.(*engine.MemorySource); ok && (overlap.DataSpans(ms.Raster(), mo.Raster()) || overlap.Words(ms.Raster(), mo.Raster())) {
			panic("resample: memory sink shares memory with the memory source")
		}
	}
	requireMask(p, dst.Masked(), src.Masked())
	return runChunked(ctx, p, dst, src, eopts)
}

// plan is one call's resampling plan and the checks it came from.
type plan struct {
	plan *resamp.Plan
}

func newPlan(dg, sg raster.Grid, opts Options, eopts engine.Options) plan {
	if opts.Method > Average {
		panic(fmt.Sprintf("resample: unknown %v", opts.Method))
	}
	if eopts.TileWidth < 0 || eopts.TileHeight < 0 || eopts.Workers < 0 {
		panic(fmt.Sprintf("resample: negative Options %+v", eopts))
	}
	if dg.CRS.Code != "" && sg.CRS.Code != "" && dg.CRS.Code != sg.CRS.Code {
		panic(fmt.Sprintf("resample: dst CRS %q differs from src CRS %q; reprojection is not supported", dg.CRS.Code, sg.CRS.Code))
	}
	checkGrid("dst", dg)
	checkGrid("src", sg)
	return plan{resamp.NewPlan(resamp.Method(opts.Method),
		resamp.Spec{N: dg.Width, Origin: dg.OriginX, Res: dg.ResolutionX, SrcN: sg.Width, SrcOrigin: sg.OriginX, SrcRes: sg.ResolutionX},
		resamp.Spec{N: dg.Height, Origin: dg.OriginY, Res: dg.ResolutionY, SrcN: sg.Height, SrcOrigin: sg.OriginY, SrcRes: sg.ResolutionY},
	)}
}

// requireMask panics if dst cannot record a cell's validity that the
// resampling may clear: when src has a mask, or when some dst cell lies
// outside the source.
func requireMask(p plan, dstMasked, srcMasked bool) {
	if dstMasked {
		return
	}
	if srcMasked {
		panic("resample: src has a validity mask but dst has none; its validity would be lost")
	}
	x, y := &p.plan.X, &p.plan.Y
	if x.Lo != 0 || x.Hi != len(x.First) || y.Lo != 0 || y.Hi != len(y.First) {
		panic("resample: dst extends beyond src, so some cells are invalid, but dst has no validity mask")
	}
}

func checkDataset(name string, d raster.Dataset) {
	if err := d.Raster.Validate(); err != nil {
		panic(fmt.Sprintf("resample: %s: %v", name, err))
	}
	if d.Grid.Width != d.Raster.Width || d.Grid.Height != d.Raster.Height {
		panic(fmt.Sprintf("resample: %s grid is %d×%d but its raster is %d×%d",
			name, d.Grid.Width, d.Grid.Height, d.Raster.Width, d.Raster.Height))
	}
}

func checkSize(name string, g raster.Grid, s interface{ Size() (int, int) }) {
	w, h := s.Size()
	if g.Width != w || g.Height != h {
		panic(fmt.Sprintf("resample: %s grid is %d×%d but its raster is %d×%d", name, g.Width, g.Height, w, h))
	}
}

func checkGrid(name string, g raster.Grid) {
	if g.Width <= 0 || g.Height <= 0 {
		panic(fmt.Sprintf("resample: %s grid is %d×%d; sizes must be positive", name, g.Width, g.Height))
	}
	for _, v := range []float64{g.OriginX, g.OriginY, g.ResolutionX, g.ResolutionY} {
		if v != v || v-v != 0 {
			panic(fmt.Sprintf("resample: %s grid has a non-finite origin or resolution", name))
		}
	}
	if g.ResolutionX == 0 || g.ResolutionY == 0 {
		panic(fmt.Sprintf("resample: %s grid has a zero resolution", name))
	}
}

// workspaces pools the workers' scratch across calls, so a call on
// rasters of a size seen before allocates only its tables.
var workspaces sync.Pool

func getWorkspaces(n int) []*resamp.Workspace {
	ws := make([]*resamp.Workspace, n)
	for i := range ws {
		if w, ok := workspaces.Get().(*resamp.Workspace); ok {
			ws[i] = w
		} else {
			ws[i] = new(resamp.Workspace)
		}
	}
	return ws
}

func putWorkspaces(ws []*resamp.Workspace) {
	for _, w := range ws {
		workspaces.Put(w)
	}
}

func workerCount(workers, units int) int {
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return max(1, min(workers, units))
}

// report adds the workers' counters to out.
func report(out *engine.Stats, stats []engine.Stats) {
	if out == nil {
		return
	}
	var s engine.Stats
	for _, w := range stats {
		s.Add(w)
	}
	s.KernelWritten = s.Cells * 4
	s.Ideal = s.Cells * 4 * 2
	out.Add(s)
}
