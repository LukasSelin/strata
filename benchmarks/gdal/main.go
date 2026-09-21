// Command strataterrain runs one terrain operation over a raw float32
// raster and reports how long it took. It is the strata side of the
// gdaldem comparison in this directory's RESULTS.md: gdalbench.sh runs
// this binary and gdaldem under the same timer, in the same container,
// over the same bytes.
//
//	strataterrain -dir out -op slope -w 11264 -h 11264 -cell 12.5 -fill 65535
//
// It reads dem.raw and writes strata-<op>.raw, both raw little-endian
// float32, with -9999 under invalid cells — the NoData value gdaldem
// writes, so that the two results can be differenced by the acceptance
// suite's gdalcompare.py without any further conversion.
//
// Two modes:
//
//   - chunked (the default) streams the file through engine.RawSource
//     and engine.RawSink with bounded memory. This is what gdaldem does,
//     so the whole process is what the comparison times.
//   - memory reads the raster in first and times only the plain call, to
//     separate the kernel's cost from the file's. gdaldem has no
//     equivalent, so that number stands on its own.
//
// The timings it prints are in-process. The wall time of the whole
// process, which is what a user waits for and what gdalbench.sh
// compares, is measured from outside.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

var (
	dir      = flag.String("dir", "out", "directory holding dem.raw")
	op       = flag.String("op", "slope", "slope, aspect or hillshade")
	mode     = flag.String("mode", "chunked", "chunked (file to file) or memory (whole raster)")
	wFlag    = flag.Int("w", 4096, "raster width in cells")
	hFlag    = flag.Int("h", 4096, "raster height in cells")
	cell     = flag.Float64("cell", 12.5, "cell size in ground units")
	fill     = flag.Float64("fill", 65535, "the NoData value in dem.raw")
	azimuth  = flag.Float64("azimuth", 315, "hillshade azimuth")
	altitude = flag.Float64("altitude", 45, "hillshade altitude")
	tile     = flag.Int("tile", 256, "tile height for the chunked runs")
	workers  = flag.Int("workers", 1, "engine workers")
	scalar   = flag.Bool("scalar", false, "force the scalar kernels")
	sync     = flag.Bool("sync", false, "fsync the output before stopping the clock")
	repeat   = flag.Int("repeat", 1, "run the operation this many times in one process")
)

// outFill is what strata writes under invalid cells, chosen to match the
// NoData value gdaldem writes into its own results.
const outFill = -9999

func main() {
	start := time.Now()
	flag.Parse()
	if err := run(start); err != nil {
		fmt.Fprintln(os.Stderr, "strataterrain:", err)
		os.Exit(1)
	}
}

func run(start time.Time) error {
	if *scalar {
		stencil.UseScalar(true)
	}
	w, h := *wFlag, *hFlag
	ctx := context.Background()

	so := terrain.SlopeOptions{CellSize: *cell, CellSizeY: *cell}
	ao := terrain.AspectOptions{CellSize: *cell, CellSizeY: *cell}
	ho := terrain.HillshadeOptions{
		CellSize: *cell, CellSizeY: *cell,
		Azimuth: *azimuth, Altitude: *altitude,
	}

	var plain func(dst, dem raster.Float32Raster)
	var chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error
	switch *op {
	case "slope":
		plain = func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, so) }
		chunked = func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
			return terrain.SlopeChunked(ctx, dst, src, so, eo)
		}
	case "aspect":
		plain = func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) }
		chunked = func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
			return terrain.AspectChunked(ctx, dst, src, ao, eo)
		}
	case "hillshade":
		plain = func(dst, dm raster.Float32Raster) { terrain.Hillshade(dst, dm, ho) }
		chunked = func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
			return terrain.HillshadeChunked(ctx, dst, src, ho, eo)
		}
	case "none":
		// Nothing to compute: the run measures what starting the
		// process costs, which is the floor under every other case.
		fmt.Printf("process ms=%.1f\n", time.Since(start).Seconds()*1000)
		return nil
	default:
		return fmt.Errorf("unknown -op %q", *op)
	}

	inOpts := engine.RawOptions{Fill: float32(*fill), HasFill: true}
	outOpts := engine.RawOptions{Fill: outFill, HasFill: true}
	demPath := filepath.Join(*dir, "dem.raw")
	outPath := filepath.Join(*dir, "strata-"+*op+".raw")
	eo := engine.Options{TileHeight: *tile, Workers: *workers}
	cells := float64(w) * float64(h)

	fmt.Printf("backend=%s workers=%d tile=%d mode=%s op=%s gomaxprocs=%d\n",
		stencil.Backend(), *workers, *tile, *mode, *op, runtime.GOMAXPROCS(0))

	report := func(i int, d time.Duration) {
		fmt.Printf("run=%d op=%s ms=%.1f Mcells_s=%.1f\n",
			i, *op, d.Seconds()*1000, cells/d.Seconds()/1e6)
	}

	switch *mode {
	case "chunked":
		// One handle per worker: calls on a single *os.File queue behind
		// each other (see engine.RawFile).
		in, err := engine.OpenRawFile(demPath, os.O_RDONLY, 0, *workers)
		if err != nil {
			return err
		}
		defer in.Close()
		for i := range *repeat {
			out, err := engine.OpenRawFile(outPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, *workers)
			if err != nil {
				return err
			}
			src := engine.NewRawSource(in, w, h, inOpts)
			sink := engine.NewRawSink(out, w, h, outOpts)
			t0 := time.Now()
			if err := chunked(ctx, sink, src, eo); err != nil {
				_ = out.Close() // the operation's error matters more
				return err
			}
			if *sync {
				if err := out.Sync(); err != nil {
					_ = out.Close() // the sync error matters more
					return err
				}
			}
			report(i, time.Since(t0))
			if err := out.Close(); err != nil {
				return err
			}
		}

	case "memory":
		in, err := engine.OpenRawFile(demPath, os.O_RDONLY, 0, *workers)
		if err != nil {
			return err
		}
		defer in.Close()
		dem := masked(w, h)
		t0 := time.Now()
		if err := engine.NewRawSource(in, w, h, inOpts).ReadWindow(ctx, dem, 0, 0); err != nil {
			return err
		}
		fmt.Printf("read ms=%.1f\n", time.Since(t0).Seconds()*1000)
		dst := raster.NewFloat32Like(dem)
		for i := range *repeat {
			t0 := time.Now()
			plain(dst, dem)
			report(i, time.Since(t0))
		}

	default:
		return fmt.Errorf("unknown -mode %q", *mode)
	}

	fmt.Printf("process ms=%.1f\n", time.Since(start).Seconds()*1000)
	return nil
}

func masked(w, h int) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	raster.MaskFillRange(r.Valid, 0, w*h, true)
	return r
}
