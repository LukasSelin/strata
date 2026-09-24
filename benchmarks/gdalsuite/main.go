// Command stratasuite runs one strata operation the way the GDAL suite
// in this directory times it, and reports how long it took. runsuite.sh
// runs it next to the GDAL command that does the same job, in the same
// container, over the same files, under the same timer.
//
//	stratasuite -op slope -mode memory  -src raw -file dem.raw -w 11264 -h 11264 -repeat 5
//	stratasuite -op slope -mode chunked -src raw -file dem.raw -w 11264 -h 11264 -dir /work
//	stratasuite -op slope -mode chunked -src cog -file dem-deflate.tif -dir /work -workers 12
//	stratasuite -list
//
// Modes:
//
//   - memory reads the input into memory first and times only the call:
//     the plain function on one worker, its Tiled form on more. After one
//     untimed run, which touches the destination's pages, it times
//     -repeat runs. This is the compute, without the file.
//   - chunked streams from the input file to strata-<op>.raw in -dir
//     through the op's Chunked form with bounded memory. With -src cog
//     the input is a GeoTIFF read by the cog module: the whole flow a
//     user runs on a real file. Every run opens a fresh source, so a COG
//     run starts with an empty block cache.
//   - none starts the process and stops: the floor under every case.
//
// Every run prints `run=<i> ms=<in-process milliseconds>`; a reduction
// adds its result, which runsuite.sh compares against GDAL's. The output
// file is raw little-endian float32 with -9999 under invalid cells, the
// NoData value gdaldem writes.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/LukasSelin/strata/cog"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

var (
	opName  = flag.String("op", "slope", "the operation; -list prints them")
	mode    = flag.String("mode", "memory", "memory, chunked or none")
	srcKind = flag.String("src", "raw", "input kind: raw (headerless float32) or cog (a GeoTIFF)")
	file    = flag.String("file", "dem.raw", "the input")
	file2   = flag.String("file2", "dem2.raw", "the second input of a two-input operation")
	dir     = flag.String("dir", ".", "where chunked writes strata-<op>.raw")
	wFlag   = flag.Int("w", 0, "raw input width in cells")
	hFlag   = flag.Int("h", 0, "raw input height in cells")
	fill    = flag.Float64("fill", 65535, "the raw input's NoData value")
	cell    = flag.Float64("cell", 12.5, "cell size in ground units, for a raw input")
	tile    = flag.Int("tile", 256, "tile height in rows")
	workers = flag.Int("workers", 1, "engine workers")
	cacheB  = flag.Int64("cache", 0, "cog.SourceOptions.CacheBytes: 0 the default, negative none")
	reuse   = flag.Bool("reuse", false, "overwrite an existing output (engine.ReuseRawFile) instead of truncating it")
	repeat  = flag.Int("repeat", 1, "timed runs in this process")
	list    = flag.Bool("list", false, "print the operations and stop")
	cpuprof = flag.String("cpuprofile", "", "write a CPU profile of the whole process here")
	kernel  = flag.Bool("kernel", false, "print conv5's weights in gdal raster neighbors syntax and stop")
)

// outFill is what strata writes under invalid cells: gdaldem's NoData.
const outFill = -9999

func main() {
	start := time.Now()
	flag.Parse()
	if *cpuprof != "" {
		f, err := os.Create(*cpuprof)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stratasuite:", err)
			os.Exit(1)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintln(os.Stderr, "stratasuite:", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "stratasuite:", err)
		pprof.StopCPUProfile()
		os.Exit(1)
	}
	if !*list && !*kernel {
		fmt.Printf("process ms=%.1f\n", time.Since(start).Seconds()*1000)
	}
}

func run() error {
	switch {
	case *list:
		for _, o := range ops() {
			fmt.Println(o.name)
		}
		return nil
	case *kernel:
		fmt.Println(gdalKernel(conv5, 5))
		return nil
	case *mode == "none":
		return nil
	}
	cellSize = *cell
	o, ok := lookup(*opName)
	if !ok {
		return fmt.Errorf("unknown -op %q (see -list)", *opName)
	}
	files := []string{*file, *file2}[:o.inputs]
	eo := engine.Options{TileHeight: *tile, Workers: *workers}
	ctx := context.Background()

	fmt.Printf("op=%s mode=%s src=%s workers=%d tile=%d gomaxprocs=%d\n",
		o.name, *mode, *srcKind, *workers, *tile, runtime.GOMAXPROCS(0))

	switch *mode {
	case "memory":
		return memory(ctx, o, files, eo)
	case "chunked":
		return chunked(ctx, o, files, eo)
	}
	return fmt.Errorf("unknown -mode %q", *mode)
}

// input is one opened input file.
type input struct {
	src   engine.RasterSource
	grid  raster.Grid
	close func() error
}

func open(name string) (input, error) {
	// One handle per worker: calls on a single *os.File queue behind
	// each other (see engine.RawFile).
	f, err := engine.OpenRawFile(name, os.O_RDONLY, 0, max(*workers, 1))
	if err != nil {
		return input{}, err
	}
	switch *srcKind {
	case "raw":
		w, h := *wFlag, *hFlag
		g := raster.Grid{Width: w, Height: h, ResolutionX: *cell, ResolutionY: -*cell}
		return input{engine.NewRawSource(f, w, h, engine.RawOptions{Fill: float32(*fill), HasFill: true}), g, f.Close}, nil
	case "cog":
		c, err := cog.Open(f)
		if err != nil {
			_ = f.Close()
			return input{}, err
		}
		s, err := c.Source(cog.SourceOptions{CacheBytes: *cacheB})
		if err != nil {
			_ = f.Close()
			return input{}, err
		}
		return input{s, s.Grid(), f.Close}, nil
	}
	_ = f.Close()
	return input{}, fmt.Errorf("unknown -src %q", *srcKind)
}

func openAll(files []string) ([]input, func(), error) {
	var ins []input
	closeAll := func() {
		for _, in := range ins {
			_ = in.close() // read-only
		}
	}
	for _, name := range files {
		in, err := open(name)
		if err != nil {
			closeAll()
			return nil, nil, err
		}
		ins = append(ins, in)
	}
	return ins, closeAll, nil
}

// dstGrid is the destination of an op that changes the resolution by
// scale, over the same extent.
func dstGrid(g raster.Grid, scale float64) raster.Grid {
	if scale == 1 {
		return g
	}
	g.Width = int(math.Round(float64(g.Width) * scale))
	g.Height = int(math.Round(float64(g.Height) * scale))
	g.ResolutionX /= scale
	g.ResolutionY /= scale
	return g
}

func memory(ctx context.Context, o op, files []string, eo engine.Options) error {
	ins, closeAll, err := openAll(files)
	if err != nil {
		return err
	}
	defer closeAll()
	t0 := time.Now()
	src := make([]raster.Dataset, len(ins))
	for i, in := range ins {
		w, h := in.src.Size()
		r := raster.NewFloat32(w, h, make([]float32, w*h))
		if in.src.Masked() {
			r.Valid = raster.NewMask(w * h)
		}
		if err := in.src.ReadWindow(ctx, r, 0, 0); err != nil {
			return err
		}
		src[i] = raster.NewDataset(in.grid, r)
	}
	fmt.Printf("read ms=%.1f\n", time.Since(t0).Seconds()*1000)

	var dst raster.Dataset
	if !o.reduces() {
		g := dstGrid(src[0].Grid, o.scale)
		r := raster.NewFloat32(g.Width, g.Height, make([]float32, g.Width*g.Height))
		r.Valid = raster.NewMask(g.Width * g.Height)
		dst = raster.NewDataset(g, r)
	}
	// One untimed run first: it faults in the destination's pages, which
	// is the allocator's cost and not the operation's.
	for i := -1; i < *repeat; i++ {
		t0 := time.Now()
		res, err := o.tiled(ctx, dst, src, eo)
		if err != nil {
			return err
		}
		if i >= 0 {
			report(i, time.Since(t0), res)
		}
	}
	return nil
}

func chunked(ctx context.Context, o op, files []string, eo engine.Options) error {
	for i := range *repeat {
		if err := chunkedOnce(ctx, o, files, eo, i); err != nil {
			return err
		}
	}
	return nil
}

func chunkedOnce(ctx context.Context, o op, files []string, eo engine.Options, i int) error {
	t0 := time.Now()
	ins, closeAll, err := openAll(files)
	if err != nil {
		return err
	}
	defer closeAll()
	srcs := make([]engine.RasterSource, len(ins))
	grids := make([]raster.Grid, len(ins))
	for k, in := range ins {
		srcs[k], grids[k] = in.src, in.grid
	}
	if o.reduces() {
		res, err := o.chunked(ctx, nil, raster.Grid{}, srcs, grids, eo)
		if err != nil {
			return err
		}
		report(i, time.Since(t0), res)
		return nil
	}
	g := dstGrid(grids[0], o.scale)
	create := engine.CreateRawFile
	if *reuse {
		create = engine.ReuseRawFile
	}
	out, err := create(filepath.Join(*dir, "strata-"+o.name+".raw"),
		4*int64(g.Width)*int64(g.Height), 0o644, max(*workers, 1))
	if err != nil {
		return err
	}
	tCreate := time.Now()
	sink := engine.NewRawSink(out, g.Width, g.Height, engine.RawOptions{Fill: outFill, HasFill: true})
	if _, err := o.chunked(ctx, sink, g, srcs, grids, eo); err != nil {
		_ = out.Close() // the operation's error matters more
		return err
	}
	tRun, mapped := time.Now(), out.Mapped()
	if err := out.Close(); err != nil {
		return err
	}
	// Where the time went: opening the inputs and creating the output
	// (which frees an earlier run's output), the operation, and closing
	// (unmapping) the output.
	fmt.Printf("phases open ms=%.1f run ms=%.1f close ms=%.1f mapped=%v\n",
		tCreate.Sub(t0).Seconds()*1000, tRun.Sub(tCreate).Seconds()*1000, time.Since(tRun).Seconds()*1000, mapped)
	report(i, time.Since(t0), fmt.Sprintf("out=%dx%d", g.Width, g.Height))
	return nil
}

func report(i int, d time.Duration, res string) {
	fmt.Printf("run=%d ms=%.2f %s\n", i, d.Seconds()*1000, res)
}
