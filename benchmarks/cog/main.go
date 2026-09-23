// Command cogbench reads a GeoTIFF through strata's cog package, or runs
// slope over one, and reports how long it took and how many blocks it
// decoded. It is the strata side of the GDAL comparison in this
// directory's RESULTS.md: cogbench.sh runs this binary and GDAL under
// the same timer, in the same container, over the same files.
//
//	cogbench -mode read  -file dem-deflate.tif -tile 256 -workers 12
//	cogbench -mode slope -file dem-deflate.tif -tile 256 -workers 1 -dir /work
//	cogbench -mode slope -src raw -file dem.raw -w 11264 -h 11264 -dir /work
//
// Modes:
//
//   - read reads the whole of band 0, level 0 through Source.ReadWindow in
//     full-width strips of -tile rows, the tiles the engine asks a source
//     for, on -workers goroutines that each reuse one buffer. Nothing is
//     computed or written: it measures the decoder.
//   - slope runs terrain.SlopeChunked from the file to a raw float32 file,
//     with -9999 under invalid cells, as benchmarks/gdal's strataterrain
//     does. -src raw reads a headerless float32 file through
//     engine.RawSource instead, strata's format-free path, so the
//     difference between the two is what reading the format costs.
//   - none starts the process and stops, the floor under every case.
//
// It counts the ReadAt calls the source makes after Open. cog reads each
// block it decodes with exactly one ReadAt when the block is at most
// 1 MiB, as every block of these files is, and makes no other reads
// after Open, so the count is the number of block decodes. reads/block
// divides it by the number of blocks in the file (-block, square): 1.00
// is every block decoded once.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LukasSelin/strata/cog"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

var (
	mode    = flag.String("mode", "read", "read, slope or none")
	srcKind = flag.String("src", "cog", "slope's input: cog (a GeoTIFF) or raw (headerless float32)")
	file    = flag.String("file", "dem.tif", "the input file")
	dir     = flag.String("dir", ".", "where slope writes strata-slope.raw")
	wFlag   = flag.Int("w", 0, "raw input width in cells")
	hFlag   = flag.Int("h", 0, "raw input height in cells")
	fill    = flag.Float64("fill", 65535, "the raw input's NoData value")
	cell    = flag.Float64("cell", 12.5, "cell size in ground units")
	tile    = flag.Int("tile", 256, "tile height in rows")
	workers = flag.Int("workers", 1, "goroutines reading, or engine workers")
	cacheB  = flag.Int64("cache", 0, "SourceOptions.CacheBytes: 0 the default, negative none")
	block   = flag.Int("block", 512, "the file's block size, for reads/block")
	repeat  = flag.Int("repeat", 1, "run this many times in one process")
	cpuprof = flag.String("cpuprofile", "", "write a CPU profile of the timed runs here")
	memprof = flag.String("memprofile", "", "write an allocation profile of the timed runs here")
)

// outFill is what slope writes under invalid cells: gdaldem's NoData.
const outFill = -9999

func main() {
	start := time.Now()
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cogbench:", err)
		os.Exit(1)
	}
	fmt.Printf("process ms=%.1f\n", time.Since(start).Seconds()*1000)
}

// counter counts the ReadAt calls and bytes that pass through it.
type counter struct {
	r            io.ReaderAt
	calls, bytes atomic.Int64
}

func (c *counter) ReadAt(p []byte, off int64) (int, error) {
	c.calls.Add(1)
	c.bytes.Add(int64(len(p)))
	return c.r.ReadAt(p, off)
}

func run() error {
	if *mode == "none" {
		return nil
	}
	if *cpuprof != "" {
		f, err := os.Create(*cpuprof)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	// One handle per worker: calls on a single *os.File queue behind
	// each other (see engine.RawFile).
	in, err := engine.OpenRawFile(*file, os.O_RDONLY, 0, max(*workers, 1))
	if err != nil {
		return err
	}
	defer in.Close()
	cnt := &counter{r: in}

	var src engine.RasterSource
	switch *srcKind {
	case "cog":
		f, err := cog.Open(cnt)
		if err != nil {
			return err
		}
		s, err := f.Source(cog.SourceOptions{CacheBytes: *cacheB})
		if err != nil {
			return err
		}
		src = s
	case "raw":
		src = engine.NewRawSource(cnt, *wFlag, *hFlag, engine.RawOptions{Fill: float32(*fill), HasFill: true})
	default:
		return fmt.Errorf("unknown -src %q", *srcKind)
	}
	w, h := src.Size()
	blocks := int64(ceilDiv(w, *block) * ceilDiv(h, *block))
	fmt.Printf("size=%dx%d masked=%v tile=%d workers=%d cache=%d gomaxprocs=%d\n",
		w, h, src.Masked(), *tile, *workers, *cacheB, runtime.GOMAXPROCS(0))
	// The bound the source chose. An interface, because the build of an
	// older reader that cogbench.sh times has no such method.
	if c, ok := src.(interface{ CacheBytes() int64 }); ok {
		fmt.Printf("cache_bytes=%d\n", c.CacheBytes())
	}

	for i := range *repeat {
		// A fresh source per run, so every run starts with an empty cache.
		if i > 0 && *srcKind == "cog" {
			f, err := cog.Open(cnt)
			if err != nil {
				return err
			}
			if src, err = f.Source(cog.SourceOptions{CacheBytes: *cacheB}); err != nil {
				return err
			}
		}
		calls0, bytes0 := cnt.calls.Load(), cnt.bytes.Load()
		var m0 runtime.MemStats
		runtime.ReadMemStats(&m0)
		stopPeak := peakHeap()
		t0 := time.Now()
		switch *mode {
		case "read":
			err = readAll(src, w, h)
		case "slope":
			err = slope(src, w, h)
		default:
			err = fmt.Errorf("unknown -mode %q", *mode)
		}
		if err != nil {
			return err
		}
		d := time.Since(t0).Seconds()
		peak := stopPeak()
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)
		calls, bytes := cnt.calls.Load()-calls0, cnt.bytes.Load()-bytes0
		fmt.Printf("run=%d ms=%.1f MBs=%.0f reads=%d readMB=%.1f reads_per_block=%.2f"+
			" allocMB=%.1f allocs=%d gcs=%d gcPauseMs=%.2f peakHeapMB=%.1f\n",
			i, d*1000, float64(w)*float64(h)*4/d/1e6, calls, float64(bytes)/1e6,
			float64(calls)/float64(blocks),
			float64(m1.TotalAlloc-m0.TotalAlloc)/1e6, m1.Mallocs-m0.Mallocs, m1.NumGC-m0.NumGC,
			float64(m1.PauseTotalNs-m0.PauseTotalNs)/1e6, float64(peak)/1e6)
	}
	if *memprof != "" {
		f, err := os.Create(*memprof)
		if err != nil {
			return err
		}
		defer f.Close()
		// Every allocation since the process started, most of them the
		// timed runs': pprof -sample_index=alloc_space shows where.
		if err := pprof.Lookup("allocs").WriteTo(f, 0); err != nil {
			return err
		}
	}
	return nil
}

// peakHeap samples the heap's objects, live and not yet swept, every
// millisecond until the function it returns is called, which returns
// the largest it saw. runtime/metrics reads it without stopping the
// world, as ReadMemStats would.
func peakHeap() func() uint64 {
	done := make(chan struct{})
	res := make(chan uint64)
	go func() {
		var peak uint64
		sample := []metrics.Sample{{Name: "/memory/classes/heap/objects:bytes"}}
		t := time.NewTicker(time.Millisecond)
		defer t.Stop()
		for {
			metrics.Read(sample)
			peak = max(peak, sample[0].Value.Uint64())
			select {
			case <-done:
				res <- peak
				return
			case <-t.C:
			}
		}
	}()
	return func() uint64 { close(done); return <-res }
}

// readAll reads the whole raster in full-width strips of -tile rows, on
// -workers goroutines, each with one buffer it reuses.
func readAll(src engine.RasterSource, w, h int) error {
	ctx := context.Background()
	strips := ceilDiv(h, *tile)
	var next atomic.Int64
	errs := make([]error, *workers)
	var wg sync.WaitGroup
	for k := range *workers {
		wg.Go(func() {
			buf := make([]float32, w**tile)
			var valid []uint64
			if src.Masked() {
				valid = raster.NewMask(w * *tile)
			}
			for {
				s := int(next.Add(1) - 1)
				if s >= strips {
					return
				}
				y := s * *tile
				rows := min(*tile, h-y)
				dst := raster.NewFloat32(w, rows, buf[:w*rows])
				if valid != nil {
					dst.Valid = valid[:raster.MaskWords(w*rows)]
				}
				if err := src.ReadWindow(ctx, dst, 0, y); err != nil {
					errs[k] = err
					return
				}
			}
		})
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// slope runs terrain.SlopeChunked from src to a raw float32 file.
func slope(src engine.RasterSource, w, h int) error {
	out, err := engine.OpenRawFile(filepath.Join(*dir, "strata-slope.raw"),
		os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, max(*workers, 1))
	if err != nil {
		return err
	}
	sink := engine.NewRawSink(out, w, h, engine.RawOptions{Fill: outFill, HasFill: true})
	so := terrain.SlopeOptions{CellSize: *cell, CellSizeY: *cell}
	eo := engine.Options{TileHeight: *tile, Workers: *workers}
	if err := terrain.SlopeChunked(context.Background(), sink, src, so, eo); err != nil {
		_ = out.Close() // the operation's error matters more
		return err
	}
	return out.Close()
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }
