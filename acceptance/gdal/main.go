// Command gdalcheck runs strata over a raw float32 raster that GDAL
// produced, so that its results can be differenced against gdaldem's.
//
//	go run ./gdal -dir out-gdal -w 4096 -h 4096 -cell 12.5 -fill 65535
//
// It reads dem.raw and writes strata-slope.raw, strata-aspect.raw,
// strata-hillshade.raw and, for Ruggedness, strata-tri.raw,
// strata-triwilson.raw, strata-tpi.raw and strata-roughness.raw, each as
// raw little-endian float32 with -9999 under invalid cells, which is the
// NoData value gdaldem writes.
//
// Every operation runs twice: once with the whole raster in memory and
// once streamed through the bounded-memory Chunked path, and the two
// files are compared here before either is written out, so a difference
// between them fails the run rather than being averaged away.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

var (
	dir      = flag.String("dir", "out-gdal", "directory holding dem.raw")
	wFlag    = flag.Int("w", 4096, "raster width in cells")
	hFlag    = flag.Int("h", 4096, "raster height in cells")
	cell     = flag.Float64("cell", 12.5, "cell size in ground units")
	fill     = flag.Float64("fill", 65535, "the NoData value in dem.raw")
	azimuth  = flag.Float64("azimuth", 315, "hillshade azimuth")
	altitude = flag.Float64("altitude", 45, "hillshade altitude")
	tile     = flag.Int("tile", 256, "tile height for the Chunked runs")
)

// outFill is what strata writes under invalid cells, chosen to match the
// NoData value gdaldem writes into its own results.
const outFill = -9999

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "gdalcheck:", err)
		os.Exit(1)
	}
}

type job struct {
	name    string
	plain   func(dst, dem raster.Float32Raster)
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error
}

func run() error {
	w, h := *wFlag, *hFlag
	ctx := context.Background()

	so := terrain.SlopeOptions{CellSize: *cell, CellSizeY: *cell}
	ao := terrain.AspectOptions{CellSize: *cell, CellSizeY: *cell}
	ho := terrain.HillshadeOptions{
		CellSize: *cell, CellSizeY: *cell,
		Azimuth: *azimuth, Altitude: *altitude,
	}
	jobs := []job{
		{"slope", func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, so) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, dst, src, so, eo)
			}},
		{"aspect", func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, dst, src, ao, eo)
			}},
		{"hillshade", func(dst, dm raster.Float32Raster) { terrain.Hillshade(dst, dm, ho) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.HillshadeChunked(ctx, dst, src, ho, eo)
			}},
	}
	for _, r := range []struct {
		name string
		t    terrain.RuggednessType
	}{
		{"tri", terrain.RuggednessTRI},
		{"triwilson", terrain.RuggednessTRIWilson},
		{"tpi", terrain.RuggednessTPI},
		{"roughness", terrain.RuggednessRoughness},
	} {
		o := terrain.RuggednessOptions{Type: r.t}
		jobs = append(jobs, job{r.name, func(dst, dm raster.Float32Raster) { terrain.Ruggedness(dst, dm, o) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.RuggednessChunked(ctx, dst, src, o, eo)
			}})
	}

	inOpts := engine.RawOptions{Fill: float32(*fill), HasFill: true}
	outOpts := engine.RawOptions{Fill: outFill, HasFill: true}

	// The whole DEM in memory, for the plain runs.
	demPath := filepath.Join(*dir, "dem.raw")
	in, err := engine.OpenRawFile(demPath, os.O_RDONLY, 0, 0)
	if err != nil {
		return err
	}
	defer in.Close()
	dem := masked(w, h)
	if err := engine.NewRawSource(in, w, h, inOpts).ReadWindow(ctx, dem, 0, 0); err != nil {
		return err
	}
	fmt.Printf("read %s (%d x %d, %d valid cells)\n", demPath, w, h, countValid(dem))

	for _, j := range jobs {
		// Plain: whole raster in memory.
		plain := raster.NewFloat32Like(dem)
		t0 := time.Now()
		j.plain(plain, dem)
		plainMS := time.Since(t0).Seconds() * 1000

		// Chunked: streamed from the file with bounded memory.
		outPath := filepath.Join(*dir, "strata-"+j.name+".raw")
		out, err := engine.OpenRawFile(outPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, 0)
		if err != nil {
			return err
		}
		src := engine.NewRawSource(in, w, h, inOpts)
		sink := engine.NewRawSink(out, w, h, outOpts)
		t0 = time.Now()
		if err := j.chunked(ctx, sink, src, engine.Options{TileHeight: *tile}); err != nil {
			out.Close()
			return err
		}
		chunkedMS := time.Since(t0).Seconds() * 1000
		if err := out.Sync(); err != nil {
			out.Close()
			return err
		}

		// The two must agree before anything is compared with gdaldem.
		back := masked(w, h)
		if err := engine.NewRawSource(out, w, h, outOpts).ReadWindow(ctx, back, 0, 0); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
		if err := sameBits(plain, back); err != nil {
			return fmt.Errorf("%s: plain and chunked disagree: %w", j.name, err)
		}
		fmt.Printf("%-10s plain %7.0f ms, chunked %7.0f ms, identical -> %s\n",
			j.name, plainMS, chunkedMS, filepath.Base(outPath))
	}
	return nil
}

func masked(w, h int) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	raster.MaskFillRange(r.Valid, 0, w*h, true)
	return r
}

func countValid(r raster.Float32Raster) int {
	n := 0
	for y := range r.Height {
		for x := range r.Width {
			if r.IsValid(x, y) {
				n++
			}
		}
	}
	return n
}

// sameBits compares two results cell by cell, over the cells that carry
// data. NaN matches NaN; data under an invalid cell is unspecified and
// is skipped, but the validity itself must match.
func sameBits(a, b raster.Float32Raster) error {
	for y := range a.Height {
		ra, rb := a.Row(y), b.Row(y)
		for x := range a.Width {
			va, vb := a.IsValid(x, y), b.IsValid(x, y)
			if va != vb {
				return fmt.Errorf("validity differs at (%d, %d)", x, y)
			}
			if !va {
				continue
			}
			fa, fb := ra[x], rb[x]
			if math.Float32bits(fa) == math.Float32bits(fb) {
				continue
			}
			if math.IsNaN(float64(fa)) && math.IsNaN(float64(fb)) {
				continue
			}
			return fmt.Errorf("at (%d, %d): %v vs %v", x, y, fa, fb)
		}
	}
	return nil
}
