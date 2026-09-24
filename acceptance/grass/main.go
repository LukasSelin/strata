// Command grasscheck runs strata's multi-scale derivatives (FitRadius)
// over a raw float32 raster, so that they can be differenced against
// GRASS GIS r.param.scale's.
//
//	go run ./grass -dir out-grass -w 4096 -h 4096 -cell 12.5 -fill 65535 -radii 1,4,8
//
// It reads dem.raw and writes, for each radius r, strata-slope-<r>.raw,
// strata-aspect-<r>.raw and strata-profile-<r>.raw, strata-plan-<r>.raw
// and strata-mean-<r>.raw (the three Curvature types), each as raw
// little-endian float32 with -9999 under invalid cells.
//
// Every operation runs twice: once with the whole raster in memory and
// once streamed through the bounded-memory Chunked path, and the two
// are compared here before either is written out, as gdal/main.go does.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

var (
	dir   = flag.String("dir", "out-grass", "directory holding dem.raw")
	wFlag = flag.Int("w", 4096, "raster width in cells")
	hFlag = flag.Int("h", 4096, "raster height in cells")
	cell  = flag.Float64("cell", 12.5, "cell size in ground units")
	fill  = flag.Float64("fill", 65535, "the NoData value in dem.raw")
	radii = flag.String("radii", "1,4,8", "FitRadius values, comma-separated")
	tile  = flag.Int("tile", 256, "tile height for the Chunked runs")
)

// outFill is what strata writes under invalid cells.
const outFill = -9999

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "grasscheck:", err)
		os.Exit(1)
	}
}

type job struct {
	name    string
	plain   func(dst, dem raster.Float32Raster)
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error
}

func jobs(r int) []job {
	so := terrain.SlopeOptions{CellSize: *cell, CellSizeY: *cell, FitRadius: r}
	ao := terrain.AspectOptions{CellSize: *cell, CellSizeY: *cell, FitRadius: r}
	js := []job{
		{"slope", func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, so) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, dst, src, so, eo)
			}},
		{"aspect", func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, dst, src, ao, eo)
			}},
	}
	for _, c := range []struct {
		name string
		t    terrain.CurvatureType
	}{
		{"profile", terrain.CurvatureProfile},
		{"plan", terrain.CurvaturePlan},
		{"mean", terrain.CurvatureMean},
	} {
		o := terrain.CurvatureOptions{CellSize: *cell, CellSizeY: *cell, FitRadius: r, Type: c.t}
		js = append(js, job{c.name, func(dst, dm raster.Float32Raster) { terrain.Curvature(dst, dm, o) },
			func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.CurvatureChunked(ctx, dst, src, o, eo)
			}})
	}
	return js
}

func run() error {
	w, h := *wFlag, *hFlag
	ctx := context.Background()
	var rs []int
	for _, s := range strings.Split(*radii, ",") {
		r, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return fmt.Errorf("-radii: %w", err)
		}
		rs = append(rs, r)
	}

	inOpts := engine.RawOptions{Fill: float32(*fill), HasFill: true}
	outOpts := engine.RawOptions{Fill: outFill, HasFill: true}

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

	for _, r := range rs {
		for _, j := range jobs(r) {
			name := fmt.Sprintf("%s-%d", j.name, r)
			plain := raster.NewFloat32Like(dem)
			t0 := time.Now()
			j.plain(plain, dem)
			plainMS := time.Since(t0).Seconds() * 1000

			outPath := filepath.Join(*dir, "strata-"+name+".raw")
			out, err := engine.CreateRawFile(outPath, 4*int64(w)*int64(h), 0o644, 0)
			if err != nil {
				return err
			}
			t0 = time.Now()
			err = j.chunked(ctx, engine.NewRawSink(out, w, h, outOpts), engine.NewRawSource(in, w, h, inOpts),
				engine.Options{TileHeight: *tile})
			if err != nil {
				out.Close()
				return err
			}
			chunkedMS := time.Since(t0).Seconds() * 1000
			back := masked(w, h)
			if err := engine.NewRawSource(out, w, h, outOpts).ReadWindow(ctx, back, 0, 0); err != nil {
				out.Close()
				return err
			}
			if err := out.Close(); err != nil {
				return err
			}
			if err := sameBits(plain, back); err != nil {
				return fmt.Errorf("%s: plain and chunked disagree: %w", name, err)
			}
			fmt.Printf("%-10s plain %7.0f ms, chunked %7.0f ms, identical -> %s\n",
				name, plainMS, chunkedMS, filepath.Base(outPath))
		}
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
