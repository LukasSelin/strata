// Command strata-acceptance drives strata through its public API only and
// writes every input and output to plain files, so that an independent
// program can judge the results without reading strata's source.
//
//	go run .              # in this directory
//	go run . -dir out     # somewhere else
//
// It builds three small DEMs, runs the terrain, focal, algebra and
// reduce operations on each in all three forms (plain, Tiled, Chunked),
// resamples windows of two of them (resample.go, judged by
// check_resample.py and gdalwarp_resample.py), puts N-dimensional arrays
// through broadcasting arithmetic and axis reductions (array.go, judged
// by check_array.py), and writes:
//
//	<dem>.f32              the elevation, raw little-endian float32
//	<dem>.mask.u8          1 per valid cell, 0 per NoData cell (masked DEMs)
//	<dem>-<op>-<form>.f32  one result
//	manifest.json          what every file is, and the scalar results
//
// Nothing here checks anything. check.py does the judging.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/terrain"
)

// Deliberately not square, not a power of two, and not a multiple of the
// tile sizes below: tiles and worker bands then land on ragged edges.
const (
	width  = 257
	height = 193
)

// The NoData value written into the raw files under invalid cells.
const fill = -9999

// Tiled and Chunked runs use these, so that every result is produced by
// more than one tiling and worker count.
var (
	tiledOpts   = engine.Options{TileWidth: 37, TileHeight: 23, Workers: 3}
	chunkedOpts = engine.Options{TileHeight: 16, Workers: 4}
)

type rasterCase struct {
	Name      string  `json:"name"`
	Surface   string  `json:"surface"`
	Op        string  `json:"op"`
	Form      string  `json:"form"`
	DEM       string  `json:"dem"`
	DEMMask   string  `json:"dem_mask,omitempty"`
	Out       string  `json:"out"`
	OutMask   string  `json:"out_mask,omitempty"`
	CellSize  float64 `json:"cell_size"`
	CellSizeY float64 `json:"cell_size_y"`
	Azimuth   float64 `json:"azimuth,omitempty"`
	Altitude  float64 `json:"altitude,omitempty"`
	// The focal operations' parameters: the radius, and the weights or
	// taps exactly as passed (float32 values, which JSON carries
	// exactly).
	Radius  int       `json:"radius,omitempty"`
	Fit     int       `json:"fit,omitempty"` // a fitted derivative's FitRadius
	Weights []float32 `json:"weights,omitempty"`
	Row     []float32 `json:"row,omitempty"`
	Col     []float32 `json:"col,omitempty"`
	// A two-input terrain operation's second input, and its mask.
	Weight     string `json:"weight,omitempty"`
	WeightMask string `json:"weight_mask,omitempty"`
}

type scalarCase struct {
	Name    string `json:"name"`
	Op      string `json:"op"`
	Form    string `json:"form"`
	DEM     string `json:"dem"`
	DEMMask string `json:"dem_mask,omitempty"`
	Min     string `json:"min,omitempty"`
	Max     string `json:"max,omitempty"`
	Count   int64  `json:"count"`
}

type manifest struct {
	Width   int          `json:"width"`
	Height  int          `json:"height"`
	Fill    float64      `json:"fill"`
	Rasters []rasterCase `json:"rasters"`
	Scalars []scalarCase `json:"scalars"`
}

var dir = flag.String("dir", "out", "directory for the generated files")

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "strata-acceptance:", err)
		os.Exit(1)
	}
}

type dem struct {
	name    string
	surface string
	r       raster.Float32Raster
	cellX   float64
	cellY   float64
}

func run() error {
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		return err
	}
	m := manifest{Width: width, Height: height, Fill: fill}

	dems := []dem{
		// A tilted plane: every interior cell has the same slope and
		// aspect, and both follow from the corner elevations alone.
		{name: "plane", surface: "plane", r: plane(0.3, -0.7), cellX: 10, cellY: 10},
		// A smooth hill on rectangular cells, so that the two cell
		// sizes have to be used separately, and every aspect occurs.
		{name: "hill", surface: "hill", r: hill(), cellX: 10, cellY: 25},
		// Noise over the hill, with two NoData regions, so validity
		// has to propagate through a 3x3 neighbourhood.
		{name: "noisy", surface: "noisy", r: noisy(), cellX: 30, cellY: 30},
	}

	for _, d := range dems {
		if err := writeRaster(filepath.Join(*dir, d.name+".f32"), d.r); err != nil {
			return err
		}
		if d.r.Valid != nil {
			if err := writeMask(filepath.Join(*dir, d.name+".mask.u8"), d.r); err != nil {
				return err
			}
		}
		if err := d.runTerrain(&m); err != nil {
			return err
		}
		if err := d.runReduce(&m); err != nil {
			return err
		}
	}
	if err := runAlgebra(&m, dems[0], dems[1]); err != nil {
		return err
	}
	if err := runWeighted(&m, dems[2]); err != nil {
		return err
	}
	if err := runResample(); err != nil {
		return err
	}
	if err := runArray(); err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(*dir, "manifest.json"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("wrote %d rasters and %d scalars to %s\n", len(m.Rasters), len(m.Scalars), *dir)
	return nil
}

// op is one operation in its three forms, with its options already bound.
type op struct {
	name     string
	azimuth  float64
	altitude float64
	radius   int
	fit      int
	weights  []float32
	row, col []float32
	plain    func(dst, dem raster.Float32Raster)
	tiled    func(ctx context.Context, dst, dem raster.Float32Raster, eo engine.Options) error
	chunked  func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error
}

func (d dem) ops() []op {
	ao := terrain.AspectOptions{CellSize: d.cellX, CellSizeY: d.cellY}
	gopts := terrain.GradientOptions{CellSize: d.cellX, CellSizeY: d.cellY}
	ho := terrain.HillshadeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Azimuth: 315, Altitude: 45}

	slope := func(u terrain.SlopeUnits) op {
		o := terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: u}
		return op{
			name:  "slope_" + unitName(u),
			plain: func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, o) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.SlopeTiled(ctx, dst, dm, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, dst, src, o, eo)
			},
		}
	}
	// Gradient writes two rasters; each component is exported as its own
	// case, with the other one thrown away.
	grad := func(wantDX bool) op {
		name := "gradient_dy"
		if wantDX {
			name = "gradient_dx"
		}
		pick := func(dst, other raster.Float32Raster) (dx, dy raster.Float32Raster) {
			if wantDX {
				return dst, other
			}
			return other, dst
		}
		return op{
			name: name,
			plain: func(dst, dm raster.Float32Raster) {
				dx, dy := pick(dst, raster.NewFloat32Like(dm))
				terrain.Gradient(dx, dy, dm, gopts)
			},
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				dx, dy := pick(dst, raster.NewFloat32Like(dm))
				return terrain.GradientTiled(ctx, dx, dy, dm, gopts, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				w, h := dst.Size()
				spare := engine.NewMemorySink(maskedLike(dst.Masked(), w, h))
				if wantDX {
					return terrain.GradientChunked(ctx, dst, spare, src, gopts, eo)
				}
				return terrain.GradientChunked(ctx, spare, dst, src, gopts, eo)
			},
		}
	}

	curvature := func(ct terrain.CurvatureType, name string) op {
		o := terrain.CurvatureOptions{CellSize: d.cellX, CellSizeY: d.cellY, Type: ct}
		return op{
			name:  "curvature_" + name,
			plain: func(dst, dm raster.Float32Raster) { terrain.Curvature(dst, dm, o) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.CurvatureTiled(ctx, dst, dm, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.CurvatureChunked(ctx, dst, src, o, eo)
			},
		}
	}

	// ruggedness is one measure over the (2r+1)² window. The 3×3 cases
	// keep their names, which gdalcompare.py matches to gdaldem's; larger
	// windows get an _r suffix and a radius in the manifest.
	ruggedness := func(rt terrain.RuggednessType, name string, r int) op {
		o := terrain.RuggednessOptions{Type: rt, Radius: r}
		name = "ruggedness_" + name
		if r > 1 {
			name += fmt.Sprintf("_r%d", r)
		} else {
			r = 0
		}
		return op{
			name:   name,
			radius: r,
			plain:  func(dst, dm raster.Float32Raster) { terrain.Ruggedness(dst, dm, o) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.RuggednessTiled(ctx, dst, dm, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.RuggednessChunked(ctx, dst, src, o, eo)
			},
		}
	}

	// surface runs terrain.Surface for all five products at once and
	// emits one: check 12 requires each to be the standalone product's
	// file bit for bit, so the multi-output pipeline is judged against
	// results check 1 and gdaldem judge on their own.
	so := terrain.SurfaceOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: terrain.SlopeDegrees,
		Azimuth: ho.Azimuth, Altitude: ho.Altitude}
	surface := func(name string, which int) op {
		outs := func(dst, dm raster.Float32Raster) (terrain.SurfaceOutputs, []raster.Float32Raster) {
			all := make([]raster.Float32Raster, 5)
			for k := range all {
				all[k] = raster.NewFloat32Like(dm)
			}
			all[which] = dst
			return terrain.SurfaceOutputs{Dx: all[0], Dy: all[1], Slope: all[2], Aspect: all[3], Hillshade: all[4]}, all
		}
		return op{
			name: "surface_" + name,
			plain: func(dst, dm raster.Float32Raster) {
				o, _ := outs(dst, dm)
				terrain.Surface(o, dm, so)
			},
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				o, _ := outs(dst, dm)
				return terrain.SurfaceTiled(ctx, o, dm, so, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				w, h := src.Size()
				sinks := make([]engine.RasterSink, 5)
				for k := range sinks {
					sinks[k] = engine.NewMemorySink(maskedLike(src.Masked(), w, h))
				}
				sinks[which] = dst
				return terrain.SurfaceChunked(ctx, terrain.SurfaceSinks{
					Dx: sinks[0], Dy: sinks[1], Slope: sinks[2], Aspect: sinks[3], Hillshade: sinks[4],
				}, src, so, eo)
			},
		}
	}

	// features runs terrain.Features for a stack of 3×3 derivatives and
	// ruggedness at three radii, and emits one output: check 13 requires
	// each to be the standalone case's file bit for bit, as check 12 does
	// for Surface. The largest radius, 8, sets the pipeline's window,
	// so the 3×3 outputs are the ones whose edge ring and validity
	// depend on the engine keeping each output's own.
	stack := []struct {
		alone  string
		radius int
		op     terrain.FeatureOp
	}{
		{"slope_deg", 0, terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: terrain.SlopeDegrees}},
		{"curvature_plan", 0, terrain.CurvatureOptions{CellSize: d.cellX, CellSizeY: d.cellY, Type: terrain.CurvaturePlan}},
		{"ruggedness_tpi", 0, terrain.RuggednessOptions{Type: terrain.RuggednessTPI}},
		{"ruggedness_tpi_r3", 3, terrain.RuggednessOptions{Type: terrain.RuggednessTPI, Radius: 3}},
		{"ruggedness_tpi_r8", 8, terrain.RuggednessOptions{Type: terrain.RuggednessTPI, Radius: 8}},
		{"ruggedness_tri_r3", 3, terrain.RuggednessOptions{Type: terrain.RuggednessTRI, Radius: 3}},
		{"ruggedness_roughness_r8", 8, terrain.RuggednessOptions{Type: terrain.RuggednessRoughness, Radius: 8}},
		{"slope_deg_fit4", 4, terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: terrain.SlopeDegrees, FitRadius: 4}},
		{"curvature_mean_fit1", 1, terrain.CurvatureOptions{CellSize: d.cellX, CellSizeY: d.cellY, Type: terrain.CurvatureMean, FitRadius: 1}},
	}
	features := func(which int) op {
		outs := func(dst, dm raster.Float32Raster) []terrain.Feature {
			out := make([]terrain.Feature, len(stack))
			for k, s := range stack {
				out[k] = terrain.Feature{Op: s.op, Dst: raster.NewFloat32Like(dm)}
			}
			out[which].Dst = dst
			return out
		}
		return op{
			name:   "features_" + stack[which].alone,
			radius: stack[which].radius,
			plain:  func(dst, dm raster.Float32Raster) { terrain.Features(outs(dst, dm), dm) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.FeaturesTiled(ctx, outs(dst, dm), dm, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				w, h := src.Size()
				sinks := make([]terrain.FeatureSink, len(stack))
				for k, s := range stack {
					sinks[k] = terrain.FeatureSink{Op: s.op, Dst: engine.NewMemorySink(maskedLike(src.Masked(), w, h))}
				}
				sinks[which].Dst = dst
				return terrain.FeaturesChunked(ctx, sinks, src, eo)
			},
		}
	}

	ops := []op{
		surface("gradient_dx", 0),
		surface("gradient_dy", 1),
		surface("slope_deg", 2),
		surface("aspect", 3),
		surface("hillshade", 4),
		slope(terrain.SlopeDegrees),
		slope(terrain.SlopeRadians),
		slope(terrain.SlopePercent),
		grad(true),
		grad(false),
		{
			name:  "aspect",
			plain: func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.AspectTiled(ctx, dst, dm, ao, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, dst, src, ao, eo)
			},
		},
		{
			// Not a terrain operation, but it needs a whole DEM and all
			// three forms, which is what this list runs.
			name:  "normalize",
			plain: func(dst, dm raster.Float32Raster) { algebra.Normalize(dst, dm) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				_, _, err := algebra.NormalizeTiled(ctx, dst, dm, eo)
				return err
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				_, _, err := algebra.NormalizeChunked(ctx, dst, src, eo)
				return err
			},
		},
		{
			name:     "hillshade",
			azimuth:  ho.Azimuth,
			altitude: ho.Altitude,
			plain:    func(dst, dm raster.Float32Raster) { terrain.Hillshade(dst, dm, ho) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.HillshadeTiled(ctx, dst, dm, ho, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.HillshadeChunked(ctx, dst, src, ho, eo)
			},
		},
		curvature(terrain.CurvatureProfile, "profile"),
		curvature(terrain.CurvaturePlan, "plan"),
		curvature(terrain.CurvatureMean, "mean"),
	}
	for _, r := range []int{1, 3, terrain.MaxRadius} {
		ops = append(ops,
			ruggedness(terrain.RuggednessTRI, "tri", r),
			ruggedness(terrain.RuggednessTRIWilson, "triwilson", r),
			ruggedness(terrain.RuggednessTPI, "tpi", r),
			ruggedness(terrain.RuggednessRoughness, "roughness", r),
		)
	}
	for _, r := range []int{1, 4} {
		ops = append(ops, d.fitted(r, ao, ho)...)
	}
	for k := range stack {
		ops = append(ops, features(k))
	}
	return ops
}

// fitted is slope, aspect, hillshade and the three curvatures from
// Wood's quadratic fitted over the (2r+1)² window, as the standalone
// functions compute them with FitRadius r. check.py judges them against
// its own least-squares solution.
func (d dem) fitted(r int, ao terrain.AspectOptions, ho terrain.HillshadeOptions) []op {
	sfx := fmt.Sprintf("_fit%d", r)
	so := terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: terrain.SlopeDegrees, FitRadius: r}
	ao.FitRadius, ho.FitRadius = r, r
	ops := []op{
		{
			name:  "slope_deg" + sfx,
			plain: func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, so) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.SlopeTiled(ctx, dst, dm, so, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, dst, src, so, eo)
			},
		},
		{
			name:  "aspect" + sfx,
			plain: func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.AspectTiled(ctx, dst, dm, ao, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, dst, src, ao, eo)
			},
		},
		{
			name:     "hillshade" + sfx,
			azimuth:  ho.Azimuth,
			altitude: ho.Altitude,
			plain:    func(dst, dm raster.Float32Raster) { terrain.Hillshade(dst, dm, ho) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.HillshadeTiled(ctx, dst, dm, ho, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.HillshadeChunked(ctx, dst, src, ho, eo)
			},
		},
	}
	for _, c := range []struct {
		t    terrain.CurvatureType
		name string
	}{{terrain.CurvatureProfile, "profile"}, {terrain.CurvaturePlan, "plan"}, {terrain.CurvatureMean, "mean"}} {
		co := terrain.CurvatureOptions{CellSize: d.cellX, CellSizeY: d.cellY, Type: c.t, FitRadius: r}
		ops = append(ops, op{
			name:  "curvature_" + c.name + sfx,
			plain: func(dst, dm raster.Float32Raster) { terrain.Curvature(dst, dm, co) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.CurvatureTiled(ctx, dst, dm, co, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.CurvatureChunked(ctx, dst, src, co, eo)
			},
		})
	}
	for i := range ops {
		ops[i].radius, ops[i].fit = r, r
	}
	return ops
}

// focalOps are the focal operations, with weights and taps that are
// asymmetric in both axes, so that a kernel applied flipped, transposed
// or with its passes swapped does not give the same result.
func focalOps() []op {
	w5 := make([]float32, 25)
	for i := range w5 {
		w5[i] = float32((i*7)%11-4) / 8
	}
	wo := focal.WeightsOptions{Radius: 2, Weights: w5}
	sep := focal.SeparableOptions{Radius: 2, Row: []float32{-0.5, 0.25, 1, 2, 0.75}, Col: []float32{1.5, -1, 0.5, 0.125, 3}}
	g := focal.Gaussian(3, 1.5)
	gauss := focal.SeparableOptions{Radius: 3, Row: g, Col: g}

	weighted := func(name string, rotate bool) op {
		plain, tiled, chunked := focal.Correlate, focal.CorrelateTiled, focal.CorrelateChunked
		if rotate {
			plain, tiled, chunked = focal.Convolve, focal.ConvolveTiled, focal.ConvolveChunked
		}
		return op{
			name: name, radius: 2, weights: w5,
			plain: func(dst, src raster.Float32Raster) { plain(dst, src, wo) },
			tiled: func(ctx context.Context, dst, src raster.Float32Raster, eo engine.Options) error {
				return tiled(ctx, dst, src, wo, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return chunked(ctx, dst, src, wo, eo)
			},
		}
	}
	separable := func(name string, o focal.SeparableOptions) op {
		return op{
			name: name, radius: o.Radius, row: o.Row, col: o.Col,
			plain: func(dst, src raster.Float32Raster) { focal.CorrelateSeparable(dst, src, o) },
			tiled: func(ctx context.Context, dst, src raster.Float32Raster, eo engine.Options) error {
				return focal.CorrelateSeparableTiled(ctx, dst, src, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return focal.CorrelateSeparableChunked(ctx, dst, src, o, eo)
			},
		}
	}
	type boxFuncs struct {
		plain   func(dst, src raster.Float32Raster, o focal.BoxOptions)
		tiled   func(ctx context.Context, dst, src raster.Float32Raster, o focal.BoxOptions, eo engine.Options) error
		chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o focal.BoxOptions, eo engine.Options) error
	}
	box := func(kind string, r int, f boxFuncs) op {
		o := focal.BoxOptions{Radius: r}
		return op{
			name: fmt.Sprintf("focal_%s_r%d", kind, r), radius: r,
			plain: func(dst, src raster.Float32Raster) { f.plain(dst, src, o) },
			tiled: func(ctx context.Context, dst, src raster.Float32Raster, eo engine.Options) error {
				return f.tiled(ctx, dst, src, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return f.chunked(ctx, dst, src, o, eo)
			},
		}
	}
	mean := boxFuncs{focal.Mean, focal.MeanTiled, focal.MeanChunked}
	mn := boxFuncs{focal.Min, focal.MinTiled, focal.MinChunked}
	mx := boxFuncs{focal.Max, focal.MaxTiled, focal.MaxChunked}
	return []op{
		weighted("focal_correlate_r2", false),
		weighted("focal_convolve_r2", true),
		separable("focal_separable_r2", sep),
		separable("focal_gaussian_r3", gauss),
		box("mean", 2, mean),
		box("min", 1, mn),
		box("max", 1, mx),
		box("min", 3, mn),
		box("max", 3, mx),
	}
}

func (d dem) runTerrain(m *manifest) error {
	ctx := context.Background()
	for _, o := range append(d.ops(), focalOps()...) {
		// Plain, in memory.
		plain := raster.NewFloat32Like(d.r)
		o.plain(plain, d.r)
		if err := d.emit(m, o, "plain", plain); err != nil {
			return err
		}

		// Tiled, in memory, on several workers with ragged tiles.
		tiled := raster.NewFloat32Like(d.r)
		if err := o.tiled(ctx, tiled, d.r, tiledOpts); err != nil {
			return err
		}
		if err := d.emit(m, o, "tiled", tiled); err != nil {
			return err
		}

		// Chunked, reading and writing raw float32 files.
		chunked, err := d.runChunked(ctx, o)
		if err != nil {
			return err
		}
		if err := d.emit(m, o, "chunked", chunked); err != nil {
			return err
		}
	}
	return nil
}

// runChunked runs o's Chunked form from the DEM file on disk into a
// temporary raw file, and reads the result back.
func (d dem) runChunked(ctx context.Context, o op) (raster.Float32Raster, error) {
	var zero raster.Float32Raster
	ro := engine.RawOptions{Fill: fill, HasFill: d.r.Valid != nil}

	in, err := engine.OpenRawFile(filepath.Join(*dir, d.name+".f32"), os.O_RDONLY, 0, 4)
	if err != nil {
		return zero, err
	}
	defer in.Close()

	tmp := filepath.Join(*dir, d.name+"-"+o.name+".chunked.tmp")
	out, err := engine.CreateRawFile(tmp, 4*int64(width)*int64(height), 0o644, 4)
	if err != nil {
		return zero, err
	}
	src := engine.NewRawSource(in, width, height, ro)
	sink := engine.NewRawSink(out, width, height, ro)
	if err := o.chunked(ctx, sink, src, chunkedOpts); err != nil {
		out.Close()
		return zero, err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return zero, err
	}
	// Read it back through the same file interface, one whole-raster
	// window, so what lands in the manifest is what is on disk.
	back := raster.NewFloat32Like(d.r)
	if err := engine.NewRawSource(out, width, height, ro).ReadWindow(ctx, back, 0, 0); err != nil {
		out.Close()
		return zero, err
	}
	if err := out.Close(); err != nil {
		return zero, err
	}
	if err := os.Remove(tmp); err != nil {
		return zero, err
	}
	return back, nil
}

func (d dem) emit(m *manifest, o op, form string, r raster.Float32Raster) error {
	base := d.name + "-" + o.name + "-" + form
	c := rasterCase{
		Name:      base,
		Surface:   d.surface,
		Op:        o.name,
		Form:      form,
		DEM:       d.name + ".f32",
		Out:       base + ".f32",
		CellSize:  d.cellX,
		CellSizeY: d.cellY,
		Azimuth:   o.azimuth,
		Altitude:  o.altitude,
		Radius:    o.radius,
		Fit:       o.fit,
		Weights:   o.weights,
		Row:       o.row,
		Col:       o.col,
	}
	if d.r.Valid != nil {
		c.DEMMask = d.name + ".mask.u8"
		c.OutMask = base + ".mask.u8"
		if err := writeMask(filepath.Join(*dir, c.OutMask), r); err != nil {
			return err
		}
	}
	if err := writeRaster(filepath.Join(*dir, c.Out), r); err != nil {
		return err
	}
	m.Rasters = append(m.Rasters, c)
	return nil
}

func (d dem) runReduce(m *manifest) error {
	ctx := context.Background()
	add := func(o, form string, mn, mx float32, n int64, hasMinMax bool) {
		s := scalarCase{Name: d.name + "-" + o + "-" + form, Op: o, Form: form, DEM: d.name + ".f32", Count: n}
		if d.r.Valid != nil {
			s.DEMMask = d.name + ".mask.u8"
		}
		if hasMinMax {
			s.Min = f32(mn)
			s.Max = f32(mx)
		}
		m.Scalars = append(m.Scalars, s)
	}

	add("count", "plain", 0, 0, reduce.Count(d.r), false)
	n, err := reduce.CountTiled(ctx, d.r, tiledOpts)
	if err != nil {
		return err
	}
	add("count", "tiled", 0, 0, n, false)

	mn, mx, n := reduce.MinMax(d.r)
	add("minmax", "plain", mn, mx, n, true)
	mn, mx, n, err = reduce.MinMaxTiled(ctx, d.r, tiledOpts)
	if err != nil {
		return err
	}
	add("minmax", "tiled", mn, mx, n, true)

	ro := engine.RawOptions{Fill: fill, HasFill: d.r.Valid != nil}
	in, err := engine.OpenRawFile(filepath.Join(*dir, d.name+".f32"), os.O_RDONLY, 0, 4)
	if err != nil {
		return err
	}
	defer in.Close()
	n, err = reduce.CountChunked(ctx, engine.NewRawSource(in, width, height, ro), chunkedOpts)
	if err != nil {
		return err
	}
	add("count", "chunked", 0, 0, n, false)
	mn, mx, n, err = reduce.MinMaxChunked(ctx, engine.NewRawSource(in, width, height, ro), chunkedOpts)
	if err != nil {
		return err
	}
	add("minmax", "chunked", mn, mx, n, true)
	return nil
}

// runAlgebra exports the pointwise operations on two of the DEMs.
func runAlgebra(m *manifest, a, b dem) error {
	ctx := context.Background()
	cases := []struct {
		name  string
		plain func(dst raster.Float32Raster)
		tiled func(ctx context.Context, dst raster.Float32Raster) error
	}{
		{"add", func(dst raster.Float32Raster) { algebra.Add(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.AddTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"sub", func(dst raster.Float32Raster) { algebra.Sub(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.SubTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"mul", func(dst raster.Float32Raster) { algebra.Mul(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MulTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"min", func(dst raster.Float32Raster) { algebra.Min(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MinTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"max", func(dst raster.Float32Raster) { algebra.Max(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MaxTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"clamp", func(dst raster.Float32Raster) { algebra.Clamp(dst, b.r, 50, 300) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.ClampTiled(ctx, dst, b.r, 50, 300, tiledOpts)
			}},
	}
	for _, c := range cases {
		for _, form := range []string{"plain", "tiled"} {
			dst := raster.NewFloat32Like(a.r)
			if form == "plain" {
				c.plain(dst)
			} else if err := c.tiled(ctx, dst); err != nil {
				return err
			}
			base := "algebra-" + c.name + "-" + form
			if err := writeRaster(filepath.Join(*dir, base+".f32"), dst); err != nil {
				return err
			}
			m.Rasters = append(m.Rasters, rasterCase{
				Name: base, Surface: "algebra", Op: "algebra_" + c.name, Form: form,
				DEM: a.name + ".f32", Out: base + ".f32",
			})
		}
	}
	return nil
}

// runWeighted exports terrain.WeightedSlope of d, in degrees, times a
// weight raster with a mask of its own, in all three forms. The weight's
// NoData is a horizontal stripe and scattered single cells, so that the
// result's validity shows whether a weight is read at its cell alone or,
// wrongly, over the slope's 3x3.
func runWeighted(m *manifest, d dem) error {
	ctx := context.Background()
	weight := weightRaster()
	if err := writeRaster(filepath.Join(*dir, "weight.f32"), weight); err != nil {
		return err
	}
	if err := writeMask(filepath.Join(*dir, "weight.mask.u8"), weight); err != nil {
		return err
	}
	o := terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY}
	name := "weighted_slope_deg"

	emit := func(form string, r raster.Float32Raster) error {
		base := d.name + "-" + name + "-" + form
		c := rasterCase{
			Name: base, Surface: d.surface + "-weighted", Op: name, Form: form,
			DEM: d.name + ".f32", Out: base + ".f32", CellSize: d.cellX, CellSizeY: d.cellY,
			Weight: "weight.f32", WeightMask: "weight.mask.u8",
			OutMask: base + ".mask.u8",
		}
		if d.r.Valid != nil {
			c.DEMMask = d.name + ".mask.u8"
		}
		if err := writeMask(filepath.Join(*dir, c.OutMask), r); err != nil {
			return err
		}
		if err := writeRaster(filepath.Join(*dir, c.Out), r); err != nil {
			return err
		}
		m.Rasters = append(m.Rasters, c)
		return nil
	}

	plain := raster.NewFloat32Like(weight)
	terrain.WeightedSlope(plain, d.r, weight, o)
	if err := emit("plain", plain); err != nil {
		return err
	}
	tiled := raster.NewFloat32Like(weight)
	if err := terrain.WeightedSlopeTiled(ctx, tiled, d.r, weight, o, tiledOpts); err != nil {
		return err
	}
	if err := emit("tiled", tiled); err != nil {
		return err
	}

	// Chunked, from the two raw files to a third.
	open := func(name string) (*engine.RawFile, error) {
		return engine.OpenRawFile(filepath.Join(*dir, name), os.O_RDONLY, 0, 4)
	}
	din, err := open(d.name + ".f32")
	if err != nil {
		return err
	}
	defer din.Close()
	win, err := open("weight.f32")
	if err != nil {
		return err
	}
	defer win.Close()
	ro := engine.RawOptions{Fill: fill, HasFill: true}
	tmp := filepath.Join(*dir, d.name+"-"+name+".chunked.tmp")
	out, err := engine.OpenRawFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, 4)
	if err != nil {
		return err
	}
	err = terrain.WeightedSlopeChunked(ctx, engine.NewRawSink(out, width, height, ro),
		engine.NewRawSource(din, width, height, engine.RawOptions{Fill: fill, HasFill: d.r.Valid != nil}),
		engine.NewRawSource(win, width, height, ro), o, chunkedOpts)
	if err != nil {
		out.Close()
		return err
	}
	back := raster.NewFloat32Like(weight)
	if err := engine.NewRawSource(out, width, height, ro).ReadWindow(ctx, back, 0, 0); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := os.Remove(tmp); err != nil {
		return err
	}
	return emit("chunked", back)
}

// --- surfaces -------------------------------------------------------

// weightRaster is a smooth weight between 0.2 and 1.8, with NoData in a
// horizontal stripe that crosses tile edges and in one cell of every 97.
func weightRaster() raster.Float32Raster {
	data := make([]float32, width*height)
	r := raster.NewFloat32(width, height, data)
	r.Valid = raster.NewMask(width * height)
	raster.MaskFillRange(r.Valid, 0, width*height, true)
	for y := range height {
		for x := range width {
			i := y*width + x
			data[i] = float32(1 + 0.8*math.Sin(float64(x)/17)*math.Cos(float64(y)/11))
			if (y >= 120 && y < 123) || i%97 == 0 {
				r.SetValid(x, y, false)
				data[i] = fill
			}
		}
	}
	return r
}

func plane(a, b float64) raster.Float32Raster {
	data := make([]float32, width*height)
	for y := range height {
		for x := range width {
			data[y*width+x] = float32(a*float64(x) + b*float64(y) + 100)
		}
	}
	return raster.NewFloat32(width, height, data)
}

func hill() raster.Float32Raster {
	data := make([]float32, width*height)
	cx, cy := float64(width)/2, float64(height)/2
	for y := range height {
		for x := range width {
			dx, dy := float64(x)-cx, float64(y)-cy
			data[y*width+x] = float32(500 * math.Exp(-(dx*dx+dy*dy)/(2*40*40)))
		}
	}
	return raster.NewFloat32(width, height, data)
}

func noisy() raster.Float32Raster {
	r := hill()
	// A fixed 64-bit LCG, so the surface is the same on every machine.
	state := uint64(0x2545F4914F6CDD1D)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>11) / float64(uint64(1)<<53)
	}
	for i := range r.Data {
		r.Data[i] += float32(40 * (next() - 0.5))
	}
	r.Valid = raster.NewMask(width * height)
	raster.MaskFillRange(r.Valid, 0, width*height, true)
	// A disc of NoData, and a vertical stripe that crosses tile edges.
	for y := range height {
		for x := range width {
			dx, dy := float64(x)-70, float64(y)-60
			if dx*dx+dy*dy < 25*25 || (x >= 180 && x < 186) {
				r.SetValid(x, y, false)
				r.Data[y*width+x] = fill
			}
		}
	}
	return r
}

func maskedLike(masked bool, w, h int) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	if masked {
		r.Valid = raster.NewMask(w * h)
		raster.MaskFillRange(r.Valid, 0, w*h, true)
	}
	return r
}

// --- output ---------------------------------------------------------

func writeRaster(path string, r raster.Float32Raster) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<16)
	buf := make([]byte, 4*r.Width)
	for y := range r.Height {
		row := r.Row(y)
		for x, v := range row[:r.Width] {
			binary.LittleEndian.PutUint32(buf[4*x:], math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func writeMask(path string, r raster.Float32Raster) error {
	buf := make([]byte, r.Width*r.Height)
	for y := range r.Height {
		for x := range r.Width {
			if r.IsValid(x, y) {
				buf[y*r.Width+x] = 1
			}
		}
	}
	return os.WriteFile(path, buf, 0o644)
}

func f32(v float32) string { return strconv.FormatFloat(float64(v), 'g', -1, 32) }

func unitName(u terrain.SlopeUnits) string {
	switch u {
	case terrain.SlopeDegrees:
		return "deg"
	case terrain.SlopeRadians:
		return "rad"
	default:
		return "pct"
	}
}
