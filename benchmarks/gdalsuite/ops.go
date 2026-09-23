package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/resample"
	"github.com/LukasSelin/strata/terrain"
)

// An op is one strata operation with a GDAL counterpart. runsuite.sh
// holds the GDAL side of each, under the same name.
//
// Every op has two entry points: tiled, the in-memory call (the plain
// function when workers is 1, since that is what a single-threaded user
// calls, and its Tiled form otherwise), and chunked, the bounded-memory
// call from sources to a sink. A reduction has no destination and returns
// a line of results instead.
type op struct {
	name   string
	inputs int // 1, or 2 for the binary algebra
	// scale is the destination's size over the source's, per axis: 1 for
	// everything but resample.
	scale float64

	tiled   func(ctx context.Context, dst raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error)
	chunked func(ctx context.Context, dst engine.RasterSink, dstGrid raster.Grid, src []engine.RasterSource, grids []raster.Grid, eo engine.Options) (string, error)
}

func (o op) reduces() bool { return o.scale == 0 }

// cellSize is set from -cell before any op runs. Terrain ops need it;
// the rest do not depend on it.
var cellSize = 12.5

// conv5 is a 5×5 weight grid asymmetric in both axes, so that a flipped
// or transposed kernel on either side shows up in the agreement check
// instead of passing. Row 0 is above the output cell. gdalKernel prints
// it in the syntax of `gdal raster neighbors --kernel`.
var conv5 = func() []float32 {
	w := make([]float32, 25)
	for j := range 5 {
		for c := range 5 {
			w[j*5+c] = float32(j+1)/16 + float32(c+1)/64
		}
	}
	return w
}()

func gdalKernel(w []float32, n int) string {
	rows := make([]string, n)
	for j := range n {
		cols := make([]string, n)
		for c := range n {
			cols[c] = fmt.Sprintf("%g", w[j*n+c])
		}
		rows[j] = "[" + strings.Join(cols, ",") + "]"
	}
	return "[" + strings.Join(rows, ",") + "]"
}

// gauss5 is GDAL's `--kernel gaussian --size 5`: the outer product of
// the binomial taps 1 4 6 4 1, normalised. strata runs it as the
// separable pair it is.
var gauss5 = []float32{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}

// Clamp bounds, in the canopy-height grid's units (decimetres).
const clampLo, clampHi = 50, 200

// unary adapts a raster-to-raster operation with one input.
func unary(
	plain func(dst, src raster.Float32Raster),
	tiled func(ctx context.Context, dst, src raster.Float32Raster, eo engine.Options) error,
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error,
) op {
	return op{
		inputs: 1, scale: 1,
		tiled: func(ctx context.Context, dst raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error) {
			if eo.Workers == 1 {
				plain(dst.Raster, src[0].Raster)
				return "", nil
			}
			return "", tiled(ctx, dst.Raster, src[0].Raster, eo)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, _ raster.Grid, src []engine.RasterSource, _ []raster.Grid, eo engine.Options) (string, error) {
			return "", chunked(ctx, dst, src[0], eo)
		},
	}
}

// binary adapts a two-input algebra operation.
func binary(
	plain func(dst, a, b raster.Float32Raster),
	tiled func(ctx context.Context, dst, a, b raster.Float32Raster, eo engine.Options) error,
	chunked func(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, eo engine.Options) error,
) op {
	return op{
		inputs: 2, scale: 1,
		tiled: func(ctx context.Context, dst raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error) {
			if eo.Workers == 1 {
				plain(dst.Raster, src[0].Raster, src[1].Raster)
				return "", nil
			}
			return "", tiled(ctx, dst.Raster, src[0].Raster, src[1].Raster, eo)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, _ raster.Grid, src []engine.RasterSource, _ []raster.Grid, eo engine.Options) (string, error) {
			return "", chunked(ctx, dst, src[0], src[1], eo)
		},
	}
}

func summary(s reduce.Summary) string {
	return fmt.Sprintf("count=%d min=%.9g max=%.9g mean=%.17g stddev=%.17g",
		s.Count, s.Min, s.Max, s.Mean, s.StdDev)
}

func minmax(mn, mx float32, n int64) string {
	return fmt.Sprintf("count=%d min=%.9g max=%.9g", n, mn, mx)
}

func resampler(m resample.Method, scale float64) op {
	o := resample.Options{Method: m}
	return op{
		inputs: 1, scale: scale,
		tiled: func(ctx context.Context, dst raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error) {
			if eo.Workers == 1 {
				resample.Resample(dst, src[0], o)
				return "", nil
			}
			return "", resample.ResampleTiled(ctx, dst, src[0], o, eo)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, dstGrid raster.Grid, src []engine.RasterSource, grids []raster.Grid, eo engine.Options) (string, error) {
			return "", resample.ResampleChunked(ctx, dst, dstGrid, src[0], grids[0], o, eo)
		},
	}
}

// ops returns the table, in the order runsuite.sh times them.
func ops() []op {
	so := func() terrain.SlopeOptions { return terrain.SlopeOptions{CellSize: cellSize, CellSizeY: cellSize} }
	ao := func() terrain.AspectOptions { return terrain.AspectOptions{CellSize: cellSize, CellSizeY: cellSize} }
	ho := func() terrain.HillshadeOptions {
		return terrain.HillshadeOptions{CellSize: cellSize, CellSizeY: cellSize, Azimuth: 315, Altitude: 45}
	}
	rugged := func(name string, t terrain.RuggednessType) op {
		ro := terrain.RuggednessOptions{Type: t}
		o := unary(
			func(d, s raster.Float32Raster) { terrain.Ruggedness(d, s, ro) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return terrain.RuggednessTiled(ctx, d, s, ro, eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return terrain.RuggednessChunked(ctx, d, s, ro, eo)
			})
		o.name = name
		return o
	}
	box := func(name string, r int,
		plain func(dst, src raster.Float32Raster, o focal.BoxOptions),
		tiled func(ctx context.Context, dst, src raster.Float32Raster, o focal.BoxOptions, eo engine.Options) error,
		chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o focal.BoxOptions, eo engine.Options) error,
	) op {
		bo := focal.BoxOptions{Radius: r}
		o := unary(
			func(d, s raster.Float32Raster) { plain(d, s, bo) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return tiled(ctx, d, s, bo, eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return chunked(ctx, d, s, bo, eo)
			})
		o.name = name
		return o
	}
	named := func(name string, o op) op { o.name = name; return o }

	wo := focal.WeightsOptions{Radius: 2, Weights: conv5}
	gso := focal.SeparableOptions{Radius: 2, Row: gauss5, Col: gauss5}

	return []op{
		// terrain: gdaldem
		named("slope", unary(
			func(d, s raster.Float32Raster) { terrain.Slope(d, s, so()) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return terrain.SlopeTiled(ctx, d, s, so(), eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, d, s, so(), eo)
			})),
		named("aspect", unary(
			func(d, s raster.Float32Raster) { terrain.Aspect(d, s, ao()) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return terrain.AspectTiled(ctx, d, s, ao(), eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, d, s, ao(), eo)
			})),
		named("hillshade", unary(
			func(d, s raster.Float32Raster) { terrain.Hillshade(d, s, ho()) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return terrain.HillshadeTiled(ctx, d, s, ho(), eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return terrain.HillshadeChunked(ctx, d, s, ho(), eo)
			})),
		rugged("tri", terrain.RuggednessTRI),
		rugged("tpi", terrain.RuggednessTPI),
		rugged("roughness", terrain.RuggednessRoughness),

		// focal: gdal raster neighbors
		box("mean3", 1, focal.Mean, focal.MeanTiled, focal.MeanChunked),
		box("mean11", 5, focal.Mean, focal.MeanTiled, focal.MeanChunked),
		box("min3", 1, focal.Min, focal.MinTiled, focal.MinChunked),
		box("max3", 1, focal.Max, focal.MaxTiled, focal.MaxChunked),
		named("gauss5", unary(
			func(d, s raster.Float32Raster) { focal.CorrelateSeparable(d, s, gso) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return focal.CorrelateSeparableTiled(ctx, d, s, gso, eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return focal.CorrelateSeparableChunked(ctx, d, s, gso, eo)
			})),
		named("conv5", unary(
			func(d, s raster.Float32Raster) { focal.Correlate(d, s, wo) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return focal.CorrelateTiled(ctx, d, s, wo, eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return focal.CorrelateChunked(ctx, d, s, wo, eo)
			})),

		// algebra: gdal raster calc and gdal_calc.py
		named("add", binary(algebra.Add, algebra.AddTiled, algebra.AddChunked)),
		named("mul", binary(algebra.Mul, algebra.MulTiled, algebra.MulChunked)),
		named("min", binary(algebra.Min, algebra.MinTiled, algebra.MinChunked)),
		named("max", binary(algebra.Max, algebra.MaxTiled, algebra.MaxChunked)),
		named("clamp", unary(
			func(d, s raster.Float32Raster) { algebra.Clamp(d, s, clampLo, clampHi) },
			func(ctx context.Context, d, s raster.Float32Raster, eo engine.Options) error {
				return algebra.ClampTiled(ctx, d, s, clampLo, clampHi, eo)
			},
			func(ctx context.Context, d engine.RasterSink, s engine.RasterSource, eo engine.Options) error {
				return algebra.ClampChunked(ctx, d, s, clampLo, clampHi, eo)
			})),

		// reduce: GDALComputeRasterStatistics and GDALComputeRasterMinMax
		{
			name: "stats", inputs: 1,
			tiled: func(ctx context.Context, _ raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error) {
				if eo.Workers == 1 {
					return summary(reduce.Stats(src[0].Raster)), nil
				}
				s, err := reduce.StatsTiled(ctx, src[0].Raster, eo)
				return summary(s), err
			},
			chunked: func(ctx context.Context, _ engine.RasterSink, _ raster.Grid, src []engine.RasterSource, _ []raster.Grid, eo engine.Options) (string, error) {
				s, err := reduce.StatsChunked(ctx, src[0], eo)
				return summary(s), err
			},
		},
		{
			name: "minmax", inputs: 1,
			tiled: func(ctx context.Context, _ raster.Dataset, src []raster.Dataset, eo engine.Options) (string, error) {
				if eo.Workers == 1 {
					return minmax(reduce.MinMax(src[0].Raster)), nil
				}
				mn, mx, n, err := reduce.MinMaxTiled(ctx, src[0].Raster, eo)
				return minmax(mn, mx, n), err
			},
			chunked: func(ctx context.Context, _ engine.RasterSink, _ raster.Grid, src []engine.RasterSource, _ []raster.Grid, eo engine.Options) (string, error) {
				mn, mx, n, err := reduce.MinMaxChunked(ctx, src[0], eo)
				return minmax(mn, mx, n), err
			},
		},

		// resample: gdalwarp
		named("near-half", resampler(resample.Nearest, 0.5)),
		named("bilinear-half", resampler(resample.Bilinear, 0.5)),
		named("cubic-half", resampler(resample.Cubic, 0.5)),
		named("lanczos-half", resampler(resample.Lanczos, 0.5)),
		named("average-half", resampler(resample.Average, 0.5)),
		named("cubic-double", resampler(resample.Cubic, 2)),
	}
}

func lookup(name string) (op, bool) {
	for _, o := range ops() {
		if o.name == name {
			return o, true
		}
	}
	return op{}, false
}
