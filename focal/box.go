package focal

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/raster"
)

// BoxOptions configures Mean, Min and Max.
type BoxOptions struct {
	// Radius is r, from 1 to MaxRadius: the neighbourhood is the
	// (2r+1)×(2r+1) square around each cell. It has no default.
	Radius int
}

// Mean writes the mean of each cell's (2r+1)×(2r+1) neighbourhood: the
// sum, columns first, divided by (2r+1)². See the package documentation
// for the evaluation order, edges and validity (every cell of the
// neighbourhood must be valid). dst and src must have the same dimensions
// and must not overlap; their strides may differ.
func Mean(dst, src raster.Float32Raster, opts BoxOptions) { run(newBox(opts, boxMean), dst, src) }

// MeanTiled is Mean run by the engine: the same operands, checks and bits
// for every engine.Options, and ctx.Err() if ctx is done before every
// cell is written. See package engine for tiling and cancellation.
func MeanTiled(ctx context.Context, dst, src raster.Float32Raster, opts BoxOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newBox(opts, boxMean), dst, src)
}

// MeanChunked is Mean run by the engine over a source and a sink with
// bounded memory, writing the bits Mean would write, for every
// engine.Options. See package engine for sources, sinks, memory,
// cancellation and errors.
func MeanChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts BoxOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newBox(opts, boxMean), dst, src)
}

// Min writes the minimum of each cell's (2r+1)×(2r+1) neighbourhood, by
// Go's builtin min: NaN if any cell is NaN, and -0 below +0. It is exact,
// so any order of evaluation gives the same bits. See the package
// documentation for edges and validity. dst and src must have the same
// dimensions and must not overlap; their strides may differ.
func Min(dst, src raster.Float32Raster, opts BoxOptions) { run(newBox(opts, boxMin), dst, src) }

// MinTiled is Min run by the engine, as MeanTiled is Mean.
func MinTiled(ctx context.Context, dst, src raster.Float32Raster, opts BoxOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newBox(opts, boxMin), dst, src)
}

// MinChunked is Min run by the engine over a source and a sink, as
// MeanChunked is Mean.
func MinChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts BoxOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newBox(opts, boxMin), dst, src)
}

// Max writes the maximum of each cell's (2r+1)×(2r+1) neighbourhood, by
// Go's builtin max: NaN if any cell is NaN, and +0 above -0. Otherwise as
// Min.
func Max(dst, src raster.Float32Raster, opts BoxOptions) { run(newBox(opts, boxMax), dst, src) }

// MaxTiled is Max run by the engine, as MeanTiled is Mean.
func MaxTiled(ctx context.Context, dst, src raster.Float32Raster, opts BoxOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newBox(opts, boxMax), dst, src)
}

// MaxChunked is Max run by the engine over a source and a sink, as
// MeanChunked is Mean.
func MaxChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts BoxOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newBox(opts, boxMax), dst, src)
}

type boxOp int

const (
	boxMean boxOp = iota
	boxMin
	boxMax
)

func newBox(opts BoxOptions, op boxOp) boxKernel {
	checkRadius(opts.Radius)
	return boxKernel{base{opts.Radius}, op}
}

// boxKernel is Mean, Min or Max, separable as separableKernel is: a
// column pass into one row of scratch, then a row pass into the output.
type boxKernel struct {
	base
	op boxOp
}

func (k boxKernel) Scratch(w, h int) exec.ScratchSize { return k.rowScratch(w, h) }

func (k boxKernel) Process(dst exec.Span, src exec.Window) {
	out, in := dst.Dst[0], src.Src[0]
	tmp := dst.Scratch.Cells[:dst.Width+2*k.r]
	n := k.size()
	cells := float32(n * n)
	for y := range dst.Height {
		col := in.Data[y*in.Stride:]
		switch k.op {
		case boxMean:
			focalrow.ColumnSum(tmp, col, in.Stride, n)
			focalrow.RowMean(out.Row(y), tmp, n, cells)
		case boxMin:
			focalrow.ColumnMin(tmp, col, in.Stride, n)
			focalrow.RowMin(out.Row(y), tmp, n)
		default:
			focalrow.ColumnMax(tmp, col, in.Stride, n)
			focalrow.RowMax(out.Row(y), tmp, n)
		}
	}
}
