package focal

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/raster"
)

// WeightsOptions configures Correlate and Convolve.
type WeightsOptions struct {
	// Radius is r, from 1 to MaxRadius. It has no default.
	Radius int
	// Weights holds (2r+1)×(2r+1) finite weights, row-major, row 0 above
	// the output cell (see the package documentation). They are copied
	// when the call starts.
	Weights []float32
}

// Correlate writes the weighted sum of each cell's (2r+1)×(2r+1)
// neighbourhood: out(x, y) = Σ_j Σ_c w[j][c] · src(x+c-r, y+j-r). See the
// package documentation for the evaluation order, edges and validity.
// dst and src must have the same dimensions and must not overlap; their
// strides may differ.
func Correlate(dst, src raster.Float32Raster, opts WeightsOptions) {
	run(newWeightsKernel(opts, false), dst, src)
}

// CorrelateTiled is Correlate run by the engine: it takes the same
// operands, applies the same checks and writes the same bits for every
// engine.Options, and returns ctx.Err() if ctx is done before every cell
// is written. See package engine for tiling and cancellation.
func CorrelateTiled(ctx context.Context, dst, src raster.Float32Raster, opts WeightsOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newWeightsKernel(opts, false), dst, src)
}

// CorrelateChunked is Correlate run by the engine over a source and a
// sink with bounded memory: it reads the input and writes the result a
// tile at a time, with Workers × tile buffers in memory. It writes the
// bits Correlate would write into in-memory rasters, for every
// engine.Options. See package engine for sources, sinks, memory,
// cancellation and errors.
func CorrelateChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts WeightsOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newWeightsKernel(opts, false), dst, src)
}

// Convolve is Correlate with the weights rotated by 180°:
// out(x, y) = Σ_j Σ_c w[j][c] · src(x-c+r, y-j+r), the convolution of
// scipy.ndimage.convolve. It writes the bits of Correlate with the
// rotated weights.
func Convolve(dst, src raster.Float32Raster, opts WeightsOptions) {
	run(newWeightsKernel(opts, true), dst, src)
}

// ConvolveTiled is Convolve run by the engine, as CorrelateTiled is
// Correlate.
func ConvolveTiled(ctx context.Context, dst, src raster.Float32Raster, opts WeightsOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newWeightsKernel(opts, true), dst, src)
}

// ConvolveChunked is Convolve run by the engine over a source and a sink,
// as CorrelateChunked is Correlate.
func ConvolveChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts WeightsOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newWeightsKernel(opts, true), dst, src)
}

// newWeightsKernel checks opts and copies the weights, rotated by 180°
// for a convolution, so the kernel always correlates.
func newWeightsKernel(opts WeightsOptions, rotate bool) weightsKernel {
	checkRadius(opts.Radius)
	k := 2*opts.Radius + 1
	if len(opts.Weights) != k*k {
		panic(fmt.Sprintf("focal: Radius %d needs %d×%d = %d Weights, got %d",
			opts.Radius, k, k, k*k, len(opts.Weights)))
	}
	checkFinite("Weights", opts.Weights)
	w := make([]float32, len(opts.Weights))
	for i, v := range opts.Weights {
		if rotate {
			i = len(w) - 1 - i
		}
		w[i] = v
	}
	return weightsKernel{base{opts.Radius}, w}
}

type weightsKernel struct {
	base
	w []float32
}

func (k weightsKernel) Process(dst exec.Span, src exec.Window) {
	out, in := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		focalrow.CorrelateRow(out.Row(y), in.Data[y*in.Stride:], in.Stride, k.w, k.size())
	}
}
