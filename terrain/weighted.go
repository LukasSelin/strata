package terrain

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// WeightedSlope computes Slope(dem) × weight: the slope of dem, as
// Slope computes it with opts, multiplied cell by cell by weight, the
// shape of a factor in a suitability or erosion model. It writes the
// bits Slope followed by algebra.Mul would write, in one pass over the
// operands rather than two, with no intermediate raster (DESIGN.md §52).
//
// Edges are Slope's: the one-cell border of dst gets NaN. If either
// input has a mask, a cell of dst is valid iff the 3×3 around it is
// valid in dem, as for Slope, and the cell itself is valid in weight:
// weight is read only at the cell, so an invalid weight invalidates that
// cell alone. dst, dem and weight must have the same dimensions, and dst
// must not overlap either input.
func WeightedSlope(dst, dem, weight raster.Float32Raster, opts SlopeOptions) {
	_ = WeightedSlopeTiled(context.Background(), dst, dem, weight, opts, engine.Options{Workers: 1})
}

// WeightedSlopeTiled is WeightedSlope run by the engine: it takes the
// same operands, applies the same checks and writes the same bits for
// every engine.Options, and returns ctx.Err() if ctx is done before every
// cell is written.
func WeightedSlopeTiled(ctx context.Context, dst, dem, weight raster.Float32Raster, opts SlopeOptions, eopts engine.Options) error {
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, []raster.Float32Raster{dem, weight},
		newWeightedSlope(opts), eopts)
}

// WeightedSlopeChunked is WeightedSlope run by the engine over sources
// and a sink with bounded memory. It writes the bits WeightedSlope would
// write into an in-memory raster, for every engine.Options.
func WeightedSlopeChunked(ctx context.Context, dst engine.RasterSink, dem, weight engine.RasterSource, opts SlopeOptions, eopts engine.Options) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{dem, weight},
		newWeightedSlope(opts), eopts)
}

// newWeightedSlope is the pipeline dem, weight → slope(dem) · weight:
// values 0 and 1 are the inputs, 2 the slope and 3 the product.
func newWeightedSlope(opts SlopeOptions) *exec.Pipeline {
	return exec.NewPipeline(2, []exec.Stage{
		{Kernel: newSlopeKernel(opts), In: []int{0}},
		{Kernel: mulKernel{}, In: []int{2, 1}},
	}, []int{3})
}

// mulKernel is algebra.Mul's arithmetic as a pipeline stage: the same
// vec kernel, so the product is the one algebra.Mul would write.
type mulKernel struct{}

func (mulKernel) Radius() int                  { return 0 }
func (mulKernel) Arity() (inputs, outputs int) { return 2, 1 }

func (mulKernel) Process(dst exec.Span, src exec.Window) {
	d, a, b := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		vec.Mul(d.Row(y), a.Row(y), b.Row(y))
	}
}
