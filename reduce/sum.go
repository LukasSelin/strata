package reduce

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/accum"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/summary"
	"github.com/LukasSelin/strata/raster"
)

// Summary is what Stats returns: the count of the valid cells and what
// they sum to, average, spread over and range between, from one pass.
type Summary struct {
	// Count is the number of valid cells.
	Count int64
	// Sum is their exact sum, correctly rounded to float64. With no valid
	// cells it is 0.
	Sum float64
	// Mean is the exact sum divided by Count, correctly rounded. With no
	// valid cells it is NaN.
	Mean float64
	// StdDev is the population standard deviation, √(Σ(x−mean)²/Count):
	// the square root of the exact variance, taken at 256 bits and
	// rounded, so within one ulp of the true value. With no valid cells
	// it is NaN, and 0 for one.
	StdDev float64
	// Min and Max are what MinMax returns.
	Min, Max float32
}

// Sum returns the sum of src's valid cells and how many there were. The
// sum is exact, then correctly rounded to float64, so it does not depend
// on the order the cells are added in: not on the layout, the tiling, the
// worker count or the backend. With no valid cells it returns 0 and 0.
//
// A NaN in a valid cell makes the sum NaN, as do infinities of both
// signs; one infinity makes it that infinity. A sum of zeros is -0 only
// when every valid cell is -0, as IEEE addition gives.
func Sum(src raster.Float32Raster) (sum float64, count int64) {
	sum, count, err := SumTiled(context.Background(), src, plain)
	must(err)
	return sum, count
}

// SumTiled is Sum run by the engine.
func SumTiled(ctx context.Context, src raster.Float32Raster, opts engine.Options) (sum float64, count int64, err error) {
	p, err := exec.Reduce(ctx, []raster.Float32Raster{src}, sumOp{}, opts)
	if err != nil {
		return 0, 0, err
	}
	return p.s.Value(), p.s.Count(), nil
}

// SumChunked is Sum run by the engine over a source, with bounded memory.
func SumChunked(ctx context.Context, src engine.RasterSource, opts engine.Options) (sum float64, count int64, err error) {
	p, err := exec.ReduceChunked(ctx, []engine.RasterSource{src}, sumOp{}, opts)
	if err != nil {
		return 0, 0, err
	}
	return p.s.Value(), p.s.Count(), nil
}

// Stats returns the Summary of src's valid cells, from one pass. Every
// field is independent of the layout, the tiling, the worker count and
// the backend. NaN and infinities in valid cells propagate as in Sum and
// MinMax, and make StdDev NaN.
func Stats(src raster.Float32Raster) Summary {
	s, err := StatsTiled(context.Background(), src, plain)
	must(err)
	return s
}

// StatsTiled is Stats run by the engine.
func StatsTiled(ctx context.Context, src raster.Float32Raster, opts engine.Options) (Summary, error) {
	p, err := exec.Reduce(ctx, []raster.Float32Raster{src}, summary.Reducer{}, opts)
	if err != nil {
		return Summary{}, err
	}
	return Summary(p.Summary()), nil
}

// StatsChunked is Stats run by the engine over a source, with bounded
// memory.
func StatsChunked(ctx context.Context, src engine.RasterSource, opts engine.Options) (Summary, error) {
	p, err := exec.ReduceChunked(ctx, []engine.RasterSource{src}, summary.Reducer{}, opts)
	if err != nil {
		return Summary{}, err
	}
	return Summary(p.Summary()), nil
}

// sumPartial is a Sum reduction's running state: the exact accumulator,
// and the buffer summary.Runs packs partly valid words into.
type sumPartial struct {
	s    accum.Sum
	runs summary.Runs
}

// sumOp adds the valid cells to an exact accumulator. The accumulator's
// bins are integers, so Combine is exact, associative and commutative,
// and the zero accumulator is its identity.
type sumOp struct{}

func (sumOp) Inputs() int { return 1 }

func (sumOp) Combine(a *sumPartial, b sumPartial) { a.s.Combine(&b.s) }

// Fold adds the valid cells in runs long enough for the vector kernels
// (summary.Runs). Data under an invalid cell is never read.
func (sumOp) Fold(p *sumPartial, c exec.Cells) {
	p.runs.Each(c, p.s.Add)
}
