package focal

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/raster"
)

// SeparableOptions configures CorrelateSeparable.
type SeparableOptions struct {
	// Radius is r, from 1 to MaxRadius. It has no default.
	Radius int
	// Row holds the 2r+1 finite taps applied along each row, left to
	// right, and Col the 2r+1 applied down each column, top to bottom.
	// They are copied when the call starts.
	Row, Col []float32
}

// CorrelateSeparable writes the weighted sum of each cell's
// (2r+1)×(2r+1) neighbourhood with the weights Col[j]·Row[c]:
// out(x, y) = Σ_c Row[c] · (Σ_j Col[j] · src(x+c-r, y+j-r)), each column
// folded first. It costs 2(2r+1) products per cell where Correlate costs
// (2r+1)². See the package documentation for the evaluation order, which
// makes the bits differ from Correlate's with the outer product as
// weights, and for edges and validity. dst and src must have the same
// dimensions and must not overlap; their strides may differ.
func CorrelateSeparable(dst, src raster.Float32Raster, opts SeparableOptions) {
	run(newSeparableKernel(opts), dst, src)
}

// CorrelateSeparableTiled is CorrelateSeparable run by the engine: it
// takes the same operands, applies the same checks and writes the same
// bits for every engine.Options, and returns ctx.Err() if ctx is done
// before every cell is written. See package engine for tiling and
// cancellation.
func CorrelateSeparableTiled(ctx context.Context, dst, src raster.Float32Raster, opts SeparableOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newSeparableKernel(opts), dst, src)
}

// CorrelateSeparableChunked is CorrelateSeparable run by the engine over
// a source and a sink with bounded memory, writing the bits
// CorrelateSeparable would write, for every engine.Options. See package
// engine for sources, sinks, memory, cancellation and errors.
func CorrelateSeparableChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts SeparableOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newSeparableKernel(opts), dst, src)
}

// Gaussian returns the 2r+1 taps of a Gaussian of standard deviation
// sigma cells, for SeparableOptions' Row and Col: exp(-x²/2σ²) for x from
// -r to r, computed and normalised to sum 1 in float64, then rounded to
// float32 (so their float32 sum may differ from 1 by a few ulps). They are
// symmetric, so CorrelateSeparable with them is also a convolution. A
// radius of about 3σ keeps all but 0.3% of the weight.
func Gaussian(radius int, sigma float64) []float32 {
	checkRadius(radius)
	if !(sigma > 0) || math.IsInf(sigma, 0) {
		panic(fmt.Sprintf("focal: Gaussian sigma must be positive and finite, got %v", sigma))
	}
	g := make([]float64, 2*radius+1)
	var sum float64
	for i := range g {
		x := float64(i - radius)
		g[i] = math.Exp(-x * x / (2 * sigma * sigma))
		sum += g[i]
	}
	taps := make([]float32, len(g))
	for i, v := range g {
		taps[i] = float32(v / sum)
	}
	return taps
}

func newSeparableKernel(opts SeparableOptions) separableKernel {
	checkRadius(opts.Radius)
	k := 2*opts.Radius + 1
	if len(opts.Row) != k || len(opts.Col) != k {
		panic(fmt.Sprintf("focal: Radius %d needs %d Row and %d Col taps, got %d and %d",
			opts.Radius, k, k, len(opts.Row), len(opts.Col)))
	}
	checkFinite("Row taps", opts.Row)
	checkFinite("Col taps", opts.Col)
	return separableKernel{base{opts.Radius},
		append([]float32(nil), opts.Row...), append([]float32(nil), opts.Col...)}
}

// separableKernel folds each output row's window in two passes: the
// column pass writes one row of W+2r column results to scratch, and the
// row pass folds each run of 2r+1 of them into an output cell. Column
// first means only the 2r cells at the ends of each row are computed for
// two output cells; a row pass first would redo 2r whole rows of it for
// every band of rows (DESIGN.md §53).
type separableKernel struct {
	base
	row, col []float32
}

func (k separableKernel) Scratch(w, h int) exec.ScratchSize { return k.rowScratch(w, h) }

func (k separableKernel) Process(dst exec.Span, src exec.Window) {
	out, in := dst.Dst[0], src.Src[0]
	tmp := dst.Scratch.Cells[:dst.Width+2*k.r]
	for y := range dst.Height {
		focalrow.ColumnCorrelate(tmp, in.Data[y*in.Stride:], in.Stride, k.col)
		focalrow.RowCorrelate(out.Row(y), tmp, k.row)
	}
}
