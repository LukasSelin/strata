package focal

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// MaxRadius is the largest radius the operations accept. The engine
// hands a kernel bands of about 2¹⁶ cells and checks for cancellation
// between them (DESIGN.md §25); at radius 8, Correlate's 289 products per
// cell already make a band some tens of milliseconds of scalar work.
const MaxRadius = 8

// checkRadius panics unless r is a radius the operations accept.
func checkRadius(r int) {
	if r < 1 || r > MaxRadius {
		panic(fmt.Sprintf("focal: Radius must be 1 to %d, got %d", MaxRadius, r))
	}
}

// checkFinite panics unless every value in v is finite, naming them what.
func checkFinite(what string, v []float32) {
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			panic(fmt.Sprintf("focal: %s must be finite, got %v at index %d", what, x, i))
		}
	}
}

// run executes k, a kernel with src as its one input, over whole rasters
// with one worker, on the calling goroutine: the plain functions start no
// goroutines (DESIGN.md §26), and callers who want workers use the Tiled
// functions. The engine applies the operand checks, the edge policy and
// the validity rules in the package documentation.
func run(k exec.Kernel, dst, src raster.Float32Raster) {
	// A background context is never done, so ProcessN cannot fail.
	_ = runTiled(context.Background(), engine.Options{Workers: 1}, k, dst, src)
}

// runTiled is run with a context and engine options.
func runTiled(ctx context.Context, eopts engine.Options, k exec.Kernel, dst, src raster.Float32Raster) error {
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, []raster.Float32Raster{src}, k, eopts)
}

// runChunked is runTiled over a source and a sink.
func runChunked(ctx context.Context, eopts engine.Options, k exec.Kernel, dst engine.RasterSink, src engine.RasterSource) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, k, eopts)
}

// base holds what every focal kernel shares: one input, one output, the
// radius, and NaN at the edges.
type base struct{ r int }

func (b base) Radius() int                { return b.r }
func (base) Arity() (inputs, outputs int) { return 1, 1 }
func (base) Edge() float32                { return float32(math.NaN()) }

// size is the neighbourhood's width, 2r+1.
func (b base) size() int { return 2*b.r + 1 }

// rowScratch is the Scratch of the separable kernels: one row of column
// results, as wide as the widest span plus the halo on both sides.
func (b base) rowScratch(w, _ int) exec.ScratchSize {
	return exec.ScratchSize{Cells: w + 2*b.r}
}
