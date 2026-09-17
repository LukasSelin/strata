package terrain

import (
	"context"
	"fmt"
	"math"

	"strata/engine"
	"strata/internal/exec"
	"strata/internal/stencil"
	"strata/raster"
)

// cellSizes resolves and checks the shared cell-size and z-factor options.
func cellSizes(cellSize, cellSizeY, zFactor float64) (kx, ky float32) {
	if !(cellSize > 0) || math.IsInf(cellSize, 0) {
		panic(fmt.Sprintf("terrain: CellSize must be positive and finite, got %v", cellSize))
	}
	if cellSizeY == 0 {
		cellSizeY = cellSize
	}
	if !(cellSizeY > 0) || math.IsInf(cellSizeY, 0) {
		panic(fmt.Sprintf("terrain: CellSizeY must be positive and finite (or 0 for CellSize), got %v", cellSizeY))
	}
	if zFactor == 0 {
		zFactor = 1
	}
	if math.IsNaN(zFactor) || math.IsInf(zFactor, 0) {
		panic(fmt.Sprintf("terrain: ZFactor must be finite, got %v", zFactor))
	}
	return stencil.HornScales(cellSize, cellSizeY, zFactor)
}

// run executes k, a kernel with the DEM as its one input, over whole
// rasters with one worker, on the calling goroutine: the plain functions
// start no goroutines (DESIGN.md §26), and callers who want workers use
// the Tiled functions. The engine applies the checks, the edge policy and
// the validity rules in the package documentation.
func run(k exec.Kernel, dem raster.Float32Raster, outs ...raster.Float32Raster) {
	// A background context is never done, so ProcessN cannot fail.
	_ = runTiled(context.Background(), engine.Options{Workers: 1}, k, dem, outs...)
}

// runTiled is run with a context and engine options.
func runTiled(ctx context.Context, eopts engine.Options, k exec.Kernel, dem raster.Float32Raster, outs ...raster.Float32Raster) error {
	return exec.ProcessN(ctx, outs, []raster.Float32Raster{dem}, k, eopts)
}

// runChunked is runTiled over a source and sinks.
func runChunked(ctx context.Context, eopts engine.Options, k exec.Kernel, dem engine.RasterSource, outs ...engine.RasterSink) error {
	return exec.ProcessChunked(ctx, outs, []engine.RasterSource{dem}, k, eopts)
}

// horn holds what every Horn kernel shares: radius 1, one DEM input,
// NaN at the edges and the resolved cell-size factors.
type horn struct{ kx, ky float32 }

func (horn) Radius() int                  { return 1 }
func (horn) Arity() (inputs, outputs int) { return 1, 1 }
func (horn) Edge() float32                { return float32(math.NaN()) }
