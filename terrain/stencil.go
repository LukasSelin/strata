package terrain

import (
	"context"
	"fmt"
	"math"

	"strata/engine"
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
// rasters on the calling goroutine. The engine applies the checks, the
// edge policy and the validity rules in the package documentation.
func run(k engine.Kernel, dem raster.Float32Raster, outs ...raster.Float32Raster) {
	// A background context is never done, so ProcessN cannot fail.
	_ = engine.ProcessN(context.Background(), outs, []raster.Float32Raster{dem}, k, engine.Options{})
}

// horn holds what every Horn kernel shares: radius 1, one DEM input,
// NaN at the edges and the resolved cell-size factors.
type horn struct{ kx, ky float32 }

func (horn) Radius() int                  { return 1 }
func (horn) Arity() (inputs, outputs int) { return 1, 1 }
func (horn) Edge() float32                { return float32(math.NaN()) }
