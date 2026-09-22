package terrain

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// resolveCells resolves and checks the shared cell-size and z-factor
// options: CellSizeY 0 means CellSize and ZFactor 0 means 1.
func resolveCells(cellSize, cellSizeY, zFactor float64) (cx, cy, z float64) {
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
	return cellSize, cellSizeY, zFactor
}

// cellSizes resolves and checks the shared options for the Horn kernels.
func cellSizes(cellSize, cellSizeY, zFactor float64) (kx, ky float32) {
	cx, cy, z := resolveCells(cellSize, cellSizeY, zFactor)
	// The kernels multiply by these factors as float32. One that
	// overflows to ±Inf turns a flat neighbourhood (0·Inf) into NaN, and
	// one that underflows to 0 flattens every gradient.
	kx, ky = stencil.HornScales(cx, cy, z)
	if !usableScale(kx) || !usableScale(ky) {
		panic(fmt.Sprintf("terrain: ZFactor/(8·CellSize) and ZFactor/(8·CellSizeY) must be finite and non-zero as float32, "+
			"got %v and %v for CellSize %v, CellSizeY %v and ZFactor %v", kx, ky, cellSize, cellSizeY, zFactor))
	}
	return kx, ky
}

func usableScale(k float32) bool { return k != 0 && !math.IsInf(float64(k), 0) }

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

// window3 holds what every kernel of this package shares: radius 1, one
// DEM input, one output unless overridden, and NaN at the edges.
type window3 struct{}

func (window3) Radius() int                  { return 1 }
func (window3) Arity() (inputs, outputs int) { return 1, 1 }
func (window3) Edge() float32                { return float32(math.NaN()) }

// horn is window3 with the resolved Horn cell-size factors.
type horn struct {
	window3
	kx, ky float32
}
