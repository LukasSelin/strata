package terrain

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// RuggednessType selects which measure Ruggedness computes.
type RuggednessType int

const (
	// RuggednessTRI is Riley et al.'s (1999) terrain ruggedness index:
	// the square root of the summed squared differences between a cell
	// and its eight neighbours. It is gdaldem TRI's default.
	RuggednessTRI RuggednessType = iota
	// RuggednessTRIWilson is Wilson et al.'s (2007) terrain ruggedness
	// index: the mean absolute difference between a cell and its eight
	// neighbours. It is gdaldem TRI -alg Wilson.
	RuggednessTRIWilson
	// RuggednessTPI is the topographic position index: a cell minus the
	// mean of its eight neighbours. Positive on ridges and hilltops,
	// negative in valleys and pits. It is gdaldem TPI.
	RuggednessTPI
	// RuggednessRoughness is the largest minus the smallest elevation of
	// the 3×3 window, centre included. It is gdaldem roughness.
	RuggednessRoughness
)

// RuggednessOptions configures Ruggedness.
type RuggednessOptions struct {
	// Type of measure. The zero value is RuggednessTRI.
	Type RuggednessType
}

// Ruggedness computes a ruggedness measure of dem over each cell's 3×3
// window, in the elevations' units. See the package documentation for
// edges and validity. dst and dem must have the same dimensions and must
// not overlap; their strides may differ.
//
// With z1..z9 the window in row-major order (z5 the centre, as in
// Gradient) and di = zi - z5:
//
//	RuggednessTRI         √(d1² + d2² + d3² + d4² + d6² + d7² + d8² + d9²)
//	RuggednessTRIWilson   (|d1| + |d2| + … + |d9|) / 8
//	RuggednessTPI         z5 - (z1 + z2 + … + z9) / 8
//	RuggednessRoughness   max(z1..z9) - min(z1..z9)
//
// with the sums over the eight neighbours only. Unlike the other
// operations of this package, none of them depends on the cell size, and
// there is no ZFactor: every measure scales with the elevations, so
// multiply the result instead (by |k| for TRI and roughness).
//
// The results are bit-identical to GDAL gdaldem's TRI, TRI -alg Wilson,
// TPI and roughness on a float32 DEM (apps/gdaldem_lib.cpp): each
// difference is rounded to float32, every sum is folded left to right in
// row-major order, and the division by 8 is a multiplication by 0.125.
// Riley's TRI squares and sums the float32 differences in float64 and
// takes the root there, as gdaldem does, before rounding to float32 once;
// it is the one kernel of this package that computes in float64, and the
// slowest of the four.
//
// Roughness uses Go's min and max, so a NaN anywhere in the window gives
// NaN. gdaldem compares with < and >, which skip a NaN unless it is the
// first cell; the two agree wherever the DEM holds no NaN it calls data.
// The result is never -0.
func Ruggedness(dst, dem raster.Float32Raster, opts RuggednessOptions) {
	run(newRuggednessKernel(opts), dem, dst)
}

// RuggednessTiled is Ruggedness run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func RuggednessTiled(ctx context.Context, dst, dem raster.Float32Raster, opts RuggednessOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, newRuggednessKernel(opts), dem, dst)
}

// RuggednessChunked is Ruggedness run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Ruggedness would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func RuggednessChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts RuggednessOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, newRuggednessKernel(opts), dem, dst)
}

// newRuggednessKernel checks opts for Ruggedness's kernel.
func newRuggednessKernel(opts RuggednessOptions) ruggednessKernel {
	var kind stencil.RuggednessKind
	switch opts.Type {
	case RuggednessTRI:
		kind = stencil.RugTRIRiley
	case RuggednessTRIWilson:
		kind = stencil.RugTRIWilson
	case RuggednessTPI:
		kind = stencil.RugTPI
	case RuggednessRoughness:
		kind = stencil.RugRoughness
	default:
		panic(fmt.Sprintf("terrain: unknown RuggednessType %d", opts.Type))
	}
	return ruggednessKernel{kind: kind}
}

type ruggednessKernel struct {
	window3
	kind stencil.RuggednessKind
}

func (k ruggednessKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		stencil.RuggednessRow(out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kind)
	}
}
