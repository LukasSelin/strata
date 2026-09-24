package terrain

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/focalrow"
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

// MaxRadius is the largest window radius Ruggedness takes, focal's
// limit (DESIGN.md §53): at r = 8 a cell reads 289 elevations.
const MaxRadius = 8

// RuggednessOptions configures Ruggedness.
type RuggednessOptions struct {
	// Type of measure. The zero value is RuggednessTRI.
	Type RuggednessType
	// Radius is the window's radius in cells: the measure is taken over
	// the (2·Radius+1)² cells around each cell. 0 means 1, the 3×3 window
	// of gdaldem; at most MaxRadius.
	Radius int
}

// Ruggedness computes a ruggedness measure of dem over each cell's
// (2r+1)×(2r+1) window, r = opts.Radius, in the elevations' units. The
// border is r cells wide, and a cell is valid iff its whole window is;
// otherwise edges and validity are as in the package documentation. dst
// and dem must have the same dimensions and must not overlap; their
// strides may differ.
//
// For the 3×3 window (r = 1), with z1..z9 the window in row-major order
// (z5 the centre, as in Gradient) and di = zi - z5:
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
//
// A larger radius measures the same thing at a coarser scale, as
// multi-scale terrain features do: the eight neighbours become the
// n = (2r+1)²−1 cells of the window other than the centre, still in
// row-major order and folded the same way, and the · 0.125 becomes a
// division by n, which at r = 1 rounds to the same bits. So TPI is the
// centre minus the mean of the window's other cells, Wilson's TRI their
// mean absolute difference from the centre, Riley's TRI the root of
// their summed squared differences (which grows with the window, as the
// definition does), and roughness the window's range. The window is a
// square, not the annulus some TPI definitions use. At r = 1 the SIMD
// kernels run. At larger radii the three sums run a scalar kernel that
// is O(r²) a cell, since their order is part of their definition, and
// roughness runs focal's separable Max and Min kernels, O(r) a cell and
// the same bits, since max and min do not depend on the order.
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
	r := opts.Radius
	if r == 0 {
		r = 1
	}
	if r < 1 || r > MaxRadius {
		panic(fmt.Sprintf("terrain: RuggednessOptions.Radius must be 0 to %d, got %d", MaxRadius, opts.Radius))
	}
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
	return ruggednessKernel{r: r, kind: kind}
}

// ruggednessKernel is window3 with the radius a field.
type ruggednessKernel struct {
	window3
	r    int
	kind stencil.RuggednessKind
}

func (k ruggednessKernel) Radius() int { return k.r }

func (k ruggednessKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	if k.r == 1 {
		for y := range dst.Height {
			stencil.RuggednessRow(out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kind)
		}
		return
	}
	if k.kind == stencil.RugRoughness {
		roughness(out, dem, 2*k.r+1)
		return
	}
	var buf [2*MaxRadius + 1][]float32
	rows := buf[:2*k.r+1]
	for y := range dst.Height {
		for j := range rows {
			rows[j] = dem.Row(y + j)
		}
		stencil.RuggednessWindowRow(out.Row(y), rows, k.kind)
	}
}

// roughnessBlock is how many cells roughness takes at a time, so that
// its column results fit on the stack rather than in engine scratch,
// which a Pipeline stage cannot have (DESIGN.md §52).
const roughnessBlock = 256

// roughness writes the range of each k×k window of dem into out, k odd:
// the column maxima and minima of the k rows under an output row, then
// the maxima and minima of each run of k of those, as focal.Max and
// focal.Min compute them. Go's max and min are associative and
// commutative, NaN and signed zeros included, so this is the brute-force
// max minus min bit for bit (DESIGN.md §53).
func roughness(out, dem raster.Float32Raster, k int) {
	var hiCol, loCol [roughnessBlock + 2*MaxRadius]float32
	var lo [roughnessBlock]float32
	for y := range out.Height {
		row := out.Row(y)
		src := dem.Data[dem.Index(0, y):]
		for x0 := 0; x0 < len(row); x0 += roughnessBlock {
			o := row[x0:min(x0+roughnessBlock, len(row))]
			cols := len(o) + k - 1
			focalrow.ColumnMax(hiCol[:cols], src[x0:], dem.Stride, k)
			focalrow.ColumnMin(loCol[:cols], src[x0:], dem.Stride, k)
			focalrow.RowMax(o, hiCol[:cols], k)
			focalrow.RowMin(lo[:len(o)], loCol[:cols], k)
			for i, l := range lo[:len(o)] {
				o[i] -= l
			}
		}
	}
}
