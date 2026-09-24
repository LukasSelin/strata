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

// HillshadeOptions configures Hillshade.
type HillshadeOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
	// FitRadius selects how the derivatives are estimated: 0 is Horn's
	// 3×3 kernel, as gdaldem uses; 1 to MaxRadius fits Wood's quadratic
	// by least squares to the (2·FitRadius+1)² window around each cell,
	// for the same measure at a coarser scale (see the package
	// documentation). At 1 the fit is not Horn's kernel.
	FitRadius int
	// Azimuth is the compass direction the light comes from, in degrees
	// clockwise from north. 0 means 315 (northwest), as in gdaldem; use
	// 360 for light from the north. It must be finite.
	Azimuth float64
	// Altitude is the light's angle above the horizon in degrees, in
	// (0, 90]. 0 means 45, as in gdaldem.
	Altitude float64
}

// Hillshade computes shaded relief of dem lit from Azimuth and Altitude,
// from its Horn gradient (dx and dy as in Gradient), as a float value in
// [0, 255]:
//
//	255 · max(0, sin(alt)·cos(slope) + cos(alt)·sin(slope)·cos(az − aspect))
//
// with aspect the compass bearing of Aspect. It is evaluated in the
// equivalent form without angles, as the cosine between the surface
// normal and the direction towards the light:
//
//	255 · (sin(alt) + cos(alt)·(dy·cos(az) − dx·sin(az))) / sqrt(1 + dx² + dy²)
//
// with the trigonometry of az and alt computed once per call in float64.
// Values below 0 (cells facing away from the light) become 0; rounding
// can push a cell facing the light squarely just above 255, which is
// clamped to 255. NaN stays NaN. A flat cell is 255·sin(alt). See the
// package documentation for edges and validity. dst and dem must have the
// same dimensions and must not overlap; their strides may differ.
//
// This is gdaldem hillshade's default algorithm (Horn, not -combined,
// -multidirectional or -igor) up to its output encoding: gdaldem reserves
// byte 0 for nodata and writes 1 + 254·max(0, cos), rounded, where
// Hillshade writes the unrounded 255·max(0, cos), because validity here
// lives in the mask. round(1 + v·254/255) converts a result v to
// gdaldem's byte. gdaldem also evaluates the square root with an
// approximation, so its values can differ slightly beyond encoding.
func Hillshade(dst, dem raster.Float32Raster, opts HillshadeOptions) {
	run(hillshadeOp(opts), dem, dst)
}

// HillshadeTiled is Hillshade run by the engine: it takes the same operands, applies
// the same checks and writes the same bits for every engine.Options, and
// returns ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation.
func HillshadeTiled(ctx context.Context, dst, dem raster.Float32Raster, opts HillshadeOptions, eopts engine.Options) error {
	return runTiled(ctx, eopts, hillshadeOp(opts), dem, dst)
}

// HillshadeChunked is Hillshade run by the engine over a source and sinks with
// bounded memory: it reads the DEM and writes the result a tile at a
// time, with Workers × tile buffers in memory. It writes the bits
// Hillshade would write into in-memory rasters, for every engine.Options.
// See package engine for sources, sinks, memory, cancellation and errors.
func HillshadeChunked(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, opts HillshadeOptions, eopts engine.Options) error {
	return runChunked(ctx, eopts, hillshadeOp(opts), dem, dst)
}

// newHillshadeKernel resolves and checks opts for Hillshade's kernel.
func newHillshadeKernel(opts HillshadeOptions) hillshadeKernel {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	c, bx, by := hillshadeLight(opts.Azimuth, opts.Altitude)
	return hillshadeKernel{horn{kx: kx, ky: ky}, c, bx, by}
}

// hillshadeLight resolves and checks the light's direction, and returns
// the kernels' constants for it.
func hillshadeLight(azimuth, altitude float64) (c, bx, by float32) {
	az, alt := azimuth, altitude
	if math.IsNaN(az) || math.IsInf(az, 0) {
		panic(fmt.Sprintf("terrain: Azimuth must be finite, got %v", az))
	}
	if az == 0 {
		az = 315
	}
	if alt == 0 {
		alt = 45
	}
	if !(alt > 0 && alt <= 90) {
		panic(fmt.Sprintf("terrain: Altitude must be in (0, 90] (or 0 for 45), got %v", altitude))
	}
	// Reduce the azimuth first: math.Mod is exact, and a huge azimuth in
	// radians would overflow to Inf, making every cell NaN.
	az = math.Mod(az, 360)
	az, alt = az*math.Pi/180, alt*math.Pi/180
	// The kernel computes (c + bx·dx + by·dy) / sqrt(1 + dx² + dy²).
	c = float32(255 * math.Sin(alt))
	bx = float32(-255 * math.Cos(alt) * math.Sin(az))
	by = float32(255 * math.Cos(alt) * math.Cos(az))
	return c, bx, by
}

type hillshadeKernel struct {
	horn
	c, bx, by float32
}

func (k hillshadeKernel) Process(dst exec.Span, src exec.Window) {
	out, dem := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		stencil.HornHillshadeRow(out.Row(y), dem.Row(y), dem.Row(y+1), dem.Row(y+2), k.kx, k.ky, k.c, k.bx, k.by)
	}
}
