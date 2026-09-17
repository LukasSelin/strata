package terrain

import (
	"fmt"
	"math"

	"strata/internal/stencil"
	"strata/raster"
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
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	az, alt := opts.Azimuth, opts.Altitude
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
		panic(fmt.Sprintf("terrain: Altitude must be in (0, 90] (or 0 for 45), got %v", opts.Altitude))
	}
	az, alt = az*math.Pi/180, alt*math.Pi/180
	// The kernel computes (c + bx·dx + by·dy) / sqrt(1 + dx² + dy²).
	c := float32(255 * math.Sin(alt))
	bx := float32(-255 * math.Cos(alt) * math.Sin(az))
	by := float32(255 * math.Cos(alt) * math.Cos(az))
	checkStencil([]string{"dem", "dst"}, dem, dst)
	forInterior(dem, []raster.Float32Raster{dst}, func(out [][]float32, r0, r1, r2 []float32) {
		stencil.HornHillshadeRow(out[0], r0, r1, r2, kx, ky, c, bx, by)
	})
	finishBorder(dst, dem)
}
