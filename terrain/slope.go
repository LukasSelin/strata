package terrain

import (
	"fmt"
	"math"

	"strata/internal/stencil"
	"strata/raster"
)

// SlopeUnits selects how Slope expresses steepness.
type SlopeUnits int

const (
	// SlopeDegrees is the slope angle in degrees, 0 (flat) to 90.
	SlopeDegrees SlopeUnits = iota
	// SlopePercent is 100 × rise / run, so 45° is 100.
	SlopePercent
	// SlopeRadians is the slope angle in radians, 0 to π/2.
	SlopeRadians
)

// SlopeOptions configures Slope.
type SlopeOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
	// Units of the result. The zero value is SlopeDegrees.
	Units SlopeUnits
}

// Slope computes the steepness of dem from the magnitude of its Horn
// gradient, m = sqrt(dx² + dy²) with dx and dy as in Gradient: atan(m) in
// degrees or radians, or 100·m in percent. See the package documentation
// for edges and validity. dst and dem must have the same dimensions and
// must not overlap; their strides may differ.
//
// Angles use a float32 arctangent (Cephes atanf) that the scalar and SIMD
// kernels share bit-for-bit. Checked over every float32 input, it is
// within 1.41e-7 radians (8.1e-6 degrees) of math.Atan.
func Slope(dst, dem raster.Float32Raster, opts SlopeOptions) {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	var scale float32
	atan := true
	switch opts.Units {
	case SlopeDegrees:
		scale = float32(180 / math.Pi)
	case SlopeRadians:
		scale = 1
	case SlopePercent:
		scale, atan = 100, false
	default:
		panic(fmt.Sprintf("terrain: unknown SlopeUnits %d", opts.Units))
	}
	checkStencil([]string{"dem", "dst"}, dem, dst)
	forInterior(dem, []raster.Float32Raster{dst}, func(out [][]float32, r0, r1, r2 []float32) {
		stencil.HornSlopeRow(out[0], r0, r1, r2, kx, ky, scale, atan)
	})
	finishBorder(dst, dem)
}
