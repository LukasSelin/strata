package terrain

import (
	"strata/internal/stencil"
	"strata/raster"
)

// AspectFlat is the aspect Aspect writes for flat cells unless
// AspectOptions.ZeroForFlat is set.
const AspectFlat = -1

// AspectOptions configures Aspect.
type AspectOptions struct {
	// CellSize is the ground width of a cell, in the same units as the
	// elevations (after ZFactor). It must be positive.
	CellSize float64
	// CellSizeY is the ground height of a cell. 0 means CellSize.
	CellSizeY float64
	// ZFactor multiplies elevations. 0 means 1.
	ZFactor float64
	// ZeroForFlat writes 0 instead of AspectFlat (-1) for flat cells, like
	// gdaldem aspect -zero_for_flat.
	ZeroForFlat bool
	// Trigonometric measures the downslope direction counterclockwise from
	// east (0 = east, 90 = north) instead of clockwise from north, like
	// gdaldem aspect -trigonometric.
	Trigonometric bool
}

// Aspect computes the downslope direction of dem from its Horn gradient
// (dx and dy as in Gradient), in degrees in [0, 360): by default a
// compass bearing, clockwise from north, so 0 is north, 90 east, 180
// south and 270 west, as gdaldem aspect writes. On a north-up grid the
// downslope direction is (-dx, dy) in (east, north) components, so the
// bearing is atan2(-dx, dy) and the trigonometric angle atan2(dy, -dx),
// folded from (-180, 0) into (180, 360) and with a result that rounds to
// 360 written as 0. See the package documentation for edges and validity.
// dst and dem must have the same dimensions and must not overlap; their
// strides may differ.
//
// A cell is flat when dx and dy are both zero, and gets AspectFlat (-1),
// or 0 with ZeroForFlat, and stays valid. gdaldem writes its nodata value
// (-9999) there instead, but validity here lives only in the mask and
// flat cells are valid data. -1 is Esri's flat aspect: it lies outside
// [0, 360) so that v < 0 finds flat cells, and unlike NaN it cannot be
// confused with the result of a NaN or infinite elevation.
//
// Unlike gdaldem, which differences raw elevations, Aspect divides by the
// cell sizes, so directions on non-square cells are those of the true
// gradient; for square cells the two agree. ZFactor only matters through
// its sign and through overflow or underflow of dx and dy.
//
// Angles use a float32 arctangent (stencil.Atan2F32, built on the Cephes
// atanf that Slope uses) that the scalar and SIMD kernels share
// bit-for-bit. Sampled over 4e7 argument pairs it is within 2.7e-7
// radians (1.6e-5 degrees) of math.Atan2. Converting to degrees and
// folding round to float32 again, which near 360 is worth up to 1.5e-5
// degrees more; against a float64 evaluation of the same gradients the
// largest difference seen in tests is 2.4e-5 degrees.
func Aspect(dst, dem raster.Float32Raster, opts AspectOptions) {
	kx, ky := cellSizes(opts.CellSize, opts.CellSizeY, opts.ZFactor)
	flat := float32(AspectFlat)
	if opts.ZeroForFlat {
		flat = 0
	}
	checkStencil([]string{"dem", "dst"}, dem, dst)
	forInterior(dem, []raster.Float32Raster{dst}, func(out [][]float32, r0, r1, r2 []float32) {
		stencil.HornAspectRow(out[0], r0, r1, r2, kx, ky, flat, opts.Trigonometric)
	})
	finishBorder(dst, dem)
}
