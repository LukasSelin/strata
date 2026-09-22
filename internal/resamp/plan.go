// Package resamp drives package resamprow's tables and row kernels over
// a raster band (DESIGN.md §54): it owns the plan of a whole resampling,
// the footprint a band reads, and the validity and NoData rules that
// turn a masked source into masked output.
//
// It is deliberately not a kernel package (§12, §14): it takes rasters
// and a mutex, where a kernel takes spans and scalars. Everything below
// that border lives in resamprow.
package resamp

import (
	"fmt"

	"github.com/LukasSelin/strata/internal/resamprow"
)

// The kernel package's table types and backend switch, re-exported so
// that a caller of resamp needs only the one import.
type (
	// Method is the resampling method.
	Method = resamprow.Method
	// Spec is one axis of a resampling.
	Spec = resamprow.Spec
	// Axis is one axis's tap table.
	Axis = resamprow.Axis
)

const (
	Nearest  = resamprow.Nearest
	Bilinear = resamprow.Bilinear
	Cubic    = resamprow.Cubic
	Lanczos  = resamprow.Lanczos
	Average  = resamprow.Average
)

// Backend names the row kernels in use, and UseScalar forces the scalar
// ones. Both are resamprow's; they are here so that a caller switching
// backends need not import it.
func Backend() string       { return resamprow.Backend() }
func UseScalar(scalar bool) { resamprow.UseScalar(scalar) }

// Plan is a whole resampling: both axes and the method's rules.
type Plan struct {
	Method Method
	resamprow.Axes
	// Cubic4 is gdalwarp's four-sample cubic, used when neither axis
	// widens: an output cell whose 4×4 taps lose one to the source edge
	// (Clipped) or, with a mask, include an invalid cell takes bilinear
	// instead, renormalised over its valid cells. BX and BY are the
	// bilinear tables for those cells.
	Cubic4 bool
	BX, BY Axis
	// ClippedX lists the output columns X.Clipped marks, so a band visits
	// only them.
	ClippedX []int32
	// HalfValid is gdalwarp's rule for Lanczos over a masked source: a
	// cell also needs at least half of the WinN[x]×WinN[y] source cells
	// its kernel reaches to be valid, unless its centre lies exactly on a
	// source centre on both axes (Exact), where the kernel copies that
	// cell. Cells outside the source do not count (measured, DESIGN.md
	// §54).
	HalfValid bool
}

// NewPlan builds the plan of a resampling from its two axes.
func NewPlan(m Method, x, y Spec) *Plan {
	if m > Average {
		panic(fmt.Sprintf("resamp: unknown %v", m))
	}
	p := &Plan{Method: m, Axes: resamprow.NewAxes(m, x, y)}
	if m == Cubic && !p.X.Widened && !p.Y.Widened {
		p.Cubic4 = true
		p.BX = resamprow.NewBilinear4(x)
		p.BY = resamprow.NewBilinear4(y)
		for c, v := range p.X.Clipped {
			if v {
				p.ClippedX = append(p.ClippedX, int32(c))
			}
		}
	}
	p.HalfValid = m == Lanczos
	return p
}

// Covered reports whether output cell (c, r) is covered on both axes.
func (p *Plan) Covered(c, r int) bool {
	return c >= p.X.Lo && c < p.X.Hi && r >= p.Y.Lo && r < p.Y.Hi
}

// Footprint returns the source window [fx0, fx1) × [fy0, fy1) that output
// cells [x0, x1) × [y0, y1) read, empty if none of them is covered. It is
// the axes' own footprint under this plan's half-valid rule.
func (p *Plan) Footprint(x0, y0, x1, y1 int) (fx0, fy0, fx1, fy1 int) {
	return p.Axes.Footprint(x0, y0, x1, y1, p.HalfValid)
}
