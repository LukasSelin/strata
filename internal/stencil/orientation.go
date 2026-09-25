package stencil

import "math"

// OrientationFromGradientRow writes northness, or eastness when east is
// set, from the gradient gx, gy that HornGradientRow (or a fit) wrote for
// the same cells. The downslope direction in (east, north) components is
// (-gx, gy), so with its component c (gy for northness, -gx for eastness)
// each cell is
//
//	c / sqrt(1 + gx² + gy²)    sin(slope) times the cosine or sine of the aspect
//	c / sqrt(gx² + gy²)        with unweighted set: the cosine or sine alone
//
// and an unweighted flat cell (gx = gy = 0) is 0. It is evaluated in
// float64, where neither square can overflow or underflow, every product
// rounded before it is added so that no multiply-add is fused, and
// rounded to float32 once. gx, gy and dst must have the same length. It
// is scalar on every build.
func OrientationFromGradientRow(dst, gx, gy []float32, east, unweighted bool) {
	requireGradient(len(dst), gx, gy)
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for i := range dst {
		dst[i] = orientation(gx[i], gy[i], east, unweighted)
	}
}

func orientation(gx, gy float32, east, unweighted bool) float32 {
	dx, dy := float64(gx), float64(gy)
	c := dy
	if east {
		c = -dx
	}
	g2 := float64(dx*dx) + float64(dy*dy)
	if !unweighted {
		return float32(c / math.Sqrt(1+g2))
	}
	if g2 == 0 {
		return 0
	}
	return float32(c / math.Sqrt(g2))
}
