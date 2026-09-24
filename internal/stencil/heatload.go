package stencil

import "math"

// HeatLoadTerms are the constants of HeatLoadFromGradientRow: one of
// McCune and Keon's (2002) regressions for a latitude and a fold of the
// aspect, rewritten as a function of the gradient (see HeatLoadFromGradientRow).
type HeatLoadTerms struct {
	// K0 is the regression's constant, and A the coefficient of cos(S).
	K0, A float64
	// BX and BY multiply dx and dy in the cos(A')·sin(S) term, E the
	// gradient's magnitude in the sin(S) term, and F the folded
	// sin(A')·sin(S) term |CX·dx + CY·dy|.
	BX, BY, E, F float64
	CX, CY       float64
	// Exp returns exp of the regression, for the equations fitted to
	// ln(radiation).
	Exp bool
}

// HeatLoadFromGradientRow writes one of McCune and Keon's regressions
// for potential direct incident radiation or heat load, from the
// gradient gx, gy that HornGradientRow (or a fit) wrote for the same
// cells. With w = 1/sqrt(1 + gx² + gy²), the cosine of the slope, each
// cell is
//
//	v = K0 + w·(A + BX·gx + BY·gy + E·sqrt(gx² + gy²) + F·|CX·gx + CY·gy|)
//
// and exp(v) when Exp is set. It is evaluated in float64, every product
// rounded before it is added so that no multiply-add is fused, and
// rounded to float32 once. gx, gy and dst must have the same length. It
// is scalar on every build.
func HeatLoadFromGradientRow(dst, gx, gy []float32, t HeatLoadTerms) {
	requireGradient(len(dst), gx, gy)
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for i := range dst {
		dst[i] = heatLoad(gx[i], gy[i], &t)
	}
}

func heatLoad(gx, gy float32, t *HeatLoadTerms) float32 {
	dx, dy := float64(gx), float64(gy)
	g2 := float64(dx*dx) + float64(dy*dy)
	w := 1 / math.Sqrt(1+g2)
	fold := math.Abs(float64(t.CX*dx) + float64(t.CY*dy))
	s := t.A + float64(t.BX*dx) + float64(t.BY*dy) + float64(t.E*math.Sqrt(g2)) + float64(t.F*fold)
	v := t.K0 + float64(w*s)
	if t.Exp {
		v = math.Exp(v)
	}
	return float32(v)
}
