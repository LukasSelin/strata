package stencil

// This file holds the scalar kernels that finish a Horn product from a
// gradient already computed: slope, aspect and hillshade from the dx and
// dy of HornGradientRow. They are the fused row kernels' second half.
// Each fused kernel computes gx and gy exactly as HornGradientRow does,
// then hands them to the same per-cell function these call (magnitude,
// aspectArgs and aspectDegrees, shade), so a gradient row followed by
// one of these is bit for bit the fused kernel. That is what lets
// terrain compute several products from one gradient (DESIGN.md §52).

// SlopeFromGradientRow writes scale·m, or scale·atan(m) when atan is
// set, for m = sqrt(gx² + gy²), with gx and gy the dx and dy
// HornGradientRow wrote for the same cells: HornSlopeRow's result, from
// the gradient rather than the elevations. gx, gy and dst must have the
// same length.
func SlopeFromGradientRow(dst, gx, gy []float32, scale float32, atan bool) {
	requireGradient(len(dst), gx, gy)
	slopeFromGradientRow(dst, gx, gy, scale, atan)
}

// AspectFromGradientRow is HornAspectRow's result from the gradient gx,
// gy that HornGradientRow wrote for the same cells.
func AspectFromGradientRow(dst, gx, gy []float32, flat float32, trig bool) {
	requireGradient(len(dst), gx, gy)
	aspectFromGradientRow(dst, gx, gy, flat, trig)
}

// HillshadeFromGradientRow is HornHillshadeRow's result from the
// gradient gx, gy that HornGradientRow wrote for the same cells.
func HillshadeFromGradientRow(dst, gx, gy []float32, c, bx, by float32) {
	requireGradient(len(dst), gx, gy)
	hillshadeFromGradientRow(dst, gx, gy, c, bx, by)
}

func requireGradient(n int, gx, gy []float32) {
	if len(gx) != n || len(gy) != n {
		panic("stencil: gx, gy and dst must have equal length")
	}
}

// scalarSlopeFromGradientRow makes two passes when atan is set, as
// scalarHornSlopeRow does and for the same reason.
func scalarSlopeFromGradientRow(dst, gx, gy []float32, scale float32, atan bool) {
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	if atan {
		for i := range dst {
			dst[i] = magnitude(gx[i], gy[i])
		}
		for i, m := range dst {
			dst[i] = float32(Atan32(m) * scale)
		}
		return
	}
	for i := range dst {
		dst[i] = float32(magnitude(gx[i], gy[i]) * scale)
	}
}

func scalarAspectFromGradientRow(dst, gx, gy []float32, flat float32, trig bool) {
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for i := range dst {
		y, x := aspectArgs(gx[i], gy[i], trig)
		dst[i] = aspectDegrees(y, x, flat)
	}
}

func scalarHillshadeFromGradientRow(dst, gx, gy []float32, c, bx, by float32) {
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for i := range dst {
		dst[i] = shade(gx[i], gy[i], c, bx, by)
	}
}
