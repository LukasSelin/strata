package simdbackend

import "math"

func scalarAdd(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scalarClamp(dst, src []float32, lo, hi float32) {
	for i, v := range src {
		dst[i] = min(max(v, lo), hi)
	}
}

// scalarSlopeRow computes the Horn slope magnitude for one output row:
// dst[j] is centred on mid[j+1], with up/down the rows above and below, so
// len(up) == len(mid) == len(down) == len(dst)+2. invDx8 and invDy8 are
// 1/(8·cellsize). The evaluation order (including f+f rather than 2*f, and
// the float32 conversions that forbid FMA fusion) is fixed so every SIMD
// variant can match it bit for bit.
func scalarSlopeRow(dst, up, mid, down []float32, invDx8, invDy8 float32) {
	for j := range dst {
		a, b, c := up[j], up[j+1], up[j+2]
		d, f := mid[j], mid[j+2]
		g, h, i := down[j], down[j+1], down[j+2]
		gx := float32(((c+(f+f))+i)-((a+(d+d))+g)) * invDx8
		gy := float32(((g+(h+h))+i)-((a+(b+b))+c)) * invDy8
		dst[j] = sqrt32(float32(gx*gx) + float32(gy*gy))
	}
}

func sqrt32(v float32) float32 { return float32(math.Sqrt(float64(v))) }
