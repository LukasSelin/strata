package resamprow

// The canonical kernels (DESIGN.md §15). Every product is rounded to
// float32 by an explicit conversion before it is added, so no compiler
// fuses a multiply-add, and every sum runs over the taps in increasing
// source index, starting from the first product rather than from zero
// (0 + -0 would be +0). The SIMD kernels perform the same operations in
// the same order in every lane.

// scalarHRows is the horizontal pass: for each of rows source rows j and
// output columns c in [c0, c1),
//
//	t[j*tStride + c-c0] = Σ_k W[Off[c]+k] · src[j*sStride + First[c]-sx0 + k]
func scalarHRows(t []float32, tStride int, src []float32, sStride, rows int, a *Axis, c0, c1, sx0 int, _ []float32) {
	for j := range rows {
		row := src[j*sStride:]
		out := t[j*tStride : j*tStride+c1-c0]
		for c := range out {
			out[c] = dot(row, a, c0+c, sx0)
		}
	}
}

// dot is one output column's horizontal sum over row.
func dot(row []float32, a *Axis, c, sx0 int) float32 {
	f := int(a.First[c]) - sx0
	n := int(a.Taps[c])
	off := int(a.Off[c])
	x := row[f : f+n : f+n]
	w := a.W[off : off+n : off+n]
	acc := float32(w[0] * x[0])
	x, w = x[1:], w[1:]
	w = w[:len(x)]
	for k, v := range x {
		acc += float32(w[k] * v)
	}
	return acc
}

// scalarVRow is the vertical pass for one output row: with n = len(w)
// taps over the rows t[k*tStride:], k in [0, n),
//
//	dst[c] = Σ_k w[k] · t[k*tStride + c]
func scalarVRow(dst, t []float32, tStride int, w []float32) {
	r0 := t[:len(dst)]
	w0 := w[0]
	for c, v := range r0 {
		dst[c] = float32(w0 * v)
	}
	for k := 1; k < len(w); k++ {
		wk := w[k]
		rk := t[k*tStride : k*tStride+len(dst)]
		rk = rk[:len(dst)]
		for c, v := range rk {
			dst[c] += float32(wk * v)
		}
	}
}

// Direct2D evaluates the axes' filter cell by cell, recomputing each
// tap row's horizontal sum rather than reusing an intermediate. It
// performs exactly the separable passes' operations for every cell, so
// it gives the same bits; it is the reference the tests hold the passes
// to, and the direct contender the benchmarks time them against
// (DESIGN.md §54). It writes dst[(r-y0)*dStride + c-x0] for the covered
// cells of [x0, x1) × [y0, y1), reading src with its row y and column x
// at src[(y-sy0)*sStride + x-sx0]; it ignores validity.
func Direct2D(dst []float32, dStride int, src []float32, sStride, sx0, sy0 int, a *Axes, x0, x1, y0, y1 int) {
	ax, ay := &a.X, &a.Y
	for r := max(y0, ay.Lo); r < min(y1, ay.Hi); r++ {
		fy := int(ay.First[r]) - sy0
		wy := ay.W[ay.Off[r] : ay.Off[r]+ay.Taps[r]]
		out := dst[(r-y0)*dStride:]
		for c := max(x0, ax.Lo); c < min(x1, ax.Hi); c++ {
			acc := float32(wy[0] * dot(src[fy*sStride:], ax, c, sx0))
			for k := 1; k < len(wy); k++ {
				acc += float32(wy[k] * dot(src[(fy+k)*sStride:], ax, c, sx0))
			}
			out[c-x0] = acc
		}
	}
}
