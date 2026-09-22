package resamprow

import (
	"math"
	"testing"
)

// TestPassesAreNotFused pins that both passes round each product before
// adding it (§15, ADR 0001): with w = [-1, 1+2⁻²³] over x = [1+2⁻²²,
// 1+2⁻²³], the rounded products cancel exactly, while a fused multiply-add
// keeps the 2⁻⁴⁶ the rounding drops. The fixture first proves it tells
// the two apart.
func TestPassesAreNotFused(t *testing.T) {
	x0 := float32(1 + 0x1p-22)
	x1 := float32(1 + 0x1p-23)
	w1 := float32(1 + 0x1p-23)
	fused := float32(math.FMA(float64(w1), float64(x1), -float64(x0)))
	if fused == 0 {
		t.Fatal("fixture does not distinguish fused from unfused")
	}
	a := &Axis{Lo: 0, Hi: 1, First: []int32{0}, Taps: []int32{2}, Off: []int32{0}, W: []float32{-1, w1}}
	check := func(backend string) {
		for _, rows := range []int{1, 4, 8, 11} {
			src := make([]float32, rows*2)
			for j := range rows {
				src[2*j], src[2*j+1] = x0, x1
			}
			tt := make([]float32, rows)
			HRows(tt, 1, src, 2, rows, a, 0, 1, 0, make([]float32, HScratch(2)))
			for j, v := range tt {
				if v != 0 {
					t.Errorf("%s HRows rows=%d: row %d = %g, want 0 (fused gives %g)", backend, rows, j, v, fused)
				}
			}
		}
		for _, n := range []int{1, 4, 8, 13} {
			rows := make([]float32, 2*n)
			for c := range n {
				rows[c], rows[n+c] = x0, x1
			}
			d := make([]float32, n)
			VRow(d, rows, n, []float32{-1, w1})
			for c, v := range d {
				if v != 0 {
					t.Errorf("%s VRow n=%d: column %d = %g, want 0 (fused gives %g)", backend, n, c, v, fused)
				}
			}
		}
	}
	defer UseScalar(false)
	UseScalar(true)
	check("scalar")
	UseScalar(false)
	check(Backend())
}
