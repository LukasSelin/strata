package overlap

import (
	"testing"

	"strata/raster"
)

func row(valid []uint64, off int) raster.Float32Raster {
	return raster.Float32Raster{Data: make([]float32, 10), Width: 10, Height: 1, Stride: 10, Valid: valid, ValidOffset: off}
}

// TestReslicedMasks checks masks that are different slices of one array:
// bit 64 of m is bit 0 of m[1:].
func TestReslicedMasks(t *testing.T) {
	m := make([]uint64, 4)
	for _, tc := range []struct {
		a, b        raster.Float32Raster
		bits        Relation
		spans, word bool
	}{
		{row(m, 64), row(m[1:], 0), Same, true, true},
		{row(m, 70), row(m[1:], 0), Partial, true, true},
		{row(m, 58), row(m[1:], 8), Disjoint, false, true},
		{row(m, 10), row(m[1:], 0), Disjoint, false, false},
		{row(m, 0), row(m[2:], 0), Disjoint, false, false},
		{row(m[3:], 10), row(m[2:], 74), Same, true, true},
		{row(m, 0), row(nil, 0), Disjoint, false, false},
	} {
		if got := Bits(tc.a, tc.b); got != tc.bits {
			t.Errorf("Bits(%d, %d) = %v, want %v", tc.a.ValidOffset, tc.b.ValidOffset, got, tc.bits)
		}
		if got := BitSpans(tc.a, tc.b); got != tc.spans {
			t.Errorf("BitSpans(%d, %d) = %v, want %v", tc.a.ValidOffset, tc.b.ValidOffset, got, tc.spans)
		}
		if got := Words(tc.a, tc.b); got != tc.word {
			t.Errorf("Words(%d, %d) = %v, want %v", tc.a.ValidOffset, tc.b.ValidOffset, got, tc.word)
		}
	}
}
