//go:build goexperiment.simd && amd64

package focalrow

import (
	"simd/archsimd"
	"testing"
)

// simdTestName is what simd_test.go expects Backend to report.
const simdTestName = "avx2"

func requireSIMD(tb testing.TB) {
	tb.Helper()
	if !archsimd.X86.AVX2() {
		tb.Skip("AVX2 not available on this CPU")
	}
}

// TestCollidingRows pins foldColumn's test to the strides measured on the
// Zen 2 desktop: those that slowed the column pass fold (more than seven
// colliding rows), those that did not stay in one pass.
func TestCollidingRows(t *testing.T) {
	for _, c := range []struct{ stride, k, want int }{
		{16384, 9, 9}, {16384, 17, 17}, {32768, 11, 11}, {49152, 11, 11}, // one set: slow
		{16400, 9, 9}, {16400, 17, 16}, // a line apart: slow
		{16416, 17, 8},                 // two lines apart: 15% slower at 17 rows
		{24576, 17, 9}, {24576, 11, 6}, // two sets: slow at 17 rows only
		{16320, 17, 4}, {16448, 17, 4}, {16512, 17, 2}, {20480, 17, 5}, // fast
		{4096, 17, 5},              // 16 KiB apart: four sets
		{16, 17, 1}, {1000, 17, 1}, // contiguous: nothing aliases
	} {
		if got := collidingRows(c.stride, c.k); got != c.want {
			t.Errorf("collidingRows(%d, %d) = %d, want %d", c.stride, c.k, got, c.want)
		}
	}
}
