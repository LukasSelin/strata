//go:build goexperiment.simd && amd64

package stencil

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

// atanLanes and atan2Lanes run atan8 and atan2_8 over whole lanes of xs
// (and ys), for simd_test.go's lane-by-lane comparisons.
func atanLanes(dst, xs []float32) {
	c := newLaneConsts()
	for i := 0; i+lane <= len(xs); i += lane {
		store8(atan8(load8(xs[i:]), &c), dst[i:])
	}
	archsimd.ClearAVXUpperBits()
}

func atan2Lanes(dst, ys, xs []float32) {
	c := newLaneConsts()
	for i := 0; i+lane <= len(ys); i += lane {
		store8(atan2_8(load8(ys[i:]), load8(xs[i:]), &c), dst[i:])
	}
	archsimd.ClearAVXUpperBits()
}
