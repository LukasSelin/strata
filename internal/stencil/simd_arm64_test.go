//go:build goexperiment.simd && arm64

package stencil

import "testing"

// simdTestName is what simd_test.go expects Backend to report.
const simdTestName = "neon"

// requireSIMD never skips: NEON is part of the arm64 baseline.
func requireSIMD(testing.TB) {}

// atanLanes and atan2Lanes run atan4 and atan2_4 over whole lanes of xs
// (and ys), for simd_test.go's lane-by-lane comparisons.
func atanLanes(dst, xs []float32) {
	c := newLaneConsts()
	for i := 0; i+lane <= len(xs); i += lane {
		store4(atan4(load4(xs[i:]), &c), dst[i:])
	}
}

func atan2Lanes(dst, ys, xs []float32) {
	c := newLaneConsts()
	for i := 0; i+lane <= len(ys); i += lane {
		store4(atan2_4(load4(ys[i:]), load4(xs[i:]), &c), dst[i:])
	}
}
