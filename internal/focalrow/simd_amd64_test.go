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
