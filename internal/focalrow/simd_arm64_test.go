//go:build goexperiment.simd && arm64

package focalrow

import "testing"

// simdTestName is what simd_test.go expects Backend to report.
const simdTestName = "neon"

// requireSIMD never skips: NEON is part of the arm64 baseline.
func requireSIMD(testing.TB) {}
