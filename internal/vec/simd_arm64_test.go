//go:build goexperiment.simd && arm64

package vec

import "testing"

// simdTestName and simdTestKernels are what simd_test.go expects this
// architecture's SIMD backend to be.
const simdTestName = "neon"

var simdTestKernels = kernelSet{
	add: addFloat32NEON, sub: subFloat32NEON, mul: mulFloat32NEON, div: divFloat32NEON,
	addScalar: addScalarFloat32NEON, mulScalar: mulScalarFloat32NEON,
	affine: affineFloat32NEON, subDiv: subDivFloat32NEON,
	min: minFloat32NEON, max: maxFloat32NEON, clamp: clampFloat32NEON,
	abs: absFloat32NEON, sqrt: sqrtFloat32NEON,
	reduceMin: reduceMinFloat32NEON, reduceMax: reduceMaxFloat32NEON,
	chain: chainFloat32NEON, validBits: validBitsNEON,
}

// requireSIMD never skips: NEON is part of the arm64 baseline.
func requireSIMD(testing.TB) {}
