//go:build goexperiment.simd && amd64

package vec

import (
	"simd/archsimd"
	"testing"
)

// simdTestName and simdTestKernels are what simd_test.go expects this
// architecture's SIMD backend to be.
const simdTestName = "avx2"

var simdTestKernels = kernelSet{
	add: addFloat32AVX2, sub: subFloat32AVX2, mul: mulFloat32AVX2, div: divFloat32AVX2,
	addScalar: addScalarFloat32AVX2, mulScalar: mulScalarFloat32AVX2,
	affine: affineFloat32AVX2, subDiv: subDivFloat32AVX2,
	min: minFloat32AVX2, max: maxFloat32AVX2, clamp: clampFloat32AVX2,
	abs: absFloat32AVX2, sqrt: sqrtFloat32AVX2,
	reduceMin: reduceMinFloat32AVX2, reduceMax: reduceMaxFloat32AVX2,
}

func requireSIMD(tb testing.TB) {
	tb.Helper()
	if !archsimd.X86.AVX2() {
		tb.Skip("AVX2 not available on this CPU")
	}
}
