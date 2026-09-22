// Package vec provides flat-slice numeric kernels for float32 data.
//
// This file holds the scalar backend: the correctness reference and
// fallback implementation for every kernel. Every SIMD backend added
// later must produce identical results to these functions, including
// on NaN, infinities, and zero-length or odd-length inputs.
//
// Each kernel first reslices its operands to the length of the slice it
// ranges over. The exported functions have already checked the lengths
// are equal; the reslice proves it to the compiler, which then drops the
// bounds check on every element (TestNoBoundsChecksInLoops).
package vec

import "math"

func scalarAddFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scalarSubFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func scalarMulFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func scalarDivFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scalarAddScalarFloat32(dst, src []float32, value float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = v + value
	}
}

func scalarMulScalarFloat32(dst, src []float32, value float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = v * value
	}
}

// scalarAffineFloat32 computes dst[i] = a*src[i] + b, the kernel of
// transfer.Rescale. The product is wrapped in an explicit float32
// conversion, which stops the compiler fusing the multiply and the add
// into one FMA: arm64 emits FMADD for a*v + b and amd64 does not, so
// without the conversion the canonical scalar result would differ
// between architectures, and the AVX2 kernel (a separate VMULPS and
// VADDPS) could not match it either (docs/adr/0001-simd-backend.md).
func scalarAffineFloat32(dst, src []float32, a, b float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = float32(v*a) + b
	}
}

// scalarSubDivFloat32 computes dst[i] = (src[i] - lo) / span, the kernel
// of algebra.Normalize. Subtracting before dividing, rather than
// multiplying by a reciprocal, is what makes src == lo give exactly 0 and
// src == lo + span exactly 1.
func scalarSubDivFloat32(dst, src []float32, lo, span float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = (v - lo) / span
	}
}

func scalarMinFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = min(a[i], b[i])
	}
}

func scalarMaxFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

func scalarClampFloat32(dst, src []float32, lo, hi float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = min(max(v, lo), hi)
	}
}

// scalarAbsFloat32 clears the sign bit directly rather than branching on
// v < 0, so that -0 maps to +0 and NaN payloads are preserved unchanged.
func scalarAbsFloat32(dst, src []float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
	}
}

func scalarSqrtFloat32(dst, src []float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = float32(math.Sqrt(float64(v)))
	}
}

// scalarReduceMinFloat32 and scalarReduceMaxFloat32 fold src into acc
// with Go's builtin min and max, which make NaN absorbing and order -0
// below +0. Both are associative and commutative under those semantics,
// so a vector backend may fold its lanes in any order and still agree
// bit for bit (which NaN's payload survives aside, as everywhere in this
// package). An empty src returns acc, so acc is the identity across
// calls and a caller folds a whole band without an empty-slice case.
func scalarReduceMinFloat32(acc float32, src []float32) float32 {
	for _, v := range src {
		acc = min(acc, v)
	}
	return acc
}

func scalarReduceMaxFloat32(acc float32, src []float32) float32 {
	for _, v := range src {
		acc = max(acc, v)
	}
	return acc
}
