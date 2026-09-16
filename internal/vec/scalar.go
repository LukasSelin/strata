// Package vec provides flat-slice numeric kernels for float32 data.
//
// This file holds the scalar backend: the correctness reference and
// fallback implementation for every kernel. Every SIMD backend added
// later must produce identical results to these functions, including
// on NaN, infinities, and zero-length or odd-length inputs.
package vec

import "math"

func scalarAddFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scalarSubFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func scalarMulFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func scalarDivFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scalarAddScalarFloat32(dst, src []float32, value float32) {
	for i, v := range src {
		dst[i] = v + value
	}
}

func scalarMulScalarFloat32(dst, src []float32, value float32) {
	for i, v := range src {
		dst[i] = v * value
	}
}

func scalarMinFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = min(a[i], b[i])
	}
}

func scalarMaxFloat32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

func scalarClampFloat32(dst, src []float32, lo, hi float32) {
	for i, v := range src {
		dst[i] = min(max(v, lo), hi)
	}
}

// scalarAbsFloat32 clears the sign bit directly rather than branching on
// v < 0, so that -0 maps to +0 and NaN payloads are preserved unchanged.
func scalarAbsFloat32(dst, src []float32) {
	for i, v := range src {
		dst[i] = math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
	}
}

func scalarSqrtFloat32(dst, src []float32) {
	for i, v := range src {
		dst[i] = float32(math.Sqrt(float64(v)))
	}
}
