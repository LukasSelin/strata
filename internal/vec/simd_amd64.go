//go:build goexperiment.simd && amd64

package vec

import "simd/archsimd"

// This file is the AVX2 backend, written with Go's simd/archsimd package
// (see docs/adr/0001-simd-backend.md). It only exists in builds with
// GOEXPERIMENT=simd; other builds keep the scalar function variables from
// dispatch.go.
//
// Every kernel must agree bit-for-bit with its scalar counterpart (any NaN
// matches any NaN). The loops follow the same shape:
//
//   - fixed-size array-pointer loads over slices that shrink by one lane
//     per iteration, which lets the compiler drop per-load bounds checks;
//   - archsimd.ClearAVXUpperBits (VZEROUPPER) before the scalar tail, since
//     the compiler does not emit it and the tail may use legacy SSE.

// avxLane is the number of float32 lanes in a YMM register.
const avxLane = 8

func init() {
	// AVX2, not just AVX: some archsimd "emulated" ops (Abs, IfElse,
	// Broadcast) need AVX2 instructions.
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{
		add:       addFloat32AVX2,
		sub:       subFloat32AVX2,
		mul:       mulFloat32AVX2,
		div:       divFloat32AVX2,
		addScalar: addScalarFloat32AVX2,
		mulScalar: mulScalarFloat32AVX2,
		min:       minFloat32AVX2,
		max:       maxFloat32AVX2,
		clamp:     clampFloat32AVX2,
		abs:       absFloat32AVX2,
		sqrt:      sqrtFloat32AVX2,
	}
	UseScalar(false)
}

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[avxLane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[avxLane]float32)(s))
}

// min8 and max8 reproduce Go's builtin min and max lanewise. VMINPS and
// VMAXPS return the second operand when the operands compare equal or are
// unordered, which gets two cases wrong: a NaN first operand, and
// min(-0, +0) / max(+0, -0). Equal lanes therefore take the bitwise OR (min)
// or AND (max) of both operands, which only differs from either operand for
// signed zeros, and NaN lanes in x are restored from x.
func min8(x, y archsimd.Float32x8) archsimd.Float32x8 {
	r := x.ToBits().Or(y.ToBits()).BitsToFloat32().IfElse(x.Equal(y), x.Min(y))
	return x.IfElse(x.IsNaN(), r)
}

func max8(x, y archsimd.Float32x8) archsimd.Float32x8 {
	r := x.ToBits().And(y.ToBits()).BitsToFloat32().IfElse(x.Equal(y), x.Max(y))
	return x.IfElse(x.IsNaN(), r)
}

func addFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(load8(a).Add(load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarAddFloat32(dst, a, b)
}

func subFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(load8(a).Sub(load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarSubFloat32(dst, a, b)
}

func mulFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(load8(a).Mul(load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarMulFloat32(dst, a, b)
}

func divFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(load8(a).Div(load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarDivFloat32(dst, a, b)
}

func minFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(min8(load8(a), load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarMinFloat32(dst, a, b)
}

func maxFloat32AVX2(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= avxLane && len(a) >= avxLane && len(b) >= avxLane {
		store8(max8(load8(a), load8(b)), dst)
		dst, a, b = dst[avxLane:], a[avxLane:], b[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarMaxFloat32(dst, a, b)
}

func addScalarFloat32AVX2(dst, src []float32, value float32) {
	v := archsimd.BroadcastFloat32x8(value)
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Add(v), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarAddScalarFloat32(dst, src, value)
}

func mulScalarFloat32AVX2(dst, src []float32, value float32) {
	v := archsimd.BroadcastFloat32x8(value)
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Mul(v), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarMulScalarFloat32(dst, src, value)
}

func clampFloat32AVX2(dst, src []float32, lo, hi float32) {
	vlo := archsimd.BroadcastFloat32x8(lo)
	vhi := archsimd.BroadcastFloat32x8(hi)
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(min8(max8(load8(src), vlo), vhi), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarClampFloat32(dst, src, lo, hi)
}

func absFloat32AVX2(dst, src []float32) {
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Abs(), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarAbsFloat32(dst, src)
}

func sqrtFloat32AVX2(dst, src []float32) {
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Sqrt(), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarSqrtFloat32(dst, src)
}
