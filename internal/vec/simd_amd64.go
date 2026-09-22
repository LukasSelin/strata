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
		affine:    affineFloat32AVX2,
		subDiv:    subDivFloat32AVX2,
		min:       minFloat32AVX2,
		max:       maxFloat32AVX2,
		clamp:     clampFloat32AVX2,
		abs:       absFloat32AVX2,
		sqrt:      sqrtFloat32AVX2,
		reduceMin: reduceMinFloat32AVX2,
		reduceMax: reduceMaxFloat32AVX2,
	}
	simdName = "avx2"
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

// affineFloat32AVX2 multiplies and adds in two separate instructions,
// VMULPS then VADDPS, never a fused VFMADD: scalarAffineFloat32 rounds
// the product before adding, and a fused lane would not match it.
func affineFloat32AVX2(dst, src []float32, a, b float32) {
	va := archsimd.BroadcastFloat32x8(a)
	vb := archsimd.BroadcastFloat32x8(b)
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Mul(va).Add(vb), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarAffineFloat32(dst, src, a, b)
}

// subDivFloat32AVX2 is VSUBPS then VDIVPS. Both are correctly rounded,
// so each lane matches scalarSubDivFloat32.
func subDivFloat32AVX2(dst, src []float32, lo, span float32) {
	vlo := archsimd.BroadcastFloat32x8(lo)
	vspan := archsimd.BroadcastFloat32x8(span)
	src = src[:len(dst)]
	for len(dst) >= avxLane && len(src) >= avxLane {
		store8(load8(src).Sub(vlo).Div(vspan), dst)
		dst, src = dst[avxLane:], src[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	scalarSubDivFloat32(dst, src, lo, span)
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

// reduceMinFloat32AVX2 and reduceMaxFloat32AVX2 keep one accumulator
// vector, so they fold the lanes in a different order from the scalar
// loop. min8 and max8 reproduce Go's builtins, which are associative and
// commutative over NaN and signed zeros alike, so the value is the same
// either way; only which NaN payload survives may differ, as elsewhere
// in this file.
//
// The lanes array is declared by the wrapper, before the first 256-bit
// instruction, because zeroing it may emit a legacy SSE store; and the
// running accumulator stays in the wrapper, so the lane function holds no
// float32 to spill inside its loop. Both are the rules of
// internal/stencil/simd_amd64.go, for the same reason: no legacy SSE
// between the first 256-bit instruction and ClearAVXUpperBits.
func reduceMinFloat32AVX2(acc float32, src []float32) float32 {
	var lanes [avxLane]float32
	i := reduceMinLanes(&lanes, src)
	if i == 0 {
		return scalarReduceMinFloat32(acc, src)
	}
	return scalarReduceMinFloat32(scalarReduceMinFloat32(acc, lanes[:]), src[i:])
}

// reduceMinLanes folds the whole lanes of src into lanes and returns how
// many cells it consumed, or 0 when src is shorter than one lane. It
// clears the upper AVX bits before it returns, so its caller's scalar
// work pays no SSE/AVX transition.
func reduceMinLanes(lanes *[avxLane]float32, src []float32) int {
	if len(src) < avxLane {
		return 0
	}
	n := len(src)
	v := load8(src)
	src = src[avxLane:]
	for len(src) >= avxLane {
		v = min8(v, load8(src))
		src = src[avxLane:]
	}
	v.StoreArray(lanes)
	archsimd.ClearAVXUpperBits()
	return n - len(src)
}

func reduceMaxFloat32AVX2(acc float32, src []float32) float32 {
	var lanes [avxLane]float32
	i := reduceMaxLanes(&lanes, src)
	if i == 0 {
		return scalarReduceMaxFloat32(acc, src)
	}
	return scalarReduceMaxFloat32(scalarReduceMaxFloat32(acc, lanes[:]), src[i:])
}

func reduceMaxLanes(lanes *[avxLane]float32, src []float32) int {
	if len(src) < avxLane {
		return 0
	}
	n := len(src)
	v := load8(src)
	src = src[avxLane:]
	for len(src) >= avxLane {
		v = max8(v, load8(src))
		src = src[avxLane:]
	}
	v.StoreArray(lanes)
	archsimd.ClearAVXUpperBits()
	return n - len(src)
}
