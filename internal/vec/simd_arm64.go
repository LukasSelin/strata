//go:build goexperiment.simd && arm64

package vec

import "simd/archsimd"

// This file is the NEON backend, written with Go's simd/archsimd package
// (see docs/adr/0001-simd-backend.md). It only exists in builds with
// GOEXPERIMENT=simd; other arm64 builds keep the scalar function variables
// from dispatch.go.
//
// Every kernel must agree bit-for-bit with its scalar counterpart (any NaN
// matches any NaN). The loops have the shape of simd_amd64.go's, four
// lanes at a time: fixed-size array-pointer loads over slices that shrink
// by one lane per iteration, so the loads carry no bounds checks, and the
// scalar kernel for the tail. There is no SSE/AVX transition on arm64, so
// nothing needs clearing before the tail.

// neonLane is the number of float32 lanes in a NEON register.
const neonLane = 4

func init() {
	// NEON is part of the arm64 baseline: no feature check.
	simdKernels = &kernelSet{
		add:       addFloat32NEON,
		sub:       subFloat32NEON,
		mul:       mulFloat32NEON,
		div:       divFloat32NEON,
		addScalar: addScalarFloat32NEON,
		mulScalar: mulScalarFloat32NEON,
		affine:    affineFloat32NEON,
		subDiv:    subDivFloat32NEON,
		min:       minFloat32NEON,
		max:       maxFloat32NEON,
		clamp:     clampFloat32NEON,
		abs:       absFloat32NEON,
		sqrt:      sqrtFloat32NEON,
		reduceMin: reduceMinFloat32NEON,
		reduceMax: reduceMaxFloat32NEON,
		chain:     chainFloat32NEON,
	}
	simdName = "neon"
	UseScalar(false)
}

func load4(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[neonLane]float32)(s))
}

func store4(v archsimd.Float32x4, s []float32) {
	v.StoreArray((*[neonLane]float32)(s))
}

// min4 and max4 are Go's builtin min and max lanewise. Unlike VMINPS and
// VMAXPS on amd64, NEON's FMIN and FMAX already are: a NaN in either
// operand gives NaN, and -0 orders below +0. So they need none of the
// repairs of min8 and max8. TestSIMDMinMaxEdgePairs holds them to that.
func min4(x, y archsimd.Float32x4) archsimd.Float32x4 { return x.Min(y) }

func max4(x, y archsimd.Float32x4) archsimd.Float32x4 { return x.Max(y) }

func addFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(load4(a).Add(load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarAddFloat32(dst, a, b)
}

func subFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(load4(a).Sub(load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarSubFloat32(dst, a, b)
}

func mulFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(load4(a).Mul(load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarMulFloat32(dst, a, b)
}

func divFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(load4(a).Div(load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarDivFloat32(dst, a, b)
}

func minFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(min4(load4(a), load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarMinFloat32(dst, a, b)
}

func maxFloat32NEON(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for len(dst) >= neonLane && len(a) >= neonLane && len(b) >= neonLane {
		store4(max4(load4(a), load4(b)), dst)
		dst, a, b = dst[neonLane:], a[neonLane:], b[neonLane:]
	}
	scalarMaxFloat32(dst, a, b)
}

func addScalarFloat32NEON(dst, src []float32, value float32) {
	v := archsimd.BroadcastFloat32x4(value)
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Add(v), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarAddScalarFloat32(dst, src, value)
}

func mulScalarFloat32NEON(dst, src []float32, value float32) {
	v := archsimd.BroadcastFloat32x4(value)
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Mul(v), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarMulScalarFloat32(dst, src, value)
}

// affineFloat32NEON multiplies and adds in two separate instructions,
// FMUL then FADD, never a fused FMLA: scalarAffineFloat32 rounds the
// product before adding, and a fused lane would not match it. The
// compiler does not fuse archsimd's Mul and Add (DESIGN.md §50,
// Determinism, has the check).
func affineFloat32NEON(dst, src []float32, a, b float32) {
	va := archsimd.BroadcastFloat32x4(a)
	vb := archsimd.BroadcastFloat32x4(b)
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Mul(va).Add(vb), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarAffineFloat32(dst, src, a, b)
}

// subDivFloat32NEON is FSUB then FDIV. Both are correctly rounded, so each
// lane matches scalarSubDivFloat32.
func subDivFloat32NEON(dst, src []float32, lo, span float32) {
	vlo := archsimd.BroadcastFloat32x4(lo)
	vspan := archsimd.BroadcastFloat32x4(span)
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Sub(vlo).Div(vspan), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarSubDivFloat32(dst, src, lo, span)
}

func clampFloat32NEON(dst, src []float32, lo, hi float32) {
	vlo := archsimd.BroadcastFloat32x4(lo)
	vhi := archsimd.BroadcastFloat32x4(hi)
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(min4(max4(load4(src), vlo), vhi), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarClampFloat32(dst, src, lo, hi)
}

func absFloat32NEON(dst, src []float32) {
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Abs(), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarAbsFloat32(dst, src)
}

func sqrtFloat32NEON(dst, src []float32) {
	src = src[:len(dst)]
	for len(dst) >= neonLane && len(src) >= neonLane {
		store4(load4(src).Sqrt(), dst)
		dst, src = dst[neonLane:], src[neonLane:]
	}
	scalarSqrtFloat32(dst, src)
}

// reduceMinFloat32NEON and reduceMaxFloat32NEON keep one accumulator
// vector and fold its lanes into acc at the end, a different order from
// the scalar loop. min4 and max4 are Go's builtins, which are associative
// and commutative over NaN and signed zeros alike, so the value is the
// same either way; only which NaN payload survives may differ.
func reduceMinFloat32NEON(acc float32, src []float32) float32 {
	if len(src) < neonLane {
		return scalarReduceMinFloat32(acc, src)
	}
	v := load4(src)
	src = src[neonLane:]
	for len(src) >= neonLane {
		v = min4(v, load4(src))
		src = src[neonLane:]
	}
	var lanes [neonLane]float32
	v.StoreArray(&lanes)
	return scalarReduceMinFloat32(scalarReduceMinFloat32(acc, lanes[:]), src)
}

func reduceMaxFloat32NEON(acc float32, src []float32) float32 {
	if len(src) < neonLane {
		return scalarReduceMaxFloat32(acc, src)
	}
	v := load4(src)
	src = src[neonLane:]
	for len(src) >= neonLane {
		v = max4(v, load4(src))
		src = src[neonLane:]
	}
	var lanes [neonLane]float32
	v.StoreArray(&lanes)
	return scalarReduceMaxFloat32(scalarReduceMaxFloat32(acc, lanes[:]), src)
}
