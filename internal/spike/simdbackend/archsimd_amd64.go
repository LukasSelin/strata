//go:build goexperiment.simd && amd64

package simdbackend

import "simd/archsimd"

// Each loop ends with ClearAVXUpperBits (VZEROUPPER): the Go 1.27 compiler
// does not emit it, and the scalar tail uses legacy SSE under GOAMD64<v3.
func addArch(dst, a, b []float32) {
	i := 0
	for ; i+8 <= len(dst); i += 8 {
		x := archsimd.LoadFloat32x8(a[i:])
		y := archsimd.LoadFloat32x8(b[i:])
		x.Add(y).Store(dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	scalarAdd(dst[i:], a[i:], b[i:])
}

// clampArch matches scalar min(max(v, lo), hi) for NaN inputs: VMAXPS and
// VMINPS return the second operand when either is NaN, so NaN lanes are
// patched back in from src explicitly.
func clampArch(dst, src []float32, lo, hi float32) {
	vlo := archsimd.BroadcastFloat32x8(lo)
	vhi := archsimd.BroadcastFloat32x8(hi)
	i := 0
	for ; i+8 <= len(src); i += 8 {
		v := archsimd.LoadFloat32x8(src[i:])
		v.IfElse(v.IsNaN(), v.Max(vlo).Min(vhi)).Store(dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	scalarClamp(dst[i:], src[i:], lo, hi)
}

func slopeRowArch(dst, up, mid, down []float32, invDx8, invDy8 float32) {
	kx := archsimd.BroadcastFloat32x8(invDx8)
	ky := archsimd.BroadcastFloat32x8(invDy8)
	j := 0
	for ; j+8 <= len(dst); j += 8 {
		a := archsimd.LoadFloat32x8(up[j:])
		b := archsimd.LoadFloat32x8(up[j+1:])
		c := archsimd.LoadFloat32x8(up[j+2:])
		d := archsimd.LoadFloat32x8(mid[j:])
		f := archsimd.LoadFloat32x8(mid[j+2:])
		g := archsimd.LoadFloat32x8(down[j:])
		h := archsimd.LoadFloat32x8(down[j+1:])
		i := archsimd.LoadFloat32x8(down[j+2:])

		gx := c.Add(f.Add(f)).Add(i).Sub(a.Add(d.Add(d)).Add(g)).Mul(kx)
		gy := g.Add(h.Add(h)).Add(i).Sub(a.Add(b.Add(b)).Add(c)).Mul(ky)
		gx.Mul(gx).Add(gy.Mul(gy)).Sqrt().Store(dst[j:])
	}
	archsimd.ClearAVXUpperBits()
	scalarSlopeRow(dst[j:], up[j:], mid[j:], down[j:], invDx8, invDy8)
}

// The *ArchBCE variants are the same kernels written so the compiler can
// drop per-load bounds checks: fixed-size array pointers over slices that
// shrink by one lane per iteration. They exist to measure how much of the
// gap to hand-written asm is codegen versus Go slice bookkeeping.

func addArchBCE(dst, a, b []float32) {
	n := len(dst)
	a, b = a[:n], b[:n]
	for len(dst) >= 8 && len(a) >= 8 && len(b) >= 8 {
		x := archsimd.LoadFloat32x8Array((*[8]float32)(a))
		y := archsimd.LoadFloat32x8Array((*[8]float32)(b))
		x.Add(y).StoreArray((*[8]float32)(dst))
		dst, a, b = dst[8:], a[8:], b[8:]
	}
	archsimd.ClearAVXUpperBits()
	scalarAdd(dst, a, b)
}

func clampArchBCE(dst, src []float32, lo, hi float32) {
	vlo := archsimd.BroadcastFloat32x8(lo)
	vhi := archsimd.BroadcastFloat32x8(hi)
	src = src[:len(dst)]
	for len(dst) >= 8 && len(src) >= 8 {
		v := archsimd.LoadFloat32x8Array((*[8]float32)(src))
		v.IfElse(v.IsNaN(), v.Max(vlo).Min(vhi)).StoreArray((*[8]float32)(dst))
		dst, src = dst[8:], src[8:]
	}
	archsimd.ClearAVXUpperBits()
	scalarClamp(dst, src, lo, hi)
}

func slopeRowArchBCE(dst, up, mid, down []float32, invDx8, invDy8 float32) {
	kx := archsimd.BroadcastFloat32x8(invDx8)
	ky := archsimd.BroadcastFloat32x8(invDy8)
	n := len(dst)
	up, mid, down = up[:n+2], mid[:n+2], down[:n+2]
	for len(dst) >= 8 && len(up) >= 10 && len(mid) >= 10 && len(down) >= 10 {
		a := archsimd.LoadFloat32x8Array((*[8]float32)(up))
		b := archsimd.LoadFloat32x8Array((*[8]float32)(up[1:]))
		c := archsimd.LoadFloat32x8Array((*[8]float32)(up[2:]))
		d := archsimd.LoadFloat32x8Array((*[8]float32)(mid))
		f := archsimd.LoadFloat32x8Array((*[8]float32)(mid[2:]))
		g := archsimd.LoadFloat32x8Array((*[8]float32)(down))
		h := archsimd.LoadFloat32x8Array((*[8]float32)(down[1:]))
		i := archsimd.LoadFloat32x8Array((*[8]float32)(down[2:]))

		gx := c.Add(f.Add(f)).Add(i).Sub(a.Add(d.Add(d)).Add(g)).Mul(kx)
		gy := g.Add(h.Add(h)).Add(i).Sub(a.Add(b.Add(b)).Add(c)).Mul(ky)
		gx.Mul(gx).Add(gy.Mul(gy)).Sqrt().StoreArray((*[8]float32)(dst))
		dst, up, mid, down = dst[8:], up[8:], mid[8:], down[8:]
	}
	archsimd.ClearAVXUpperBits()
	scalarSlopeRow(dst, up, mid, down, invDx8, invDy8)
}
