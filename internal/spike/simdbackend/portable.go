//go:build goexperiment.simd

package simdbackend

import "simd"

// The portable variants are width-agnostic: the compiler multi-versions each
// function for the vector sizes the target supports (amd64: emulated, 128,
// 256, 512; arm64: emulated, 128) and picks one at run time.

func addPortable(dst, a, b []float32) {
	var zero simd.Float32s
	w := zero.Len()
	i := 0
	for ; i+w <= len(dst); i += w {
		x := simd.LoadFloat32s(a[i:])
		y := simd.LoadFloat32s(b[i:])
		x.Add(y).Store(dst[i:])
	}
	scalarAdd(dst[i:], a[i:], b[i:])
}

func clampPortable(dst, src []float32, lo, hi float32) {
	vlo := simd.BroadcastFloat32s(lo)
	vhi := simd.BroadcastFloat32s(hi)
	w := vlo.Len()
	i := 0
	for ; i+w <= len(src); i += w {
		v := simd.LoadFloat32s(src[i:])
		v.IfElse(v.NotEqual(v), v.Max(vlo).Min(vhi)).Store(dst[i:])
	}
	scalarClamp(dst[i:], src[i:], lo, hi)
}

func slopeRowPortable(dst, up, mid, down []float32, invDx8, invDy8 float32) {
	kx := simd.BroadcastFloat32s(invDx8)
	ky := simd.BroadcastFloat32s(invDy8)
	w := kx.Len()
	j := 0
	for ; j+w <= len(dst); j += w {
		a := simd.LoadFloat32s(up[j:])
		b := simd.LoadFloat32s(up[j+1:])
		c := simd.LoadFloat32s(up[j+2:])
		d := simd.LoadFloat32s(mid[j:])
		f := simd.LoadFloat32s(mid[j+2:])
		g := simd.LoadFloat32s(down[j:])
		h := simd.LoadFloat32s(down[j+1:])
		i := simd.LoadFloat32s(down[j+2:])

		gx := c.Add(f.Add(f)).Add(i).Sub(a.Add(d.Add(d)).Add(g)).Mul(kx)
		gy := g.Add(h.Add(h)).Add(i).Sub(a.Add(b.Add(b)).Add(c)).Mul(ky)
		gx.Mul(gx).Add(gy.Mul(gy)).Sqrt().Store(dst[j:])
	}
	scalarSlopeRow(dst[j:], up[j:], mid[j:], down[j:], invDx8, invDy8)
}
