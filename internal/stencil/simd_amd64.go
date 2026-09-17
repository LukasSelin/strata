//go:build goexperiment.simd && amd64

package stencil

import "simd/archsimd"

// This file is the AVX2 backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). Each lane evaluates the scalar formula
// in horn.go operation for operation, so results match bit-for-bit.
//
// Every kernel is split in two, as in benchmarks/nodata: a *Lanes function
// runs the vector loop, clears the upper AVX bits and returns how many
// cells it wrote, and a wrapper runs the scalar tail. Keeping the float32
// parameters out of the loop function's live-after-loop set stops the
// compiler reloading a spilled one with a legacy-SSE MOVSS inside the AVX
// loop (golang/go#80835). Loops use array-pointer loads over slices that
// shrink by one lane per iteration so the loads carry no bounds checks.

const lane = 8

func init() {
	// AVX2, not just AVX: IfElse and Broadcast need AVX2 instructions.
	if !archsimd.X86.AVX2() {
		return
	}
	simdGradient = hornGradientRowAVX2
	simdSlope = hornSlopeRowAVX2
	UseScalar(false)
}

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// hornDiff8 is hornDX and hornDY for eight adjacent cells, unscaled. The
// rows must have at least lane+2 cells.
func hornDiff8(r0, r1, r2 []float32) (dx, dy archsimd.Float32x8) {
	z1, z2, z3 := load8(r0), load8(r0[1:]), load8(r0[2:])
	z4, z6 := load8(r1), load8(r1[2:])
	z7, z8, z9 := load8(r2), load8(r2[1:]), load8(r2[2:])
	dx = z3.Add(z9).Add(z6.Add(z6)).Sub(z1.Add(z7).Add(z4.Add(z4)))
	dy = z7.Add(z9).Add(z8.Add(z8)).Sub(z1.Add(z3).Add(z2.Add(z2)))
	return dx, dy
}

func hornGradientRowAVX2(dx, dy, r0, r1, r2 []float32, kx, ky float32) {
	i := hornGradientLanes(dx, dy, r0, r1, r2, kx, ky)
	n := len(dx)
	scalarHornGradientRow(dx[i:], dy[i:n], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky)
}

func hornGradientLanes(dx, dy, r0, r1, r2 []float32, kx, ky float32) int {
	vkx := archsimd.BroadcastFloat32x8(kx)
	vky := archsimd.BroadcastFloat32x8(ky)
	n := len(dx)
	dy = dy[:n]
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dx) >= lane && len(dy) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		gx, gy := hornDiff8(r0, r1, r2)
		store8(gx.Mul(vkx), dx)
		store8(gy.Mul(vky), dy)
		dx, dy, r0, r1, r2 = dx[lane:], dy[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dx)
}

func hornSlopeRowAVX2(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool) {
	var i int
	if atan {
		i = hornSlopeAtanLanes(dst, r0, r1, r2, kx, ky, scale)
	} else {
		i = hornSlopeLanes(dst, r0, r1, r2, kx, ky, scale)
	}
	n := len(dst)
	scalarHornSlopeRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky, scale, atan)
}

// hornMagnitude8 is sqrt(gx² + gy²) for eight adjacent cells.
func hornMagnitude8(r0, r1, r2 []float32, kx, ky archsimd.Float32x8) archsimd.Float32x8 {
	dx, dy := hornDiff8(r0, r1, r2)
	gx, gy := dx.Mul(kx), dy.Mul(ky)
	return gx.Mul(gx).Add(gy.Mul(gy)).Sqrt()
}

func hornSlopeLanes(dst, r0, r1, r2 []float32, kx, ky, scale float32) int {
	vkx := archsimd.BroadcastFloat32x8(kx)
	vky := archsimd.BroadcastFloat32x8(ky)
	vs := archsimd.BroadcastFloat32x8(scale)
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		store8(hornMagnitude8(r0, r1, r2, vkx, vky).Mul(vs), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func hornSlopeAtanLanes(dst, r0, r1, r2 []float32, kx, ky, scale float32) int {
	vkx := archsimd.BroadcastFloat32x8(kx)
	vky := archsimd.BroadcastFloat32x8(ky)
	vs := archsimd.BroadcastFloat32x8(scale)
	c := newAtanConsts()
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		m := hornMagnitude8(r0, r1, r2, vkx, vky)
		store8(atan8(m, &c).Mul(vs), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

type atanConsts struct {
	tan3pi8, tanpi8, pi2, pi4, one, negOne, zero, c4, c3, c2, c1 archsimd.Float32x8
}

func newAtanConsts() atanConsts {
	b := archsimd.BroadcastFloat32x8
	return atanConsts{
		tan3pi8: b(atanTan3Pi8), tanpi8: b(atanTanPi8),
		pi2: b(atanPi2), pi4: b(atanPi4),
		one: b(1), negOne: b(-1),
		c4: b(atanC4), c3: b(atanC3), c2: b(atanC2), c1: b(atanC1),
	}
}

// atan8 is Atan32 lanewise. The two argument reductions are both
// computed and selected with IfElse, in the same order as Atan32's two
// ifs, so a lane takes exactly the scalar path's num, den and y0.
func atan8(x archsimd.Float32x8, c *atanConsts) archsimd.Float32x8 {
	mid := x.Greater(c.tanpi8)
	big := x.Greater(c.tan3pi8)
	num := c.negOne.IfElse(big, x.Sub(c.one).IfElse(mid, x))
	den := x.IfElse(big, x.Add(c.one).IfElse(mid, c.one))
	y0 := c.pi2.IfElse(big, c.pi4.IfElse(mid, c.zero))
	t := num.Div(den)
	z := t.Mul(t)
	p := c.c4.Mul(z).Sub(c.c3)
	p = p.Mul(z).Add(c.c2)
	p = p.Mul(z).Sub(c.c1)
	return y0.Add(p.Mul(z).Mul(t).Add(t))
}
