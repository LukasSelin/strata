//go:build goexperiment.simd && arm64

package stencil

import (
	"math"
	"simd/archsimd"
)

// This file is the NEON backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). It is simd_amd64.go four lanes at a
// time: each lane evaluates the scalar formula in horn.go operation for
// operation, so results match bit-for-bit.
//
// The kernels keep the amd64 file's split into a *Lanes function, which
// runs the vector loop and returns how many cells it wrote, and a wrapper
// that finishes the row. On amd64 the split keeps legacy SSE out of the
// AVX loop; arm64 has no such transition, but the same shape keeps the two
// files reviewable side by side, and Aspect's padded tail needs the lane
// function on its own anyway. Constant vectors live in consts, built once
// by init, as there. Loops use array-pointer loads over slices that shrink
// by one lane per iteration so the loads carry no bounds checks.

const lane = 4

func init() {
	// NEON is part of the arm64 baseline: no feature check.
	consts = newLaneConsts()
	simdName = "neon"
	simdGradient = hornGradientRowNEON
	simdSlope = hornSlopeRowNEON
	simdAspect = hornAspectRowNEON
	simdHillshade = hornHillshadeRowNEON
	simdCurvature = ztCurvatureRowNEON
	UseScalar(false)
}

func load4(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[lane]float32)(s))
}

func store4(v archsimd.Float32x4, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// hornDiff4 is hornDX and hornDY for four adjacent cells, unscaled. The
// rows must have at least lane+2 cells.
func hornDiff4(r0, r1, r2 []float32) (dx, dy archsimd.Float32x4) {
	z1, z2, z3 := load4(r0), load4(r0[1:]), load4(r0[2:])
	z4, z6 := load4(r1), load4(r1[2:])
	z7, z8, z9 := load4(r2), load4(r2[1:]), load4(r2[2:])
	d := z6.Sub(z4)
	dx = z3.Sub(z1).Add(z9.Sub(z7)).Add(d.Add(d))
	d = z8.Sub(z2)
	dy = z7.Sub(z1).Add(z9.Sub(z3)).Add(d.Add(d))
	return dx, dy
}

func hornGradientRowNEON(dx, dy, r0, r1, r2 []float32, kx, ky float32) {
	i := hornGradientLanes(dx, dy, r0, r1, r2, kx, ky)
	n := len(dx)
	scalarHornGradientRow(dx[i:], dy[i:n], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky)
}

func hornGradientLanes(dx, dy, r0, r1, r2 []float32, kx, ky float32) int {
	vkx := archsimd.BroadcastFloat32x4(kx)
	vky := archsimd.BroadcastFloat32x4(ky)
	n := len(dx)
	dy = dy[:n]
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dx) >= lane && len(dy) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		gx, gy := hornDiff4(r0, r1, r2)
		store4(gx.Mul(vkx), dx)
		store4(gy.Mul(vky), dy)
		dx, dy, r0, r1, r2 = dx[lane:], dy[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dx)
}

func hornSlopeRowNEON(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool) {
	var i int
	if atan {
		i = hornSlopeAtanLanes(dst, r0, r1, r2, kx, ky, scale)
	} else {
		i = hornSlopeLanes(dst, r0, r1, r2, kx, ky, scale)
	}
	n := len(dst)
	scalarHornSlopeRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky, scale, atan)
}

// hornMagnitude4 is sqrt(gx² + gy²) for four adjacent cells.
func hornMagnitude4(r0, r1, r2 []float32, kx, ky archsimd.Float32x4) archsimd.Float32x4 {
	dx, dy := hornDiff4(r0, r1, r2)
	gx, gy := dx.Mul(kx), dy.Mul(ky)
	return gx.Mul(gx).Add(gy.Mul(gy)).Sqrt()
}

func hornSlopeLanes(dst, r0, r1, r2 []float32, kx, ky, scale float32) int {
	vkx := archsimd.BroadcastFloat32x4(kx)
	vky := archsimd.BroadcastFloat32x4(ky)
	vs := archsimd.BroadcastFloat32x4(scale)
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		store4(hornMagnitude4(r0, r1, r2, vkx, vky).Mul(vs), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

func hornSlopeAtanLanes(dst, r0, r1, r2 []float32, kx, ky, scale float32) int {
	vkx := archsimd.BroadcastFloat32x4(kx)
	vky := archsimd.BroadcastFloat32x4(ky)
	vs := archsimd.BroadcastFloat32x4(scale)
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		m := hornMagnitude4(r0, r1, r2, vkx, vky)
		store4(atan4(m, &consts).Mul(vs), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

// laneConsts are the constant vectors of the lane functions.
type laneConsts struct {
	tan3pi8, tanpi8, pi, pi2, pi4, one, negOne, zero, maxf, c4, c3, c2, c1 archsimd.Float32x4
	deg, full, hi                                                          archsimd.Float32x4
	sign                                                                   archsimd.Uint32x4
	zeroInt                                                                archsimd.Int32x4
}

// consts is set once by init, before any kernel runs, and only read
// afterwards.
var consts laneConsts

func newLaneConsts() laneConsts {
	b := archsimd.BroadcastFloat32x4
	return laneConsts{
		tan3pi8: b(atanTan3Pi8), tanpi8: b(atanTanPi8),
		pi: b(atan2Pi), pi2: b(atanPi2), pi4: b(atanPi4),
		one: b(1), negOne: b(-1), zero: b(0), maxf: b(math.MaxFloat32),
		c4: b(atanC4), c3: b(atanC3), c2: b(atanC2), c1: b(atanC1),
		deg: b(radToDeg), full: b(360), hi: b(255),
		sign:    archsimd.BroadcastUint32x4(signBit32),
		zeroInt: archsimd.BroadcastInt32x4(0),
	}
}

// atan4 is Atan32 lanewise. The two argument reductions are both
// computed and selected with IfElse, in the same order as Atan32's two
// ifs, so a lane takes exactly the scalar path's num, den and y0.
func atan4(x archsimd.Float32x4, c *laneConsts) archsimd.Float32x4 {
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

func hornAspectRowNEON(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool) {
	i := hornAspectLanes(dst, r0, r1, r2, kx, ky, flat, trig)
	n := len(dst)
	if i == n {
		return
	}
	// The scalar tail would run Atan2F32 per cell, which on amd64 costs
	// about 15% at 254 cells per row (BenchmarkRowWidth). Instead, copy the last
	// n%4 cells' inputs into one lane padded with zeros, run the lane
	// function over it and copy the results back. A lane computes each
	// cell exactly as the scalar kernel does, so the bits are the same.
	// Only aspect gains from this: the other kernels' scalar tails cost
	// no more than the copies.
	var t0, t1, t2 [lane + 2]float32
	var out [lane]float32
	copy(t0[:], r0[i:n+2])
	copy(t1[:], r1[i:n+2])
	copy(t2[:], r2[i:n+2])
	hornAspectLanes(out[:], t0[:], t1[:], t2[:], kx, ky, flat, trig)
	copy(dst[i:], out[:])
}

func hornAspectLanes(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool) int {
	vkx := archsimd.BroadcastFloat32x4(kx)
	vky := archsimd.BroadcastFloat32x4(ky)
	vflat := archsimd.BroadcastFloat32x4(flat)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		dx, dy := hornDiff4(r0, r1, r2)
		gx, gy := dx.Mul(vkx), dy.Mul(vky)
		y, x := c.zero.Sub(gx), gy
		if trig {
			y, x = gy, c.zero.Sub(gx)
		}
		// aspectDegrees lanewise.
		d := atan2_4(y, x, c).Mul(c.deg)
		d = d.Add(c.full.IfElse(d.Less(c.zero), c.zero))
		d = c.zero.IfElse(d.GreaterEqual(c.full), d)
		d = vflat.IfElse(y.Equal(c.zero).And(x.Equal(c.zero)), d)
		store4(d, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

// atan2_4 is Atan2F32 lanewise, with the same reductions in the same
// order. Absolute value, sign tests and copysign work on the bits.
func atan2_4(y, x archsimd.Float32x4, c *laneConsts) archsimd.Float32x4 {
	sign := c.sign
	yb, xb := y.ToBits(), x.ToBits()
	ay, ax := yb.AndNot(sign).BitsToFloat32(), xb.AndNot(sign).BitsToFloat32()
	a := atan4(ay.Div(ax), c)
	a = c.zero.IfElse(ay.Equal(c.zero).And(ax.Equal(c.zero)), a)
	a = c.pi4.IfElse(ay.Equal(ax).And(ay.Greater(c.maxf)), a)
	a = c.pi.Sub(a).IfElse(xb.BitsToInt32().Less(c.zeroInt), a)
	return a.ToBits().Or(yb.And(sign)).BitsToFloat32()
}

func hornHillshadeRowNEON(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) {
	i := hornHillshadeLanes(dst, r0, r1, r2, kx, ky, c, bx, by)
	n := len(dst)
	scalarHornHillshadeRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky, c, bx, by)
}

func hornHillshadeLanes(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) int {
	b := archsimd.BroadcastFloat32x4
	vkx, vky := b(kx), b(ky)
	vc, vbx, vby := b(c), b(bx), b(by)
	k := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		dx, dy := hornDiff4(r0, r1, r2)
		gx, gy := dx.Mul(vkx), dy.Mul(vky)
		num := vc.Add(vbx.Mul(gx).Add(vby.Mul(gy)))
		den := k.one.Add(gx.Mul(gx).Add(gy.Mul(gy))).Sqrt()
		v := num.Div(den)
		v = k.zero.IfElse(v.Less(k.zero), v)
		v = k.hi.IfElse(v.Greater(k.hi), v)
		store4(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

func ztCurvatureRowNEON(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32, kind CurvatureKind) {
	var i int
	switch kind {
	case CurvProfile:
		i = ztProfileLanes(dst, r0, r1, r2, kp, kq, kr, kt, ks)
	case CurvPlan:
		i = ztPlanLanes(dst, r0, r1, r2, kp, kq, kr, kt, ks)
	default:
		i = ztMeanLanes(dst, r0, r1, r2, kp, kq, kr, kt, ks)
	}
	n := len(dst)
	scalarZTCurvatureRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kp, kq, kr, kt, ks, kind)
}

// ztDerivs4 is ztDerivs for four adjacent cells. The rows must have at
// least lane+2 cells.
func ztDerivs4(r0, r1, r2 []float32, kp, kq, kr, kt, ks archsimd.Float32x4) (p, q, r, s, t archsimd.Float32x4) {
	z1, z2, z3 := load4(r0), load4(r0[1:]), load4(r0[2:])
	z4, z5, z6 := load4(r1), load4(r1[1:]), load4(r1[2:])
	z7, z8, z9 := load4(r2), load4(r2[1:]), load4(r2[2:])
	p = z6.Sub(z4).Mul(kp)
	q = z8.Sub(z2).Mul(kq)
	c := z5.Add(z5)
	r = z4.Add(z6).Sub(c).Mul(kr)
	t = z2.Add(z8).Sub(c).Mul(kt)
	s = z1.Add(z9).Sub(z3.Add(z7)).Mul(ks)
	return p, q, r, s, t
}

func ztProfileLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x4
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs4(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		g := p2.Add(q2)
		w := c.one.Add(g)
		num := p2.Mul(r).Add(pq.Add(pq).Mul(s)).Add(q2.Mul(t))
		v := c.zero.Sub(num.Div(g).Div(w.Mul(w.Sqrt())))
		v = num.Add(c.zero).IfElse(g.Equal(c.zero), v)
		store4(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

func ztPlanLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x4
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs4(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		g := p2.Add(q2)
		num := q2.Mul(r).Sub(pq.Add(pq).Mul(s)).Add(p2.Mul(t))
		v := c.zero.Sub(num.Div(g).Div(g.Sqrt()))
		v = num.Add(c.zero).IfElse(g.Equal(c.zero), v)
		store4(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}

func ztMeanLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x4
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs4(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		w := c.one.Add(p2.Add(q2))
		num := c.one.Add(q2).Mul(r).Sub(pq.Add(pq).Mul(s)).Add(c.one.Add(p2).Mul(t))
		d := w.Mul(w.Sqrt())
		store4(c.zero.Sub(num.Div(d.Add(d))), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	return n - len(dst)
}
