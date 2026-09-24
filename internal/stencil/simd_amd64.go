//go:build goexperiment.simd && amd64

package stencil

import (
	"math"
	"simd/archsimd"
)

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
//
// Between the first 256-bit instruction of a *Lanes function and its
// ClearAVXUpperBits there must be no legacy (non-VEX) SSE instruction.
// On the Zen 2 machine of benchmarks/terrain/RESULTS.md each one costs
// about 65 ns there (an SSE/AVX transition, golang/go#80835), as much as
// 50 cells of Slope in degrees, paid once per row. The compiler emits them
// for struct copies and zeroing (MOVUPS), for some float32 spills and
// reloads (MOVSS) and for zero vectors (XORPS). So constant vectors live
// in consts, built once by init, instead of being broadcast into a struct
// per call; lane functions broadcast only their float32 parameters, as
// the first thing they do. BenchmarkRowWidth measures the per-call cost:
// with a transition in the lane function it is 70–250 ns, without one
// 5–20 ns.

const lane = 8

func init() {
	// AVX2, not just AVX: IfElse and Broadcast need AVX2 instructions.
	if !archsimd.X86.AVX2() {
		return
	}
	consts = newLaneConsts()
	archsimd.ClearAVXUpperBits()
	simdName = "avx2"
	simdGradient = hornGradientRowAVX2
	simdSlope = hornSlopeRowAVX2
	simdAspect = hornAspectRowAVX2
	simdHillshade = hornHillshadeRowAVX2
	simdCurvature = ztCurvatureRowAVX2
	simdRuggedness = ruggednessRowAVX2
	simdSlopeGrad = slopeFromGradientRowAVX2
	simdAspectGrad = aspectFromGradientRowAVX2
	simdHillshadeGrad = hillshadeFromGradientRowAVX2
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
	d := z6.Sub(z4)
	dx = z3.Sub(z1).Add(z9.Sub(z7)).Add(d.Add(d))
	d = z8.Sub(z2)
	dy = z7.Sub(z1).Add(z9.Sub(z3)).Add(d.Add(d))
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
	return magnitude8(dx.Mul(kx), dy.Mul(ky))
}

// magnitude8 is magnitude lanewise.
func magnitude8(gx, gy archsimd.Float32x8) archsimd.Float32x8 {
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
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		m := hornMagnitude8(r0, r1, r2, vkx, vky)
		store8(atan8(m, &consts).Mul(vs), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// laneConsts are the constant vectors of the lane functions.
type laneConsts struct {
	tan3pi8, tanpi8, pi, pi2, pi4, one, negOne, zero, maxf, c4, c3, c2, c1 archsimd.Float32x8
	deg, full, hi, eighth, nan                                             archsimd.Float32x8
	sign                                                                   archsimd.Uint32x8
	zeroInt                                                                archsimd.Int32x8
}

// consts is set once by init on CPUs with AVX2, before any kernel runs,
// and only read afterwards.
var consts laneConsts

func newLaneConsts() laneConsts {
	b := archsimd.BroadcastFloat32x8
	return laneConsts{
		tan3pi8: b(atanTan3Pi8), tanpi8: b(atanTanPi8),
		pi: b(atan2Pi), pi2: b(atanPi2), pi4: b(atanPi4),
		one: b(1), negOne: b(-1), zero: b(0), maxf: b(math.MaxFloat32),
		c4: b(atanC4), c3: b(atanC3), c2: b(atanC2), c1: b(atanC1),
		deg: b(radToDeg), full: b(360), hi: b(255), eighth: b(0.125),
		nan:     b(float32(math.NaN())),
		sign:    archsimd.BroadcastUint32x8(signBit32),
		zeroInt: archsimd.BroadcastInt32x8(0),
	}
}

// atan8 is Atan32 lanewise. The two argument reductions are both
// computed and selected with IfElse, in the same order as Atan32's two
// ifs, so a lane takes exactly the scalar path's num, den and y0.
func atan8(x archsimd.Float32x8, c *laneConsts) archsimd.Float32x8 {
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

func hornAspectRowAVX2(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool) {
	i := hornAspectLanes(dst, r0, r1, r2, kx, ky, flat, trig)
	n := len(dst)
	if i == n {
		return
	}
	// The scalar tail would run Atan2F32 per cell, which costs about 15%
	// at 254 cells per row (BenchmarkRowWidth). Instead, copy the last
	// n%8 cells' inputs into one lane padded with zeros, run the lane
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
	vkx := archsimd.BroadcastFloat32x8(kx)
	vky := archsimd.BroadcastFloat32x8(ky)
	vflat := archsimd.BroadcastFloat32x8(flat)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		dx, dy := hornDiff8(r0, r1, r2)
		gx, gy := dx.Mul(vkx), dy.Mul(vky)
		// aspect8, written out: it does not inline, and the call costs
		// this loop 8% at 254 cells a row (BenchmarkRowWidth).
		// TestFromGradientMatchesFused holds the two to the same bits.
		y, x := c.zero.Sub(gx), gy
		if trig {
			y, x = gy, c.zero.Sub(gx)
		}
		d := atan2_8(y, x, c).Mul(c.deg)
		d = d.Add(c.full.IfElse(d.Less(c.zero), c.zero))
		d = c.zero.IfElse(d.GreaterEqual(c.full), d)
		d = vflat.IfElse(y.Equal(c.zero).And(x.Equal(c.zero)), d)
		store8(d, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// aspect8 is aspectArgs and aspectDegrees lanewise.
func aspect8(gx, gy, vflat archsimd.Float32x8, trig bool, c *laneConsts) archsimd.Float32x8 {
	y, x := c.zero.Sub(gx), gy
	if trig {
		y, x = gy, c.zero.Sub(gx)
	}
	d := atan2_8(y, x, c).Mul(c.deg)
	d = d.Add(c.full.IfElse(d.Less(c.zero), c.zero))
	d = c.zero.IfElse(d.GreaterEqual(c.full), d)
	return vflat.IfElse(y.Equal(c.zero).And(x.Equal(c.zero)), d)
}

// atan2_8 is Atan2F32 lanewise, with the same reductions in the same
// order. Absolute value, sign tests and copysign work on the bits.
func atan2_8(y, x archsimd.Float32x8, c *laneConsts) archsimd.Float32x8 {
	sign := c.sign
	yb, xb := y.ToBits(), x.ToBits()
	ay, ax := yb.AndNot(sign).BitsToFloat32(), xb.AndNot(sign).BitsToFloat32()
	a := atan8(ay.Div(ax), c)
	a = c.zero.IfElse(ay.Equal(c.zero).And(ax.Equal(c.zero)), a)
	a = c.pi4.IfElse(ay.Equal(ax).And(ay.Greater(c.maxf)), a)
	a = c.pi.Sub(a).IfElse(xb.BitsToInt32().Less(c.zeroInt), a)
	return a.ToBits().Or(yb.And(sign)).BitsToFloat32()
}

func hornHillshadeRowAVX2(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) {
	i := hornHillshadeLanes(dst, r0, r1, r2, kx, ky, c, bx, by)
	n := len(dst)
	scalarHornHillshadeRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kx, ky, c, bx, by)
}

func hornHillshadeLanes(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) int {
	b := archsimd.BroadcastFloat32x8
	vkx, vky := b(kx), b(ky)
	vc, vbx, vby := b(c), b(bx), b(by)
	k := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		dx, dy := hornDiff8(r0, r1, r2)
		store8(shade8(dx.Mul(vkx), dy.Mul(vky), vc, vbx, vby, k), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// shade8 is shade lanewise.
func shade8(gx, gy, vc, vbx, vby archsimd.Float32x8, k *laneConsts) archsimd.Float32x8 {
	num := vc.Add(vbx.Mul(gx).Add(vby.Mul(gy)))
	den := k.one.Add(gx.Mul(gx).Add(gy.Mul(gy))).Sqrt()
	v := num.Div(den)
	v = k.zero.IfElse(v.Less(k.zero), v)
	return k.hi.IfElse(v.Greater(k.hi), v)
}

// The from-gradient kernels (gradrow.go): the fused kernels' lanes after
// hornDiff8 and the scaling, over gradient rows instead.

func slopeFromGradientRowAVX2(dst, gx, gy []float32, scale float32, atan bool) {
	i := slopeFromGradientLanes(dst, gx, gy, scale, atan)
	scalarSlopeFromGradientRow(dst[i:], gx[i:], gy[i:], scale, atan)
}

func slopeFromGradientLanes(dst, gx, gy []float32, scale float32, atan bool) int {
	vs := archsimd.BroadcastFloat32x8(scale)
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for len(dst) >= lane && len(gx) >= lane && len(gy) >= lane {
		m := magnitude8(load8(gx), load8(gy))
		if atan {
			m = atan8(m, &consts)
		}
		store8(m.Mul(vs), dst)
		dst, gx, gy = dst[lane:], gx[lane:], gy[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func aspectFromGradientRowAVX2(dst, gx, gy []float32, flat float32, trig bool) {
	i := aspectFromGradientLanes(dst, gx, gy, flat, trig)
	scalarAspectFromGradientRow(dst[i:], gx[i:], gy[i:], flat, trig)
}

func aspectFromGradientLanes(dst, gx, gy []float32, flat float32, trig bool) int {
	vflat := archsimd.BroadcastFloat32x8(flat)
	c := &consts
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for len(dst) >= lane && len(gx) >= lane && len(gy) >= lane {
		store8(aspect8(load8(gx), load8(gy), vflat, trig, c), dst)
		dst, gx, gy = dst[lane:], gx[lane:], gy[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func hillshadeFromGradientRowAVX2(dst, gx, gy []float32, c, bx, by float32) {
	i := hillshadeFromGradientLanes(dst, gx, gy, c, bx, by)
	scalarHillshadeFromGradientRow(dst[i:], gx[i:], gy[i:], c, bx, by)
}

func hillshadeFromGradientLanes(dst, gx, gy []float32, c, bx, by float32) int {
	b := archsimd.BroadcastFloat32x8
	vc, vbx, vby := b(c), b(bx), b(by)
	k := &consts
	n := len(dst)
	gx, gy = gx[:n], gy[:n]
	for len(dst) >= lane && len(gx) >= lane && len(gy) >= lane {
		store8(shade8(load8(gx), load8(gy), vc, vbx, vby, k), dst)
		dst, gx, gy = dst[lane:], gx[lane:], gy[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func ztCurvatureRowAVX2(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32, kind CurvatureKind) {
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

// ztDerivs8 is ztDerivs for eight adjacent cells. The rows must have at
// least lane+2 cells.
func ztDerivs8(r0, r1, r2 []float32, kp, kq, kr, kt, ks archsimd.Float32x8) (p, q, r, s, t archsimd.Float32x8) {
	z1, z2, z3 := load8(r0), load8(r0[1:]), load8(r0[2:])
	z4, z5, z6 := load8(r1), load8(r1[1:]), load8(r1[2:])
	z7, z8, z9 := load8(r2), load8(r2[1:]), load8(r2[2:])
	p = z6.Sub(z4).Mul(kp)
	q = z8.Sub(z2).Mul(kq)
	c := z5.Add(z5)
	r = z4.Add(z6).Sub(c).Mul(kr)
	t = z2.Add(z8).Sub(c).Mul(kt)
	s = z1.Add(z9).Sub(z3.Add(z7)).Mul(ks)
	return p, q, r, s, t
}

func ztProfileLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x8
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs8(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		g := p2.Add(q2)
		w := c.one.Add(g)
		num := p2.Mul(r).Add(pq.Add(pq).Mul(s)).Add(q2.Mul(t))
		v := c.zero.Sub(num.Div(g).Div(w.Mul(w.Sqrt())))
		v = num.Add(c.zero).IfElse(g.Equal(c.zero), v)
		store8(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func ztPlanLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x8
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs8(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		g := p2.Add(q2)
		num := q2.Mul(r).Sub(pq.Add(pq).Mul(s)).Add(p2.Mul(t))
		v := c.zero.Sub(num.Div(g).Div(g.Sqrt()))
		v = num.Add(c.zero).IfElse(g.Equal(c.zero), v)
		store8(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func ztMeanLanes(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32) int {
	b := archsimd.BroadcastFloat32x8
	vkp, vkq, vkr, vkt, vks := b(kp), b(kq), b(kr), b(kt), b(ks)
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		p, q, r, s, t := ztDerivs8(r0, r1, r2, vkp, vkq, vkr, vkt, vks)
		p2, q2, pq := p.Mul(p), q.Mul(q), p.Mul(q)
		w := c.one.Add(p2.Add(q2))
		num := c.one.Add(q2).Mul(r).Sub(pq.Add(pq).Mul(s)).Add(c.one.Add(p2).Mul(t))
		d := w.Mul(w.Sqrt())
		store8(c.zero.Sub(num.Div(d.Add(d))), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func ruggednessRowAVX2(dst, r0, r1, r2 []float32, kind RuggednessKind) {
	var i int
	switch kind {
	case RugTRIRiley:
		i = rileyLanes(dst, r0, r1, r2)
	case RugTRIWilson:
		i = wilsonLanes(dst, r0, r1, r2)
	case RugTPI:
		i = tpiLanes(dst, r0, r1, r2)
	default:
		i = roughnessLanes(dst, r0, r1, r2)
	}
	n := len(dst)
	scalarRuggednessRow(dst[i:], r0[i:n+2], r1[i:n+2], r2[i:n+2], kind)
}

// window8 loads the nine cells of the windows centred on eight adjacent
// cells. The rows must have at least lane+2 cells.
func window8(r0, r1, r2 []float32) (z1, z2, z3, z4, z5, z6, z7, z8, z9 archsimd.Float32x8) {
	return load8(r0), load8(r0[1:]), load8(r0[2:]),
		load8(r1), load8(r1[1:]), load8(r1[2:]),
		load8(r2), load8(r2[1:]), load8(r2[2:])
}

// rileyLanes widens each float32 difference to float64, a half vector
// at a time (VCVTPS2PD), and sums the squares there, as sq64 does.
func rileyLanes(dst, r0, r1, r2 []float32) int {
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		z1, z2, z3, z4, z5, z6, z7, z8, z9 := window8(r0, r1, r2)
		lo1, hi1 := sq4x2(z1.Sub(z5))
		lo2, hi2 := sq4x2(z2.Sub(z5))
		lo3, hi3 := sq4x2(z3.Sub(z5))
		lo4, hi4 := sq4x2(z4.Sub(z5))
		lo6, hi6 := sq4x2(z6.Sub(z5))
		lo7, hi7 := sq4x2(z7.Sub(z5))
		lo8, hi8 := sq4x2(z8.Sub(z5))
		lo9, hi9 := sq4x2(z9.Sub(z5))
		lo := lo1.Add(lo2).Add(lo3).Add(lo4).Add(lo6).Add(lo7).Add(lo8).Add(lo9)
		hi := hi1.Add(hi2).Add(hi3).Add(hi4).Add(hi6).Add(hi7).Add(hi8).Add(hi9)
		store8(c.zero.SetLo(lo.Sqrt().ConvertToFloat32()).SetHi(hi.Sqrt().ConvertToFloat32()), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// sq4x2 is sq64 of the low and the high four lanes of d.
func sq4x2(d archsimd.Float32x8) (lo, hi archsimd.Float64x4) {
	lo, hi = d.GetLo().ConvertToFloat64(), d.GetHi().ConvertToFloat64()
	return lo.Mul(lo), hi.Mul(hi)
}

func wilsonLanes(dst, r0, r1, r2 []float32) int {
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	abs := func(x archsimd.Float32x8) archsimd.Float32x8 { return x.ToBits().AndNot(c.sign).BitsToFloat32() }
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		z1, z2, z3, z4, z5, z6, z7, z8, z9 := window8(r0, r1, r2)
		s := abs(z1.Sub(z5)).Add(abs(z2.Sub(z5))).Add(abs(z3.Sub(z5))).Add(abs(z4.Sub(z5))).
			Add(abs(z6.Sub(z5))).Add(abs(z7.Sub(z5))).Add(abs(z8.Sub(z5))).Add(abs(z9.Sub(z5)))
		store8(s.Mul(c.eighth), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func tpiLanes(dst, r0, r1, r2 []float32) int {
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		z1, z2, z3, z4, z5, z6, z7, z8, z9 := window8(r0, r1, r2)
		s := z1.Add(z2).Add(z3).Add(z4).Add(z6).Add(z7).Add(z8).Add(z9)
		store8(z5.Sub(s.Mul(c.eighth)), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// roughnessLanes uses VMAXPS and VMINPS as they are and repairs the
// difference once. They differ from Go's max and min only on NaN and on
// zeros of opposite sign. A NaN anywhere in the window must give NaN, so
// those lanes are selected afterwards. Picking the wrong zero as the
// largest or smallest cell changes nothing unless both are zeros, when
// the difference can come out -0; Go's max - min is never -0, and adding
// +0 turns -0 into +0 and leaves every other value alone.
func roughnessLanes(dst, r0, r1, r2 []float32) int {
	c := &consts
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		z1, z2, z3, z4, z5, z6, z7, z8, z9 := window8(r0, r1, r2)
		hi := z1.Max(z2).Max(z3.Max(z4)).Max(z5.Max(z6).Max(z7.Max(z8))).Max(z9)
		lo := z1.Min(z2).Min(z3.Min(z4)).Min(z5.Min(z6).Min(z7.Min(z8))).Min(z9)
		nan := z1.IsNaN().Or(z2.IsNaN()).Or(z3.IsNaN()).Or(z4.IsNaN()).Or(z5.IsNaN()).
			Or(z6.IsNaN()).Or(z7.IsNaN()).Or(z8.IsNaN()).Or(z9.IsNaN())
		v := hi.Sub(lo).Add(c.zero)
		store8(c.nan.IfElse(nan, v), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}
