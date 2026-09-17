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
	simdGradient = hornGradientRowAVX2
	simdSlope = hornSlopeRowAVX2
	simdAspect = hornAspectRowAVX2
	simdHillshade = hornHillshadeRowAVX2
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
	deg, full, hi                                                          archsimd.Float32x8
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
		deg: b(radToDeg), full: b(360), hi: b(255),
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
		y, x := c.zero.Sub(gx), gy
		if trig {
			y, x = gy, c.zero.Sub(gx)
		}
		// aspectDegrees lanewise.
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
		gx, gy := dx.Mul(vkx), dy.Mul(vky)
		num := vc.Add(vbx.Mul(gx).Add(vby.Mul(gy)))
		den := k.one.Add(gx.Mul(gx).Add(gy.Mul(gy))).Sqrt()
		v := num.Div(den)
		v = k.zero.IfElse(v.Less(k.zero), v)
		v = k.hi.IfElse(v.Greater(k.hi), v)
		store8(v, dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}
