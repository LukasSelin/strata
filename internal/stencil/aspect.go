package stencil

import "math"

// This file holds the scalar aspect and hillshade kernels and the
// arctangent they build on. Like horn.go it is the canonical backend.

// Atan2F32 constants.
const (
	atan2Pi   = float32(math.Pi)
	radToDeg  = float32(180 / math.Pi)
	signBit32 = uint32(1) << 31
)

// Atan2F32 is the float32 two-argument arctangent used by the aspect
// kernels, built on Atan32: the angle of (x, y) in radians, in [-π, π].
// SIMD lanes match it bit-for-bit.
//
// It reduces to the first quadrant with a = Atan32(|y| / |x|), fixes the
// two ratios that are NaN there (a = 0 when both are zero, π/4 when both
// are infinite), reflects to a = π - a when x has its sign bit set, and
// gives the result the sign of y. Special cases follow math.Atan2: signed
// zeros select the quadrant (Atan2F32(+0, -0) = π, Atan2F32(-0, +0) = -0),
// infinities give multiples of π/4, and NaN in either argument gives NaN.
//
// Against math.Atan2 (go1.27.0) the largest error found is 2.69e-7
// radians (1.54e-5 degrees), over 4e7 random argument pairs of both signs
// with magnitudes from 2^-100 to 2^100 (half of them with nearly equal
// magnitudes), and every pair of zeros, infinities, NaN, extremes and
// subnormals. The error is at most Atan32's 1.41e-7 plus float32 rounding
// of π - a near ±π. Every result has the sign math.Atan2 gives.
// TestAtan2F32Accuracy checks a 3e-7 bound on a 4e6-pair sample.
func Atan2F32(y, x float32) float32 {
	ax := math.Float32frombits(math.Float32bits(x) &^ signBit32)
	ay := math.Float32frombits(math.Float32bits(y) &^ signBit32)
	a := Atan32(ay / ax)
	if ay == 0 && ax == 0 {
		a = 0
	}
	if ay == ax && ay > math.MaxFloat32 {
		a = atanPi4
	}
	if math.Float32bits(x)&signBit32 != 0 {
		a = atan2Pi - a
	}
	return math.Float32frombits(math.Float32bits(a) | math.Float32bits(y)&signBit32)
}

// HornAspectRow computes the direction of the Horn gradient for one row
// (see HornGradientRow), with gx = dx·kx and gy = dy·ky, in degrees in
// [0, 360). By default it is the compass bearing of the downslope
// direction, Atan2F32(-gx, gy), measured clockwise from decreasing row
// index (north on a north-up grid). With trig it is the downslope angle
// counterclockwise from increasing column index (east),
// Atan2F32(gy, -gx). The negation is 0 - gx, so it never makes -0.
//
// The angle in degrees d is folded into [0, 360) with d += 360 if d < 0,
// else d += 0 (which turns -0 into +0), then d = 0 if d >= 360. Cells
// with gx == 0 and gy == 0 (after scaling, either sign) get flat instead;
// a NaN gradient gives NaN.
func HornAspectRow(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool) {
	requireRows(len(dst), r0, r1, r2)
	hornAspectRow(dst, r0, r1, r2, kx, ky, flat, trig)
}

// aspectChunk is how many cells scalarHornAspectRow computes gradients
// for before the angle pass. See scalarHornSlopeRow for why the passes
// are separate.
const aspectChunk = 64

func scalarHornAspectRow(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool) {
	var ys, xs [aspectChunk]float32
	for len(dst) > 0 {
		m := min(len(dst), aspectChunk)
		n := m + 2
		scalarHornAspectArgs(ys[:m], xs[:m], r0[:n], r1[:n], r2[:n], kx, ky, trig)
		for i := range m {
			dst[i] = aspectDegrees(ys[i], xs[i], flat)
		}
		dst, r0, r1, r2 = dst[m:], r0[m:], r1[m:], r2[m:]
	}
}

// scalarHornAspectArgs writes the atan2 arguments of each cell.
func scalarHornAspectArgs(ys, xs, r0, r1, r2 []float32, kx, ky float32, trig bool) {
	n := len(ys)
	xs = xs[:n]
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for i := range ys {
		z1, z2, z3 := r0[i], r0[i+1], r0[i+2]
		z4, z6 := r1[i], r1[i+2]
		z7, z8, z9 := r2[i], r2[i+1], r2[i+2]
		gx := float32(hornDX(z1, z3, z4, z6, z7, z9) * kx)
		gy := float32(hornDY(z1, z2, z3, z7, z8, z9) * ky)
		if trig {
			ys[i], xs[i] = gy, 0-gx
		} else {
			ys[i], xs[i] = 0-gx, gy
		}
	}
}

func aspectDegrees(y, x, flat float32) float32 {
	d := float32(Atan2F32(y, x) * radToDeg)
	off := float32(0)
	if d < 0 {
		off = 360
	}
	d += off
	if d >= 360 {
		d = 0
	}
	if y == 0 && x == 0 {
		d = flat
	}
	return d
}

// HornHillshadeRow computes the shaded relief of one row from its Horn
// gradient (see HornGradientRow), gx = dx·kx and gy = dy·ky:
//
//	v = (c + (bx·gx + by·gy)) / sqrt(1 + (gx² + gy²))
//
// clamped to [0, 255] (v < 0 gives 0, v > 255 gives 255, NaN stays NaN).
// With axes (column, row, up), that is the cosine of the angle between
// the surface normal (-gx, -gy, 1) and the direction towards the light
// (-bx, -by, c), times that vector's length: callers pass a light vector
// of length 255.
func HornHillshadeRow(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) {
	requireRows(len(dst), r0, r1, r2)
	hornHillshadeRow(dst, r0, r1, r2, kx, ky, c, bx, by)
}

func scalarHornHillshadeRow(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32) {
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for i := range dst {
		z1, z2, z3 := r0[i], r0[i+1], r0[i+2]
		z4, z6 := r1[i], r1[i+2]
		z7, z8, z9 := r2[i], r2[i+1], r2[i+2]
		gx := float32(hornDX(z1, z3, z4, z6, z7, z9) * kx)
		gy := float32(hornDY(z1, z2, z3, z7, z8, z9) * ky)
		num := c + (float32(bx*gx) + float32(by*gy))
		den := float32(math.Sqrt(float64(1 + (float32(gx*gx) + float32(gy*gy)))))
		v := num / den
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		dst[i] = v
	}
}
