package stencil

import (
	"fmt"
	"math"
)

// This file holds the scalar Zevenbergen–Thorne curvature kernels. Like
// horn.go it is the canonical backend.

// CurvatureKind selects which curvature ZTCurvatureRow writes.
type CurvatureKind int

const (
	// CurvProfile is the profile (vertical) curvature, the normal
	// curvature along the gradient direction.
	CurvProfile CurvatureKind = iota
	// CurvPlan is the plan (contour) curvature, the curvature of the
	// contour line through the cell.
	CurvPlan
	// CurvMean is the mean curvature of the surface.
	CurvMean
)

// ZTScales returns the factors ZTCurvatureRow multiplies its differences
// by: zFactor/(2·cellSizeX), zFactor/(2·cellSizeY), zFactor/cellSizeX²,
// zFactor/cellSizeY² and zFactor/(4·cellSizeX·cellSizeY), each computed
// in float64 and rounded once.
func ZTScales(cellSizeX, cellSizeY, zFactor float64) (kp, kq, kr, kt, ks float32) {
	return float32(zFactor / (2 * cellSizeX)), float32(zFactor / (2 * cellSizeY)),
		float32(zFactor / (cellSizeX * cellSizeX)), float32(zFactor / (cellSizeY * cellSizeY)),
		float32(zFactor / (4 * cellSizeX * cellSizeY))
}

// ZTCurvatureRow computes one row of a curvature from the
// Zevenbergen–Thorne derivatives of the 3×3 window (z1..z9 as in
// HornGradientRow, z5 the centre):
//
//	p = (z6 - z4)·kp                 ∂z/∂x
//	q = (z8 - z2)·kq                 ∂z/∂y (y with the row index)
//	r = ((z4 + z6) - (z5 + z5))·kr   ∂²z/∂x²
//	t = ((z2 + z8) - (z5 + z5))·kt   ∂²z/∂y²
//	s = ((z1 + z9) - (z3 + z7))·ks   ∂²z/∂x∂y
//
// With p2 = p², q2 = q², pq2 = 2pq, g = p2 + q2 and w = 1 + g, kind
// selects (Florinsky's normal-section curvatures, positive where the
// surface is convex):
//
//	CurvProfile  0 - ((p2·r + pq2·s) + q2·t) / g / (w·√w)
//	CurvPlan     0 - ((q2·r - pq2·s) + p2·t) / g / √g
//	CurvMean     0 - (((1+q2)·r - pq2·s) + (1+p2)·t) / (w·√w + w·√w)
//
// Dividing by g before w·√w keeps the bounded ratio num/g from
// overflowing on steep ground; 0 - x makes a zero result +0. Where g is
// zero, profile and plan write num + 0 instead, which is +0 when r, s and
// t are finite and NaN otherwise. The formulas are invariant under
// flipping the y axis, so north-up and south-up grids agree.
func ZTCurvatureRow(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32, kind CurvatureKind) {
	requireRows(len(dst), r0, r1, r2)
	if kind < CurvProfile || kind > CurvMean {
		panic(fmt.Sprintf("stencil: unknown CurvatureKind %d", kind))
	}
	ztCurvatureRow(dst, r0, r1, r2, kp, kq, kr, kt, ks, kind)
}

// ztViews is hornViews with the centre cell: vN[i] is cell N of the
// window centred on dst[i].
func ztViews(n int, r0, r1, r2 []float32) (v1, v2, v3, v4, v5, v6, v7, v8, v9 []float32) {
	return r0[0:n], r0[1 : n+1], r0[2 : n+2],
		r1[0:n], r1[1 : n+1], r1[2 : n+2],
		r2[0:n], r2[1 : n+1], r2[2 : n+2]
}

// ztDerivs is the five scaled Zevenbergen–Thorne derivatives of one
// window. The grouping is shared with the SIMD kernels.
func ztDerivs(z1, z2, z3, z4, z5, z6, z7, z8, z9, kp, kq, kr, kt, ks float32) (p, q, r, s, t float32) {
	p = float32((z6 - z4) * kp)
	q = float32((z8 - z2) * kq)
	c := z5 + z5
	r = float32(((z4 + z6) - c) * kr)
	t = float32(((z2 + z8) - c) * kt)
	s = float32(((z1 + z9) - (z3 + z7)) * ks)
	return p, q, r, s, t
}

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func scalarZTCurvatureRow(dst, r0, r1, r2 []float32, kp, kq, kr, kt, ks float32, kind CurvatureKind) {
	v1, v2, v3, v4, v5, v6, v7, v8, v9 := ztViews(len(dst), r0, r1, r2)
	switch kind {
	case CurvProfile:
		for i := range dst {
			p, q, r, s, t := ztDerivs(v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i], v8[i], v9[i], kp, kq, kr, kt, ks)
			p2, q2 := float32(p*p), float32(q*q)
			pq := float32(p * q)
			g := p2 + q2
			w := 1 + g
			num := (float32(p2*r) + float32((pq+pq)*s)) + float32(q2*t)
			v := 0 - num/g/float32(w*sqrt32(w))
			if g == 0 {
				v = num + 0
			}
			dst[i] = v
		}
	case CurvPlan:
		for i := range dst {
			p, q, r, s, t := ztDerivs(v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i], v8[i], v9[i], kp, kq, kr, kt, ks)
			p2, q2 := float32(p*p), float32(q*q)
			pq := float32(p * q)
			g := p2 + q2
			num := (float32(q2*r) - float32((pq+pq)*s)) + float32(p2*t)
			v := 0 - num/g/sqrt32(g)
			if g == 0 {
				v = num + 0
			}
			dst[i] = v
		}
	default:
		for i := range dst {
			p, q, r, s, t := ztDerivs(v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i], v8[i], v9[i], kp, kq, kr, kt, ks)
			p2, q2 := float32(p*p), float32(q*q)
			pq := float32(p * q)
			w := 1 + (p2 + q2)
			num := (float32((1+q2)*r) - float32((pq+pq)*s)) + float32((1+p2)*t)
			d := float32(w * sqrt32(w))
			dst[i] = 0 - num/(d+d)
		}
	}
}

// CurvatureFromDerivsRow writes a curvature from derivatives already
// computed: p, q, r, s and t for each cell, as ZTCurvatureRow computes
// them from the 3×3 window, or as another fit estimates them. It is
// ZTCurvatureRow's second half, the same float32 operations in the same
// order, so ZT derivatives followed by this are ZTCurvatureRow bit for
// bit. All six slices must have the same length. It is scalar on every
// build.
func CurvatureFromDerivsRow(dst, p, q, r, s, t []float32, kind CurvatureKind) {
	n := len(dst)
	if len(p) != n || len(q) != n || len(r) != n || len(s) != n || len(t) != n {
		panic("stencil: p, q, r, s, t and dst must have equal length")
	}
	if kind < CurvProfile || kind > CurvMean {
		panic(fmt.Sprintf("stencil: unknown CurvatureKind %d", kind))
	}
	p, q, r, s, t = p[:n], q[:n], r[:n], s[:n], t[:n]
	switch kind {
	case CurvProfile:
		for i := range dst {
			p2, q2 := float32(p[i]*p[i]), float32(q[i]*q[i])
			pq := float32(p[i] * q[i])
			g := p2 + q2
			w := 1 + g
			num := (float32(p2*r[i]) + float32((pq+pq)*s[i])) + float32(q2*t[i])
			v := 0 - num/g/float32(w*sqrt32(w))
			if g == 0 {
				v = num + 0
			}
			dst[i] = v
		}
	case CurvPlan:
		for i := range dst {
			p2, q2 := float32(p[i]*p[i]), float32(q[i]*q[i])
			pq := float32(p[i] * q[i])
			g := p2 + q2
			num := (float32(q2*r[i]) - float32((pq+pq)*s[i])) + float32(p2*t[i])
			v := 0 - num/g/sqrt32(g)
			if g == 0 {
				v = num + 0
			}
			dst[i] = v
		}
	default:
		for i := range dst {
			p2, q2 := float32(p[i]*p[i]), float32(q[i]*q[i])
			pq := float32(p[i] * q[i])
			w := 1 + (p2 + q2)
			num := (float32((1+q2)*r[i]) - float32((pq+pq)*s[i])) + float32((1+p2)*t[i])
			d := float32(w * sqrt32(w))
			dst[i] = 0 - num/(d+d)
		}
	}
}
