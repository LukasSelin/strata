//go:build goexperiment.simd && amd64

package nodata

import "simd/archsimd"

// The vectorized variants, written with simd/archsimd like internal/vec
// (docs/adr/0001-simd-backend.md). They exist only in GOEXPERIMENT=simd
// builds; other builds get the scalar fallbacks in simd_other.go.
//
// Loops use the ADR's bounds-check-free form (array-pointer loads over
// slices that shrink by one lane per iteration) and clear the upper AVX
// bits before the scalar tail. Arithmetic follows horn/hornNaN in slope.go
// operation for operation, so valid results are bit-identical to scalar.

// HaveAVX2 reports whether the AVX2 variants can run on this machine.
var HaveAVX2 = archsimd.X86.AVX2()

const lane = 8

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// Each kernel is split in two: a *Lanes function that runs the vector
// loop and returns how many cells it wrote, and a wrapper that runs the
// scalar tail. If the float32 parameters stay live across the loop (the
// tail needs them), the Go 1.27 compiler may spill one and reload it with
// a legacy-SSE MOVSS inside the AVX loop, which costs an SSE/AVX transition
// penalty on every iteration (golang/go#80835). An early single-function
// sentinel slope kernel hit this and ran at 11 ns/cell, slower than scalar.

// AddSentinelAVX2 computes a + b, compares both inputs against nd and
// blends nd into the flagged lanes, with the select form for the tail.
func AddSentinelAVX2(dst, a, b []float32, nd float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	i := addSentinelLanes(dst, a, b, nd)
	AddSentinelSelect(dst[i:], a[i:], b[i:], nd)
}

func addSentinelLanes(dst, a, b []float32, nd float32) int {
	vnd := archsimd.BroadcastFloat32x8(nd)
	n := len(dst)
	a, b = a[:n], b[:n]
	for len(dst) >= lane && len(a) >= lane && len(b) >= lane {
		x, y := load8(a), load8(b)
		bad := x.Equal(vnd).Or(y.Equal(vnd))
		store8(vnd.IfElse(bad, x.Add(y)), dst)
		dst, a, b = dst[lane:], a[lane:], b[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// hornLanes is the vector form of the shared part of horn: the sum of
// squared gradients, before the square root.
func hornLanes(r0, r1, r2 []float32, kx, ky archsimd.Float32x8) archsimd.Float32x8 {
	z1, z2, z3 := load8(r0), load8(r0[1:]), load8(r0[2:])
	z4, z6 := load8(r1), load8(r1[2:])
	z7, z8, z9 := load8(r2), load8(r2[1:]), load8(r2[2:])
	gx := z3.Add(z9).Add(z6.Add(z6)).Sub(z1.Add(z7).Add(z4.Add(z4))).Mul(kx)
	gy := z7.Add(z9).Add(z8.Add(z8)).Sub(z1.Add(z3).Add(z2.Add(z2))).Mul(ky)
	return gx.Mul(gx).Add(gy.Mul(gy))
}

// rowTail trims the rows to n+2 cells and returns the tails after i cells.
func rowTail(dst, r0, r1, r2 []float32, i int) ([]float32, []float32, []float32, []float32) {
	n := len(dst)
	return dst[i:], r0[i : n+2], r1[i : n+2], r2[i : n+2]
}

func slopeRowAVX2(dst, r0, r1, r2 []float32, invx, invy float32) {
	i := slopeLanes(dst, r0, r1, r2, invx, invy, false)
	dst, r0, r1, r2 = rowTail(dst, r0, r1, r2, i)
	slopeRowScalar(dst, r0, r1, r2, invx, invy)
}

// slopeRowNaNAVX2 adds hornNaN's z5·0 term so a NaN centre propagates.
func slopeRowNaNAVX2(dst, r0, r1, r2 []float32, invx, invy float32) {
	i := slopeLanes(dst, r0, r1, r2, invx, invy, true)
	dst, r0, r1, r2 = rowTail(dst, r0, r1, r2, i)
	slopeRowNaNScalar(dst, r0, r1, r2, invx, invy)
}

// slopeLanes is the Horn row loop; centre adds the z5·0 term. The branch
// on centre is outside the loop, so each loop compiles without it.
func slopeLanes(dst, r0, r1, r2 []float32, invx, invy float32, centre bool) int {
	kx := archsimd.BroadcastFloat32x8(invx)
	ky := archsimd.BroadcastFloat32x8(invy)
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	if centre {
		var zero archsimd.Float32x8
		for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
			s := hornLanes(r0, r1, r2, kx, ky).Add(load8(r1[1:]).Mul(zero))
			store8(s.Sqrt(), dst)
			dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
		}
	} else {
		for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
			store8(hornLanes(r0, r1, r2, kx, ky).Sqrt(), dst)
			dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
		}
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// slopeRowSentinelAVX2 adds nine compares against nd and a final blend.
func slopeRowSentinelAVX2(dst, r0, r1, r2 []float32, invx, invy, nd float32) {
	i := slopeSentinelLanes(dst, r0, r1, r2, invx, invy, nd)
	dst, r0, r1, r2 = rowTail(dst, r0, r1, r2, i)
	slopeRowSentinelSelect(dst, r0, r1, r2, invx, invy, nd)
}

func slopeSentinelLanes(dst, r0, r1, r2 []float32, invx, invy, nd float32) int {
	kx := archsimd.BroadcastFloat32x8(invx)
	ky := archsimd.BroadcastFloat32x8(invy)
	vnd := archsimd.BroadcastFloat32x8(nd)
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	for len(dst) >= lane && len(r0) >= lane+2 && len(r1) >= lane+2 && len(r2) >= lane+2 {
		bad := load8(r0).Equal(vnd).
			Or(load8(r0[1:]).Equal(vnd)).
			Or(load8(r0[2:]).Equal(vnd)).
			Or(load8(r1).Equal(vnd)).
			Or(load8(r1[1:]).Equal(vnd)).
			Or(load8(r1[2:]).Equal(vnd)).
			Or(load8(r2).Equal(vnd)).
			Or(load8(r2[1:]).Equal(vnd)).
			Or(load8(r2[2:]).Equal(vnd))
		store8(vnd.IfElse(bad, hornLanes(r0, r1, r2, kx, ky).Sqrt()), dst)
		dst, r0, r1, r2 = dst[lane:], r0[lane:], r1[lane:], r2[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// SlopeNaNAVX2: vectorized Horn kernel; NaN propagates through the
// arithmetic, with the extra z5·0 term (see hornNaN).
func SlopeNaNAVX2(dst, src []float32, w, h int, cellSize float32) {
	fillBorder(dst, w, h, NaN32)
	slopeRows(dst, src, w, h, cellSize, slopeRowNaNAVX2)
}

// SlopeSentinelAVX2: vectorized Horn kernel with nine compares and a blend.
func SlopeSentinelAVX2(dst, src []float32, w, h int, cellSize, nd float32) {
	fillBorder(dst, w, h, nd)
	invx, invy := hornScales(cellSize)
	for y := 1; y < h-1; y++ {
		slopeRowSentinelAVX2(dst[y*w+1:y*w+w-1], src[(y-1)*w:y*w], src[y*w:(y+1)*w], src[(y+1)*w:(y+2)*w], invx, invy, nd)
	}
}

// SlopeMaskAVX2: the vectorized Horn kernel over every interior cell (no
// centre term needed), plus the word-level SlopeMask pass.
func SlopeMaskAVX2(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64) {
	slopeRows(dst, src, w, h, cellSize, slopeRowAVX2)
	SlopeMask(dstValid, valid, w, h, scratch)
}

// SlopeMaskAVX2Fill additionally writes fill under invalid output cells.
func SlopeMaskAVX2Fill(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64, fill float32) {
	SlopeMaskAVX2(dst, src, dstValid, valid, w, h, cellSize, scratch)
	FillInvalid(dst, dstValid, fill)
}
