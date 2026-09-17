package nodata

import "math"

// 3×3 Horn gradient magnitude over a w×h raster (stride == w):
//
//	z1 z2 z3
//	z4 z5 z6
//	z7 z8 z9
//
//	dz/dx = ((z3 + 2·z6 + z9) - (z1 + 2·z4 + z7)) / (8·cellSize)
//	dz/dy = ((z7 + 2·z8 + z9) - (z1 + 2·z2 + z3)) / (8·cellSize)
//	out   = sqrt(dz/dx² + dz/dy²)
//
// Any NoData neighbour (z5 included) makes the output NoData, and the
// one-cell border is NoData. Every variant evaluates the arithmetic in
// the same order as slopeRowScalar and the archsimd kernels, and explicit
// float32 conversions stop the compiler fusing into FMA, so all
// variants agree bit-for-bit on valid cells.

func hornScales(cellSize float32) (invx, invy float32) {
	inv := 1 / (8 * cellSize)
	return inv, inv
}

// horn evaluates one cell from its eight contributing neighbours.
func horn(z1, z2, z3, z4, z6, z7, z8, z9, invx, invy float32) float32 {
	gx := float32((((z3 + z9) + (z6 + z6)) - ((z1 + z7) + (z4 + z4))) * invx)
	gy := float32((((z7 + z9) + (z8 + z8)) - ((z1 + z3) + (z2 + z2))) * invy)
	return float32(math.Sqrt(float64(float32(gx*gx) + float32(gy*gy))))
}

// hornNaN is horn plus a z5·0 term. The Horn stencil never reads the
// centre cell, so without it a NaN centre would not propagate: with NaN
// as NoData every kernel must deliberately touch its whole footprint.
// The sum of squares is >= +0 and +0 + ±0 = +0, so valid results stay
// bit-identical to horn (a ±Inf centre does become NaN, though).
func hornNaN(z1, z2, z3, z4, z5, z6, z7, z8, z9, invx, invy float32) float32 {
	gx := float32((((z3 + z9) + (z6 + z6)) - ((z1 + z7) + (z4 + z4))) * invx)
	gy := float32((((z7 + z9) + (z8 + z8)) - ((z1 + z3) + (z2 + z2))) * invy)
	s := float32(gx*gx) + float32(gy*gy)
	s = s + float32(z5*0)
	return float32(math.Sqrt(float64(s)))
}

// slopeRowNaNScalar is slopeRowScalar using hornNaN.
func slopeRowNaNScalar(dst, r0, r1, r2 []float32, invx, invy float32) {
	n := len(dst)
	r0 = r0[:n+2]
	r1 = r1[:n+2]
	r2 = r2[:n+2]
	for i := range dst {
		dst[i] = hornNaN(r0[i], r0[i+1], r0[i+2], r1[i], r1[i+1], r1[i+2], r2[i], r2[i+1], r2[i+2], invx, invy)
	}
}

// slopeRowScalar fills dst[0:n]; r0, r1, r2 are rows y-1, y, y+1 starting
// at column x-1 of dst[0], with at least n+2 elements.
func slopeRowScalar(dst, r0, r1, r2 []float32, invx, invy float32) {
	n := len(dst)
	r0 = r0[:n+2]
	r1 = r1[:n+2]
	r2 = r2[:n+2]
	for i := range dst {
		dst[i] = horn(r0[i], r0[i+1], r0[i+2], r1[i], r1[i+2], r2[i], r2[i+1], r2[i+2], invx, invy)
	}
}

// fillBorder writes v into the one-cell border of dst.
func fillBorder(dst []float32, w, h int, v float32) {
	for x := 0; x < w; x++ {
		dst[x] = v
		dst[(h-1)*w+x] = v
	}
	for y := 1; y < h-1; y++ {
		dst[y*w] = v
		dst[y*w+w-1] = v
	}
}

type rowKernel func(dst, r0, r1, r2 []float32, invx, invy float32)

func slopeRows(dst, src []float32, w, h int, cellSize float32, row rowKernel) {
	invx, invy := hornScales(cellSize)
	for y := 1; y < h-1; y++ {
		row(dst[y*w+1:y*w+w-1], src[(y-1)*w:y*w], src[y*w:(y+1)*w], src[(y+1)*w:(y+2)*w], invx, invy)
	}
}

// SlopeNaNScalar: plain loop, NaN propagates through the arithmetic.
func SlopeNaNScalar(dst, src []float32, w, h int, cellSize float32) {
	fillBorder(dst, w, h, NaN32)
	slopeRows(dst, src, w, h, cellSize, slopeRowNaNScalar)
}

// SlopeSentinelBranchy checks the nine inputs before computing.
func SlopeSentinelBranchy(dst, src []float32, w, h int, cellSize, nd float32) {
	fillBorder(dst, w, h, nd)
	invx, invy := hornScales(cellSize)
	for y := 1; y < h-1; y++ {
		r0 := src[(y-1)*w : y*w]
		r1 := src[y*w : (y+1)*w]
		r2 := src[(y+1)*w : (y+2)*w]
		out := dst[y*w : (y+1)*w]
		for x := 1; x < w-1; x++ {
			z1, z2, z3 := r0[x-1], r0[x], r0[x+1]
			z4, z5, z6 := r1[x-1], r1[x], r1[x+1]
			z7, z8, z9 := r2[x-1], r2[x], r2[x+1]
			if z1 == nd || z2 == nd || z3 == nd ||
				z4 == nd || z5 == nd || z6 == nd ||
				z7 == nd || z8 == nd || z9 == nd {
				out[x] = nd
				continue
			}
			out[x] = horn(z1, z2, z3, z4, z6, z7, z8, z9, invx, invy)
		}
	}
}

// slopeRowSentinelSelect computes unconditionally, then selects nd where
// any input matched, without short-circuit evaluation.
func slopeRowSentinelSelect(dst, r0, r1, r2 []float32, invx, invy, nd float32) {
	n := len(dst)
	r0 = r0[:n+2]
	r1 = r1[:n+2]
	r2 = r2[:n+2]
	for i := range dst {
		z1, z2, z3 := r0[i], r0[i+1], r0[i+2]
		z4, z5, z6 := r1[i], r1[i+1], r1[i+2]
		z7, z8, z9 := r2[i], r2[i+1], r2[i+2]
		s := horn(z1, z2, z3, z4, z6, z7, z8, z9, invx, invy)
		bad := b2u(z1 == nd) | b2u(z2 == nd) | b2u(z3 == nd) |
			b2u(z4 == nd) | b2u(z5 == nd) | b2u(z6 == nd) |
			b2u(z7 == nd) | b2u(z8 == nd) | b2u(z9 == nd)
		if bad != 0 {
			s = nd
		}
		dst[i] = s
	}
}

// SlopeSentinelSelect: compute unconditionally, then select.
func SlopeSentinelSelect(dst, src []float32, w, h int, cellSize, nd float32) {
	fillBorder(dst, w, h, nd)
	invx, invy := hornScales(cellSize)
	for y := 1; y < h-1; y++ {
		slopeRowSentinelSelect(dst[y*w+1:y*w+w-1], src[(y-1)*w:y*w], src[y*w:(y+1)*w], src[(y+1)*w:(y+2)*w], invx, invy, nd)
	}
}

// SlopeMaskBranchy tests the nine validity bits per cell and computes
// only valid cells. Data under invalid output cells is left untouched.
func SlopeMaskBranchy(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32) {
	clear(dstValid)
	invx, invy := hornScales(cellSize)
	bit := func(i int) uint64 { return valid[i>>6] >> uint(i&63) & 1 }
	for y := 1; y < h-1; y++ {
		r0 := src[(y-1)*w : y*w]
		r1 := src[y*w : (y+1)*w]
		r2 := src[(y+1)*w : (y+2)*w]
		i0, i1, i2 := (y-1)*w, y*w, (y+1)*w
		for x := 1; x < w-1; x++ {
			ok := bit(i0+x-1) & bit(i0+x) & bit(i0+x+1) &
				bit(i1+x-1) & bit(i1+x) & bit(i1+x+1) &
				bit(i2+x-1) & bit(i2+x) & bit(i2+x+1)
			if ok == 0 {
				continue
			}
			dst[i1+x] = horn(r0[x-1], r0[x], r0[x+1], r1[x-1], r1[x+1], r2[x-1], r2[x], r2[x+1], invx, invy)
			dstValid[(i1+x)>>6] |= 1 << uint((i1+x)&63)
		}
	}
}

// SlopeMaskScalar: plain loop over every interior cell plus the
// word-level SlopeMask pass. Data under invalid output cells is garbage.
func SlopeMaskScalar(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64) {
	slopeRows(dst, src, w, h, cellSize, slopeRowScalar)
	SlopeMask(dstValid, valid, w, h, scratch)
}
