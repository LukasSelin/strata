// Package stencil holds row kernels for radius-1 (3×3) neighbourhood
// operations such as terrain gradients, slope, aspect and hillshade, and
// the word-level validity erosion that goes with them.
//
// A row kernel fills dst[0:n] from three input rows r0, r1, r2 (rows y-1,
// y and y+1). Each input row starts one column left of dst[0] and has at
// least n+2 cells, so dst[i] is centred on r1[i+1]. Kernels know nothing
// about rasters, strides, masks or borders: callers slice rows out of
// their data and handle edges. That keeps them wrappable by the Kernel
// abstraction (DESIGN.md §22) and by tiled execution with halos (§23, §25).
//
// This file holds the scalar backend, which is canonical (DESIGN.md §15).
// SIMD backends (simd_amd64.go and simd_arm64.go, built with
// GOEXPERIMENT=simd) must agree with it bit-for-bit, any NaN matching any
// NaN. To make that possible the scalar code fixes its evaluation order
// and wraps every product in an explicit float32 conversion, which stops
// the compiler fusing a multiply-add into FMA (see
// docs/adr/0001-simd-backend.md).
//
// Like internal/vec, exported functions panic on mismatched lengths.
package stencil

import (
	"fmt"
	"math"
)

// Backend function variables, swapped in init by SIMD builds.
var (
	hornGradientRow  = scalarHornGradientRow
	hornSlopeRow     = scalarHornSlopeRow
	hornAspectRow    = scalarHornAspectRow
	hornHillshadeRow = scalarHornHillshadeRow
)

// simdGradient, simdSlope, simdAspect and simdHillshade are the SIMD set,
// or nil when this build or CPU has none. SIMD builds set all or none,
// and simdName, what Backend reports for them.
var (
	simdName      string
	simdGradient  func(dx, dy, r0, r1, r2 []float32, kx, ky float32)
	simdSlope     func(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool)
	simdAspect    func(dst, r0, r1, r2 []float32, kx, ky, flat float32, trig bool)
	simdHillshade func(dst, r0, r1, r2 []float32, kx, ky, c, bx, by float32)
)

// Backend names the kernels currently in use: "avx2", "neon" or "scalar".
func Backend() string {
	if simdSlope != nil && !usingScalar {
		return simdName
	}
	return "scalar"
}

var usingScalar bool

// UseScalar forces the scalar kernels (true) or restores the best
// available backend (false). It exists for equivalence tests and
// scalar-vs-SIMD benchmarks, and must not be called while kernels run.
func UseScalar(scalar bool) {
	usingScalar = scalar
	if scalar || simdSlope == nil {
		hornGradientRow, hornSlopeRow = scalarHornGradientRow, scalarHornSlopeRow
		hornAspectRow, hornHillshadeRow = scalarHornAspectRow, scalarHornHillshadeRow
		return
	}
	hornGradientRow, hornSlopeRow = simdGradient, simdSlope
	hornAspectRow, hornHillshadeRow = simdAspect, simdHillshade
}

func requireRows(n int, r0, r1, r2 []float32) {
	if len(r0) < n+2 || len(r1) < n+2 || len(r2) < n+2 {
		panic(fmt.Sprintf("stencil: input rows have %d, %d, %d cells, need %d",
			len(r0), len(r1), len(r2), n+2))
	}
}

// HornScales returns the per-axis factors the Horn row kernels multiply
// their weighted differences by: zFactor / (8·cellSizeX) and
// zFactor / (8·cellSizeY), computed in float64 and rounded once.
func HornScales(cellSizeX, cellSizeY, zFactor float64) (kx, ky float32) {
	return float32(zFactor / (8 * cellSizeX)), float32(zFactor / (8 * cellSizeY))
}

// HornGradientRow computes Horn's 3×3 gradient for one row:
//
//	z1 z2 z3        dx = ((z3 + 2·z6 + z9) - (z1 + 2·z4 + z7)) · kx
//	z4 z5 z6        dy = ((z7 + 2·z8 + z9) - (z1 + 2·z2 + z3)) · ky
//	z7 z8 z9
//
// with z1..z3 from r0, z4..z6 from r1 and z7..z9 from r2. dx is positive
// when values rise with the column index and dy when they rise with the
// row index. z5 is not used. dx and dy must have equal length n, and each
// input row at least n+2 cells.
func HornGradientRow(dx, dy, r0, r1, r2 []float32, kx, ky float32) {
	if len(dy) != len(dx) {
		panic("stencil: dx and dy must have equal length")
	}
	requireRows(len(dx), r0, r1, r2)
	hornGradientRow(dx, dy, r0, r1, r2, kx, ky)
}

// HornSlopeRow computes the Horn gradient magnitude m = sqrt(dx² + dy²)
// for one row (see HornGradientRow) and writes scale·m, or scale·atan(m)
// when atan is set. Slope in degrees is atan with scale 180/π, radians is
// atan with scale 1, percent is scale 100 without atan. atan is Atan32.
func HornSlopeRow(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool) {
	requireRows(len(dst), r0, r1, r2)
	hornSlopeRow(dst, r0, r1, r2, kx, ky, scale, atan)
}

// hornDX and hornDY are the weighted differences before scaling. The
// grouping is shared with the SIMD kernels. Opposite cells are subtracted
// first: a DEM's neighbours lie within a factor of two of each other, so
// each difference is exact (Sterbenz) and only the small sums round.
// Adding the elevations first would round at the ulp of four times the
// elevation (5.9e-3 at 8800 m) and bury the gradient of gentle terrain
// (TestHornGradientNearFlat, tools/herbie/RESULTS.md).
func hornDX(z1, z3, z4, z6, z7, z9 float32) float32 {
	d := z6 - z4
	return ((z3 - z1) + (z9 - z7)) + (d + d)
}

func hornDY(z1, z2, z3, z7, z8, z9 float32) float32 {
	d := z8 - z2
	return ((z7 - z1) + (z9 - z3)) + (d + d)
}

// hornViews returns the eight views of the input rows that a 3×3 kernel
// reads, one per cell of the window except the unused centre: vN[i] is
// cell N of the window centred on dst[i]. Indexing every operand with
// the loop variable alone is what lets the compiler drop the bounds
// checks; it cannot prove r0[i+1] and r0[i+2] in bounds from a range
// over a slice two cells shorter, and charged two checks per cell for
// them (TestNoBoundsChecksInLoops, DESIGN.md §39). The eight reslices
// here are checked once per row.
func hornViews(n int, r0, r1, r2 []float32) (v1, v2, v3, v4, v6, v7, v8, v9 []float32) {
	return r0[0:n], r0[1 : n+1], r0[2 : n+2],
		r1[0:n], r1[2 : n+2],
		r2[0:n], r2[1 : n+1], r2[2 : n+2]
}

func scalarHornGradientRow(dx, dy, r0, r1, r2 []float32, kx, ky float32) {
	n := len(dx)
	dy = dy[:n]
	v1, v2, v3, v4, v6, v7, v8, v9 := hornViews(n, r0, r1, r2)
	for i := range dx {
		z1, z2, z3 := v1[i], v2[i], v3[i]
		z4, z6 := v4[i], v6[i]
		z7, z8, z9 := v7[i], v8[i], v9[i]
		dx[i] = float32(hornDX(z1, z3, z4, z6, z7, z9) * kx)
		dy[i] = float32(hornDY(z1, z2, z3, z7, z8, z9) * ky)
	}
}

// scalarHornSlopeRow makes two passes when atan is set: magnitudes, then
// Atan32 of each in place. Atan32 is too big to inline, and calling it
// inside the stencil loop costs a spill and reload of every float
// register per cell (9.8 vs 6.8 ns/cell on Zen 2). Results are identical.
func scalarHornSlopeRow(dst, r0, r1, r2 []float32, kx, ky, scale float32, atan bool) {
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	if atan {
		scalarHornMagnitudeRow(dst, r0, r1, r2, kx, ky, 1)
		for i, m := range dst {
			dst[i] = float32(Atan32(m) * scale)
		}
		return
	}
	scalarHornMagnitudeRow(dst, r0, r1, r2, kx, ky, scale)
}

func scalarHornMagnitudeRow(dst, r0, r1, r2 []float32, kx, ky, scale float32) {
	v1, v2, v3, v4, v6, v7, v8, v9 := hornViews(len(dst), r0, r1, r2)
	for i := range dst {
		z1, z2, z3 := v1[i], v2[i], v3[i]
		z4, z6 := v4[i], v6[i]
		z7, z8, z9 := v7[i], v8[i], v9[i]
		gx := float32(hornDX(z1, z3, z4, z6, z7, z9) * kx)
		gy := float32(hornDY(z1, z2, z3, z7, z8, z9) * ky)
		dst[i] = float32(float32(math.Sqrt(float64(float32(gx*gx)+float32(gy*gy)))) * scale)
	}
}

// Atan32 constants: the Cephes atanf polynomial (S. L. Moshier), with the
// argument reduced into [-tan(π/8), tan(π/8)] around 0, π/4 or π/2.
const (
	atanTan3Pi8 = float32(2.414213562373095)  // tan(3π/8)
	atanTanPi8  = float32(0.4142135623730950) // tan(π/8)
	atanPi2     = float32(math.Pi / 2)
	atanPi4     = float32(math.Pi / 4)
	atanC4      = float32(8.05374449538e-2)
	atanC3      = float32(1.38776856032e-1)
	atanC2      = float32(1.99777106478e-1)
	atanC1      = float32(3.33329491539e-1)
)

// Atan32 is the arctangent used by the slope kernels, for x >= 0 (a
// gradient magnitude), +Inf or NaN. It is branch-free in the SIMD kernels
// and this is its canonical scalar form: SIMD lanes match it bit-for-bit.
//
// Checked exhaustively over every non-negative float32 against
// math.Atan (go1.27.0): the largest error is 1.41e-7 radians (8.1e-6
// degrees, at x ≈ 2.547) and never more than 3 float32 ULPs of the result;
// 95.1% of inputs give the correctly rounded float32(math.Atan(x)), 4.6%
// are one ULP off. TestAtan32Accuracy samples it against a 1.5e-7 bound. It
// returns +0 for +0, π/2 (as float32) for +Inf and NaN for NaN. Negative
// inputs are outside its contract.
func Atan32(x float32) float32 {
	num, den, y0 := x, float32(1), float32(0)
	if x > atanTanPi8 {
		num, den, y0 = x-1, x+1, atanPi4
	}
	if x > atanTan3Pi8 {
		num, den, y0 = -1, x, atanPi2
	}
	t := num / den
	z := float32(t * t)
	p := float32(atanC4*z) - atanC3
	p = float32(p*z) + atanC2
	p = float32(p*z) - atanC1
	return y0 + (float32(float32(p*z)*t) + t)
}
