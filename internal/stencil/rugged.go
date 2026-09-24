package stencil

import (
	"fmt"
	"math"
)

// This file holds the scalar ruggedness kernels. Like horn.go it is the
// canonical backend.

// RuggednessKind selects which measure RuggednessRow writes.
type RuggednessKind int

const (
	// RugTRIRiley is Riley's terrain ruggedness index: the root of the
	// summed squared differences between the centre and its neighbours.
	RugTRIRiley RuggednessKind = iota
	// RugTRIWilson is Wilson's terrain ruggedness index: the mean
	// absolute difference between the centre and its neighbours.
	RugTRIWilson
	// RugTPI is the topographic position index: the centre minus the
	// mean of its neighbours.
	RugTPI
	// RugRoughness is the largest minus the smallest of all nine cells.
	RugRoughness
)

// RuggednessRow computes one row of a ruggedness measure of the 3×3
// window (z1..z9 as in HornGradientRow, z5 the centre). Each formula is
// GDAL gdaldem's (apps/gdaldem_lib.cpp, for float32 input) operation for
// operation, so that the results are bit-identical to gdaldem's. With
// di = zi - z5 rounded to float32 and the eight neighbours taken in
// row-major order (1, 2, 3, 4, 6, 7, 8, 9), kind selects:
//
//	RugTRIRiley   float32(√(d1² + d2² + … + d9²)), squares, sum and root in float64
//	RugTRIWilson  (|d1| + |d2| + … + |d9|) · 0.125
//	RugTPI        z5 - (z1 + z2 + … + z9) · 0.125
//	RugRoughness  max(z1..z9) - min(z1..z9)
//
// with every sum folded left to right. Riley's squares of float32
// differences are exact in float64, so only the sum and the root round
// before the final conversion. Roughness uses Go's min and max: a NaN
// anywhere in the window gives NaN. Its result does not depend on the
// order.
func RuggednessRow(dst, r0, r1, r2 []float32, kind RuggednessKind) {
	requireRows(len(dst), r0, r1, r2)
	if kind < RugTRIRiley || kind > RugRoughness {
		panic(fmt.Sprintf("stencil: unknown RuggednessKind %d", kind))
	}
	ruggednessRow(dst, r0, r1, r2, kind)
}

func abs32(x float32) float32 { return math.Float32frombits(math.Float32bits(x) &^ signBit32) }

// sq64 is the square of d in float64, which is exact.
func sq64(d float32) float64 { x := float64(d); return float64(x * x) }

func scalarRuggednessRow(dst, r0, r1, r2 []float32, kind RuggednessKind) {
	v1, v2, v3, v4, v5, v6, v7, v8, v9 := ztViews(len(dst), r0, r1, r2)
	switch kind {
	case RugTRIRiley:
		for i := range dst {
			c := v5[i]
			s := sq64(v1[i]-c) + sq64(v2[i]-c)
			s += sq64(v3[i] - c)
			s += sq64(v4[i] - c)
			s += sq64(v6[i] - c)
			s += sq64(v7[i] - c)
			s += sq64(v8[i] - c)
			s += sq64(v9[i] - c)
			dst[i] = float32(math.Sqrt(s))
		}
	case RugTRIWilson:
		for i := range dst {
			c := v5[i]
			s := abs32(v1[i]-c) + abs32(v2[i]-c)
			s += abs32(v3[i] - c)
			s += abs32(v4[i] - c)
			s += abs32(v6[i] - c)
			s += abs32(v7[i] - c)
			s += abs32(v8[i] - c)
			s += abs32(v9[i] - c)
			dst[i] = float32(s * 0.125)
		}
	case RugTPI:
		for i := range dst {
			s := v1[i] + v2[i]
			s += v3[i]
			s += v4[i]
			s += v6[i]
			s += v7[i]
			s += v8[i]
			s += v9[i]
			dst[i] = v5[i] - float32(s*0.125)
		}
	default:
		for i := range dst {
			hi := max(v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i], v8[i], v9[i])
			lo := min(v1[i], v2[i], v3[i], v4[i], v5[i], v6[i], v7[i], v8[i], v9[i])
			dst[i] = hi - lo
		}
	}
}

// RuggednessWindowRow computes one row of a ruggedness measure over the
// (2r+1)×(2r+1) window around each cell, for r ≥ 1. rows holds the
// window's 2r+1 input rows, top to bottom, each starting r columns left
// of dst[0] with at least len(dst)+2r cells, so dst[i] is centred on
// rows[r][i+r]. It is RuggednessRow with the eight neighbours replaced by
// the n = (2r+1)²−1 cells of the window other than the centre, taken in
// row-major order, and the · 0.125 by a division by n:
//
//	RugTRIRiley   float32(√(Σ di²)), squares, sum and root in float64
//	RugTRIWilson  (Σ |di|) / n
//	RugTPI        z_centre - (Σ zi) / n
//	RugRoughness  max over the window - min over the window
//
// with every sum folded left to right from its first term. Dividing by 8
// and multiplying by 0.125 round the same exact quotient, so at r = 1 it
// writes RuggednessRow's bits.
//
// It is scalar on every build. Each cell is summed afresh, O(r²), in the
// same order whatever the row or the tiling, so its results do not
// depend on where a band starts (DESIGN.md §53, "why nothing slides").
func RuggednessWindowRow(dst []float32, rows [][]float32, kind RuggednessKind) {
	if len(rows) < 3 || len(rows)%2 == 0 {
		panic(fmt.Sprintf("stencil: a ruggedness window needs an odd number of rows, at least 3, got %d", len(rows)))
	}
	w := len(dst) + len(rows) - 1
	for j, row := range rows {
		if len(row) < w {
			panic(fmt.Sprintf("stencil: input row %d has %d cells, need %d", j, len(row), w))
		}
	}
	if kind < RugTRIRiley || kind > RugRoughness {
		panic(fmt.Sprintf("stencil: unknown RuggednessKind %d", kind))
	}
	scalarRuggednessWindowRow(dst, rows, kind)
}

// rugBlock is how many cells scalarRuggednessWindowRow accumulates at a
// time: the terms of the window are the outer loop and the cells the
// inner one, as in internal/focalrow's scalar kernels, so each term is
// a pass over a block that stays in L1, and Riley's float64 sums fit on
// the stack. Each pass is a helper that takes eight cells a step: a
// one-cell loop runs at half speed when it spans two 64-byte lines of
// code, which is up to the rest of the binary (DESIGN.md §53, "Loop
// placement").
const rugBlock = 256

func scalarRuggednessWindowRow(dst []float32, rows [][]float32, kind RuggednessKind) {
	d := len(rows)        // the window's side, 2r+1
	r := d / 2            // its radius
	n := float32(d*d - 1) // the neighbours, exact in float32 up to d = 4096
	var acc [rugBlock]float64
	var lo [rugBlock]float32
	for i0 := 0; i0 < len(dst); i0 += rugBlock {
		out := dst[i0:min(i0+rugBlock, len(dst))]
		c := rows[r][i0+r:][:len(out)]
		a, l := acc[:len(out)], lo[:len(out)]
		first := true
		for j := range d {
			for k := range d {
				// v is the window cell at row j, column k under each
				// cell of the block.
				v := rows[j][i0+k:][:len(out)]
				switch {
				case kind == RugRoughness && first:
					copy(out, v)
					copy(l, v)
				case kind == RugRoughness:
					maxMinRow(out, l, v)
				case j == r && k == r:
					continue // the centre is not a neighbour
				case kind == RugTRIRiley && first:
					setSqDiffRow(a, v, c)
				case kind == RugTRIRiley:
					addSqDiffRow(a, v, c)
				case kind == RugTRIWilson && first:
					setAbsDiffRow(out, v, c)
				case kind == RugTRIWilson:
					addAbsDiffRow(out, v, c)
				case first:
					copy(out, v)
				default:
					addRow(out, v)
				}
				first = false
			}
		}
		switch kind {
		case RugTRIRiley:
			for i := range out {
				out[i] = float32(math.Sqrt(a[i]))
			}
		case RugTRIWilson:
			for i := range out {
				out[i] /= n
			}
		case RugTPI:
			for i := range out {
				out[i] = c[i] - out[i]/n
			}
		default:
			for i := range out {
				out[i] -= l[i]
			}
		}
	}
}

// addRow sets dst[i] += v[i]. v must have len(dst) cells.
//
//go:noinline
func addRow(dst, v []float32) {
	v = v[:len(dst)]
	for len(dst) > 8 && len(v) > 8 {
		dst[0] += v[0]
		dst[1] += v[1]
		dst[2] += v[2]
		dst[3] += v[3]
		dst[4] += v[4]
		dst[5] += v[5]
		dst[6] += v[6]
		dst[7] += v[7]
		dst, v = dst[8:], v[8:]
	}
	v = v[:len(dst)]
	for i := range dst {
		dst[i] += v[i]
	}
}

// setAbsDiffRow sets dst[i] = |v[i] - c[i]|, and addAbsDiffRow adds it.
// v and c must have len(dst) cells.
//
//go:noinline
func setAbsDiffRow(dst, v, c []float32) {
	for i := range dst {
		dst[i] = 0
	}
	addAbsDiffRow(dst, v, c)
}

// addAbsDiffRow is setAbsDiffRow's sum. Starting setAbsDiffRow from +0
// gives the first term's bits: |d| is never -0, and +0 + x is x for
// every x that is not -0.
//
//go:noinline
func addAbsDiffRow(dst, v, c []float32) {
	v, c = v[:len(dst)], c[:len(dst)]
	for len(dst) > 8 && len(v) > 8 && len(c) > 8 {
		dst[0] += abs32(v[0] - c[0])
		dst[1] += abs32(v[1] - c[1])
		dst[2] += abs32(v[2] - c[2])
		dst[3] += abs32(v[3] - c[3])
		dst[4] += abs32(v[4] - c[4])
		dst[5] += abs32(v[5] - c[5])
		dst[6] += abs32(v[6] - c[6])
		dst[7] += abs32(v[7] - c[7])
		dst, v, c = dst[8:], v[8:], c[8:]
	}
	v, c = v[:len(dst)], c[:len(dst)]
	for i := range dst {
		dst[i] += abs32(v[i] - c[i])
	}
}

// setSqDiffRow sets a[i] = (v[i] - c[i])² in float64, and addSqDiffRow
// adds it. Starting from +0 is exact for the same reason as for
// setAbsDiffRow: a square is never -0.
//
//go:noinline
func setSqDiffRow(a []float64, v, c []float32) {
	for i := range a {
		a[i] = 0
	}
	addSqDiffRow(a, v, c)
}

//go:noinline
func addSqDiffRow(a []float64, v, c []float32) {
	v, c = v[:len(a)], c[:len(a)]
	for len(a) > 8 && len(v) > 8 && len(c) > 8 {
		a[0] += sq64(v[0] - c[0])
		a[1] += sq64(v[1] - c[1])
		a[2] += sq64(v[2] - c[2])
		a[3] += sq64(v[3] - c[3])
		a[4] += sq64(v[4] - c[4])
		a[5] += sq64(v[5] - c[5])
		a[6] += sq64(v[6] - c[6])
		a[7] += sq64(v[7] - c[7])
		a, v, c = a[8:], v[8:], c[8:]
	}
	v, c = v[:len(a)], c[:len(a)]
	for i := range a {
		a[i] += sq64(v[i] - c[i])
	}
}

// maxMinRow sets hi[i] = max(hi[i], v[i]) and lo[i] = min(lo[i], v[i]).
// v and lo must have len(hi) cells.
//
//go:noinline
func maxMinRow(hi, lo, v []float32) {
	lo, v = lo[:len(hi)], v[:len(hi)]
	for i := range hi {
		hi[i] = max(hi[i], v[i])
		lo[i] = min(lo[i], v[i])
	}
}
