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
