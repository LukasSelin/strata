//go:build goexperiment.simd && arm64

package focalrow

import "simd/archsimd"

// This file is the NEON backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). It is simd_amd64.go four lanes at a
// time: each lane holds one output cell's accumulator and folds that
// cell's terms in the scalar kernel's order, so results match bit for
// bit. The compiler does not fuse archsimd's Mul and Add into FMLA.
//
// A block is four vectors, 16 adjacent cells, with an accumulator each,
// read through array pointers sliced with constant bounds, so each term
// costs one bounds check per block and none per load. There is no
// SSE/AVX transition on arm64, so nothing is cleared before the tail;
// the shapes are kept so the two files read side by side.

const (
	lane  = 4
	block = 4 * lane
)

func init() {
	// NEON is part of the arm64 baseline: no feature check.
	simdKernels = &kernelSet{
		correlateRow:    correlateRowNEON,
		columnCorrelate: columnCorrelateNEON,
		columnSum:       columnSumNEON,
		columnMin:       columnMinNEON,
		columnMax:       columnMaxNEON,
		rowCorrelate:    rowCorrelateNEON,
		rowMean:         rowMeanNEON,
		rowMin:          rowMinNEON,
		rowMax:          rowMaxNEON,
	}
	simdName = "neon"
	UseScalar(false)
}

func load4(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[lane]float32)(s))
}

func store4(v archsimd.Float32x4, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// quad is one block of cells read through an array pointer.
type quad = [block]float32

func loadQuad(p *quad) (a, b, c, d archsimd.Float32x4) {
	return archsimd.LoadFloat32x4Array((*[lane]float32)(p[0:lane])),
		archsimd.LoadFloat32x4Array((*[lane]float32)(p[lane : 2*lane])),
		archsimd.LoadFloat32x4Array((*[lane]float32)(p[2*lane : 3*lane])),
		archsimd.LoadFloat32x4Array((*[lane]float32)(p[3*lane : 4*lane]))
}

func storeQuad(p *quad, a, b, c, d archsimd.Float32x4) {
	a.StoreArray((*[lane]float32)(p[0:lane]))
	b.StoreArray((*[lane]float32)(p[lane : 2*lane]))
	c.StoreArray((*[lane]float32)(p[2*lane : 3*lane]))
	d.StoreArray((*[lane]float32)(p[3*lane : 4*lane]))
}

// min4 and max4 are Go's builtin min and max lanewise. NEON's FMIN and
// FMAX already are: a NaN in either operand gives NaN, and -0 orders
// below +0 (internal/vec/simd_arm64.go).
func min4(x, y archsimd.Float32x4) archsimd.Float32x4 { return x.Min(y) }

func max4(x, y archsimd.Float32x4) archsimd.Float32x4 { return x.Max(y) }

// weightedLanes writes dst[i] = Σ_t w[t]·src[i+t·step] over whole
// vectors and returns how many cells it wrote. The caller guarantees
// src[i+t·step] exists for every such cell.
func weightedLanes(dst, src []float32, step int, w []float32) int {
	n := len(dst)
	w0, ws := w[0], w[1:]
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		v0 := archsimd.BroadcastFloat32x4(w0)
		x0, x1, x2, x3 := loadQuad((*quad)(s))
		a0, a1, a2, a3 := v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
		off := 0
		for _, wt := range ws {
			off += step
			vw := archsimd.BroadcastFloat32x4(wt)
			x0, x1, x2, x3 := loadQuad((*quad)(s[off:]))
			a0, a1 = a0.Add(vw.Mul(x0)), a1.Add(vw.Mul(x1))
			a2, a3 = a2.Add(vw.Mul(x2)), a3.Add(vw.Mul(x3))
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= n; i += lane {
		s := src[i:]
		a := archsimd.BroadcastFloat32x4(w0).Mul(load4(s))
		off := 0
		for _, wt := range ws {
			off += step
			a = a.Add(archsimd.BroadcastFloat32x4(wt).Mul(load4(s[off:])))
		}
		store4(a, dst[i:])
	}
	return i
}

// correlate2DLanes is weightedLanes over a k×k neighbourhood: the term
// of weight w[j·k+c] reads src[i+j·stride+c].
func correlate2DLanes(dst, src []float32, stride int, w []float32, k int) int {
	n := len(dst)
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		v0 := archsimd.BroadcastFloat32x4(w[0])
		x0, x1, x2, x3 := loadQuad((*quad)(s))
		a0, a1, a2, a3 := v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
		for j := range k {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 {
				c0 = 1
			}
			for c := c0; c < len(wr); c++ {
				vw := archsimd.BroadcastFloat32x4(wr[c])
				x0, x1, x2, x3 := loadQuad((*quad)(row[c:]))
				a0, a1 = a0.Add(vw.Mul(x0)), a1.Add(vw.Mul(x1))
				a2, a3 = a2.Add(vw.Mul(x2)), a3.Add(vw.Mul(x3))
			}
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= n; i += lane {
		s := src[i:]
		a := archsimd.BroadcastFloat32x4(w[0]).Mul(load4(s))
		for j := range k {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 {
				c0 = 1
			}
			for c := c0; c < len(wr); c++ {
				a = a.Add(archsimd.BroadcastFloat32x4(wr[c]).Mul(load4(row[c:])))
			}
		}
		store4(a, dst[i:])
	}
	return i
}

// sumLanes writes dst[i] = src[i] + src[i+step] + … (k terms, in order),
// divided by n when div is set, over whole vectors.
func sumLanes(dst, src []float32, step, k int, div bool, n float32) int {
	m := len(dst)
	i := 0
	for ; i+block <= m; i += block {
		s := src[i:]
		a0, a1, a2, a3 := loadQuad((*quad)(s))
		off := 0
		for range k - 1 {
			off += step
			x0, x1, x2, x3 := loadQuad((*quad)(s[off:]))
			a0, a1, a2, a3 = a0.Add(x0), a1.Add(x1), a2.Add(x2), a3.Add(x3)
		}
		if div {
			vn := archsimd.BroadcastFloat32x4(n)
			a0, a1, a2, a3 = a0.Div(vn), a1.Div(vn), a2.Div(vn), a3.Div(vn)
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= m; i += lane {
		s := src[i:]
		a := load4(s)
		off := 0
		for range k - 1 {
			off += step
			a = a.Add(load4(s[off:]))
		}
		if div {
			a = a.Div(archsimd.BroadcastFloat32x4(n))
		}
		store4(a, dst[i:])
	}
	return i
}

// extremeLanes writes the min (or max) of src[i], src[i+step], … (k
// terms) over whole vectors.
func extremeLanes(dst, src []float32, step, k int, isMax bool) int {
	m := len(dst)
	i := 0
	for ; i+block <= m; i += block {
		s := src[i:]
		a0, a1, a2, a3 := loadQuad((*quad)(s))
		off := 0
		for range k - 1 {
			off += step
			x0, x1, x2, x3 := loadQuad((*quad)(s[off:]))
			if isMax {
				a0, a1, a2, a3 = max4(a0, x0), max4(a1, x1), max4(a2, x2), max4(a3, x3)
			} else {
				a0, a1, a2, a3 = min4(a0, x0), min4(a1, x1), min4(a2, x2), min4(a3, x3)
			}
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= m; i += lane {
		s := src[i:]
		a := load4(s)
		off := 0
		for range k - 1 {
			off += step
			if isMax {
				a = max4(a, load4(s[off:]))
			} else {
				a = min4(a, load4(s[off:]))
			}
		}
		store4(a, dst[i:])
	}
	return i
}

func correlateRowNEON(dst, src []float32, stride int, w []float32, k int) {
	i := correlate2DLanes(dst, src, stride, w, k)
	scalarCorrelateRow(dst[i:], src[i:], stride, w, k)
}

func columnCorrelateNEON(dst, src []float32, stride int, taps []float32) {
	i := weightedLanes(dst, src, stride, taps)
	scalarColumnCorrelate(dst[i:], src[i:], stride, taps)
}

func columnSumNEON(dst, src []float32, stride, k int) {
	i := sumLanes(dst, src, stride, k, false, 0)
	scalarColumnSum(dst[i:], src[i:], stride, k)
}

func columnMinNEON(dst, src []float32, stride, k int) {
	i := extremeLanes(dst, src, stride, k, false)
	scalarColumnMin(dst[i:], src[i:], stride, k)
}

func columnMaxNEON(dst, src []float32, stride, k int) {
	i := extremeLanes(dst, src, stride, k, true)
	scalarColumnMax(dst[i:], src[i:], stride, k)
}

func rowCorrelateNEON(dst, src, taps []float32) {
	i := weightedLanes(dst, src, 1, taps)
	scalarRowCorrelate(dst[i:], src[i:], taps)
}

func rowMeanNEON(dst, src []float32, k int, n float32) {
	i := sumLanes(dst, src, 1, k, true, n)
	scalarRowMean(dst[i:], src[i:], k, n)
}

func rowMinNEON(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, false)
	scalarRowMin(dst[i:], src[i:], k)
}

func rowMaxNEON(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, true)
	scalarRowMax(dst[i:], src[i:], k)
}
