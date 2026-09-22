//go:build goexperiment.simd && amd64

package focalrow

import "simd/archsimd"

// This file is the AVX2 backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). Each lane holds one output cell's
// accumulator and folds that cell's terms in the scalar kernel's order,
// so results match bit for bit.
//
// A block is four vectors, 32 adjacent cells, with an accumulator each:
// four independent add chains hide the latency of one, and every term
// costs one weight broadcast and one bounds check per block, not per
// vector. The weights are broadcast from memory with VBROADCASTSS after a
// VEX VMOVSS, so no legacy SSE runs inside the loop. Blocks are read
// through array pointers, sliced with constant bounds, which is what
// leaves one check per term (the conversion) and none per load. Whole
// vectors follow the blocks, then archsimd.ClearAVXUpperBits and the
// scalar kernel for the tail.

const (
	lane  = 8
	block = 4 * lane
)

func init() {
	// AVX2, not just AVX, as in internal/vec and internal/stencil.
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{
		correlateRow:    correlateRowAVX2,
		columnCorrelate: columnCorrelateAVX2,
		columnSum:       columnSumAVX2,
		columnMin:       columnMinAVX2,
		columnMax:       columnMaxAVX2,
		rowCorrelate:    rowCorrelateAVX2,
		rowMean:         rowMeanAVX2,
		rowMin:          rowMinAVX2,
		rowMax:          rowMaxAVX2,
	}
	simdName = "avx2"
	UseScalar(false)
}

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// quad is one block of cells read through an array pointer.
type quad = [block]float32

func loadQuad(p *quad) (a, b, c, d archsimd.Float32x8) {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(p[0:lane])),
		archsimd.LoadFloat32x8Array((*[lane]float32)(p[lane : 2*lane])),
		archsimd.LoadFloat32x8Array((*[lane]float32)(p[2*lane : 3*lane])),
		archsimd.LoadFloat32x8Array((*[lane]float32)(p[3*lane : 4*lane]))
}

func storeQuad(p *quad, a, b, c, d archsimd.Float32x8) {
	a.StoreArray((*[lane]float32)(p[0:lane]))
	b.StoreArray((*[lane]float32)(p[lane : 2*lane]))
	c.StoreArray((*[lane]float32)(p[2*lane : 3*lane]))
	d.StoreArray((*[lane]float32)(p[3*lane : 4*lane]))
}

// min8 and max8 are Go's builtin min and max lanewise. VMINPS and VMAXPS
// return the second operand for equal or unordered operands, so equal
// lanes take the OR (min) or AND (max) of both, which fixes the signed
// zeros, and a lane where either operand is NaN must be repaired from
// both operands: the compiler treats x.Min(y) as commutative and may emit
// VMINPS with the operands swapped to suit its register allocation, which
// it does inside extremeLanes' fold, so restoring only x's NaN would lose
// a NaN in y.
//
// min8 folds both repairs into one blend: x|y is NaN when either operand
// is, so equal and unordered lanes both take it. max8 cannot, since the
// AND it needs for equal lanes can turn a NaN into ±Inf, so its NaN lanes
// take x+y instead.
func min8(x, y archsimd.Float32x8) archsimd.Float32x8 {
	o := x.ToBits().Or(y.ToBits()).BitsToFloat32()
	return o.IfElse(x.Equal(y).Or(x.IsNaN().Or(y.IsNaN())), x.Min(y))
}

func max8(x, y archsimd.Float32x8) archsimd.Float32x8 {
	r := x.ToBits().And(y.ToBits()).BitsToFloat32().IfElse(x.Equal(y), x.Max(y))
	return x.Add(y).IfElse(x.IsNaN().Or(y.IsNaN()), r)
}

// weightedLanes writes dst[i] = Σ_t w[t]·src[i+t·step] over whole
// vectors and returns how many cells it wrote. The caller guarantees
// src[i+t·step] exists for every such cell.
func weightedLanes(dst, src []float32, step int, w []float32) int {
	n := len(dst)
	w0, ws := w[0], w[1:]
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		v0 := archsimd.BroadcastFloat32x8(w0)
		x0, x1, x2, x3 := loadQuad((*quad)(s))
		a0, a1, a2, a3 := v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
		off := 0
		for _, wt := range ws {
			off += step
			vw := archsimd.BroadcastFloat32x8(wt)
			x0, x1, x2, x3 := loadQuad((*quad)(s[off:]))
			a0, a1 = a0.Add(vw.Mul(x0)), a1.Add(vw.Mul(x1))
			a2, a3 = a2.Add(vw.Mul(x2)), a3.Add(vw.Mul(x3))
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= n; i += lane {
		s := src[i:]
		a := archsimd.BroadcastFloat32x8(w0).Mul(load8(s))
		off := 0
		for _, wt := range ws {
			off += step
			a = a.Add(archsimd.BroadcastFloat32x8(wt).Mul(load8(s[off:])))
		}
		store8(a, dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// correlate2DLanes is weightedLanes over a k×k neighbourhood: the term
// of weight w[j·k+c] reads src[i+j·stride+c].
func correlate2DLanes(dst, src []float32, stride int, w []float32, k int) int {
	n := len(dst)
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		v0 := archsimd.BroadcastFloat32x8(w[0])
		x0, x1, x2, x3 := loadQuad((*quad)(s))
		a0, a1, a2, a3 := v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
		for j := range k {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 {
				c0 = 1
			}
			for c := c0; c < len(wr); c++ {
				vw := archsimd.BroadcastFloat32x8(wr[c])
				x0, x1, x2, x3 := loadQuad((*quad)(row[c:]))
				a0, a1 = a0.Add(vw.Mul(x0)), a1.Add(vw.Mul(x1))
				a2, a3 = a2.Add(vw.Mul(x2)), a3.Add(vw.Mul(x3))
			}
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= n; i += lane {
		s := src[i:]
		a := archsimd.BroadcastFloat32x8(w[0]).Mul(load8(s))
		for j := range k {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 {
				c0 = 1
			}
			for c := c0; c < len(wr); c++ {
				a = a.Add(archsimd.BroadcastFloat32x8(wr[c]).Mul(load8(row[c:])))
			}
		}
		store8(a, dst[i:])
	}
	archsimd.ClearAVXUpperBits()
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
			vn := archsimd.BroadcastFloat32x8(n)
			a0, a1, a2, a3 = a0.Div(vn), a1.Div(vn), a2.Div(vn), a3.Div(vn)
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= m; i += lane {
		s := src[i:]
		a := load8(s)
		off := 0
		for range k - 1 {
			off += step
			a = a.Add(load8(s[off:]))
		}
		if div {
			a = a.Div(archsimd.BroadcastFloat32x8(n))
		}
		store8(a, dst[i:])
	}
	archsimd.ClearAVXUpperBits()
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
				a0, a1, a2, a3 = max8(a0, x0), max8(a1, x1), max8(a2, x2), max8(a3, x3)
			} else {
				a0, a1, a2, a3 = min8(a0, x0), min8(a1, x1), min8(a2, x2), min8(a3, x3)
			}
		}
		storeQuad((*quad)(dst[i:]), a0, a1, a2, a3)
	}
	for ; i+lane <= m; i += lane {
		s := src[i:]
		a := load8(s)
		off := 0
		for range k - 1 {
			off += step
			if isMax {
				a = max8(a, load8(s[off:]))
			} else {
				a = min8(a, load8(s[off:]))
			}
		}
		store8(a, dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	return i
}

func correlateRowAVX2(dst, src []float32, stride int, w []float32, k int) {
	i := correlate2DLanes(dst, src, stride, w, k)
	scalarCorrelateRow(dst[i:], src[i:], stride, w, k)
}

func columnCorrelateAVX2(dst, src []float32, stride int, taps []float32) {
	i := weightedLanes(dst, src, stride, taps)
	scalarColumnCorrelate(dst[i:], src[i:], stride, taps)
}

func columnSumAVX2(dst, src []float32, stride, k int) {
	i := sumLanes(dst, src, stride, k, false, 0)
	scalarColumnSum(dst[i:], src[i:], stride, k)
}

func columnMinAVX2(dst, src []float32, stride, k int) {
	i := extremeLanes(dst, src, stride, k, false)
	scalarColumnMin(dst[i:], src[i:], stride, k)
}

func columnMaxAVX2(dst, src []float32, stride, k int) {
	i := extremeLanes(dst, src, stride, k, true)
	scalarColumnMax(dst[i:], src[i:], stride, k)
}

func rowCorrelateAVX2(dst, src, taps []float32) {
	i := weightedLanes(dst, src, 1, taps)
	scalarRowCorrelate(dst[i:], src[i:], taps)
}

func rowMeanAVX2(dst, src []float32, k int, n float32) {
	i := sumLanes(dst, src, 1, k, true, n)
	scalarRowMean(dst[i:], src[i:], k, n)
}

func rowMinAVX2(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, false)
	scalarRowMin(dst[i:], src[i:], k)
}

func rowMaxAVX2(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, true)
	scalarRowMax(dst[i:], src[i:], k)
}
