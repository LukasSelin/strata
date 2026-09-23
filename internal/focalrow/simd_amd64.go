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
// src[i+t·step] exists for every such cell. With acc set it adds the
// terms to dst[i] instead, continuing a fold that foldColumn split into
// groups of rows.
func weightedLanes(dst, src []float32, step int, w []float32, acc bool) int {
	n := len(dst)
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		var a0, a1, a2, a3 archsimd.Float32x8
		off, ws := 0, w
		if acc {
			a0, a1, a2, a3 = loadQuad((*quad)(dst[i:]))
			off = -step
		} else {
			v0 := archsimd.BroadcastFloat32x8(w[0])
			x0, x1, x2, x3 := loadQuad((*quad)(s))
			a0, a1, a2, a3 = v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
			ws = w[1:]
		}
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
		var a archsimd.Float32x8
		off, ws := 0, w
		if acc {
			a, off = load8(dst[i:]), -step
		} else {
			a, ws = archsimd.BroadcastFloat32x8(w[0]).Mul(load8(s)), w[1:]
		}
		for _, wt := range ws {
			off += step
			a = a.Add(archsimd.BroadcastFloat32x8(wt).Mul(load8(s[off:])))
		}
		store8(a, dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// correlate2DLanes is weightedLanes over len(w)/k rows of a neighbourhood
// k cells wide: the term of weight w[j·k+c] reads src[i+j·stride+c].
func correlate2DLanes(dst, src []float32, stride int, w []float32, k int, acc bool) int {
	n, rows := len(dst), len(w)/k
	i := 0
	for ; i+block <= n; i += block {
		s := src[i:]
		var a0, a1, a2, a3 archsimd.Float32x8
		if acc {
			a0, a1, a2, a3 = loadQuad((*quad)(dst[i:]))
		} else {
			v0 := archsimd.BroadcastFloat32x8(w[0])
			x0, x1, x2, x3 := loadQuad((*quad)(s))
			a0, a1, a2, a3 = v0.Mul(x0), v0.Mul(x1), v0.Mul(x2), v0.Mul(x3)
		}
		for j := range rows {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 && !acc {
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
		var a archsimd.Float32x8
		if acc {
			a = load8(dst[i:])
		} else {
			a = archsimd.BroadcastFloat32x8(w[0]).Mul(load8(s))
		}
		for j := range rows {
			row, wr, c0 := s[j*stride:], w[j*k:j*k+k], 0
			if j == 0 && !acc {
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
// divided by n when div is set, over whole vectors. With acc set it adds
// the k terms to dst[i] instead, as weightedLanes does.
func sumLanes(dst, src []float32, step, k int, div bool, n float32, acc bool) int {
	m := len(dst)
	i := 0
	for ; i+block <= m; i += block {
		s := src[i:]
		var a0, a1, a2, a3 archsimd.Float32x8
		off, terms := 0, k-1
		if acc {
			a0, a1, a2, a3 = loadQuad((*quad)(dst[i:]))
			off, terms = -step, k
		} else {
			a0, a1, a2, a3 = loadQuad((*quad)(s))
		}
		for range terms {
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
		a, off, terms := load8(s), 0, k-1
		if acc {
			a, off, terms = load8(dst[i:]), -step, k
		}
		for range terms {
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
// terms) over whole vectors. With acc set it folds dst[i] in too.
func extremeLanes(dst, src []float32, step, k int, isMax, acc bool) int {
	m := len(dst)
	i := 0
	for ; i+block <= m; i += block {
		s := src[i:]
		var a0, a1, a2, a3 archsimd.Float32x8
		off, terms := 0, k-1
		if acc {
			a0, a1, a2, a3 = loadQuad((*quad)(dst[i:]))
			off, terms = -step, k
		} else {
			a0, a1, a2, a3 = loadQuad((*quad)(s))
		}
		for range terms {
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
		a, off, terms := load8(s), 0, k-1
		if acc {
			a, off, terms = load8(dst[i:]), -step, k
		}
		for range terms {
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

// Rows whose stride is a multiple of 64 KiB (16384 float32 columns) all
// fall into one set of Zen 2's L2, which is 8-way with sets repeating
// every 64 KiB. A block of register accumulators reads its rows in
// lockstep, so a column pass over more than 8 such rows evicts its own
// lines, and the prefetcher's, before it uses them, and runs at a third
// of its speed or less (benchmarks/focal/RESULTS.md, DESIGN.md §53).
// Rows a few cache lines apart modulo 64 KiB collide nearly as badly.
//
// foldColumn therefore reads at most foldRows such rows at once: when
// more than foldRows of the k rows collide, it folds them a group at a
// time over foldChunk cells, the first group writing dst and the others
// adding to it, which keeps the partial sums in L1. Each cell still
// folds its rows in order, and dst holds a float32, as the scalar
// kernels' accumulator does, so the bits do not change. Rows that do not
// collide are read in one pass, as the extra loads and stores of dst
// cost 4–19% there.
const (
	foldRows  = 7
	foldChunk = 4096
)

// collidingRows counts the rows of a k-row neighbourhood, stride cells
// apart, that lie within 1 KiB of the first modulo 64 KiB but not within
// 63 KiB of it in memory, the first included. (Rows closer than that in
// memory share lines or sit in neighbouring sets, as any contiguous data
// does.) On the Zen 2 desktop a column pass slows down when this exceeds
// 7 and not otherwise, at strides from 16320 to 24576 cells and 9 to 17
// rows (benchmarks/focal/RESULTS.md).
func collidingRows(stride, k int) int {
	const period, near = 64 << 10, 1 << 10 // bytes
	n := 1
	for d := 1; d < k; d++ {
		dist := d * stride * 4
		if p := dist % period; dist >= period-near && (p < near || p > period-near) {
			n++
		}
	}
	return n
}

// foldColumn runs lanes over dst and the k rows of src, grouped as above
// when more than foldRows of them collide. lanes gets a stretch of dst,
// src from row j0 at that stretch's first cell, the rows j0 to j1 it
// folds, and whether to add to dst; it returns how many cells it wrote,
// all but a tail shorter than a vector. foldColumn returns how many cells
// of dst are written, all but that tail.
func foldColumn(dst, src []float32, stride, k int, lanes func(d, s []float32, j0, j1 int, acc bool) int) int {
	if k <= foldRows || collidingRows(stride, k) <= foldRows {
		return lanes(dst, src, 0, k, false)
	}
	n, groups := len(dst), (k+foldRows-1)/foldRows
	for c := 0; c < n; c += foldChunk {
		d := dst[c:min(c+foldChunk, n)]
		done := 0
		for g := range groups {
			j0, j1 := g*k/groups, (g+1)*k/groups
			done = lanes(d, src[j0*stride+c:], j0, j1, g > 0)
		}
		if done < len(d) {
			return c + done
		}
	}
	return n
}

func correlateRowAVX2(dst, src []float32, stride int, w []float32, k int) {
	i := foldColumn(dst, src, stride, k, func(d, s []float32, j0, j1 int, acc bool) int {
		return correlate2DLanes(d, s, stride, w[j0*k:j1*k], k, acc)
	})
	scalarCorrelateRow(dst[i:], src[i:], stride, w, k)
}

func columnCorrelateAVX2(dst, src []float32, stride int, taps []float32) {
	i := foldColumn(dst, src, stride, len(taps), func(d, s []float32, j0, j1 int, acc bool) int {
		return weightedLanes(d, s, stride, taps[j0:j1], acc)
	})
	scalarColumnCorrelate(dst[i:], src[i:], stride, taps)
}

func columnSumAVX2(dst, src []float32, stride, k int) {
	i := foldColumn(dst, src, stride, k, func(d, s []float32, j0, j1 int, acc bool) int {
		return sumLanes(d, s, stride, j1-j0, false, 0, acc)
	})
	scalarColumnSum(dst[i:], src[i:], stride, k)
}

func columnMinAVX2(dst, src []float32, stride, k int) {
	i := foldColumn(dst, src, stride, k, func(d, s []float32, j0, j1 int, acc bool) int {
		return extremeLanes(d, s, stride, j1-j0, false, acc)
	})
	scalarColumnMin(dst[i:], src[i:], stride, k)
}

func columnMaxAVX2(dst, src []float32, stride, k int) {
	i := foldColumn(dst, src, stride, k, func(d, s []float32, j0, j1 int, acc bool) int {
		return extremeLanes(d, s, stride, j1-j0, true, acc)
	})
	scalarColumnMax(dst[i:], src[i:], stride, k)
}

func rowCorrelateAVX2(dst, src, taps []float32) {
	i := weightedLanes(dst, src, 1, taps, false)
	scalarRowCorrelate(dst[i:], src[i:], taps)
}

func rowMeanAVX2(dst, src []float32, k int, n float32) {
	i := sumLanes(dst, src, 1, k, true, n, false)
	scalarRowMean(dst[i:], src[i:], k, n)
}

func rowMinAVX2(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, false, false)
	scalarRowMin(dst[i:], src[i:], k)
}

func rowMaxAVX2(dst, src []float32, k int) {
	i := extremeLanes(dst, src, 1, k, true, false)
	scalarRowMax(dst[i:], src[i:], k)
}
