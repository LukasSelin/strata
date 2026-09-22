// Package focalrow holds the row kernels of package focal: weighted sums,
// sums, minima and maxima over a (2r+1)-cell neighbourhood, one output row
// at a time (DESIGN.md §53).
//
// A 2-D kernel (CorrelateRow) reads 2r+1 input rows. The separable
// operations read them in two passes: a column pass (Column*) folds 2r+1
// rows into one row of W+2r cells, and a row pass (Row*) folds each run of
// 2r+1 of those cells into one output cell. Input rows are given as one
// slice and a stride, row j starting at src[j*stride:], rather than as a
// slice of rows: a slice of slices built per call would escape through
// the dispatch variables below and allocate once per band (DESIGN.md
// §26).
//
// This file holds the scalar backend, which is canonical (DESIGN.md §15).
// SIMD backends (simd_amd64.go and simd_arm64.go, built with
// GOEXPERIMENT=simd) must agree with it bit for bit, any NaN matching any
// NaN. Every kernel fixes the order in which it folds a cell's terms:
//
//   - A weighted sum starts from the first product, not from +0, and adds
//     each later product in order: acc = float32(acc + float32(w·v)).
//     Starting from the first product keeps −0 when every product is −0.
//     The explicit conversions stop the compiler fusing a multiply-add
//     (docs/adr/0001-simd-backend.md), and every term is evaluated, zero
//     weights included, so 0·Inf is NaN.
//   - A plain sum starts from the first value.
//   - Minima and maxima use Go's builtin min and max, which are
//     associative and commutative (NaN wins, −0 orders below +0), so their
//     order does not matter.
//
// The scalar kernels loop over the terms on the outside and the cells on
// the inside, accumulating in dst, which is the same per-cell order as a
// register accumulator and keeps every cell loop indexed by its loop
// variable alone (DESIGN.md §39).
//
// Like internal/vec, exported functions panic on mismatched lengths, with
// a "focalrow: " prefix.
//
//strata:kernel
package focalrow

import "fmt"

// Backend function variables, swapped in init by SIMD builds.
var (
	correlateRow    = scalarCorrelateRow
	columnCorrelate = scalarColumnCorrelate
	columnSum       = scalarColumnSum
	columnMin       = scalarColumnMin
	columnMax       = scalarColumnMax
	rowCorrelate    = scalarRowCorrelate
	rowMean         = scalarRowMean
	rowMin          = scalarRowMin
	rowMax          = scalarRowMax
)

// kernelSet is one backend's kernels.
type kernelSet struct {
	correlateRow    func(dst, src []float32, stride int, w []float32, k int)
	columnCorrelate func(dst, src []float32, stride int, taps []float32)
	columnSum       func(dst, src []float32, stride, k int)
	columnMin       func(dst, src []float32, stride, k int)
	columnMax       func(dst, src []float32, stride, k int)
	rowCorrelate    func(dst, src, taps []float32)
	rowMean         func(dst, src []float32, k int, n float32)
	rowMin          func(dst, src []float32, k int)
	rowMax          func(dst, src []float32, k int)
}

var scalarKernels = kernelSet{
	correlateRow:    scalarCorrelateRow,
	columnCorrelate: scalarColumnCorrelate,
	columnSum:       scalarColumnSum,
	columnMin:       scalarColumnMin,
	columnMax:       scalarColumnMax,
	rowCorrelate:    scalarRowCorrelate,
	rowMean:         scalarRowMean,
	rowMin:          scalarRowMin,
	rowMax:          scalarRowMax,
}

// simdKernels is the SIMD set, or nil when this build or CPU has none,
// and simdName what Backend reports for it.
var (
	simdKernels *kernelSet
	simdName    string
)

func (k *kernelSet) install() {
	correlateRow, columnCorrelate = k.correlateRow, k.columnCorrelate
	columnSum, columnMin, columnMax = k.columnSum, k.columnMin, k.columnMax
	rowCorrelate, rowMean = k.rowCorrelate, k.rowMean
	rowMin, rowMax = k.rowMin, k.rowMax
}

// Backend names the kernels currently in use: "avx2", "neon" or "scalar".
func Backend() string {
	if simdKernels != nil && !usingScalar {
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
	if scalar || simdKernels == nil {
		scalarKernels.install()
		return
	}
	simdKernels.install()
}

// requireRows panics unless src holds k rows of at least cells cells,
// row j starting at j*stride.
func requireRows(src []float32, stride, k, cells int) {
	if k < 1 {
		panic(fmt.Sprintf("focalrow: need at least one row, got %d", k))
	}
	if k > 1 && stride < cells {
		panic(fmt.Sprintf("focalrow: stride %d is shorter than a row of %d cells", stride, cells))
	}
	if need := (k-1)*stride + cells; len(src) < need {
		panic(fmt.Sprintf("focalrow: src has %d cells, need %d for %d rows of %d at stride %d",
			len(src), need, k, cells, stride))
	}
}

func requireOdd(k int) {
	if k < 1 || k%2 == 0 {
		panic(fmt.Sprintf("focalrow: a neighbourhood must be a positive odd number of cells, got %d", k))
	}
}

// CorrelateRow writes one output row of a 2-D correlation with the k×k
// weights w (row-major, k = 2r+1 odd): dst[i] = Σ_j Σ_c w[j·k+c]·row_j[i+c],
// folded in row-major order of the weights. Row j of the neighbourhood is
// src[j*stride:] and must have at least len(dst)+k-1 cells.
func CorrelateRow(dst, src []float32, stride int, w []float32, k int) {
	requireOdd(k)
	if len(w) != k*k {
		panic(fmt.Sprintf("focalrow: %d weights for a %d×%d neighbourhood", len(w), k, k))
	}
	requireRows(src, stride, k, len(dst)+k-1)
	correlateRow(dst, src, stride, w, k)
}

// ColumnCorrelate folds len(taps) rows into one: dst[i] = Σ_j taps[j]·row_j[i],
// in order of j. Row j is src[j*stride:] and must have len(dst) cells.
func ColumnCorrelate(dst, src []float32, stride int, taps []float32) {
	requireOdd(len(taps))
	requireRows(src, stride, len(taps), len(dst))
	columnCorrelate(dst, src, stride, taps)
}

// ColumnSum folds k rows into one by addition, top to bottom.
func ColumnSum(dst, src []float32, stride, k int) {
	requireOdd(k)
	requireRows(src, stride, k, len(dst))
	columnSum(dst, src, stride, k)
}

// ColumnMin folds k rows into one by Go's builtin min.
func ColumnMin(dst, src []float32, stride, k int) {
	requireOdd(k)
	requireRows(src, stride, k, len(dst))
	columnMin(dst, src, stride, k)
}

// ColumnMax folds k rows into one by Go's builtin max.
func ColumnMax(dst, src []float32, stride, k int) {
	requireOdd(k)
	requireRows(src, stride, k, len(dst))
	columnMax(dst, src, stride, k)
}

func requireRun(dst, src []float32, k int) {
	requireOdd(k)
	if len(src) < len(dst)+k-1 {
		panic(fmt.Sprintf("focalrow: src has %d cells, need %d", len(src), len(dst)+k-1))
	}
}

// RowCorrelate writes dst[i] = Σ_c taps[c]·src[i+c], in order of c. src
// must have at least len(dst)+len(taps)-1 cells.
func RowCorrelate(dst, src, taps []float32) {
	requireRun(dst, src, len(taps))
	rowCorrelate(dst, src, taps)
}

// RowMean writes dst[i] = (src[i] + … + src[i+k-1]) / n, summed left to
// right and divided (not multiplied by a reciprocal) at the end.
func RowMean(dst, src []float32, k int, n float32) {
	requireRun(dst, src, k)
	rowMean(dst, src, k, n)
}

// RowMin writes dst[i] = min(src[i], …, src[i+k-1]).
func RowMin(dst, src []float32, k int) {
	requireRun(dst, src, k)
	rowMin(dst, src, k)
}

// RowMax writes dst[i] = max(src[i], …, src[i+k-1]).
func RowMax(dst, src []float32, k int) {
	requireRun(dst, src, k)
	rowMax(dst, src, k)
}

// The scalar kernels. Each takes the operands its wrapper checked and
// reslices them to exact lengths once per term, so the cell loops carry
// no bounds checks.

func scalarCorrelateRow(dst, src []float32, stride int, w []float32, k int) {
	n := len(dst)
	for j := range k {
		row := src[j*stride : j*stride+n+k-1]
		for c := range k {
			wt, v := w[j*k+c], row[c:c+n]
			if j == 0 && c == 0 {
				for i := range dst {
					dst[i] = float32(wt * v[i])
				}
				continue
			}
			for i := range dst {
				dst[i] = float32(dst[i] + float32(wt*v[i]))
			}
		}
	}
}

func scalarColumnCorrelate(dst, src []float32, stride int, taps []float32) {
	n := len(dst)
	for j, wt := range taps {
		v := src[j*stride:][:n]
		if j == 0 {
			for i := range dst {
				dst[i] = float32(wt * v[i])
			}
			continue
		}
		for i := range dst {
			dst[i] = float32(dst[i] + float32(wt*v[i]))
		}
	}
}

func scalarColumnSum(dst, src []float32, stride, k int) {
	n := len(dst)
	copy(dst, src[:n])
	for j := 1; j < k; j++ {
		v := src[j*stride:][:n]
		for i := range dst {
			dst[i] += v[i]
		}
	}
}

func scalarColumnMin(dst, src []float32, stride, k int) {
	n := len(dst)
	copy(dst, src[:n])
	for j := 1; j < k; j++ {
		v := src[j*stride:][:n]
		for i := range dst {
			dst[i] = min(dst[i], v[i])
		}
	}
}

func scalarColumnMax(dst, src []float32, stride, k int) {
	n := len(dst)
	copy(dst, src[:n])
	for j := 1; j < k; j++ {
		v := src[j*stride:][:n]
		for i := range dst {
			dst[i] = max(dst[i], v[i])
		}
	}
}

func scalarRowCorrelate(dst, src, taps []float32) {
	n := len(dst)
	for c, wt := range taps {
		v := src[c : c+n]
		if c == 0 {
			for i := range dst {
				dst[i] = float32(wt * v[i])
			}
			continue
		}
		for i := range dst {
			dst[i] = float32(dst[i] + float32(wt*v[i]))
		}
	}
}

func scalarRowMean(dst, src []float32, k int, n float32) {
	m := len(dst)
	copy(dst, src[:m])
	for c := 1; c < k; c++ {
		v := src[c : c+m]
		for i := range dst {
			dst[i] += v[i]
		}
	}
	for i := range dst {
		dst[i] /= n
	}
}

func scalarRowMin(dst, src []float32, k int) {
	m := len(dst)
	copy(dst, src[:m])
	for c := 1; c < k; c++ {
		v := src[c : c+m]
		for i := range dst {
			dst[i] = min(dst[i], v[i])
		}
	}
}

func scalarRowMax(dst, src []float32, k int) {
	m := len(dst)
	copy(dst, src[:m])
	for c := 1; c < k; c++ {
		v := src[c : c+m]
		for i := range dst {
			dst[i] = max(dst[i], v[i])
		}
	}
}
