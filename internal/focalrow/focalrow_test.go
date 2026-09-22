package focalrow

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// The kernels, by number, so that tests can drive every one of them
// through one table.
const (
	kCorrelateRow = iota
	kColumnCorrelate
	kColumnSum
	kColumnMin
	kColumnMax
	kRowCorrelate
	kRowMean
	kRowMin
	kRowMax
	numKernels
)

var kernelNames = [numKernels]string{
	"CorrelateRow", "ColumnCorrelate", "ColumnSum", "ColumnMin", "ColumnMax",
	"RowCorrelate", "RowMean", "RowMin", "RowMax",
}

// shape says how many input rows a kernel reads for an output of n cells
// and a neighbourhood of k cells, how many cells each row has, and how
// many weights it takes.
func shape(kind, n, k int) (rows, cells, weights int) {
	switch kind {
	case kCorrelateRow:
		return k, n + k - 1, k * k
	case kColumnCorrelate:
		return k, n, k
	case kColumnSum, kColumnMin, kColumnMax:
		return k, n, 0
	case kRowCorrelate:
		return 1, n + k - 1, k
	default:
		return 1, n + k - 1, 0
	}
}

// meanDivisor is the n RowMean divides by in these tests: the cell count
// of a k×k neighbourhood, as package focal passes it.
func meanDivisor(k int) float32 { return float32(k * k) }

// call runs kernel kind through its exported wrapper.
func call(kind int, dst, src []float32, stride int, w []float32, k int) {
	switch kind {
	case kCorrelateRow:
		CorrelateRow(dst, src, stride, w, k)
	case kColumnCorrelate:
		ColumnCorrelate(dst, src, stride, w)
	case kColumnSum:
		ColumnSum(dst, src, stride, k)
	case kColumnMin:
		ColumnMin(dst, src, stride, k)
	case kColumnMax:
		ColumnMax(dst, src, stride, k)
	case kRowCorrelate:
		RowCorrelate(dst, src, w)
	case kRowMean:
		RowMean(dst, src, k, meanDivisor(k))
	case kRowMin:
		RowMin(dst, src, k)
	case kRowMax:
		RowMax(dst, src, k)
	}
}

// naive computes kernel kind one cell at a time, folding each cell's
// terms in the documented order with an explicit rounding after every
// operation. It shares no code with the kernels.
func naive(kind int, dst, src []float32, stride int, w []float32, k int) {
	for i := range dst {
		var terms []float32 // values in fold order
		var ws []float32    // their weights, for the weighted kernels
		switch kind {
		case kCorrelateRow:
			for j := range k {
				for c := range k {
					terms = append(terms, src[j*stride+i+c])
					ws = append(ws, w[j*k+c])
				}
			}
		case kColumnCorrelate, kColumnSum, kColumnMin, kColumnMax:
			for j := range k {
				terms = append(terms, src[j*stride+i])
			}
			ws = w
		default:
			for c := range k {
				terms = append(terms, src[i+c])
			}
			ws = w
		}
		var acc float32
		switch kind {
		case kCorrelateRow, kColumnCorrelate, kRowCorrelate:
			acc = float32(ws[0] * terms[0])
			for t := 1; t < len(terms); t++ {
				p := float32(ws[t] * terms[t])
				acc = float32(acc + p)
			}
		case kColumnSum, kRowMean:
			acc = terms[0]
			for _, v := range terms[1:] {
				acc = float32(acc + v)
			}
			if kind == kRowMean {
				acc = float32(acc / meanDivisor(k))
			}
		case kColumnMin, kRowMin:
			acc = terms[0]
			for _, v := range terms[1:] {
				acc = min(acc, v)
			}
		default:
			acc = terms[0]
			for _, v := range terms[1:] {
				acc = max(acc, v)
			}
		}
		dst[i] = acc
	}
}

// sameBits compares bit patterns, except that any NaN matches any NaN.
func sameBits(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// hazards are the values kernels get wrong.
var hazards = []float32{
	float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
	0, float32(math.Copysign(0, -1)), math.MaxFloat32, -math.MaxFloat32,
	1e30, -1e30, math.SmallestNonzeroFloat32, 1e-30,
}

// value draws smooth data, a hazard with probability special, or a
// value whose products and sums are exact (to exercise signed zeros).
func value(rng *rand.Rand, special float64) float32 {
	switch {
	case rng.Float64() < special:
		return hazards[rng.IntN(len(hazards))]
	case rng.IntN(8) == 0:
		return float32(rng.IntN(5) - 2)
	default:
		return float32(rng.NormFloat64()*50 + 500)
	}
}

// operands builds kernel kind's inputs for n outputs and k-cell
// neighbourhoods, with pad cells after every row and extra cells after
// the last, all holding poison. It returns src, the stride and the
// weights.
func operands(rng *rand.Rand, kind, n, k, pad, extra int, special float64, poison float32) ([]float32, int, []float32) {
	rows, cells, nw := shape(kind, n, k)
	stride := cells + pad
	src := make([]float32, (rows-1)*stride+cells+extra)
	for i := range src {
		src[i] = poison
	}
	for j := range rows {
		for c := range cells {
			src[j*stride+c] = value(rng, special)
		}
	}
	w := make([]float32, nw)
	for i := range w {
		if rng.IntN(6) == 0 {
			w[i] = hazards[rng.IntN(len(hazards))]
			if math.IsNaN(float64(w[i])) || math.IsInf(float64(w[i]), 0) {
				w[i] = 0
			}
		} else {
			w[i] = float32(rng.NormFloat64())
		}
	}
	return src, stride, w
}

// TestScalarMatchesNaive holds the canonical backend to the per-cell
// definition, bit for bit, over hazards, odd lengths and strides.
func TestScalarMatchesNaive(t *testing.T) {
	UseScalar(true)
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(1, 2))
	for kind := range numKernels {
		for _, k := range []int{1, 3, 5, 7, 11, 17} {
			for _, n := range []int{0, 1, 2, 7, 8, 9, 31, 33, 100} {
				src, stride, w := operands(rng, kind, n, k, rng.IntN(5), 3, 0.2, 1e30)
				want := make([]float32, n)
				naive(kind, want, src, stride, w, k)
				got := make([]float32, n+2)
				got[n], got[n+1] = 7, 7
				call(kind, got[:n], src, stride, w, k)
				for i := range want {
					if !sameBits(got[i], want[i]) {
						t.Fatalf("%s n=%d k=%d: cell %d = %v, want %v", kernelNames[kind], n, k, i, got[i], want[i])
					}
				}
				if got[n] != 7 || got[n+1] != 7 {
					t.Fatalf("%s n=%d k=%d: wrote past dst", kernelNames[kind], n, k)
				}
			}
		}
	}
}

// TestSignedZero checks the first-term rule: a weighted sum whose every
// product is -0 is -0, which a fold started from +0 would lose.
func TestSignedZero(t *testing.T) {
	negz := float32(math.Copysign(0, -1))
	for _, scalar := range []bool{true, false} {
		UseScalar(scalar)
		for _, n := range []int{1, 9, 40} {
			src := make([]float32, n+2)
			for i := range src {
				src[i] = negz
			}
			dst := make([]float32, n)
			RowCorrelate(dst, src, []float32{1, 2, 3})
			for i, v := range dst {
				if math.Float32bits(v) != math.Float32bits(negz) {
					t.Fatalf("%s n=%d: cell %d = %v, want -0", Backend(), n, i, v)
				}
			}
		}
	}
	UseScalar(false)
}

func TestPanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if s, ok := r.(string); !ok || !strings.HasPrefix(s, "focalrow: ") {
				t.Errorf("%s: panic %v, want a focalrow: message", name, r)
			}
		}()
		f()
	}
	dst, src := make([]float32, 4), make([]float32, 64)
	mustPanic("even k", func() { RowMin(dst, src, 2) })
	mustPanic("zero k", func() { ColumnSum(dst, src, 8, 0) })
	mustPanic("short row", func() { RowMax(dst, src[:5], 3) })
	mustPanic("short src", func() { ColumnMin(dst, src[:19], 8, 3) })
	mustPanic("short stride", func() { ColumnMax(dst, src, 3, 3) })
	mustPanic("weights", func() { CorrelateRow(dst, src, 8, make([]float32, 8), 3) })
	mustPanic("even taps", func() { RowCorrelate(dst, src, make([]float32, 4)) })
	mustPanic("2-D short", func() { CorrelateRow(dst, src[:21], 8, make([]float32, 9), 3) })
}
