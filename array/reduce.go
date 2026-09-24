package array

import (
	"fmt"
	"math"
	"slices"

	"github.com/LukasSelin/strata/raster"
)

// The axis reductions fold src along one or more axes into dst. dst's
// shape is src's with those axes removed, or with them kept at length 1
// (numpy's keepdims), whichever the caller allocated:
//
//	mean := array.New[float64](ny, nx)     // or (1, ny, nx)
//	array.MeanOver(mean, stack, 0)         // stack is [time, y, x]
//
// Only valid elements take part (DESIGN.md §31). Each result is
// independent of the order elements are visited in, so the layout of src
// — compact, strided, transposed, a view of a view — never changes a bit
// of dst: counts are integers, Min and Max are associative under Go's
// semantics (a NaN result is the canonical quiet NaN), and Sum and Mean
// are exact until one final rounding (DESIGN.md §49).

// CountOver writes to each element of dst how many valid elements of src
// fold into it. Every element of dst is valid.
func CountOver[T Number](dst Array[int64], src Array[T], axes ...int) {
	r := planReduction("array.CountOver", dst, src, axes, false)
	reduceOver(r, src, func(a *countAcc[T], off int) {
		dst.Data[off] = a.n
		r.setValid(dst.Valid, dst.ValidOffset+off, true)
	})
}

// SumOver writes to each element of dst the sum of the valid elements of
// src that fold into it, exact and then correctly rounded to float64, as
// reduce.Sum computes for a raster. Every element of dst is valid; with
// no valid elements the sum is 0.
//
// Values convert to float64 exactly, except int64 and uint64 beyond
// 2^53, which round first. A NaN makes the sum NaN, as do infinities of
// both signs; one infinity makes it that infinity. A sum of zeros is -0
// only when every value is -0. A float64 sum past the float64 range is
// the infinity of that sign.
func SumOver[T Number](dst Array[float64], src Array[T], axes ...int) {
	r := planReduction("array.SumOver", dst, src, axes, false)
	reduceOver(r, src, func(a *sumAcc[T], off int) {
		dst.Data[off] = a.s.value()
		r.setValid(dst.Valid, dst.ValidOffset+off, true)
	})
}

// MeanOver writes to each element of dst the mean of the valid elements
// of src that fold into it: their exact sum divided by their count,
// correctly rounded to float64, which reduce.Stats's Mean also is.
// Non-finite values propagate as in SumOver. An element with no valid
// elements is invalid (and NaN), so if src has a validity mask dst must
// have one.
func MeanOver[T Number](dst Array[float64], src Array[T], axes ...int) {
	r := planReduction("array.MeanOver", dst, src, axes, true)
	var scratch []float64
	reduceOver(r, src, func(a *sumAcc[T], off int) {
		dst.Data[off], scratch = a.s.mean(scratch)
		r.setValid(dst.Valid, dst.ValidOffset+off, a.s.n > 0)
	})
}

// MinOver writes to each element of dst the smallest valid element of
// src that folds into it, by Go's builtin min: a NaN makes the result
// NaN, and -0 is below +0. An element with no valid elements is invalid
// (and 0), so if src has a validity mask dst must have one.
func MinOver[T Number](dst, src Array[T], axes ...int) {
	r := planReduction("array.MinOver", dst, src, axes, true)
	reduceOver(r, src, func(a *minAcc[T], off int) {
		dst.Data[off] = canonical(a.v)
		r.setValid(dst.Valid, dst.ValidOffset+off, a.n > 0)
	})
}

// MaxOver is MinOver with Go's builtin max.
func MaxOver[T Number](dst, src Array[T], axes ...int) {
	r := planReduction("array.MaxOver", dst, src, axes, true)
	reduceOver(r, src, func(a *maxAcc[T], off int) {
		dst.Data[off] = canonical(a.v)
		r.setValid(dst.Valid, dst.ValidOffset+off, a.n > 0)
	})
}

// reduction is the layout of one axis reduction: the kept axes, with
// dst's and src's strides along them, and the reduced axes, with src's.
type reduction struct {
	kept, dstStride, srcKept []int
	reduced, srcReduced      []int
	masked                   bool // src has a mask
}

// setValid records an output element's validity if dst has a mask.
func (r reduction) setValid(m []uint64, bit int, valid bool) {
	if m != nil {
		raster.MaskSet(m, bit, valid)
	}
}

// planReduction checks a reduction's operands and lays it out. empty
// says whether an output can be empty, and so invalid, which needs a
// mask on dst when src has one.
func planReduction[T, U Number](op string, dst Array[U], src Array[T], axes []int, empty bool) reduction {
	dst.must(op, "dst")
	src.must(op, "src")
	if len(axes) == 0 {
		panic(op + ": no axes to reduce")
	}
	rank := len(src.Shape)
	isReduced := make([]bool, rank)
	for _, ax := range axes {
		if uint(ax) >= uint(rank) || isReduced[ax] {
			panic(fmt.Sprintf("%s: axes %v are not distinct axes of a rank-%d array", op, axes, rank))
		}
		isReduced[ax] = true
	}
	var r reduction
	keepdims := len(dst.Shape) == rank
	for k, n := range src.Shape {
		if isReduced[k] {
			r.reduced = append(r.reduced, n)
			r.srcReduced = append(r.srcReduced, src.Stride[k])
			continue
		}
		r.kept = append(r.kept, n)
		r.srcKept = append(r.srcKept, src.Stride[k])
		if keepdims {
			r.dstStride = append(r.dstStride, dst.Stride[k])
		}
	}
	want := slices.Clone(src.Shape)
	if keepdims {
		for _, ax := range axes {
			want[ax] = 1
		}
	} else {
		want = r.kept
		r.dstStride = dst.Stride
	}
	if !slices.Equal(dst.Shape, want) {
		panic(fmt.Sprintf("%s: dst shape %v, want %v for src %v reduced over %v", op, dst.Shape, want, src.Shape, axes))
	}
	requireWritable(op, dst)
	if empty {
		requireMask(op, dst.Valid != nil, src.Valid != nil)
	}
	requireDisjoint(op, "src", dst, src)
	r.masked = src.Valid != nil
	return r
}

// block is how many output elements one pass over the reduced axes
// feeds, so that reducing a leading axis reads src in runs rather than
// one element per cache line.
const block = 256

// reduceOver folds src into accumulators, one per output element, and
// hands each finished one to write with its dst offset. Outputs are taken
// in blocks along their innermost run: for each element of the reduced
// axes the whole block is fed, unless the reduced axes are themselves
// the contiguous ones, in which case each output takes its run at once.
// Which way round only changes the order of adds, which no accumulator
// can see.
func reduceOver[T Number, A any, PA interface {
	*A
	reset()
	add(T)
}](r reduction, src Array[T], write func(PA, int)) {
	outer := newLoop(r.kept, true, r.srcKept, r.dstStride)
	inner := newLoop(r.reduced, true, r.srcReduced)
	accs := make([]A, block)
	for ; !outer.done; outer.next() {
		n, step := outer.run()
		for j0 := 0; j0 < n; j0 += block {
			b := min(block, n-j0)
			for j := range b {
				PA(&accs[j]).reset()
			}
			base := outer.off[0] + j0*step[0]
			for inner.restart(); !inner.done; inner.next() {
				m, rs := inner.run()
				feed[T, A, PA](accs[:b], src, base+inner.off[0], step[0], rs[0], m, r.masked)
			}
			for j := range b {
				write(PA(&accs[j]), outer.off[1]+(j0+j)*step[1])
			}
		}
	}
}

// feed adds to accumulator j the m elements of src at off + j*js + i*is,
// skipping invalid ones when masked.
func feed[T Number, A any, PA interface {
	*A
	reset()
	add(T)
}](accs []A, src Array[T], off, js, is, m int, masked bool) {
	d := src.Data
	if is <= js || len(accs) == 1 {
		for j := range accs {
			a := PA(&accs[j])
			o := off + j*js
			for i := range m {
				if !masked || raster.MaskGet(src.Valid, src.ValidOffset+o+i*is) {
					a.add(d[o+i*is])
				}
			}
		}
		return
	}
	for i := range m {
		o := off + i*is
		for j := range accs {
			if !masked || raster.MaskGet(src.Valid, src.ValidOffset+o+j*js) {
				PA(&accs[j]).add(d[o+j*js])
			}
		}
	}
}

type countAcc[T Number] struct{ n int64 }

func (a *countAcc[T]) reset() { a.n = 0 }
func (a *countAcc[T]) add(T)  { a.n++ }

type sumAcc[T Number] struct{ s exactSum }

func (a *sumAcc[T]) reset()  { a.s.reset() }
func (a *sumAcc[T]) add(v T) { a.s.add(float64(v)) }

// minAcc and maxAcc hold the extreme so far, meaningful once n > 0: no
// value of T is outside what src may hold, so the empty marker is the
// count rather than a sentinel.
type minAcc[T Number] struct {
	v T
	n int64
}

func (a *minAcc[T]) reset() { *a = minAcc[T]{} }

func (a *minAcc[T]) add(v T) {
	if a.n == 0 {
		a.v = v
	} else {
		a.v = min(a.v, v)
	}
	a.n++
}

type maxAcc[T Number] struct {
	v T
	n int64
}

func (a *maxAcc[T]) reset() { *a = maxAcc[T]{} }

func (a *maxAcc[T]) add(v T) {
	if a.n == 0 {
		a.v = v
	} else {
		a.v = max(a.v, v)
	}
	a.n++
}

// canonical replaces a NaN with the canonical quiet NaN. Go's min and
// max do not say which operand's payload a NaN result carries, so it
// would otherwise depend on the visiting order (reduce's canonicalNaN).
func canonical[T Number](v T) T {
	if v != v {
		return T(nan)
	}
	return v
}

var nan = math.NaN()
