package array_test

import (
	"fmt"
	"math"
	"math/big"
	"slices"
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/array"
	"github.com/LukasSelin/strata/reduce"
)

// refFold is the reference for one output element: the valid values
// that fold into it, in index order.
func refFold[T array.Number](src array.Array[T], axes []int, out []int, keepdims bool) []T {
	isReduced := make([]bool, src.Rank())
	for _, ax := range axes {
		isReduced[ax] = true
	}
	var vals []T
	for _, idx := range indices(src.Shape) {
		j, match := 0, true
		for k, i := range idx {
			if isReduced[k] {
				if keepdims {
					j++
				}
				continue
			}
			if out[j] != i {
				match = false
				break
			}
			j++
		}
		if match && src.IsValid(idx...) {
			vals = append(vals, src.At(idx...))
		}
	}
	return vals
}

// refSum is the correctly rounded sum by math/big, with the package's
// rules for non-finite values and signed zeros.
func refSum[T array.Number](vals []T) float64 {
	s := new(big.Float).SetPrec(4000)
	var nan, pinf, ninf bool
	allNegZero := len(vals) > 0
	for _, v := range vals {
		f := float64(v)
		switch {
		case f != f:
			nan = true
		case math.IsInf(f, 1):
			pinf = true
		case math.IsInf(f, -1):
			ninf = true
		default:
			s.Add(s, new(big.Float).SetFloat64(f))
		}
		if f != 0 || !math.Signbit(f) {
			allNegZero = false
		}
	}
	switch {
	case nan || pinf && ninf:
		return math.NaN()
	case pinf:
		return math.Inf(1)
	case ninf:
		return math.Inf(-1)
	case allNegZero:
		return math.Copysign(0, -1)
	}
	f, _ := s.Float64()
	return f
}

// refMean is the correctly rounded mean by math/big.
func refMean[T array.Number](vals []T) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	sum := refSum(vals)
	if math.IsNaN(sum) || math.IsInf(sum, 0) || sum == 0 {
		return sum / float64(len(vals))
	}
	exact := new(big.Rat)
	for _, v := range vals {
		exact.Add(exact, new(big.Rat).SetFloat64(float64(v)))
	}
	exact.Quo(exact, new(big.Rat).SetInt64(int64(len(vals))))
	f, _ := exact.Float64()
	return f
}

// drawAxes draws a non-empty set of distinct axes of a rank-n array.
func drawAxes(t *rapid.T, n int) []int {
	perm := rapid.Permutation(seq(n)).Draw(t, "axes-perm")
	return perm[:rapid.IntRange(1, n).Draw(t, "axes-n")]
}

// reducedShape is src's shape after reducing axes.
func reducedShape(shape, axes []int, keepdims bool) []int {
	var out []int
	for k, n := range shape {
		switch {
		case !slices.Contains(axes, k):
			out = append(out, n)
		case keepdims:
			out = append(out, 1)
		}
	}
	return out
}

// TestReductions runs testReductions for several element types.
func TestReductions(t *testing.T) {
	t.Run("float32", func(t *testing.T) { testReductions[float32](t) })
	t.Run("float64", func(t *testing.T) { testReductions[float64](t) })
	t.Run("int16", func(t *testing.T) { testReductions[int16](t) })
	t.Run("uint8", func(t *testing.T) { testReductions[uint8](t) })
}

// testReductions holds each reduction, over random axes, layouts and
// masks, in both dst forms, to the reference.
func testReductions[T array.Number](t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		shape := rapid.SliceOfN(rapid.IntRange(1, 5), 1, 4).Draw(rt, "shape")
		src := drawArray[T](rt, "src", shape, true)
		axes := drawAxes(rt, len(shape))
		keepdims := rapid.Bool().Draw(rt, "keepdims")
		out := reducedShape(shape, axes, keepdims)

		cnt := array.NewMasked[int64](out...)
		sum := array.NewMasked[float64](out...)
		mean := array.NewMasked[float64](out...)
		mn := array.NewMasked[T](out...)
		mx := array.NewMasked[T](out...)
		array.CountOver(cnt, src, axes...)
		array.SumOver(sum, src, axes...)
		array.MeanOver(mean, src, axes...)
		array.MinOver(mn, src, axes...)
		array.MaxOver(mx, src, axes...)

		for _, idx := range indices(out) {
			vals := refFold(src, axes, idx, keepdims)
			if got := cnt.At(idx...); got != int64(len(vals)) || !cnt.IsValid(idx...) {
				rt.Fatalf("CountOver at %v: %d, want %d", idx, got, len(vals))
			}
			if got, want := sum.At(idx...), refSum(vals); !sameValue(got, want) || !sum.IsValid(idx...) {
				rt.Fatalf("SumOver at %v: %v, want %v (values %v)", idx, got, want, vals)
			}
			if !mean.IsValid(idx...) != (len(vals) == 0) || mn.IsValid(idx...) != (len(vals) > 0) || mx.IsValid(idx...) != (len(vals) > 0) {
				rt.Fatalf("validity at %v with %d values", idx, len(vals))
			}
			if len(vals) == 0 {
				continue
			}
			if got, want := mean.At(idx...), refMean(vals); !sameValue(got, want) {
				rt.Fatalf("MeanOver at %v: %v, want %v (values %v)", idx, got, want, vals)
			}
			wmn, wmx := vals[0], vals[0]
			for _, v := range vals[1:] {
				wmn, wmx = min(wmn, v), max(wmx, v)
			}
			if !sameValue(mn.At(idx...), wmn) || !sameValue(mx.At(idx...), wmx) {
				rt.Fatalf("Min/MaxOver at %v: %v %v, want %v %v", idx, mn.At(idx...), mx.At(idx...), wmn, wmx)
			}
		}
	})
}

// TestReductionLayoutIndependence reduces one array through two
// different layouts of the same elements, and requires every bit of
// every result to match, NaN payloads included.
func TestReductionLayoutIndependence(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		shape := rapid.SliceOfN(rapid.IntRange(1, 6), 1, 4).Draw(rt, "shape")
		a := drawArray[float32](rt, "a", shape, true)
		b := array.NewLike[float32](a)
		array.Copy(b, a)
		axes := drawAxes(rt, len(shape))
		out := reducedShape(shape, axes, false)
		for name, f := range map[string]func(array.Array[float32]) array.Array[float64]{
			"sum": func(s array.Array[float32]) array.Array[float64] {
				d := array.New[float64](out...)
				array.SumOver(d, s, axes...)
				return d
			},
			"mean": func(s array.Array[float32]) array.Array[float64] {
				d := array.NewMasked[float64](out...)
				array.MeanOver(d, s, axes...)
				return d
			},
		} {
			x, y := f(a), f(b)
			for i := range x.Data {
				if math.Float64bits(x.Data[i]) != math.Float64bits(y.Data[i]) {
					rt.Fatalf("%s: element %d differs across layouts: %v, %v", name, i, x.Data[i], y.Data[i])
				}
			}
		}
		x, y := array.NewMasked[float32](out...), array.NewMasked[float32](out...)
		array.MinOver(x, a, axes...)
		array.MinOver(y, b, axes...)
		for i := range x.Data {
			if math.Float32bits(x.Data[i]) != math.Float32bits(y.Data[i]) {
				rt.Fatalf("min: element %d differs across layouts", i)
			}
		}
	})
}

// TestAgreesWithReduce reduces a float32 array over every axis and
// requires reduce's answer for the same cells as a raster, bit for bit:
// two different exact accumulators, one rounding each.
func TestAgreesWithReduce(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		shape := rapid.SliceOfN(rapid.IntRange(1, 9), 2, 2).Draw(rt, "shape")
		a := drawArray[float32](rt, "a", shape, true)
		// A raster must have unit steps along x: copy into a compact array
		// with the same mask.
		c := array.NewLike[float32](a)
		array.Copy(c, a)
		r := array.ToRaster(c)

		sum := array.New[float64]()
		array.SumOver(sum, a, 0, 1)
		wantSum, wantN := reduce.Sum(r)
		cnt := array.New[int64]()
		array.CountOver(cnt, a, 1, 0)
		if !sameValue(sum.At(), wantSum) || cnt.At() != wantN {
			rt.Fatalf("SumOver %v count %d, reduce.Sum %v count %d", sum.At(), cnt.At(), wantSum, wantN)
		}
		if wantN == 0 {
			return
		}
		stats := reduce.Stats(r)
		mean := array.NewMasked[float64](1, 1)
		array.MeanOver(mean, a, 0, 1)
		if !sameValue(mean.At(0, 0), stats.Mean) {
			rt.Fatalf("MeanOver %v, reduce.Stats Mean %v", mean.At(0, 0), stats.Mean)
		}
		mn, mx := array.NewMasked[float32](), array.NewMasked[float32]()
		array.MinOver(mn, a, 0, 1)
		array.MaxOver(mx, a, 0, 1)
		if math.Float32bits(mn.At()) != math.Float32bits(stats.Min) || math.Float32bits(mx.At()) != math.Float32bits(stats.Max) {
			rt.Fatalf("Min/MaxOver %v %v, reduce %v %v", mn.At(), mx.At(), stats.Min, stats.Max)
		}
	})
}

// TestExactSums puts values of wildly different magnitudes, and sums
// that cancel to a tie, through SumOver and MeanOver, where a float64
// running sum would be wrong.
func TestExactSums(t *testing.T) {
	cases := [][]float64{
		{1e308, 1e-308, -1e308},
		{1, 0x1p-53, 0x1p-106},        // a tie broken by the lowest partial
		{1, 0x1p-53, -0x1p-106},       // ... the other way
		{0x1p53, 1, -0x1p53, 0x1p-60}, // cancels to 1 + tiny
		{3, 3, 4},                     // mean 10/3
		{0.1, 0.2, 0.3, 0.4, -1},
		{math.Copysign(0, -1), math.Copysign(0, -1)},
		{math.Copysign(0, -1), 0},
		{math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64},
	}
	for _, vals := range cases {
		a := array.Wrap(vals, len(vals))
		sum, mean := array.New[float64](), array.New[float64]()
		array.SumOver(sum, a, 0)
		array.MeanOver(mean, a, 0)
		wantSum := refSum(vals)
		if vals[0] == math.MaxFloat64 {
			wantSum = math.Inf(1) // the documented overflow
		}
		if !sameValue(sum.At(), wantSum) {
			t.Errorf("SumOver %v = %v, want %v", vals, sum.At(), wantSum)
		}
		if wantMean := refMean(vals); vals[0] != math.MaxFloat64 && !sameValue(mean.At(), wantMean) {
			t.Errorf("MeanOver %v = %v, want %v", vals, mean.At(), wantMean)
		}
	}
}

// TestMeanRounding draws sums whose quotient lands near a rounding
// boundary, where dividing the rounded sum would miss by an ulp.
func TestMeanRounding(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 40).Draw(rt, "n")
		vals := make([]float64, n)
		for i := range vals {
			m := rapid.Int64Range(-(1<<53), 1<<53).Draw(rt, "m")
			e := rapid.IntRange(-80, 80).Draw(rt, "e")
			vals[i] = math.Ldexp(float64(m), e)
		}
		a := array.Wrap(vals, n)
		mean := array.New[float64]()
		array.MeanOver(mean, a, 0)
		if want := refMean(vals); !sameValue(mean.At(), want) {
			rt.Fatalf("MeanOver %v = %v, want %v", vals, mean.At(), want)
		}
		sum := array.New[float64]()
		array.SumOver(sum, a, 0)
		if want := refSum(vals); !sameValue(sum.At(), want) {
			rt.Fatalf("SumOver %v = %v, want %v", vals, sum.At(), want)
		}
	})
}

func TestReductionSpecials(t *testing.T) {
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	a := array.Wrap([]float32{
		1, inf, // one infinity
		inf, -inf, // both: NaN
		nan, 2, // NaN
		0, 0, // made -0, +0 below: min -0, max +0
	}, 4, 2)
	negZero := float32(math.Copysign(0, -1))
	a.Set(negZero, 3, 0)
	sum, mn, mx := array.New[float64](4), array.New[float32](4), array.New[float32](4)
	array.SumOver(sum, a, 1)
	array.MinOver(mn, a, 1)
	array.MaxOver(mx, a, 1)
	if !math.IsInf(sum.At(0), 1) || !math.IsNaN(sum.At(1)) || !math.IsNaN(sum.At(2)) || sum.At(3) != 0 || math.Signbit(sum.At(3)) {
		t.Errorf("sums %v", sum.Data)
	}
	if math.Float32bits(mn.At(2)) != 0x7fc00000 || math.Float32bits(mx.At(2)) != 0x7fc00000 {
		t.Errorf("NaN results not canonical: %#x %#x", math.Float32bits(mn.At(2)), math.Float32bits(mx.At(2)))
	}
	if !math.Signbit(float64(mn.At(3))) || math.Signbit(float64(mx.At(3))) {
		t.Errorf("min/max of -0 and +0: %v %v", mn.At(3), mx.At(3))
	}
}

func ExampleMeanOver() {
	// The mean over time of a [time, y, x] stack whose second step has
	// no data at (0, 1).
	stack := array.NewMasked[float32](2, 1, 2)
	copy(stack.Data, []float32{1, 2, 3, 99})
	stack.SetValid(false, 1, 0, 1)
	mean := array.NewMasked[float64](1, 2)
	array.MeanOver(mean, stack, 0)
	n := array.New[int64](1, 2)
	array.CountOver(n, stack, 0)
	fmt.Println(mean.Data, n.Data)
	// Output: [2 2] [2 1]
}

// TestReductionBlocks covers outputs longer than one block of
// accumulators, reduced both along a leading axis (blocks fed a slab at
// a time) and along the contiguous one (each output its own run).
func TestReductionBlocks(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		long := rapid.IntRange(250, 700).Draw(rt, "long")
		short := rapid.IntRange(1, 4).Draw(rt, "short")
		shape := []int{short, long}
		if rapid.Bool().Draw(rt, "swap") {
			shape = []int{long, short}
		}
		src := array.NewMasked[float32](shape...)
		for i := range src.Data {
			src.Data[i] = drawValue[float32](rt, "v")
			if rapid.IntRange(0, 5).Draw(rt, "bit") == 0 {
				src.Valid[i>>6] &^= 1 << uint(i&63)
			}
		}
		axis := rapid.IntRange(0, 1).Draw(rt, "axis")
		out := reducedShape(shape, []int{axis}, false)
		sum, mn := array.New[float64](out...), array.NewMasked[float32](out...)
		array.SumOver(sum, src, axis)
		array.MinOver(mn, src, axis)
		for _, idx := range indices(out) {
			vals := refFold(src, []int{axis}, idx, false)
			if !sameValue(sum.At(idx...), refSum(vals)) {
				rt.Fatalf("SumOver at %v: %v, want %v", idx, sum.At(idx...), refSum(vals))
			}
			if mn.IsValid(idx...) != (len(vals) > 0) {
				rt.Fatalf("MinOver validity at %v", idx)
			}
			if len(vals) > 0 {
				w := vals[0]
				for _, v := range vals[1:] {
					w = min(w, v)
				}
				if !sameValue(mn.At(idx...), w) {
					rt.Fatalf("MinOver at %v: %v, want %v", idx, mn.At(idx...), w)
				}
			}
		}
	})
}
