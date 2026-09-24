package array_test

import (
	"math"
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/array"
	"github.com/LukasSelin/strata/raster"
)

// The tests judge the package against a reference that knows nothing of
// strides, runs or merged dimensions: it enumerates logical indices and
// reads each element with At, the way the arithmetic is defined. Exact
// sums and means come from math/big.

// indices returns every index of shape in row-major order. A rank-0
// shape has one, the empty index.
func indices(shape []int) [][]int {
	out := [][]int{{}}
	for _, n := range shape {
		var next [][]int
		for _, idx := range out {
			for i := range n {
				next = append(next, append(append([]int{}, idx...), i))
			}
		}
		out = next
	}
	return out
}

// bcast maps an index of a broadcast target shape to one of a, by
// numpy's rules.
func bcast[T array.Number](a array.Array[T], idx []int) []int {
	lead := len(idx) - a.Rank()
	out := make([]int, a.Rank())
	for k := range out {
		if a.Shape[k] != 1 {
			out[k] = idx[lead+k]
		}
	}
	return out
}

// sameValue compares two elements bit for bit, except that any NaN
// equals any NaN: Go's min and max, and the vector kernels, do not
// promise which operand's payload a NaN result carries.
func sameValue[T array.Number](a, b T) bool {
	if a != a && b != b {
		return true
	}
	switch x := any(a).(type) {
	case float32:
		return math.Float32bits(x) == math.Float32bits(any(b).(float32))
	case float64:
		return math.Float64bits(x) == math.Float64bits(any(b).(float64))
	}
	return a == b
}

// drawValue draws an element: for floats, a mix of ordinary values,
// signed zeros, infinities, NaN and subnormals; for integers, anything
// the type holds.
func drawValue[T array.Number](t *rapid.T, label string) T {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
		special := []float64{0, math.Copysign(0, -1), math.Inf(1), math.Inf(-1), math.NaN(), 1e-40, -3e-39, 1, -1, 0.5}
		switch rapid.IntRange(0, 4).Draw(t, label+"-kind") {
		case 0:
			return T(special[rapid.IntRange(0, len(special)-1).Draw(t, label+"-special")])
		case 1:
			// Any float32 magnitude, so that sums span many exponents.
			m := rapid.Int64Range(-(1<<24), 1<<24).Draw(t, label+"-m")
			return T(float32(math.Ldexp(float64(m), rapid.IntRange(-149, 103).Draw(t, label+"-e"))))
		}
		return T(rapid.Float64Range(-1e6, 1e6).Draw(t, label))
	case int8, int16, int32, int64:
		return T(rapid.Int64Range(math.MinInt8, math.MaxInt8).Draw(t, label))
	default:
		return T(rapid.Uint64Range(0, math.MaxUint8).Draw(t, label))
	}
}

// drawArray draws an array of the given shape as a random view of a
// larger parent: padded, sliced and transposed, with a random mask that
// may be nil, so the operations see every kind of layout.
func drawArray[T array.Number](t *rapid.T, label string, shape []int, maskedOK bool) array.Array[T] {
	rank := len(shape)
	// Draw a permutation: the parent holds the axes in this order.
	perm := rapid.Permutation(seq(rank)).Draw(t, label+"-perm")
	parentShape := make([]int, rank)
	start := make([]int, rank)
	for k := range rank {
		pad := rapid.IntRange(0, 2).Draw(t, label+"-pad")
		start[k] = rapid.IntRange(0, pad).Draw(t, label+"-start")
		parentShape[k] = shape[perm[k]] + pad
	}
	var parent array.Array[T]
	masked := maskedOK && rapid.Bool().Draw(t, label+"-masked")
	if masked {
		parent = array.NewMasked[T](parentShape...)
	} else {
		parent = array.New[T](parentShape...)
	}
	if rank == 0 && rapid.Bool().Draw(t, label+"-rank0-from-1") {
		// A rank-0 view taken by Select, not a rank-0 root.
		p := array.New[T](3)
		if masked {
			p.Valid = raster.NewMask(3)
		}
		parent = p.Select(0, rapid.IntRange(0, 2).Draw(t, label+"-sel"))
	}
	for i := range parent.Data {
		parent.Data[i] = drawValue[T](t, label+"-v")
	}
	if masked {
		for i := range len(parent.Data) {
			raster.MaskSet(parent.Valid, parent.ValidOffset+i, rapid.IntRange(0, 3).Draw(t, label+"-bit") != 0)
		}
	}
	size := make([]int, rank)
	for k := range rank {
		size[k] = shape[perm[k]]
	}
	v := parent
	if rank > 0 {
		v = parent.Window(start, size)
	}
	// Undo the permutation: axis k of the result is parent axis inv[k].
	inv := make([]int, rank)
	for k, p := range perm {
		inv[p] = k
	}
	return v.Transpose(inv...)
}

// drawShape draws a shape of rank 0 to 4 with lengths 1 to 5.
func drawShape(t *rapid.T, label string) []int {
	return rapid.SliceOfN(rapid.IntRange(1, 5), 0, 4).Draw(t, label)
}

// drawBroadcastable draws a shape that broadcasts to shape: a suffix of
// it with some lengths set to 1.
func drawBroadcastable(t *rapid.T, label string, shape []int) []int {
	drop := rapid.IntRange(0, len(shape)).Draw(t, label+"-drop")
	out := append([]int{}, shape[drop:]...)
	for k := range out {
		if rapid.IntRange(0, 3).Draw(t, label+"-one") == 0 {
			out[k] = 1
		}
	}
	return out
}

func seq(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

// mustPanic fails t unless f panics.
func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: did not panic", name)
		}
	}()
	f()
}
