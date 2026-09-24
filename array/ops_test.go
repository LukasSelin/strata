package array_test

import (
	"fmt"
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/array"
	"github.com/LukasSelin/strata/internal/vec"
)

type binaryOp[T array.Number] struct {
	name string
	f    func(dst, a, b array.Array[T])
	ref  func(x, y T) T
}

func binaryOps[T array.Number]() []binaryOp[T] {
	return []binaryOp[T]{
		{"Add", array.Add[T], func(x, y T) T { return x + y }},
		{"Sub", array.Sub[T], func(x, y T) T { return x - y }},
		{"Mul", array.Mul[T], func(x, y T) T { return x * y }},
		{"Min", array.Min[T], func(x, y T) T { return min(x, y) }},
		{"Max", array.Max[T], func(x, y T) T { return max(x, y) }},
	}
}

// TestBinary holds every elementwise operation, on random shapes,
// broadcasts, layouts and masks, to the reference: each dst element is
// the operation on the broadcast inputs' elements, and is valid iff both
// are. Elements of dst's parent outside dst are never touched.
func TestBinary(t *testing.T) {
	t.Run("float32", func(t *testing.T) { testBinary[float32](t) })
	t.Run("float32-scalar", func(t *testing.T) {
		vec.UseScalar(true)
		defer vec.UseScalar(false)
		testBinary[float32](t)
	})
	t.Run("float64", func(t *testing.T) { testBinary[float64](t) })
	t.Run("int16", func(t *testing.T) { testBinary[int16](t) })
	t.Run("uint8", func(t *testing.T) { testBinary[uint8](t) })
}

func testBinary[T array.Number](t *testing.T) {
	ops := binaryOps[T]()
	rapid.Check(t, func(rt *rapid.T) {
		op := ops[rapid.IntRange(0, len(ops)-1).Draw(rt, "op")]
		shape := drawShape(rt, "shape")
		a := drawArray[T](rt, "a", drawBroadcastable(rt, "ashape", shape), true)
		b := drawArray[T](rt, "b", drawBroadcastable(rt, "bshape", shape), true)
		dst := drawArray[T](rt, "dst", shape, false)
		if a.Valid != nil || b.Valid != nil || rapid.Bool().Draw(rt, "dst-masked") {
			dst = withMask(rt, dst)
		}
		// Snapshot dst's whole parent span, to check nothing outside dst
		// is written.
		before := snapshot(dst)
		wantV, wantOK := make([]T, 0), make([]bool, 0)
		for _, idx := range indices(shape) {
			ai, bi := bcast(a, idx), bcast(b, idx)
			wantV = append(wantV, op.ref(a.At(ai...), b.At(bi...)))
			wantOK = append(wantOK, a.IsValid(ai...) && b.IsValid(bi...))
		}
		op.f(dst, a, b)
		for i, idx := range indices(shape) {
			if !sameValue(dst.At(idx...), wantV[i]) {
				rt.Fatalf("%s at %v: got %v, want %v", op.name, idx, dst.At(idx...), wantV[i])
			}
			if dst.Valid != nil && dst.IsValid(idx...) != wantOK[i] {
				rt.Fatalf("%s at %v: valid %v, want %v", op.name, idx, dst.IsValid(idx...), wantOK[i])
			}
		}
		before.checkOutside(rt, dst)
	})
}

// withMask gives a drawn dst a mask with random bits, so stale bits must
// be overwritten.
func withMask[T array.Number](rt *rapid.T, a array.Array[T]) array.Array[T] {
	if a.Valid != nil {
		return a
	}
	// a's Data starts at its first element; give it a mask of its own
	// whose bit 0 is that element, sized for its span.
	m := array.NewMasked[T](len(a.Data)).Valid
	for i := range len(a.Data) {
		if rapid.Bool().Draw(rt, "stale") {
			m[i>>6] &^= 1 << uint(i&63)
		}
	}
	a.Valid = m
	return a
}

// span is a copy of an array's whole Data span and its mask bits, to
// check that an operation writes only the array's own elements.
type span[T array.Number] struct {
	data []T
	bits []bool
}

func snapshot[T array.Number](a array.Array[T]) span[T] {
	s := span[T]{data: append([]T{}, a.Data...)}
	if a.Valid != nil {
		for i := range len(a.Data) {
			s.bits = append(s.bits, a.Valid[(a.ValidOffset+i)>>6]>>uint((a.ValidOffset+i)&63)&1 != 0)
		}
	}
	return s
}

func (s span[T]) checkOutside(rt *rapid.T, a array.Array[T]) {
	own := make([]bool, len(a.Data))
	for _, idx := range indices(a.Shape) {
		own[a.Offset(idx...)] = true
	}
	for i := range a.Data {
		if own[i] {
			continue
		}
		if !sameValue(a.Data[i], s.data[i]) {
			rt.Fatalf("Data[%d] outside the array was written", i)
		}
		if s.bits != nil && (a.Valid[(a.ValidOffset+i)>>6]>>uint((a.ValidOffset+i)&63)&1 != 0) != s.bits[i] {
			rt.Fatalf("mask bit of Data[%d] outside the array was written", i)
		}
	}
}

// TestInPlace runs each operation with dst the same array as an input,
// through a non-compact view, against the same operation into a fresh
// array.
func TestInPlace(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		shape := drawShape(rt, "shape")
		a := drawArray[float32](rt, "a", shape, true)
		b := drawArray[float32](rt, "b", drawBroadcastable(rt, "bshape", shape), true)
		a = withMask(rt, a)
		for _, op := range binaryOps[float32]() {
			want := array.NewMasked[float32](shape...)
			op.f(want, a, b)
			got := array.NewMasked[float32](shape...)
			array.Copy(got, a)
			view := got.Transpose()
			op.f(view, view, b.BroadcastTo(shape...).Transpose())
			for _, idx := range indices(shape) {
				if !sameValue(got.At(idx...), want.At(idx...)) || got.IsValid(idx...) != want.IsValid(idx...) {
					rt.Fatalf("%s in place at %v: %v, want %v", op.name, idx, got.At(idx...), want.At(idx...))
				}
			}
		}
	})
}

// TestConvert copies across element types and checks the values are Go's
// conversions and the validity is copied.
func TestConvert(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		shape := drawShape(rt, "shape")
		src := drawArray[int16](rt, "src", drawBroadcastable(rt, "sshape", shape), true)
		dst := withMask(rt, drawArray[float32](rt, "dst", shape, false))
		array.Convert(dst, src)
		for _, idx := range indices(shape) {
			si := bcast(src, idx)
			if dst.At(idx...) != float32(src.At(si...)) || dst.IsValid(idx...) != src.IsValid(si...) {
				rt.Fatalf("at %v: %v valid %v, want %v valid %v", idx, dst.At(idx...), dst.IsValid(idx...), src.At(si...), src.IsValid(si...))
			}
		}
		back := array.NewLike[int16](dst)
		array.Convert(back, dst)
		for _, idx := range indices(shape) {
			if back.At(idx...) != src.At(bcast(src, idx)...) {
				rt.Fatalf("round trip at %v", idx)
			}
		}
	})
}

func TestFill(t *testing.T) {
	a := array.NewMasked[uint16](4, 6)
	a.SetValid(false, 1, 1)
	w := a.Window([]int{1, 1}, []int{2, 3})
	array.Fill(w, 9)
	for _, idx := range indices(a.Shape) {
		in := idx[0] >= 1 && idx[0] < 3 && idx[1] >= 1 && idx[1] < 4
		if (a.At(idx...) == 9) != in || !a.IsValid(idx...) {
			t.Fatalf("at %v: %v valid %v", idx, a.At(idx...), a.IsValid(idx...))
		}
	}
}

func TestOperandPanics(t *testing.T) {
	a := array.NewMasked[float32](3, 4)
	b := array.New[float32](3, 4)
	plain := array.New[float32](3, 4)
	cases := map[string]func(){
		"shape":          func() { array.Add(plain, b, array.New[float32](4, 3)) },
		"dst broadcast":  func() { array.Add(array.New[float32](1, 4).BroadcastTo(3, 4), b, b) },
		"missing mask":   func() { array.Add(plain, a, b) },
		"partial alias":  func() { array.Add(b.Slice(1, 0, 3), b.Slice(1, 1, 4), b.Slice(1, 0, 3)) },
		"transposed":     func() { sq := array.New[float32](3, 3); array.Add(sq, sq.Transpose(), sq) },
		"invalid input":  func() { bad := b; bad.Data = bad.Data[:5]; array.Add(plain, bad, b) },
		"bits alias":     func() { c := array.New[float32](3, 4); c.Valid = a.Valid; c.ValidOffset = 1; array.Add(c, a, a) },
		"convert alias":  func() { array.Convert(array.Wrap(b.Data[1:], 11), b.Reshape(12).Slice(0, 0, 11)) },
		"reduce alias":   func() { array.MinOver(b.Select(0, 0), b, 0) },
		"reduce shape":   func() { array.SumOver(array.New[float64](4), b, 1) },
		"reduce axes":    func() { array.SumOver(array.New[float64](), b, 0, 0) },
		"reduce no axes": func() { array.SumOver(array.New[float64](3, 4), b) },
		"reduce mask":    func() { array.MeanOver(array.New[float64](4), a, 0) },
	}
	for name, f := range cases {
		mustPanic(t, name, f)
	}
	// The same elements in the same layout are fine: that is in place.
	array.Add(b, b, b)
	array.Add(b.Transpose(), b.Transpose(), b.Transpose())
	// So are disjoint windows of one array.
	big := array.New[float32](2, 8)
	array.Add(big.Slice(0, 0, 1), big.Slice(0, 1, 2), big.Slice(0, 1, 2))
	// Count and Sum need no mask on dst.
	array.SumOver(array.New[float64](4), a, 0)
	array.CountOver(array.New[int64](4), a, 0)
}

func ExampleAdd() {
	// A [time, y, x] stack minus its per-pixel mean over time: the
	// anomaly, by broadcasting the [y, x] mean along time.
	stack := array.Wrap([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	mean64 := array.New[float64](2, 2)
	array.MeanOver(mean64, stack, 0)
	mean := array.New[float32](2, 2)
	array.Convert(mean, mean64)
	anomaly := array.New[float32](2, 2, 2)
	array.Sub(anomaly, stack, mean)
	fmt.Println(mean.Data, anomaly.Data)
	// Output: [3 4 5 6] [-2 -2 -2 -2 2 2 2 2]
}
