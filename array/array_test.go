package array_test

import (
	"slices"
	"testing"

	"github.com/LukasSelin/strata/array"
	"github.com/LukasSelin/strata/raster"
)

func TestNewAndWrap(t *testing.T) {
	a := array.New[float32](2, 3, 4)
	if !slices.Equal(a.Stride, []int{12, 4, 1}) || len(a.Data) != 24 || a.Len() != 24 || a.Rank() != 3 {
		t.Fatalf("New: shape %v strides %v len %d", a.Shape, a.Stride, len(a.Data))
	}
	if !a.Compact() || a.Valid != nil {
		t.Error("New: want compact and unmasked")
	}
	a.Set(7, 1, 2, 3)
	if a.Data[23] != 7 || a.At(1, 2, 3) != 7 {
		t.Error("Set/At: last element is not Data[23]")
	}
	w := array.Wrap(make([]int16, 10), 3, 3)
	if len(w.Data) != 9 || cap(w.Data) != 9 {
		t.Errorf("Wrap: len %d cap %d, want 9 9", len(w.Data), cap(w.Data))
	}
	s := array.New[uint8]()
	if s.Len() != 1 || len(s.Data) != 1 || s.Rank() != 0 {
		t.Errorf("rank 0: len %d data %d", s.Len(), len(s.Data))
	}
	s.Set(9)
	if s.At() != 9 {
		t.Error("rank 0: Set/At")
	}
	m := array.NewLike[float64](array.NewMasked[int32](2, 2))
	if m.Valid == nil || !m.IsValid(1, 1) {
		t.Error("NewLike of a masked array: want an all-valid mask")
	}
	mustPanic(t, "New(0)", func() { array.New[float32](3, 0) })
	mustPanic(t, "Wrap short", func() { array.Wrap(make([]float32, 5), 2, 3) })
	mustPanic(t, "At out of range", func() { a.At(2, 0, 0) })
	mustPanic(t, "At wrong rank", func() { a.At(0, 0) })
	mustPanic(t, "SetValid unmasked", func() { a.SetValid(false, 0, 0, 0) })
}

func TestValidate(t *testing.T) {
	good := array.NewMasked[float32](3, 4)
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := map[string]func(a *array.Array[float32]){
		"stride count":   func(a *array.Array[float32]) { a.Stride = a.Stride[:1] },
		"zero length":    func(a *array.Array[float32]) { a.Shape = []int{0, 4} },
		"negative step":  func(a *array.Array[float32]) { a.Stride = []int{-4, 1} },
		"short data":     func(a *array.Array[float32]) { a.Data = a.Data[:11] },
		"negative bit":   func(a *array.Array[float32]) { a.ValidOffset = -1 },
		"short mask":     func(a *array.Array[float32]) { a.ValidOffset = 60 },
		"span overflows": func(a *array.Array[float32]) { a.Shape = []int{3, 4}; a.Stride = []int{1 << 62, 1} },
	}
	for name, f := range bad {
		a := array.NewMasked[float32](3, 4)
		a.Shape, a.Stride = slices.Clone(a.Shape), slices.Clone(a.Stride)
		f(&a)
		if a.Validate() == nil {
			t.Errorf("%s: Validate accepted it", name)
		}
	}
}

func TestViews(t *testing.T) {
	a := array.NewMasked[float32](3, 4, 5)
	for i := range a.Data {
		a.Data[i] = float32(i)
	}
	at := func(v array.Array[float32], idx ...int) float32 { return v.At(idx...) }

	s := a.Slice(1, 1, 3)
	if !slices.Equal(s.Shape, []int{3, 2, 5}) || at(s, 2, 1, 4) != at(a, 2, 2, 4) {
		t.Errorf("Slice: shape %v", s.Shape)
	}
	w := a.Window([]int{1, 2, 3}, []int{2, 2, 2})
	if at(w, 1, 1, 1) != at(a, 2, 3, 4) || at(w, 0, 0, 0) != at(a, 1, 2, 3) {
		t.Error("Window: wrong elements")
	}
	sel := a.Select(2, 4)
	if !slices.Equal(sel.Shape, []int{3, 4}) || at(sel, 2, 3) != at(a, 2, 3, 4) {
		t.Errorf("Select: shape %v", sel.Shape)
	}
	tr := a.Transpose(2, 0, 1)
	if !slices.Equal(tr.Shape, []int{5, 3, 4}) || at(tr, 4, 2, 1) != at(a, 2, 1, 4) {
		t.Errorf("Transpose: shape %v", tr.Shape)
	}
	rev := a.Transpose()
	if !slices.Equal(rev.Shape, []int{5, 4, 3}) || at(rev, 1, 2, 0) != at(a, 0, 2, 1) {
		t.Error("Transpose(): want axes reversed")
	}
	r := a.Reshape(6, 10)
	if at(r, 5, 9) != 59 || at(r, 1, 0) != 10 {
		t.Error("Reshape: wrong order")
	}
	b := a.Select(1, 0).ExpandDims(1).BroadcastTo(2, 3, 4, 5)
	if !slices.Equal(b.Shape, []int{2, 3, 4, 5}) || at(b, 1, 2, 3, 4) != at(a, 2, 0, 4) {
		t.Errorf("ExpandDims+BroadcastTo: shape %v strides %v", b.Shape, b.Stride)
	}
	if b.Validate() != nil {
		t.Error("a broadcast view must validate")
	}

	// Views share the mask, and ValidOffset follows the first element.
	w.SetValid(false, 1, 1, 1)
	if a.IsValid(2, 3, 4) || !a.IsValid(2, 3, 3) {
		t.Error("SetValid through a window is not seen by the root")
	}
	if tr.IsValid(4, 2, 3) {
		t.Error("a transposed view does not see the bit")
	}
	// A view's shape is its own.
	s.Shape[0] = 99
	if a.Shape[0] != 3 {
		t.Error("changing a view's Shape changed its parent's")
	}

	mustPanic(t, "Slice empty", func() { a.Slice(0, 2, 2) })
	mustPanic(t, "Slice past end", func() { a.Slice(0, 1, 4) })
	mustPanic(t, "Select range", func() { a.Select(0, 3) })
	mustPanic(t, "Transpose dup", func() { a.Transpose(0, 0, 1) })
	mustPanic(t, "Reshape count", func() { a.Reshape(7, 9) })
	mustPanic(t, "Reshape non-compact", func() { tr.Reshape(60) })
	mustPanic(t, "BroadcastTo", func() { a.BroadcastTo(3, 5, 5) })
	mustPanic(t, "Window", func() { a.Window([]int{2, 0, 0}, []int{2, 1, 1}) })
}

func TestBroadcastShape(t *testing.T) {
	got, err := array.BroadcastShape([]int{4, 1, 3}, []int{5, 1}, nil, []int{1})
	if err != nil || !slices.Equal(got, []int{4, 5, 3}) {
		t.Errorf("got %v, %v", got, err)
	}
	if _, err := array.BroadcastShape([]int{2, 3}, []int{3, 3}); err == nil {
		t.Error("2×3 and 3×3 broadcast")
	}
}

func TestRasterBridge(t *testing.T) {
	root := raster.NewFloat32Stride(6, 5, 8, make([]float32, 4*8+6))
	root.Valid = raster.NewMask(len(root.Data))
	r := root.Window(1, 2, 4, 3)
	a := array.FromRaster(r)
	if !slices.Equal(a.Shape, []int{3, 4}) || !slices.Equal(a.Stride, []int{8, 1}) {
		t.Fatalf("FromRaster: shape %v strides %v", a.Shape, a.Stride)
	}
	a.Set(5, 2, 3)
	a.SetValid(false, 1, 0)
	if r.Data[r.Index(3, 2)] != 5 || r.IsValid(0, 1) || !r.IsValid(1, 1) {
		t.Error("FromRaster does not share Data and Valid")
	}
	back := array.ToRaster(a)
	if back.Width != 4 || back.Height != 3 || back.Stride != 8 || back.ValidOffset != r.ValidOffset || &back.Data[0] != &r.Data[0] {
		t.Errorf("ToRaster(FromRaster(r)) = %+v", back)
	}
	// A time slice of a stack is a raster.
	stack := array.NewMasked[float32](4, 3, 5)
	stack.Set(1, 2, 1, 1)
	layer := array.ToRaster(stack.Select(0, 2))
	if layer.Data[layer.Index(1, 1)] != 1 || layer.ValidOffset != 30 {
		t.Error("ToRaster of a Select: wrong element or offset")
	}
	one := array.ToRaster(stack.Select(0, 0).Slice(0, 1, 2))
	if one.Height != 1 || one.Stride != 5 {
		t.Errorf("one row: %+v", one)
	}
	mustPanic(t, "rank 3", func() { array.ToRaster(stack) })
	mustPanic(t, "transposed", func() { array.ToRaster(stack.Select(0, 0).Transpose()) })
	mustPanic(t, "broadcast rows", func() { array.ToRaster(stack.Select(0, 0).Slice(0, 0, 1).BroadcastTo(3, 5)) })
}
