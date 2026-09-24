package array

import (
	"fmt"
	"slices"
	"unsafe"
)

// requireWritable panics unless every element of dst has its own memory,
// in Data and so in Valid: an operation writing a broadcast view, or any
// layout whose elements share a cell, would write one cell twice with
// different values. The test is the usual sufficient one: ordered by
// stride, each axis steps past everything the smaller axes span. Every
// view the methods make of a New or Wrap array passes it except
// BroadcastTo's.
func requireWritable[T Number](op string, dst Array[T]) {
	type axis struct{ n, stride int }
	axes := make([]axis, 0, len(dst.Shape))
	for k, n := range dst.Shape {
		if n > 1 {
			axes = append(axes, axis{n, dst.Stride[k]})
		}
	}
	slices.SortFunc(axes, func(a, b axis) int { return a.stride - b.stride })
	reach := 1 // elements spanned by the axes checked so far
	for _, a := range axes {
		if a.stride < reach {
			panic(fmt.Sprintf("%s: dst elements share memory (shape %v, strides %v)", op, dst.Shape, dst.Stride))
		}
		reach = (a.n-1)*a.stride + reach
	}
}

// requireBroadcast returns in's strides repeated to shape, and panics if
// in does not broadcast to it.
func requireBroadcast[T Number](op, name string, in Array[T], shape []int) []int {
	st, ok := broadcastStrides(in.Shape, in.Stride, shape)
	if !ok {
		panic(fmt.Sprintf("%s: %s of shape %v does not broadcast to dst shape %v", op, name, in.Shape, shape))
	}
	return st
}

// requireMask panics if an input has a validity mask and dst does not.
// An operation cannot attach one itself: a parent sharing dst's Data
// would never see it.
func requireMask(op string, dstMasked, inputMasked bool) {
	if inputMasked && !dstMasked {
		panic(op + ": an input has a validity mask but dst does not")
	}
}

// requireApart panics if in shares memory with dst other than by being
// the same elements in the same layout, which is how an operation runs
// in place. inStride is in's strides broadcast to dst's shape. The test
// for Data is conservative, like internal/overlap's with different
// strides: two views whose spans meet are rejected even if their
// elements interleave without touching. Validity bits are held to the
// same rule.
func requireApart[T, U Number](op, name string, dst Array[T], in Array[U], inStride []int) {
	same := unsafe.Sizeof(*new(T)) == unsafe.Sizeof(*new(U)) && sameSteps(dst.Shape, dst.Stride, inStride)
	d0, i0 := dataAddr(dst.Data), dataAddr(in.Data)
	if d0 != i0 || !same {
		if meet(d0, dst.span()*elemSize[T](), i0, in.span()*elemSize[U]()) {
			panic(fmt.Sprintf("%s: dst and %s share some but not all elements", op, name))
		}
	}
	if dst.Valid == nil || in.Valid == nil {
		return
	}
	db, ib := bitAddr(dst.Valid, dst.ValidOffset), bitAddr(in.Valid, in.ValidOffset)
	if db != ib || !same {
		if meet(db, dst.span(), ib, in.span()) {
			panic(fmt.Sprintf("%s: dst and %s share some but not all validity bits", op, name))
		}
	}
}

// requireDisjoint panics if in shares any memory with dst, in Data or in
// Valid: for operations whose result cannot overwrite an input it is
// still reading, such as a reduction.
func requireDisjoint[T, U Number](op, name string, dst Array[T], in Array[U]) {
	if meet(dataAddr(dst.Data), dst.span()*elemSize[T](), dataAddr(in.Data), in.span()*elemSize[U]()) {
		panic(fmt.Sprintf("%s: dst shares memory with %s", op, name))
	}
	if dst.Valid != nil && in.Valid != nil &&
		meet(bitAddr(dst.Valid, dst.ValidOffset), dst.span(), bitAddr(in.Valid, in.ValidOffset), in.span()) {
		panic(fmt.Sprintf("%s: dst shares validity bits with %s", op, name))
	}
}

// sameSteps reports whether two layouts step alike along every axis of
// shape that has more than one element.
func sameSteps(shape, a, b []int) bool {
	for k, n := range shape {
		if n > 1 && a[k] != b[k] {
			return false
		}
	}
	return true
}

func elemSize[T Number]() int { return int(unsafe.Sizeof(*new(T))) }

func dataAddr[T Number](d []T) int {
	return int(uintptr(unsafe.Pointer(unsafe.SliceData(d))))
}

// bitAddr is the address of a mask bit counted in bits, so that bits of
// different slices of one backing array compare correctly.
func bitAddr(m []uint64, off int) int {
	return int(uintptr(unsafe.Pointer(unsafe.SliceData(m))))*8 + off
}

func meet(a0, an, b0, bn int) bool { return a0 < b0+bn && b0 < a0+an }
