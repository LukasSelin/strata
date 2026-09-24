package array

import (
	"fmt"
	"slices"
)

// The view methods return an Array that shares a's Data and Valid, so
// writes through either are visible in both. Nothing is copied except
// the small Shape and Stride slices, which every view owns, so a view's
// Shape can never change its parent's. Like raster.Window they panic on
// arguments outside the array, and like it they point ValidOffset at the
// view's first element.

// view is a over the elements starting at Data[off] with the given
// layout. It takes ownership of shape and stride.
func (a Array[T]) view(off int, shape, stride []int) Array[T] {
	n, _ := spanLen(shape, stride)
	v := Array[T]{
		Data:   a.Data[off : off+n : off+n],
		Shape:  shape,
		Stride: stride,
		Valid:  a.Valid,
	}
	if a.Valid != nil {
		v.ValidOffset = a.ValidOffset + off
	}
	return v
}

// Slice returns the elements with index in [start, end) along axis, all
// others kept. It panics unless 0 <= start < end <= a.Shape[axis].
func (a Array[T]) Slice(axis, start, end int) Array[T] {
	a.requireAxis("Slice", axis)
	if start < 0 || end <= start || end > a.Shape[axis] {
		panic(fmt.Sprintf("array: Slice [%d, %d) outside [0, %d) on axis %d", start, end, a.Shape[axis], axis))
	}
	shape := clone(a.Shape)
	shape[axis] = end - start
	return a.view(start*a.Stride[axis], shape, clone(a.Stride))
}

// Window returns the block of the given shape whose first element is at
// start, the N-dimensional raster.Window. It panics if start and shape
// do not have a's rank, a length is not positive, or the block extends
// outside a.
func (a Array[T]) Window(start, shape []int) Array[T] {
	if len(start) != len(a.Shape) || len(shape) != len(a.Shape) {
		panic(fmt.Sprintf("array: Window of rank %d and %d on a rank-%d array", len(start), len(shape), len(a.Shape)))
	}
	off := 0
	for k := range a.Shape {
		if shape[k] <= 0 || start[k] < 0 || start[k] > a.Shape[k]-shape[k] {
			panic(fmt.Sprintf("array: Window %v at %v outside shape %v", shape, start, a.Shape))
		}
		off += start[k] * a.Stride[k]
	}
	return a.view(off, clone(shape), clone(a.Stride))
}

// Select returns the elements with index i along axis, with that axis
// removed: Select(0, t) of a [time, y, x] array is the [y, x] array at
// time t.
func (a Array[T]) Select(axis, i int) Array[T] {
	a.requireAxis("Select", axis)
	if uint(i) >= uint(a.Shape[axis]) {
		panic(fmt.Sprintf("array: Select index %d out of range [0, %d) on axis %d", i, a.Shape[axis], axis))
	}
	return a.view(i*a.Stride[axis], slices.Delete(clone(a.Shape), axis, axis+1),
		slices.Delete(clone(a.Stride), axis, axis+1))
}

// Transpose returns a with its axes permuted: axis k of the result is
// axis perm[k] of a. With no arguments it reverses the axes. It panics
// if perm is not a permutation of a's axes.
func (a Array[T]) Transpose(perm ...int) Array[T] {
	n := len(a.Shape)
	if len(perm) == 0 {
		perm = make([]int, n)
		for k := range perm {
			perm[k] = n - 1 - k
		}
	}
	if len(perm) != n {
		panic(fmt.Sprintf("array: Transpose of %d axes on a rank-%d array", len(perm), n))
	}
	seen := make([]bool, n)
	shape, stride := make([]int, n), make([]int, n)
	for k, p := range perm {
		if uint(p) >= uint(n) || seen[p] {
			panic(fmt.Sprintf("array: Transpose %v is not a permutation of %d axes", perm, n))
		}
		seen[p] = true
		shape[k], stride[k] = a.Shape[p], a.Stride[p]
	}
	return a.view(0, shape, stride)
}

// Reshape returns a compact array's elements in a new shape with the
// same element count, in the same row-major order. It panics if a is not
// Compact (copy it into a New array first) or the counts differ.
func (a Array[T]) Reshape(shape ...int) Array[T] {
	n := requireShape("array: Reshape", shape)
	if n != a.Len() {
		panic(fmt.Sprintf("array: Reshape %v to %v changes the element count", a.Shape, shape))
	}
	if !a.Compact() {
		panic(fmt.Sprintf("array: Reshape of a non-compact array (shape %v, strides %v)", a.Shape, a.Stride))
	}
	return a.view(0, clone(shape), compactStrides(shape))
}

// BroadcastTo returns a repeated to the given shape by numpy's rules: the
// shapes are aligned at their last axis, a's length on each axis must
// equal the target's or be 1, and missing leading axes count as 1. A
// repeated axis gets stride 0, so nothing is copied, and the result must
// not be used as a destination. It panics if a cannot broadcast to shape.
func (a Array[T]) BroadcastTo(shape ...int) Array[T] {
	requireShape("array: BroadcastTo", shape)
	stride, ok := broadcastStrides(a.Shape, a.Stride, shape)
	if !ok {
		panic(fmt.Sprintf("array: shape %v does not broadcast to %v", a.Shape, shape))
	}
	return a.view(0, clone(shape), stride)
}

// ExpandDims returns a with a new axis of length 1 inserted at position
// axis, which may be 0 through a.Rank(): the way to line an axis up for
// broadcasting, such as a [y, x] mean against a [y, time, x] array.
func (a Array[T]) ExpandDims(axis int) Array[T] {
	if axis < 0 || axis > len(a.Shape) {
		panic(fmt.Sprintf("array: ExpandDims axis %d outside [0, %d]", axis, len(a.Shape)))
	}
	return a.view(0, slices.Insert(clone(a.Shape), axis, 1), slices.Insert(clone(a.Stride), axis, 0))
}

// BroadcastShape returns the shape that arrays of the given shapes
// broadcast to under numpy's rules, or an error if they do not: aligned
// at their last axis, each axis's lengths must be equal or 1.
func BroadcastShape(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, s := range shapes {
		rank = max(rank, len(s))
	}
	out := make([]int, rank)
	for k := range out {
		out[k] = 1
	}
	for _, s := range shapes {
		for k, n := range s {
			j := rank - len(s) + k
			switch {
			case n <= 0:
				return nil, fmt.Errorf("array: shape %v has a non-positive length", s)
			case out[j] == 1:
				out[j] = n
			case n != 1 && n != out[j]:
				return nil, fmt.Errorf("array: shapes %v do not broadcast together", shapes)
			}
		}
	}
	return out, nil
}

// broadcastStrides returns the strides of an array of the given shape
// and strides repeated to target, or false if it does not broadcast.
func broadcastStrides(shape, stride, target []int) ([]int, bool) {
	if len(shape) > len(target) {
		return nil, false
	}
	out := make([]int, len(target))
	lead := len(target) - len(shape)
	for k, n := range shape {
		switch {
		case n == target[lead+k]:
			if n != 1 {
				out[lead+k] = stride[k]
			}
		case n != 1:
			return nil, false
		}
	}
	return out, true
}

func (a Array[T]) requireAxis(op string, axis int) {
	if uint(axis) >= uint(len(a.Shape)) {
		panic(fmt.Sprintf("array: %s axis %d outside a rank-%d array", op, axis, len(a.Shape)))
	}
}
