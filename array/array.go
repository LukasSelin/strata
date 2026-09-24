package array

import (
	"fmt"
	"math"

	"github.com/LukasSelin/strata/raster"
)

// Number is the element types an Array may hold: the fixed-width
// integers and the two float widths. int and uint are left out on
// purpose, because their width depends on the platform and a stored
// array's does not.
type Number interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Array is an N-dimensional array of T, or a view into one. Views are
// also Arrays, so anything that accepts an array accepts a view, and
// views can be viewed again.
type Array[T Number] struct {
	// Data holds the elements. Element (i0, …, iN-1) is
	// Data[i0*Stride[0] + … + iN-1*Stride[N-1]].
	Data []T

	// Shape is the length of each dimension, outermost first. Every
	// length is positive. An empty Shape is a rank-0 array of one
	// element.
	Shape []int
	// Stride is the step in Data, in elements, between neighbours along
	// each dimension. Strides are never negative; a stride of 0 repeats
	// one element along that dimension, which is how BroadcastTo works.
	Stride []int

	// Valid is the validity mask shared by an array and all its views,
	// or nil when every element is valid. The bit of Data[i] is
	// ValidOffset+i, laid out as in package raster.
	Valid []uint64
	// ValidOffset is the bit index in Valid of Data[0].
	ValidOffset int
}

// New allocates a zeroed, compact array of the given shape with no
// validity mask. It panics if a length is not positive or the element
// count overflows an int.
func New[T Number](shape ...int) Array[T] {
	n := requireShape("array.New", shape)
	return Array[T]{
		Data:   make([]T, n),
		Shape:  clone(shape),
		Stride: compactStrides(shape),
	}
}

// NewMasked is New with an all-valid validity mask attached.
func NewMasked[T Number](shape ...int) Array[T] {
	a := New[T](shape...)
	a.Valid = raster.NewMask(len(a.Data))
	return a
}

// Wrap wraps data as a compact array of the given shape, with no
// validity mask. data is not copied. It panics if a length is not
// positive or data holds fewer elements than the shape. Elements past
// the shape's count are not part of the array.
func Wrap[T Number](data []T, shape ...int) Array[T] {
	n := requireShape("array.Wrap", shape)
	if len(data) < n {
		panic(fmt.Sprintf("array.Wrap: data has %d elements, need %d for shape %v", len(data), n, shape))
	}
	return Array[T]{
		Data:   data[:n:n],
		Shape:  clone(shape),
		Stride: compactStrides(shape),
	}
}

// NewLike allocates a zeroed, compact array with a's shape. If a has a
// validity mask the result gets its own all-valid mask; if a has none,
// neither does the result.
func NewLike[T, U Number](a Array[U]) Array[T] {
	if a.Valid != nil {
		return NewMasked[T](a.Shape...)
	}
	return New[T](a.Shape...)
}

// Rank returns the number of dimensions.
func (a Array[T]) Rank() int { return len(a.Shape) }

// Len returns the number of elements: the product of the lengths.
func (a Array[T]) Len() int {
	n := 1
	for _, s := range a.Shape {
		n *= s
	}
	return n
}

// Compact reports whether a's elements are laid out row-major with no
// gaps and no repeats, so that Data[:a.Len()] is every element in order.
func (a Array[T]) Compact() bool {
	want := 1
	for k := len(a.Shape) - 1; k >= 0; k-- {
		if a.Shape[k] != 1 && a.Stride[k] != want {
			return false
		}
		want *= a.Shape[k]
	}
	return true
}

// Offset returns the Data index of the element at idx. It panics if idx
// has the wrong length or any index is out of range.
func (a Array[T]) Offset(idx ...int) int {
	if len(idx) != len(a.Shape) {
		panic(fmt.Sprintf("array: %d indices for a rank-%d array", len(idx), len(a.Shape)))
	}
	off := 0
	for k, i := range idx {
		if uint(i) >= uint(a.Shape[k]) {
			panic(fmt.Sprintf("array: index %d out of range [0, %d) on axis %d", i, a.Shape[k], k))
		}
		off += i * a.Stride[k]
	}
	return off
}

// At returns the element at idx.
func (a Array[T]) At(idx ...int) T { return a.Data[a.Offset(idx...)] }

// Set stores v at idx.
func (a Array[T]) Set(v T, idx ...int) { a.Data[a.Offset(idx...)] = v }

// IsValid reports whether the element at idx holds data. It is true for
// every element of an array without a mask.
func (a Array[T]) IsValid(idx ...int) bool {
	off := a.Offset(idx...)
	return a.Valid == nil || raster.MaskGet(a.Valid, a.ValidOffset+off)
}

// SetValid marks the element at idx valid or invalid in the shared mask.
// It panics if a has no mask: as with rasters, a mask must be attached to
// the root array before taking views, or the root would not see it.
func (a Array[T]) SetValid(valid bool, idx ...int) {
	off := a.Offset(idx...)
	if a.Valid == nil {
		panic("array: SetValid on an array without a validity mask")
	}
	raster.MaskSet(a.Valid, a.ValidOffset+off, valid)
}

// Validate reports whether a's fields are consistent: Shape and Stride
// of equal length, positive lengths, non-negative strides, Data long
// enough for the last element, and, if Valid is set, a non-negative
// ValidOffset with Valid long enough to cover every element. It does not
// inspect values or mask bits, and it accepts strides that make
// elements share memory (broadcast views); operations that write check
// that separately.
func (a Array[T]) Validate() error {
	if len(a.Stride) != len(a.Shape) {
		return fmt.Errorf("array: %d strides for %d dimensions", len(a.Stride), len(a.Shape))
	}
	for k, s := range a.Shape {
		if s <= 0 {
			return fmt.Errorf("array: length %d on axis %d is not positive", s, k)
		}
		if a.Stride[k] < 0 {
			return fmt.Errorf("array: negative stride %d on axis %d", a.Stride[k], k)
		}
	}
	n, ok := spanLen(a.Shape, a.Stride)
	if !ok {
		return fmt.Errorf("array: shape %v with strides %v spans more elements than an int holds", a.Shape, a.Stride)
	}
	if len(a.Data) < n {
		return fmt.Errorf("array: data has %d elements, need %d", len(a.Data), n)
	}
	if a.Valid != nil {
		if a.ValidOffset < 0 {
			return fmt.Errorf("array: negative ValidOffset %d", a.ValidOffset)
		}
		if bits := len(a.Valid) * 64; a.ValidOffset > bits-n {
			return fmt.Errorf("array: mask has %d bits, need %d (offset %d + %d elements)",
				bits, a.ValidOffset+n, a.ValidOffset, n)
		}
	}
	return nil
}

// must panics with op's name if a is inconsistent.
func (a Array[T]) must(op, name string) {
	if err := a.Validate(); err != nil {
		panic(fmt.Sprintf("%s: %s: %v", op, name, err))
	}
}

// span is the number of Data elements from a's first element to its
// last, inclusive: the length its Data must have.
func (a Array[T]) span() int {
	n, _ := spanLen(a.Shape, a.Stride)
	return n
}

// spanLen is 1 + Σ(shape[k]-1)*stride[k], with ok false on overflow.
// It assumes positive lengths and non-negative strides.
func spanLen(shape, stride []int) (n int, ok bool) {
	n = 1
	for k, s := range shape {
		step := stride[k]
		if step == 0 || s == 1 {
			continue
		}
		if s-1 > (math.MaxInt-n)/step {
			return 0, false
		}
		n += (s - 1) * step
	}
	return n, true
}

// requireShape panics on a non-positive length or an element count that
// overflows an int, and returns the count.
func requireShape(op string, shape []int) int {
	n := 1
	for k, s := range shape {
		if s <= 0 {
			panic(fmt.Sprintf("%s: length %d on axis %d is not positive", op, s, k))
		}
		if n > math.MaxInt/s {
			panic(fmt.Sprintf("%s: shape %v has more elements than an int holds", op, shape))
		}
		n *= s
	}
	return n
}

// compactStrides returns the row-major strides of shape.
func compactStrides(shape []int) []int {
	st := make([]int, len(shape))
	n := 1
	for k := len(shape) - 1; k >= 0; k-- {
		st[k] = n
		n *= shape[k]
	}
	return st
}

func clone(s []int) []int { return append([]int{}, s...) }
