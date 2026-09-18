// Package overlap decides whether two rasters share memory, in their
// Data or in their validity bits. Package algebra and package engine use
// it to reject operands that would read cells an operation has already
// overwritten.
//
// Both rasters must have passed raster.Float32Raster.Validate. A raster's
// cells are the Width cells at the start of each of its Height rows, so a
// raster occupies span = (Height-1)*Stride + Width elements from its
// first cell; row padding between its cells is not its own.
package overlap

import (
	"unsafe"

	"github.com/LukasSelin/strata/raster"
)

// Relation is how the cells of two rasters with equal dimensions relate
// in memory.
type Relation int

const (
	// Disjoint rasters share no cells.
	Disjoint Relation = iota
	// Same rasters are the same cells in the same layout.
	Same
	// Partial rasters share some cells without being the same. With
	// different strides the test is conservative: interleaved rasters
	// whose spans meet are Partial even if no cell is shared.
	Partial
)

// Data relates the Data cells of a and b, which must have equal Width
// and Height. It is exact for equal strides, so disjoint windows of one
// raster, such as side-by-side tiles, are Disjoint even though each
// one's span runs through the other's rows.
func Data(a, b raster.Float32Raster) Relation {
	const size = int(unsafe.Sizeof(float32(0)))
	a0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(a.Data))))
	b0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(b.Data))))
	if (a0-b0)%size != 0 {
		// Misaligned float32 views of one buffer: only the spans can say.
		if spansMeet(a0, span(a)*size, b0, span(b)*size) {
			return Partial
		}
		return Disjoint
	}
	return relate(a0/size, a.Stride, b0/size, b.Stride, a.Width, a.Height)
}

// Bits relates the validity bits of a and b, which must have equal Width
// and Height, like Data. Rasters without a mask are Disjoint from all.
func Bits(a, b raster.Float32Raster) Relation {
	if a.Valid == nil || b.Valid == nil {
		return Disjoint
	}
	return relate(bitAddr(a), a.Stride, bitAddr(b), b.Stride, a.Width, a.Height)
}

// DataSpans reports whether the Data spans of a and b meet, whatever
// their dimensions: any shared memory, including being the same cells.
func DataSpans(a, b raster.Float32Raster) bool {
	const size = int(unsafe.Sizeof(float32(0)))
	a0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(a.Data))))
	b0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(b.Data))))
	return spansMeet(a0, span(a)*size, b0, span(b)*size)
}

// BitSpans is DataSpans for validity bits. Rasters without a mask meet
// nothing.
func BitSpans(a, b raster.Float32Raster) bool {
	if a.Valid == nil || b.Valid == nil {
		return false
	}
	return spansMeet(bitAddr(a), span(a), bitAddr(b), span(b))
}

// bitAddr is the absolute address, in bits, of r's first validity bit:
// bit i of a mask is bit i&63 of the word 8·(i>>6) bytes into it, 8·addr + i
// bits into memory, so masks that are different slices of one array
// compare correctly.
func bitAddr(r raster.Float32Raster) int {
	return int(uintptr(unsafe.Pointer(unsafe.SliceData(r.Valid))))*8 + r.ValidOffset
}

func span(r raster.Float32Raster) int { return (r.Height-1)*r.Stride + r.Width }

func spansMeet(a0, an, b0, bn int) bool { return a0 < b0+bn && b0 < a0+an }

// relate compares two w×h grids of cells starting at addresses a0 and b0
// (in cells) with the given strides.
func relate(a0, aStride, b0, bStride, w, h int) Relation {
	if !spansMeet(a0, (h-1)*aStride+w, b0, (h-1)*bStride+w) {
		return Disjoint
	}
	if a0 == b0 {
		if aStride == bStride || h == 1 {
			return Same
		}
		return Partial
	}
	if aStride != bStride {
		return Partial
	}
	// a's first cell is dy rows and dx columns from b's, with 0 <= dx <
	// stride. Its columns [dx, dx+w) meet b's columns [0, w) in the same
	// row, or, wrapping around the stride, one row further down.
	stride := aStride
	diff := a0 - b0
	dy, dx := diff/stride, diff%stride
	if dx < 0 {
		dy, dx = dy-1, dx+stride
	}
	rowsMeet := func(dy int) bool { return dy > -h && dy < h }
	if (dx < w && rowsMeet(dy)) || (dx > stride-w && rowsMeet(dy+1)) {
		return Partial
	}
	return Disjoint
}

// Words reports whether any validity word holding one of a's bits also
// holds one of b's, whatever their dimensions. Code that updates a's bits
// a word at a time races with code reading b's bits of the same word,
// even when no bit is shared. Rasters without a mask meet nothing.
func Words(a, b raster.Float32Raster) bool {
	if a.Valid == nil || b.Valid == nil {
		return false
	}
	words := func(r raster.Float32Raster) (first, n int) {
		first = bitAddr(r) >> 6
		return first, (bitAddr(r)+span(r)+63)>>6 - first
	}
	a0, an := words(a)
	b0, bn := words(b)
	return spansMeet(a0, an, b0, bn)
}
