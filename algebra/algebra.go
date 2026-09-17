package algebra

import (
	"unsafe"

	"strata/internal/vec"
	"strata/raster"
)

// Add computes dst = a + b cell by cell.
func Add(dst, a, b raster.Float32Raster) { binary("algebra.Add", dst, a, b, vec.Add) }

// Sub computes dst = a - b cell by cell.
func Sub(dst, a, b raster.Float32Raster) { binary("algebra.Sub", dst, a, b, vec.Sub) }

// Mul computes dst = a * b cell by cell.
func Mul(dst, a, b raster.Float32Raster) { binary("algebra.Mul", dst, a, b, vec.Mul) }

// Min computes dst = min(a, b) cell by cell.
func Min(dst, a, b raster.Float32Raster) { binary("algebra.Min", dst, a, b, vec.Min) }

// Max computes dst = max(a, b) cell by cell.
func Max(dst, a, b raster.Float32Raster) { binary("algebra.Max", dst, a, b, vec.Max) }

// Clamp computes dst = min(max(src, lo), hi) cell by cell.
func Clamp(dst, src raster.Float32Raster, lo, hi float32) {
	const op = "algebra.Clamp"
	check(op, "dst", dst, dst)
	check(op, "src", dst, src)
	clamp(op, dst, src, lo, hi, compact(dst) && compact(src))
}

// clamp is Clamp after the checks. whole selects one kernel call over all
// cells, which requires every operand to be compact (see binaryApply).
func clamp(op string, dst, src raster.Float32Raster, lo, hi float32, whole bool) {
	if whole {
		n := dst.Width * dst.Height
		vec.Clamp(dst.Data[:n], src.Data[:n], lo, hi)
	} else {
		for y := range dst.Height {
			vec.Clamp(dst.Row(y), src.Row(y), lo, hi)
		}
	}
	unaryValidity(op, dst, src, whole)
}

// binaryKernel is an internal/vec kernel over equal-length flat slices.
// It is called once per raster or once per row, never per cell.
type binaryKernel func(dst, a, b []float32)

func binary(op string, dst, a, b raster.Float32Raster, kernel binaryKernel) {
	check(op, "dst", dst, dst)
	check(op, "a", dst, a)
	check(op, "b", dst, b)
	binaryApply(op, dst, a, b, kernel, compact(dst) && compact(a) && compact(b))
}

// binaryApply is binary after the checks. whole selects one kernel call
// and one mask pass over all cells instead of one per row. It requires
// every operand to be compact: sharing a wider stride is not enough,
// because the span would then cover dst's row padding, which in a window
// is its parent's cells.
func binaryApply(op string, dst, a, b raster.Float32Raster, kernel binaryKernel, whole bool) {
	if whole {
		n := dst.Width * dst.Height
		kernel(dst.Data[:n], a.Data[:n], b.Data[:n])
	} else {
		for y := range dst.Height {
			kernel(dst.Row(y), a.Row(y), b.Row(y))
		}
	}
	binaryValidity(op, dst, a, b, whole)
}

func compact(r raster.Float32Raster) bool { return r.Stride == r.Width }

// check panics unless r is a consistent raster with dst's dimensions whose
// cells either are dst's cells or do not overlap them.
func check(op, name string, dst, r raster.Float32Raster) {
	if err := r.Validate(); err != nil {
		panic(op + ": " + name + ": " + err.Error())
	}
	if r.Width != dst.Width || r.Height != dst.Height {
		panic(op + ": " + name + " dimensions differ from dst")
	}
	if name == "dst" {
		return
	}
	if overlaps(dst, r) {
		panic(op + ": dst overlaps " + name + " at a different offset or stride")
	}
}

// overlaps reports whether dst and r share memory other than by being the
// same cells. With equal strides it is exact, so disjoint windows of one
// raster, such as side-by-side tiles, are accepted even though each one's
// Data span runs through the other's rows. With different strides it
// conservatively compares the spans.
func overlaps(dst, r raster.Float32Raster) bool {
	const size = int(unsafe.Sizeof(float32(0)))
	d0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(dst.Data))))
	r0 := int(uintptr(unsafe.Pointer(unsafe.SliceData(r.Data))))
	if d0 >= r0+span(r)*size || r0 >= d0+span(dst)*size {
		return false
	}
	if d0 == r0 {
		return dst.Stride != r.Stride && dst.Height > 1
	}
	if dst.Stride != r.Stride {
		return true
	}
	// dst's first cell is dy rows and dx columns from r's, with 0 <= dx <
	// Stride. Its columns [dx, dx+Width) meet r's columns [0, Width) in the
	// same row, or, wrapping around the stride, one row further down.
	diff := (d0 - r0) / size
	dy, dx := diff/dst.Stride, diff%dst.Stride
	if dx < 0 {
		dy, dx = dy-1, dx+dst.Stride
	}
	rowsMeet := func(dy int) bool { return dy > -dst.Height && dy < dst.Height }
	return (dx < dst.Width && rowsMeet(dy)) || (dx > dst.Stride-dst.Width && rowsMeet(dy+1))
}

// span is the length of the Data prefix holding r's cells.
func span(r raster.Float32Raster) int { return (r.Height-1)*r.Stride + r.Width }

// binaryValidity sets dst's validity to the AND of a's and b's.
func binaryValidity(op string, dst, a, b raster.Float32Raster, whole bool) {
	switch {
	case a.Valid == nil:
		unaryValidity(op, dst, b, whole)
		return
	case b.Valid == nil:
		unaryValidity(op, dst, a, whole)
		return
	}
	requireDstMask(op, dst)
	if whole {
		raster.MaskAndRange(dst.Valid, dst.ValidOffset, a.Valid, a.ValidOffset,
			b.Valid, b.ValidOffset, dst.Width*dst.Height)
		return
	}
	for y := range dst.Height {
		raster.MaskAndRange(dst.Valid, dst.ValidOffset+y*dst.Stride,
			a.Valid, a.ValidOffset+y*a.Stride, b.Valid, b.ValidOffset+y*b.Stride, dst.Width)
	}
}

// unaryValidity sets dst's validity to src's.
func unaryValidity(op string, dst, src raster.Float32Raster, whole bool) {
	if src.Valid == nil {
		fillValid(dst, whole)
		return
	}
	requireDstMask(op, dst)
	if sameBits(dst, src) {
		return // in place: dst's bits already are src's
	}
	if whole {
		raster.MaskCopyRange(dst.Valid, dst.ValidOffset, src.Valid, src.ValidOffset,
			dst.Width*dst.Height)
		return
	}
	for y := range dst.Height {
		raster.MaskCopyRange(dst.Valid, dst.ValidOffset+y*dst.Stride,
			src.Valid, src.ValidOffset+y*src.Stride, dst.Width)
	}
}

// fillValid marks every cell of dst valid. Without a mask there is nothing
// to do: that is the all-valid fast path.
func fillValid(dst raster.Float32Raster, whole bool) {
	if dst.Valid == nil {
		return
	}
	if whole {
		raster.MaskFillRange(dst.Valid, dst.ValidOffset, dst.Width*dst.Height, true)
		return
	}
	for y := range dst.Height {
		raster.MaskFillRange(dst.Valid, dst.ValidOffset+y*dst.Stride, dst.Width, true)
	}
}

func requireDstMask(op string, dst raster.Float32Raster) {
	if dst.Valid == nil {
		panic(op + ": an input has a validity mask but dst.Valid is nil; allocate dst " +
			"with raster.NewFloat32Like, or set Valid on its root raster before windowing")
	}
}

// sameBits reports whether dst and src are backed by the same bits of the
// same mask, as when dst is src.
func sameBits(dst, src raster.Float32Raster) bool {
	return unsafe.SliceData(dst.Valid) == unsafe.SliceData(src.Valid) &&
		dst.ValidOffset == src.ValidOffset && (dst.Stride == src.Stride || dst.Height == 1)
}
