package algebra

import (
	"github.com/LukasSelin/strata/internal/pointwise"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
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
	checkUnary("algebra.Clamp", dst, src)
	clamp(dst, src, lo, hi, compact(dst) && compact(src))
}

// Mask writes src into dst, valid where src and mask are both valid.
// Only mask's validity is read, never its values; a nil mask on either
// input means all valid.
func Mask(dst, src, mask raster.Float32Raster) {
	const op = "algebra.Mask"
	pointwise.Check(op, "dst", dst, dst)
	pointwise.Check(op, "src", dst, src)
	pointwise.Check(op, "mask", dst, mask)
	pointwise.CheckMasks(op, dst, src, mask)
	// Values and validity choose the whole-raster path independently:
	// the copy never touches mask, and mask's stride only constrains the
	// bits. Like the other operations, values are written first.
	pointwise.CopyValues(dst, src, compact(dst) && compact(src))
	pointwise.BinaryValidity(dst, src, mask, compact(dst) && compact(src) && compact(mask))
}

// clamp is Clamp after the checks. whole selects one kernel call over all
// cells, which requires every operand to be compact (see binaryApply).
func clamp(dst, src raster.Float32Raster, lo, hi float32, whole bool) {
	clampValues(dst, src, lo, hi, whole)
	pointwise.UnaryValidity(dst, src, whole)
}

// clampValues is the arithmetic of clamp, without validity.
func clampValues(dst, src raster.Float32Raster, lo, hi float32, whole bool) {
	if whole {
		n := dst.Width * dst.Height
		vec.Clamp(dst.Data[:n], src.Data[:n], lo, hi)
		return
	}
	for y := range dst.Height {
		vec.Clamp(dst.Row(y), src.Row(y), lo, hi)
	}
}

// compact is internal/pointwise's, named locally because the operations
// ask it about every operand.
func compact(r raster.Float32Raster) bool { return pointwise.Compact(r) }

// binaryKernel is an internal/vec kernel over equal-length flat slices.
// It is called once per raster or once per row, never per cell.
type binaryKernel func(dst, a, b []float32)

func binary(op string, dst, a, b raster.Float32Raster, kernel binaryKernel) {
	pointwise.Check(op, "dst", dst, dst)
	pointwise.Check(op, "a", dst, a)
	pointwise.Check(op, "b", dst, b)
	pointwise.CheckMasks(op, dst, a, b)
	binaryApply(dst, a, b, kernel, compact(dst) && compact(a) && compact(b))
}

// binaryApply is binary after the checks. whole selects one kernel call
// and one mask pass over all cells instead of one per row. It requires
// every operand to be compact: sharing a wider stride is not enough,
// because the span would then cover dst's row padding, which in a window
// is its parent's cells.
func binaryApply(dst, a, b raster.Float32Raster, kernel binaryKernel, whole bool) {
	binaryValues(dst, a, b, kernel, whole)
	pointwise.BinaryValidity(dst, a, b, whole)
}

// binaryValues is the arithmetic of binaryApply, without validity.
func binaryValues(dst, a, b raster.Float32Raster, kernel binaryKernel, whole bool) {
	if whole {
		n := dst.Width * dst.Height
		kernel(dst.Data[:n], a.Data[:n], b.Data[:n])
		return
	}
	for y := range dst.Height {
		kernel(dst.Row(y), a.Row(y), b.Row(y))
	}
}
