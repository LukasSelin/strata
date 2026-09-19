// Package pointwise holds the operand rules and validity bookkeeping
// shared by the radius-0 packages whose plain functions do not go
// through the engine: algebra and transfer. Those functions promise to
// allocate nothing, which internal/exec's per-call setup cannot, so they
// run their vector kernels directly — and then need the checks and the
// mask work internal/exec would otherwise have done.
//
// Every function takes the operation's name, so a panic carries the
// caller's own prefix ("algebra.Add", "transfer.Reclass") rather than
// this package's. The rules are documented once, in the calling
// packages' doc comments, and are identical in both:
//
//   - dst and every input must have the same Width and Height, and must
//     each pass raster.Float32Raster.Validate. Any stride or window is
//     accepted.
//   - dst may be exactly an input, which computes in place, or may share
//     no cells with one. Sharing some but not all panics.
//   - An output cell is valid iff it is valid in every input. If any
//     input has a mask, dst must have one too.
//
// The whole argument that several of these take selects one pass over
// all cells instead of one per row. It requires every operand to be
// compact: sharing a wider stride is not enough, because the span would
// then cover dst's row padding, which in a window is its parent's cells.
package pointwise

import (
	"unsafe"

	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/raster"
)

// Compact reports whether r's rows are one contiguous run of cells.
func Compact(r raster.Float32Raster) bool { return r.Stride == r.Width }

// Check panics unless r is a consistent raster with dst's dimensions
// whose cells either are dst's cells or do not overlap them. Pass
// name == "dst" for the destination itself, which is only validated.
func Check(op, name string, dst, r raster.Float32Raster) {
	if err := r.Validate(); err != nil {
		panic(op + ": " + name + ": " + err.Error())
	}
	if r.Width != dst.Width || r.Height != dst.Height {
		panic(op + ": " + name + " dimensions differ from dst")
	}
	if name == "dst" {
		return
	}
	if overlap.Data(dst, r) == overlap.Partial {
		panic(op + ": dst overlaps " + name + " at a different offset or stride")
	}
}

// CheckMasks panics if an input has a validity mask and dst has none,
// before anything is written.
func CheckMasks(op string, dst raster.Float32Raster, inputs ...raster.Float32Raster) {
	if dst.Valid != nil {
		return
	}
	for _, in := range inputs {
		if in.Valid != nil {
			panic(op + ": an input has a validity mask but dst.Valid is nil; allocate dst " +
				"with raster.NewFloat32Like, or set Valid on its root raster before windowing")
		}
	}
}

// CopyValues copies src's cells into dst. It needs no internal/vec
// kernel: copy is a memmove, which already moves cells at memory speed,
// and there is nothing to compute. dst's cells being src's own, as in
// place, copies nothing.
func CopyValues(dst, src raster.Float32Raster, whole bool) {
	if overlap.Data(dst, src) == overlap.Same {
		return
	}
	if whole {
		n := dst.Width * dst.Height
		copy(dst.Data[:n], src.Data[:n])
		return
	}
	for y := range dst.Height {
		copy(dst.Row(y), src.Row(y))
	}
}

// BinaryValidity sets dst's validity to the AND of a's and b's. dst has
// a mask if an input does.
func BinaryValidity(dst, a, b raster.Float32Raster, whole bool) {
	switch {
	case a.Valid == nil:
		UnaryValidity(dst, b, whole)
		return
	case b.Valid == nil:
		UnaryValidity(dst, a, whole)
		return
	}
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

// UnaryValidity sets dst's validity to src's. dst has a mask if src does.
func UnaryValidity(dst, src raster.Float32Raster, whole bool) {
	if src.Valid == nil {
		FillValid(dst, whole)
		return
	}
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

// FillValid marks every cell of dst valid. Without a mask there is
// nothing to do: that is the all-valid fast path.
func FillValid(dst raster.Float32Raster, whole bool) {
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

// sameBits reports whether dst and src are backed by the same bits of
// the same mask, as when dst is src.
func sameBits(dst, src raster.Float32Raster) bool {
	return unsafe.SliceData(dst.Valid) == unsafe.SliceData(src.Valid) &&
		dst.ValidOffset == src.ValidOffset && (dst.Stride == src.Stride || dst.Height == 1)
}
