package algebra

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/pointwise"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// Normalize maps src's valid cells onto [0, 1]: with lo and hi the
// smallest and largest of them, as reduce.MinMax finds them, it computes
// dst = (src - lo) / (hi - lo) cell by cell in float32, rounding the
// difference and then the quotient. It returns lo and hi, so a caller can
// map a result back or detect a degenerate range.
//
// The difference-then-quotient form is what makes the endpoints exact: a
// cell equal to lo gives +0 and a cell equal to hi gives exactly 1, and
// since both steps round monotonically, every valid cell lands in
// [0, 1]. transfer.RescaleRange(dst, src, lo, hi, 0, 1) is the same map
// as one multiply-add, but its rounded coefficients put hi near 1 rather
// than on it: on elevation-like data about one raster in three comes out
// with a maximum other than 1, half of those above it, by up to tens of
// ulps. The values are those of
// (x - x.min()) / (x.max() - x.min()) over a float32 array in NumPy,
// except that a -0 cell gives +0 here: MinMax orders -0 below +0, so lo
// is -0 whenever one is present.
//
// Degenerate ranges follow the arithmetic rather than being special
// cases, as elsewhere in this package:
//
//   - A constant raster (lo == hi) gives 0/0, NaN in every finite cell.
//   - A NaN in a valid cell makes lo and hi NaN (see reduce.MinMax), and
//     so every cell NaN.
//   - An infinite valid cell, or a range wider than float32 can hold,
//     such as -MaxFloat32 to MaxFloat32, makes the span +Inf, and cells
//     give 0 or NaN as IEEE division falls.
//   - With no valid cells, lo and hi are NaN and dst is all invalid.
//
// Validity is src's, as for Clamp. dst may be src: the reduction finishes
// before the first cell is written. Unlike the other plain functions,
// Normalize allocates a few small slices per call, in the reduction.
func Normalize(dst, src raster.Float32Raster) (lo, hi float32) {
	const op = "algebra.Normalize"
	checkUnary(op, dst, src)
	lo, hi, _ = reduce.MinMax(src)
	normalize(dst, src, lo, hi, compact(dst) && compact(src))
	return lo, hi
}

// NormalizeTiled is Normalize run by the engine: one reduction and one
// map, both tiled. If ctx is done it returns ctx.Err() and zero bounds,
// and dst may hold some tiles of the result, as the other Tiled
// functions leave it.
func NormalizeTiled(ctx context.Context, dst, src raster.Float32Raster, opts engine.Options) (lo, hi float32, err error) {
	checkUnary("algebra.NormalizeTiled", dst, src)
	lo, hi, _, err = reduce.MinMaxTiled(ctx, src, opts)
	if err != nil {
		return 0, 0, err
	}
	if err := exec.Process(ctx, dst, src, normalizeKernel{lo, hi - lo}, opts); err != nil {
		return 0, 0, err
	}
	return lo, hi, nil
}

// NormalizeChunked is Normalize run by the engine over a source and a
// sink. It reads src twice, a reduction and then a map, so it costs two
// full passes over a file where the other Chunked functions cost one
// (DESIGN.md §49). If a read or write fails, or ctx is done, it returns
// the error and zero bounds; after a failed map pass the sink may hold
// some tiles of the result.
func NormalizeChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts engine.Options) (lo, hi float32, err error) {
	lo, hi, _, err = reduce.MinMaxChunked(ctx, src, opts)
	if err != nil {
		return 0, 0, err
	}
	k := normalizeKernel{lo, hi - lo}
	if err := exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, k, opts); err != nil {
		return 0, 0, err
	}
	return lo, hi, nil
}

// checkUnary applies the operand rules to a one-input, one-output
// operation. Normalize runs it before reducing, so a bad dst panics
// before a whole pass over src is spent on it.
func checkUnary(op string, dst, src raster.Float32Raster) {
	pointwise.Check(op, "dst", dst, dst)
	pointwise.Check(op, "src", dst, src)
	pointwise.CheckMasks(op, dst, src)
}

// normalize is Normalize after the checks and the reduction.
func normalize(dst, src raster.Float32Raster, lo, hi float32, whole bool) {
	normalizeValues(dst, src, lo, hi-lo, whole)
	pointwise.UnaryValidity(dst, src, whole)
}

// normalizeValues is the arithmetic of normalize, without validity.
func normalizeValues(dst, src raster.Float32Raster, lo, span float32, whole bool) {
	if whole {
		n := dst.Width * dst.Height
		vec.SubDiv(dst.Data[:n], src.Data[:n], lo, span)
		return
	}
	for y := range dst.Height {
		vec.SubDiv(dst.Row(y), src.Row(y), lo, span)
	}
}

type normalizeKernel struct{ lo, span float32 }

func (normalizeKernel) Radius() int                  { return 0 }
func (normalizeKernel) Arity() (inputs, outputs int) { return 1, 1 }

// Fuse lets a Pipeline run Normalize's second pass as one step of a
// fused chain rather than as a pass of its own (DESIGN.md §29).
// vec.OpSubDiv is the kernel normalizeValues calls, subtraction before
// division and all, so the bits are the same either way.
func (k normalizeKernel) Fuse() (vec.Step, bool) {
	return vec.Step{Op: vec.OpSubDiv, K: [2]float32{k.lo, k.span}}, true
}

func (k normalizeKernel) Process(dst exec.Span, src exec.Window) {
	d, s := dst.Dst[0], src.Src[0]
	normalizeValues(d, s, k.lo, k.span, compact(d) && compact(s))
}
