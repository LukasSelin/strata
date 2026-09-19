package transfer

import (
	"fmt"

	"github.com/LukasSelin/strata/internal/curve"
	"github.com/LukasSelin/strata/internal/pointwise"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Reclass computes dst = values[k] cell by cell, where k is the number
// of breaks less than or equal to the cell: values[i] covers
// [breaks[i-1], breaks[i]), values[0] every cell below the first break,
// and the last value every cell from the last break up. A cell exactly
// on a break takes the class above it. A NaN cell gives that same NaN.
//
// breaks must be strictly increasing and free of NaN, and values must
// hold one more element than breaks. Neither is copied; see the package
// documentation for what that means during a Tiled or Chunked run.
func Reclass(dst, src raster.Float32Raster, breaks, values []float32) {
	const op = "transfer.Reclass"
	k := newReclassKernel(op, breaks, values)
	check(op, dst, src)
	apply(dst, src, k)
}

// Lookup computes dst by linear interpolation along the polyline through
// the knots (xs[i], ys[i]), bounded: a cell at or below xs[0] gives
// ys[0] and a cell at or above the last x gives the last y. Every knot
// comes back exactly. A NaN cell gives that same NaN.
//
// Lookup interpolates. The operation that picks a table entry without
// interpolating is Reclass.
//
// xs must be strictly increasing and finite, and ys must be the same
// length, at least one. ys is not constrained: it need not be monotone,
// which is what lets it describe a curve that rises and falls, and a NaN
// or an infinity in it is an ordinary value that spreads over the
// segments touching it, though not over its own knot.
func Lookup(dst, src raster.Float32Raster, xs, ys []float32) {
	const op = "transfer.Lookup"
	k := newLookupKernel(op, xs, ys)
	check(op, dst, src)
	apply(dst, src, k)
}

// Rescale computes dst = a*src + b cell by cell, with the multiply and
// the add rounded separately: never a fused multiply-add, so the result
// is the same on every architecture and backend.
//
// Any a and b are accepted and the arithmetic is IEEE, so a zero a over
// an infinite cell gives NaN. Note that Rescale(dst, src, 1, 0) is not
// quite the identity: v + 0 turns -0 into +0. Rescale with b a negative
// zero is the identity for every cell, because v + -0 is v.
func Rescale(dst, src raster.Float32Raster, a, b float32) {
	const op = "transfer.Rescale"
	check(op, dst, src)
	apply(dst, src, rescaleKernel{a, b})
}

// RescaleRange is the affine map that sends inLo to outLo and inHi to
// outHi, applied cell by cell. It is Rescale with the coefficients that
// pair of intervals determines, so it writes the same bits and runs the
// same kernel.
//
// It does not clamp, so cells outside [inLo, inHi] are extrapolated. A
// clamping version would be exactly the two-knot curve
//
//	transfer.Lookup(dst, src, []float32{inLo, inHi}, []float32{outLo, outHi})
//
// which this package already has, and two spellings of one operation is
// what DESIGN.md §18 exists to prevent. On the affine path, follow
// RescaleRange with algebra.Clamp for the same effect.
//
// Unlike Rescale's a and b, these four numbers describe two intervals
// rather than being the arithmetic, so they are checked: all four must
// be finite, inLo and inHi must differ, and the coefficients they give
// must themselves be finite. outLo may equal outHi, which is a constant.
//
// The endpoints land near outLo and outHi rather than on them, to
// float32 precision relative to the output span; inLo maps to outLo
// exactly when it is zero, which is the common case. A caller who needs
// both endpoints exact for any interval wants the two-knot Lookup above,
// which reproduces its knots by construction — that, and not the clamp,
// is the substantive difference between the two.
func RescaleRange(dst, src raster.Float32Raster, inLo, inHi, outLo, outHi float32) {
	const op = "transfer.RescaleRange"
	a, b := rescaleCoeffs(op, inLo, inHi, outLo, outHi)
	check(op, dst, src)
	apply(dst, src, rescaleKernel{a, b})
}

// check applies the operand rules to a one-input, one-output operation.
func check(op string, dst, src raster.Float32Raster) {
	pointwise.Check(op, "dst", dst, dst)
	pointwise.Check(op, "src", dst, src)
	pointwise.CheckMasks(op, dst, src)
}

// kernel is what the three operations have in common: a span of cells
// they can fill, and a raster they can fill a row at a time. Both live
// on the same values the engine kernels in tiled.go are built from, so
// the plain, Tiled and Chunked forms of an operation run the same code
// over the same cells and write the same bits.
//
// apply is generic rather than taking this as an interface because a
// kernel holding two slices would have to be boxed to satisfy one, and
// that is an allocation per call — which is exactly what the plain
// functions promise not to do (TestNoAllocs).
type kernel interface {
	span(dst, src []float32)
	rows(dst, src raster.Float32Raster)
}

// apply runs an operation's arithmetic and then its validity: one pass
// over all cells when both operands are compact, and one per row
// otherwise. Sharing a wider stride is not enough for the whole-raster
// path, because the span would then cover dst's row padding, which in a
// window is its parent's cells.
func apply[K kernel](dst, src raster.Float32Raster, k K) {
	whole := compact(dst) && compact(src)
	if whole {
		n := dst.Width * dst.Height
		k.span(dst.Data[:n], src.Data[:n])
	} else {
		k.rows(dst, src)
	}
	pointwise.UnaryValidity(dst, src, whole)
}

// compact is internal/pointwise's, named locally because every operation
// asks it about both operands.
func compact(r raster.Float32Raster) bool { return pointwise.Compact(r) }

// The kernels. Each holds its table or its coefficients by reference and
// nothing else, so one value serves every worker of a Tiled or Chunked
// run (internal/exec.Kernel).

type reclassKernel struct{ breaks, values []float32 }

func (k reclassKernel) span(dst, src []float32) { curve.Reclass(dst, src, k.breaks, k.values) }

func (k reclassKernel) rows(dst, src raster.Float32Raster) {
	for y := range dst.Height {
		curve.Reclass(dst.Row(y), src.Row(y), k.breaks, k.values)
	}
}

type lookupKernel struct{ xs, ys []float32 }

func (k lookupKernel) span(dst, src []float32) { curve.Lookup(dst, src, k.xs, k.ys) }

func (k lookupKernel) rows(dst, src raster.Float32Raster) {
	for y := range dst.Height {
		curve.Lookup(dst.Row(y), src.Row(y), k.xs, k.ys)
	}
}

type rescaleKernel struct{ a, b float32 }

func (k rescaleKernel) span(dst, src []float32) { vec.Affine(dst, src, k.a, k.b) }

func (k rescaleKernel) rows(dst, src raster.Float32Raster) {
	for y := range dst.Height {
		vec.Affine(dst.Row(y), src.Row(y), k.a, k.b)
	}
}

// newReclassKernel and newLookupKernel resolve and check a table once,
// before any cell is written, as terrain's newSlopeKernel checks its
// options. The plain, Tiled and Chunked forms all go through them, so
// all three reject the same tables with the same message.

func newReclassKernel(op string, breaks, values []float32) reclassKernel {
	if len(values) != len(breaks)+1 {
		panic(fmt.Sprintf("%s: values must hold one more element than breaks, got %d values and %d breaks",
			op, len(values), len(breaks)))
	}
	checkIncreasing(op, "breaks", breaks)
	return reclassKernel{breaks, values}
}

func newLookupKernel(op string, xs, ys []float32) lookupKernel {
	if len(xs) != len(ys) {
		panic(fmt.Sprintf("%s: xs and ys must have equal length, got %d and %d", op, len(xs), len(ys)))
	}
	if len(xs) == 0 {
		panic(op + ": the table needs at least one knot")
	}
	checkIncreasing(op, "xs", xs)
	// An infinite knot would make every segment touching it flat, since
	// the cell's distance along it rounds to zero. The clamp already
	// carries the curve out to the infinities, so such a knot buys
	// nothing and only hides a mistake.
	for i, x := range xs {
		if x-x != 0 {
			panic(fmt.Sprintf("%s: xs must be finite, but xs[%d] = %v", op, i, x))
		}
	}
	return lookupKernel{xs, ys}
}

// checkIncreasing panics unless xs holds no NaN and strictly increases.
// internal/curve's scan relies on the order, and equal neighbours would
// make a class or a segment unreachable. NaN is rejected on its own
// rather than left to the comparison, which would miss it in a
// one-element table and would report it as an ordering fault in a longer
// one.
func checkIncreasing(op, name string, xs []float32) {
	for i, x := range xs {
		if x != x {
			panic(fmt.Sprintf("%s: %s must not hold NaN, but %s[%d] is NaN", op, name, name, i))
		}
	}
	for i := 1; i < len(xs); i++ {
		if !(xs[i-1] < xs[i]) {
			panic(fmt.Sprintf("%s: %s must be strictly increasing, but %s[%d] = %v is not above %s[%d] = %v",
				op, name, name, i, xs[i], name, i-1, xs[i-1]))
		}
	}
}

// rescaleCoeffs resolves two intervals into Rescale's a and b. It works
// in float64 and rounds once, as stencil.HornScales does, so the
// coefficients do not depend on the order of a float32 subtraction; the
// inner conversion in b stops the compiler fusing the multiply and the
// subtract, for the reason vec.Affine wraps its product. b is derived
// from the rounded a rather than independently, so it complements the
// coefficient the kernel will actually use.
//
// Neither endpoint is exact in general -- both a and b are rounded, and
// the kernel rounds again -- but a zero inLo makes b outLo itself, and
// then the kernel's float32(0*a) + outLo is outLo. Lookup is the
// operation whose endpoints are exact for every interval.
func rescaleCoeffs(op string, inLo, inHi, outLo, outHi float32) (a, b float32) {
	if !finite(inLo) || !finite(inHi) || inLo == inHi {
		panic(fmt.Sprintf("%s: inLo and inHi must be finite and different, got %v and %v", op, inLo, inHi))
	}
	if !finite(outLo) || !finite(outHi) {
		panic(fmt.Sprintf("%s: outLo and outHi must be finite, got %v and %v", op, outLo, outHi))
	}
	a = float32((float64(outHi) - float64(outLo)) / (float64(inHi) - float64(inLo)))
	b = float32(float64(outLo) - float64(float64(a)*float64(inLo)))
	if !finite(a) || !finite(b) {
		panic(fmt.Sprintf("%s: [%v, %v] to [%v, %v] gives a scale of %v and an offset of %v; both must be finite",
			op, inLo, inHi, outLo, outHi, a, b))
	}
	return a, b
}

// finite reports whether v is neither NaN nor an infinity: v-v is 0 for
// every finite v and NaN for the rest.
func finite(v float32) bool { return v-v == 0 }
