package algebra

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/pointwise"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// The Tiled functions run an operation through the engine: they take the
// same operands, apply the same operand, in-place and validity rules
// (and also reject a dst whose mask bits partly overlap an input's) and
// write the same bits as the plain function for every engine.Options,
// and return ctx.Err() if ctx is done before every cell is written. See
// package engine for tiling and cancellation. Unlike the plain functions
// they allocate a few small slices per call.

// AddTiled is Add run by the engine.
func AddTiled(ctx context.Context, dst, a, b raster.Float32Raster, opts engine.Options) error {
	return binaryTiled(ctx, dst, a, b, vec.OpAdd, opts)
}

// SubTiled is Sub run by the engine.
func SubTiled(ctx context.Context, dst, a, b raster.Float32Raster, opts engine.Options) error {
	return binaryTiled(ctx, dst, a, b, vec.OpSub, opts)
}

// MulTiled is Mul run by the engine.
func MulTiled(ctx context.Context, dst, a, b raster.Float32Raster, opts engine.Options) error {
	return binaryTiled(ctx, dst, a, b, vec.OpMul, opts)
}

// MinTiled is Min run by the engine.
func MinTiled(ctx context.Context, dst, a, b raster.Float32Raster, opts engine.Options) error {
	return binaryTiled(ctx, dst, a, b, vec.OpMin, opts)
}

// MaxTiled is Max run by the engine.
func MaxTiled(ctx context.Context, dst, a, b raster.Float32Raster, opts engine.Options) error {
	return binaryTiled(ctx, dst, a, b, vec.OpMax, opts)
}

// MaskTiled is Mask run by the engine.
func MaskTiled(ctx context.Context, dst, src, mask raster.Float32Raster, opts engine.Options) error {
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, []raster.Float32Raster{src, mask}, maskOp{}, opts)
}

// ClampTiled is Clamp run by the engine.
func ClampTiled(ctx context.Context, dst, src raster.Float32Raster, lo, hi float32, opts engine.Options) error {
	return exec.Process(ctx, dst, src, clampKernel{lo, hi}, opts)
}

// The Chunked functions run an operation through the engine over sources
// and sinks with bounded memory: they read the inputs and write dst a tile
// at a time, with Workers × tile buffers in memory, and write the bits the
// plain function would write into in-memory rasters for every
// engine.Options. They return an error if a source or sink fails, or
// ctx.Err() if ctx is done before every tile is written. See package
// engine for sources, sinks, memory, cancellation and errors.

// AddChunked is Add run by the engine over sources and a sink.
func AddChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, opts engine.Options) error {
	return binaryChunked(ctx, dst, a, b, vec.OpAdd, opts)
}

// SubChunked is Sub run by the engine over sources and a sink.
func SubChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, opts engine.Options) error {
	return binaryChunked(ctx, dst, a, b, vec.OpSub, opts)
}

// MulChunked is Mul run by the engine over sources and a sink.
func MulChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, opts engine.Options) error {
	return binaryChunked(ctx, dst, a, b, vec.OpMul, opts)
}

// MinChunked is Min run by the engine over sources and a sink.
func MinChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, opts engine.Options) error {
	return binaryChunked(ctx, dst, a, b, vec.OpMin, opts)
}

// MaxChunked is Max run by the engine over sources and a sink.
func MaxChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, opts engine.Options) error {
	return binaryChunked(ctx, dst, a, b, vec.OpMax, opts)
}

// MaskChunked is Mask run by the engine over sources and a sink. The
// mask source is read like any other, so its validity must come from the
// source itself: a raw file carries none unless RawOptions.Fill marks it.
func MaskChunked(ctx context.Context, dst engine.RasterSink, src, mask engine.RasterSource, opts engine.Options) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src, mask}, maskOp{}, opts)
}

// ClampChunked is Clamp run by the engine over a source and a sink.
func ClampChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, lo, hi float32, opts engine.Options) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, clampKernel{lo, hi}, opts)
}

func binaryChunked(ctx context.Context, dst engine.RasterSink, a, b engine.RasterSource, op vec.Op, opts engine.Options) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{a, b}, newBinaryOp(op), opts)
}

func binaryTiled(ctx context.Context, dst, a, b raster.Float32Raster, op vec.Op, opts engine.Options) error {
	return exec.ProcessN(ctx, []raster.Float32Raster{dst}, []raster.Float32Raster{a, b}, newBinaryOp(op), opts)
}

type clampKernel struct{ lo, hi float32 }

func (clampKernel) Radius() int                  { return 0 }
func (clampKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k clampKernel) Process(dst exec.Span, src exec.Window) {
	d, s := dst.Dst[0], src.Src[0]
	clampValues(d, s, k.lo, k.hi, compact(d) && compact(s))
}

// Fuse lets a Pipeline run Clamp as one step of a fused chain rather
// than as a pass of its own (DESIGN.md §29). vec.OpClamp is the very
// kernel clampValues calls, so the bits are the same either way.
func (k clampKernel) Fuse() (vec.Step, bool) {
	return vec.Step{Op: vec.OpClamp, K: [2]float32{k.lo, k.hi}}, true
}

// maskOp copies the first input's cells. The second input is the mask:
// the engine ANDs every input's validity into dst, which is all Mask
// needs from it, so its values are never read.
type maskOp struct{}

func (maskOp) Radius() int                  { return 0 }
func (maskOp) Arity() (inputs, outputs int) { return 2, 1 }

func (maskOp) Process(dst exec.Span, src exec.Window) {
	d, s := dst.Dst[0], src.Src[0]
	pointwise.CopyValues(d, s, compact(d) && compact(s))
}

// binaryOp is one of the two-operand algebra operations as a Kernel. It
// carries the operation as a vec.Op as well as the function that runs
// it: the function is what Process calls, and the op is the identity a
// Pipeline needs to fuse the stage into a chain, which a bare
// binaryKernel does not carry.
type binaryOp struct {
	op     vec.Op
	kernel binaryKernel
}

func newBinaryOp(op vec.Op) binaryOp {
	var kernel binaryKernel
	switch op {
	case vec.OpAdd:
		kernel = vec.Add
	case vec.OpSub:
		kernel = vec.Sub
	case vec.OpMul:
		kernel = vec.Mul
	case vec.OpDiv:
		kernel = vec.Div
	case vec.OpMin:
		kernel = vec.Min
	case vec.OpMax:
		kernel = vec.Max
	default:
		panic("algebra: no binary kernel for " + op.String())
	}
	return binaryOp{op: op, kernel: kernel}
}

func (binaryOp) Radius() int                  { return 0 }
func (binaryOp) Arity() (inputs, outputs int) { return 2, 1 }

func (k binaryOp) Process(dst exec.Span, src exec.Window) {
	d, a, b := dst.Dst[0], src.Src[0], src.Src[1]
	binaryValues(d, a, b, k.kernel, compact(d) && compact(a) && compact(b))
}

// Fuse lets a Pipeline run this operation as one step of a fused chain
// rather than as a pass of its own (DESIGN.md §29). The step is the same
// vec kernel Process calls, so the bits are the same either way.
func (k binaryOp) Fuse() (vec.Step, bool) { return vec.Step{Op: k.op}, true }
