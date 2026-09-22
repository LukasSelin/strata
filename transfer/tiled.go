package transfer

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// The Tiled functions run an operation through the engine: they take the
// same operands and tables, apply the same operand, table, in-place and
// validity rules (and also reject a dst whose mask bits partly overlap
// src's) and write the same bits as the plain function for every
// engine.Options, and return ctx.Err() if ctx is done before every cell
// is written. See package engine for tiling and cancellation. Unlike the
// plain functions they allocate a few small slices per call.
//
// The tables are still not copied, and every worker reads them, so a
// caller must not mutate one while a call is in flight.

// ReclassTiled is Reclass run by the engine.
func ReclassTiled(ctx context.Context, dst, src raster.Float32Raster, breaks, values []float32, opts engine.Options) error {
	return exec.Process(ctx, dst, src, newReclassKernel("transfer.ReclassTiled", breaks, values), opts)
}

// LookupTiled is Lookup run by the engine.
func LookupTiled(ctx context.Context, dst, src raster.Float32Raster, xs, ys []float32, opts engine.Options) error {
	return exec.Process(ctx, dst, src, newLookupKernel("transfer.LookupTiled", xs, ys), opts)
}

// RescaleTiled is Rescale run by the engine.
func RescaleTiled(ctx context.Context, dst, src raster.Float32Raster, a, b float32, opts engine.Options) error {
	return exec.Process(ctx, dst, src, rescaleKernel{a, b}, opts)
}

// RescaleRangeTiled is RescaleRange run by the engine.
func RescaleRangeTiled(ctx context.Context, dst, src raster.Float32Raster, inLo, inHi, outLo, outHi float32, opts engine.Options) error {
	a, b := rescaleCoeffs("transfer.RescaleRangeTiled", inLo, inHi, outLo, outHi)
	return exec.Process(ctx, dst, src, rescaleKernel{a, b}, opts)
}

// The Chunked functions run an operation through the engine over a
// source and a sink with bounded memory: they read src and write dst a
// tile at a time, with Workers × tile buffers in memory, and write the
// bits the plain function would write into in-memory rasters for every
// engine.Options. They return an error if the source or the sink fails,
// or ctx.Err() if ctx is done before every tile is written. See package
// engine for sources, sinks, memory, cancellation and errors.

// ReclassChunked is Reclass run by the engine over a source and a sink.
func ReclassChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, breaks, values []float32, opts engine.Options) error {
	k := newReclassKernel("transfer.ReclassChunked", breaks, values)
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, k, opts)
}

// LookupChunked is Lookup run by the engine over a source and a sink.
func LookupChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, xs, ys []float32, opts engine.Options) error {
	k := newLookupKernel("transfer.LookupChunked", xs, ys)
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, k, opts)
}

// RescaleChunked is Rescale run by the engine over a source and a sink.
func RescaleChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, a, b float32, opts engine.Options) error {
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, rescaleKernel{a, b}, opts)
}

// RescaleRangeChunked is RescaleRange run by the engine over a source
// and a sink.
func RescaleRangeChunked(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, inLo, inHi, outLo, outHi float32, opts engine.Options) error {
	a, b := rescaleCoeffs("transfer.RescaleRangeChunked", inLo, inHi, outLo, outHi)
	return exec.ProcessChunked(ctx, []engine.RasterSink{dst}, []engine.RasterSource{src}, rescaleKernel{a, b}, opts)
}

// The engine side of the three kernels. All are pointwise, so they read
// no neighbours and have one input and one output; Process fills the
// span it is given with the same span method the plain functions use,
// and the engine derives the validity.

func (reclassKernel) Radius() int                  { return 0 }
func (reclassKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k reclassKernel) Process(dst exec.Span, src exec.Window) {
	process(dst, src, k)
}

func (lookupKernel) Radius() int                  { return 0 }
func (lookupKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k lookupKernel) Process(dst exec.Span, src exec.Window) {
	process(dst, src, k)
}

func (rescaleKernel) Radius() int                  { return 0 }
func (rescaleKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k rescaleKernel) Process(dst exec.Span, src exec.Window) {
	process(dst, src, k)
}

// Fuse lets a Pipeline run Rescale as one step of a fused chain rather
// than as a pass of its own (DESIGN.md §29). vec.OpAffine is the kernel
// rescaleKernel.span calls, unfused multiply-add and all, so the bits
// are the same either way. Reclass and Lookup have no Fuse: their inner
// scan over a table is per cell and data-dependent, which is not a step
// a lane of a vector can take (internal/curve).
func (k rescaleKernel) Fuse() (vec.Step, bool) {
	return vec.Step{Op: vec.OpAffine, K: [2]float32{k.a, k.b}}, true
}

// process is the body every Process shares: fill the output view from
// the input one, in one call when both are compact and a row at a time
// otherwise, exactly as the plain path does. It writes no validity bits,
// which belong to the engine.
func process[K kernel](dst exec.Span, src exec.Window, k K) {
	d, s := dst.Dst[0], src.Src[0]
	if compact(d) && compact(s) {
		n := d.Width * d.Height
		k.span(d.Data[:n], s.Data[:n])
		return
	}
	k.rows(d, s)
}
