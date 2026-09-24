// Package array holds strata's N-dimensional arrays: Array[T], flat
// data of one numeric type with a shape, per-axis strides and the same
// validity mask as a raster, zero-copy views onto it, elementwise
// operations that broadcast, and reductions along axes (DESIGN.md §10).
//
// Environmental data is often not a single grid: a [time, y, x] stack of
// scenes, a [level, time, y, x] model output, a [scenario, time, y, x]
// ensemble. These are Arrays, and a 2-D slice of one is a raster:
//
//	stack := array.NewMasked[float32](nt, ny, nx)
//	scene := array.ToRaster(stack.Select(0, t))     // time step t, for terrain, focal, …
//	mean := array.NewMasked[float64](ny, nx)
//	array.MeanOver(mean, stack, 0)                  // per-pixel mean over time
//
// # Layout
//
// Element (i0, …, iN-1) is Data[i0*Stride[0] + … + iN-1*Stride[N-1]].
// Arrays from New and Wrap are compact and row-major, the last axis
// contiguous. Views keep the parent's Data and step through it with
// their own strides, so Slice, Window, Select, Transpose, Reshape,
// BroadcastTo and ExpandDims copy nothing. A view's Data starts at its
// first element and ends at its last, as a raster window's does.
// Strides are never negative; a stride of 0 repeats one element, which is
// what BroadcastTo makes, and such a view may be read but not written.
//
// # Element types
//
// T is any fixed-width integer or float type. The raster compute type
// stays float32 (DESIGN.md §9): the SIMD kernels, and every package that
// takes a raster, work on float32. Arrays of other types carry what
// storage holds — class labels, counts, float64 model output — with
// scalar loops, and Convert moves between types at the boundary.
// Elementwise operations on float32 arrays run the same internal/vec
// kernels as package algebra over every contiguous run.
//
// # Validity
//
// As for rasters (DESIGN.md §31): Valid is a bitmap shared by an array
// and its views, nil means all valid, bit ValidOffset+i is the validity
// of Data[i], and data under a cleared bit is unspecified. Views keep the
// parent's Valid and move ValidOffset with their first element, so the
// bits follow the strides without a copy. Attach a mask to the root array
// before taking views, as with rasters.
//
// # Elementwise operations
//
// Add, Sub, Mul, Min, Max, Copy, Convert and Fill write into a
// caller-supplied dst. Inputs are broadcast to dst's shape by numpy's
// rules: shapes are aligned at their last axis, and an input's length on
// an axis must equal dst's or be 1, with missing leading axes counting as
// 1. BroadcastShape computes the shape to allocate.
//
// Values follow Go: +, -, * and the builtin min and max, so integers
// wrap, NaN propagates and min(-0, +0) is -0. A result is valid iff every
// input is valid there. If any input has a mask dst must have one; if
// none does, dst's elements are marked valid, and with no masks anywhere
// no mask work is done.
//
// dst may be the same elements as an input in the same layout, which
// computes in place, or share no memory with it. Anything between panics:
// the test is conservative, so two views whose spans meet are rejected
// even if their elements interleave without touching. dst must not repeat
// an element (a broadcast view). Only dst's elements are written, in Data
// and in Valid.
//
// # Reductions
//
// CountOver, SumOver, MeanOver, MinOver and MaxOver fold src along one or
// more axes into dst, whose shape is src's without those axes, or with
// them kept at length 1. Only valid elements take part. Sum and Mean are
// exact, then correctly rounded to float64, so they agree bit for bit
// with package reduce on the same cells; Min and Max are those of Go's
// builtins, with a NaN result canonical. No result depends on src's
// layout or the order elements are visited in. An output with nothing to
// fold is invalid for Mean, Min and Max, and 0 for Count and Sum.
//
// A reduction over every axis of a float32 array, with the rest of
// package reduce's statistics, is package reduce applied to ToRaster of a
// 2-D view or a Reshape of a compact array.
//
// # Errors and allocation
//
// Like package raster, invalid arguments panic: they are programming
// errors. Validate checks a hand-built array. Operations allocate a few
// small slices per call to plan their loops, and the reductions one block
// of accumulators, never anything per element. They run on the calling
// goroutine; there is no tiled or chunked form yet (DESIGN.md §10).
package array
