// Package reduce folds a raster to numbers: the count, smallest and
// largest of its valid cells, their exact sum, and a Summary with the
// mean and standard deviation beside them (DESIGN.md §49). It is the other half of a
// map — package algebra and package terrain turn rasters into rasters,
// and this turns a raster into a value — so it is what lets a caller ask
// for the range of a DEM, or how many cells of a chunked output are
// valid, without reading the whole thing back.
//
//	mn, mx, n := reduce.MinMax(dem)
//	s := reduce.Stats(dem) // Count, Sum, Mean, StdDev, Min, Max
//	s, err := reduce.StatsChunked(ctx, demSource, engine.Options{TileHeight: 256})
//
// The operations are named, not a per-cell callback, for the reasons of
// DESIGN.md §19. Histograms and quantiles are a later operation, because they return a vector rather than a scalar
// and their bin edges are a question of policy this package should not
// answer.
//
// # Operands
//
// Any stride, window, or window of a window. A raster that fails Validate
// panics, as elsewhere; so does a negative engine.Options. Only
// cancellation and a source's IO failure come back as errors.
//
// # Values and validity
//
// Only valid cells take part (DESIGN.md §31). A nil mask means every cell
// counts, which is also what a source that is not Masked means. Data
// under an invalid cell is unspecified and is never read, so scribbling
// over it cannot change an answer.
//
// Validity is never inferred from Data. A NaN in a valid cell is an
// ordinary value, not NoData: it propagates through Min and Max with the
// semantics of Go's builtins, so both come back NaN. A caller who wants
// it skipped clears the bit. -0 and +0 follow the builtins too, so a
// raster holding both has -0 as its minimum and +0 as its maximum.
//
// The count comes back beside every value, so an empty reduction is
// visible rather than disguised: with no valid cells MinMax returns NaN,
// NaN and 0. Since Validate forbids a raster with no cells, that can only
// mean a mask with no bits set.
//
// # Determinism
//
// A reduction returns the same bits for every tile size, worker count and
// backend. Min and Max keep that promise without machinery: under Go's
// semantics NaN is absorbing and -0 sorts below +0, which makes them
// associative and commutative, so the cells may be split into tiles and
// bands and folded by any number of workers, and the vector kernels may
// keep eight accumulator lanes, and the answer cannot move.
//
// The one thing that would move is which NaN's payload survives, since
// Go's min and max do not say. A NaN result is therefore the canonical
// quiet NaN rather than a payload copied out of the data.
//
// Sum, Mean and StdDev keep it with an accumulator that does not round as
// it goes: floating-point addition is not associative, so a float64 sum
// would depend on where the tiles cut the raster. Every finite float32 is
// an integer times a power of two, so the sum and the sum of squares are
// kept as integers, which add in any order, and rounded once when the
// result is read. Sum, Mean and the variance under StdDev are therefore
// correctly rounded; StdDev itself is a 256-bit square root of the exact
// variance, rounded, so within one ulp and still order-independent.
//
// A NaN in a valid cell makes Sum and Mean NaN, and so do infinities of
// both signs; one infinity makes them that infinity. Any NaN or infinity
// makes StdDev NaN. A sum of zeros is -0 only when every valid cell is
// -0, as IEEE addition gives.
//
// # Tiled and chunked execution
//
// The Tiled functions run a reduction through the engine over a raster in
// memory; the Chunked ones run it over an engine.RasterSource with
// bounded memory, reading a tile at a time. Both give the plain
// function's bits for every engine.Options.
//
// Cancellation differs from a map pass. A cancelled map leaves a prefix
// of finished tiles in its sinks, which is useful; a fold over an unknown
// subset of a raster is not, so a cancelled or failed reduction returns
// ctx.Err(), or the source's error, and no value at all.
//
// # Allocation
//
// Unlike package algebra's plain functions, these allocate a few small
// slices per call: they run through the engine, as package terrain's do,
// so that the plain, Tiled and Chunked forms cannot drift apart. Nothing
// is allocated per tile, band, row or cell. Like the rest of the module
// outside the engine, this package starts no goroutines of its own.
package reduce
