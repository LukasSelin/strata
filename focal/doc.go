// Package focal computes neighbourhood (focal) operations over rasters:
// weighted sums with a caller's (2r+1)×(2r+1) weights (Correlate,
// Convolve), separable weighted sums (CorrelateSeparable, with Gaussian
// taps), and the focal Mean, Min and Max of each cell's (2r+1)×(2r+1)
// neighbourhood. DESIGN.md §53 records the design.
//
// # Conventions
//
// Weights are row-major, (2r+1)×(2r+1): row 0 is the row above the
// output cell by r (dy = -r), column 0 the column left of it by r. The
// weight at row j and column c multiplies the input cell at (x+c-r,
// y+j-r):
//
//	Correlate: out(x, y) = Σ_j Σ_c w[j][c] · in(x+c-r, y+j-r)
//	Convolve:  out(x, y) = Σ_j Σ_c w[j][c] · in(x-c+r, y-j+r)
//
// so Convolve is Correlate with the weights rotated by 180°, the
// convention of scipy.ndimage.convolve and of textbooks, and Correlate is
// the one of scipy.ndimage.correlate, OpenCV's filter2D and GIS "focal
// weights". For weights symmetric under that rotation they agree.
// CorrelateSeparable applies a row of taps and a column of taps; for
// symmetric taps, such as Gaussian's, it is also a convolution.
//
// The radius is required and must be 1 to MaxRadius. It has no default:
// a Mean whose radius was forgotten would otherwise quietly become 3×3.
//
// # Evaluation order
//
// Every cell's terms are folded in a fixed order, which is what makes
// the Tiled and Chunked forms and the SIMD kernels give the same bits as
// the plain functions:
//
//   - Correlate adds its products in row-major order of the input cells
//     they read (top row first, left to right), so Convolve(w) gives the
//     bits of Correlate with the rotated weights. The sum starts from the
//     first product rather than from +0, and every product and sum is
//     rounded to float32 separately: there is never a fused multiply-add.
//     Every weight is used, zeros included, so a zero weight on an
//     infinite cell gives NaN, as it does in numpy.
//   - CorrelateSeparable folds each column of the neighbourhood with the
//     column taps first, top to bottom, and then those column results
//     with the row taps, left to right. Its result is therefore not
//     bit-for-bit Correlate's with the outer product of the taps as
//     weights, though it is the same sum.
//   - Mean adds the neighbourhood the same way, columns then row, and
//     divides the sum by (2r+1)², rather than multiplying by the
//     reciprocal. So the mean of a constant raster need not be exactly
//     that constant.
//   - Min and Max use Go's builtin min and max: a NaN anywhere in the
//     neighbourhood gives NaN, and -0 is below +0. Their result does not
//     depend on the order.
//
// A sliding (running) sum would cost less for large radii, but its
// rounding would depend on where each band of rows starts, which
// breaks the tiling guarantee below, so every cell is summed afresh.
//
// # Edges
//
// A cell within r of the edge of the rasters passed in has an incomplete
// neighbourhood. Those cells of the output get NaN in Data and, if the
// output has a validity mask, a cleared validity bit. A raster no more
// than 2r cells wide or tall is all edge. The edge is the edge of the
// rasters passed in, even when they are windows whose parent has data
// beyond them.
//
// # Tiled execution
//
// The Tiled functions run the same operations in tiles on
// engine.Options.Workers goroutines (by default one per GOMAXPROCS) with a
// context, and return ctx.Err() if cancelled (see package engine). Cells
// on tile boundaries read their neighbours from the input, so the result
// is bit-for-bit the plain function's for every tiling and worker count.
// The plain functions run the same kernels as one tile with one worker,
// on the calling goroutine.
//
// The Chunked functions read the input from an engine.RasterSource and
// write to an engine.RasterSink a tile at a time, so rasters larger than
// memory run in Workers × tile buffers (DESIGN.md §27). They give the same
// bits as the plain functions, for every tiling and worker count.
//
// engine.Stats counts each band's window, halo included, as kernel reads,
// and bands are about 2¹⁶ cells of whole tile rows: with the default
// full-width tile and a radius of 5, a 4096-wide raster gets 16-row bands
// that each read 26 rows. That is what Stats reports, not work done twice.
//
// # Validity
//
// Validity is never inferred from Data: every interior cell is computed,
// and a NaN or ±Inf input with its bit set is an ordinary value that
// flows through IEEE arithmetic. If the input has a mask, an output cell
// is valid iff it is not an edge cell and every cell of its
// (2r+1)×(2r+1) neighbourhood is valid, whatever its weight. Data under an
// invalid output cell is unspecified. Operations that skip invalid cells
// (a mean of the valid cells only, weights renormalised over them) are
// not provided; DESIGN.md §53 says why.
//
// If the input has a mask, the output must have one too, or the call
// panics; allocate outputs with raster.NewFloat32Like(src). If the input
// has no mask, an output without one stays without one, and an output
// with one gets every interior cell marked valid and its edge cleared.
//
// # Errors
//
// Functions panic on programming errors. Invalid options (a radius out of
// range, weights or taps of the wrong length or not finite) panic with a
// message starting "focal:", before anything is written. Operand errors
// (rasters that fail Validate, mismatched dimensions, an output
// overlapping the input) are the engine's, and their messages start with
// "engine:". The Tiled functions return an error only for cancellation,
// and the Chunked functions also for errors of their sources and sinks.
//
// # Backends
//
// Row kernels live in internal/focalrow. Builds with GOEXPERIMENT=simd run
// vectorized kernels on amd64 CPUs with AVX2 and on arm64 (NEON) that
// agree bit-for-bit with the scalar ones; other builds run scalar.
package focal
