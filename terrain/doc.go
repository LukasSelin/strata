// Package terrain computes terrain derivatives of elevation rasters from
// Horn's 3×3 gradient: Gradient, Slope, Aspect and Hillshade.
//
// # Conventions
//
// x increases with the column index and y with the row index. On a
// north-up grid that is east and south. Gradient's dx is ∂z/∂x (positive
// when elevation rises eastward) and dy is ∂z/∂y (positive when it rises
// southward), so the northward gradient of a north-up DEM is -dy. These
// are the dx and dy of GDAL gdaldem's Horn aspect kernel (east minus
// west, bottom row minus top row). gdaldem's slope kernel uses west minus
// east instead, which squaring hides.
//
// Directions are in degrees. Aspect and Hillshade's Azimuth are compass
// bearings, clockwise from north (decreasing row index), so a cell whose
// elevation falls eastward has aspect 90. The downslope direction in
// (east, north) components is (-dx, dy).
//
// Cell sizes are positive ground distances in the elevation's units, not
// signed geotransform resolutions: pass abs(ResolutionY) for a north-up
// raster.Grid. A ZFactor multiplies elevations before differencing, for
// elevations in different units from the cell size; gdaldem's -s scale
// corresponds to ZFactor 1/scale.
//
// # Edges
//
// Horn's method needs all eight neighbours, which the one-cell border of
// the raster does not have. Border cells of every output get NaN in Data
// and, if the output has a validity mask, a cleared validity bit. A
// raster narrower or shorter than three cells is all border.
//
// The edge is the edge of the rasters passed in, even when they are
// windows whose parent has data beyond them.
//
// # Tiled execution
//
// SlopeTiled, AspectTiled, HillshadeTiled and GradientTiled run the same
// operations in tiles with engine.Options and a context, and return
// ctx.Err() if cancelled (see package engine). Cells on tile boundaries
// read their neighbours from the DEM, so the result is bit-for-bit the
// plain function's for every tiling. The plain functions run the same
// kernels as one tile.
//
// # Validity
//
// Validity is never inferred from Data: every interior cell is computed,
// and a NaN or ±Inf elevation with its bit set is an ordinary value that
// flows through IEEE arithmetic. Values that are defined but degenerate,
// such as the aspect of a flat cell, are ordinary valid values too. If the DEM has a mask, an output cell is
// valid iff it is interior and all nine cells of its 3×3 neighbourhood are
// valid (the centre too, although Horn gives it zero weight). Data under
// an invalid output cell is unspecified. The output masks are computed
// with word-level operations, separately from the arithmetic.
//
// If the DEM has a mask, every output must have one too, or the call
// panics; allocate outputs with raster.NewFloat32Like(dem). If the DEM
// has no mask (all valid), no erosion is done: an output without a mask
// stays without one, and an output with a mask gets every interior cell
// marked valid and its border cleared, so stale bits from earlier use do
// not survive. This matches package algebra.
//
// # Errors
//
// Like package raster, functions panic on programming errors: rasters
// that fail Validate, mismatched dimensions, outputs whose Data overlaps
// another raster's Data (the stencil reads neighbours of cells it has not
// written yet), overlapping mask bits, and invalid options. The operand
// checks are shared with the Tiled functions, and their panic messages
// start with "engine:". The Tiled functions return an error only for
// cancellation.
//
// # Backends
//
// Row kernels live in internal/stencil. Builds with GOEXPERIMENT=simd on
// amd64 CPUs with AVX2 run vectorized kernels that agree bit-for-bit with
// the scalar ones; other builds run scalar.
package terrain
