// Package terrain computes terrain derivatives of elevation rasters:
// Horn gradients and slope so far.
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
// windows whose parent has data beyond them. Reading a halo from the
// parent belongs to tiled execution (DESIGN.md §16–17) and is not done
// here.
//
// # Validity
//
// Validity is never inferred from Data: every interior cell is computed,
// and a NaN or ±Inf elevation with its bit set is an ordinary value that
// flows through IEEE arithmetic. If the DEM has a mask, an output cell is
// valid iff it is interior and all nine cells of its 3×3 neighbourhood are
// valid (the centre too, although Horn gives it zero weight). Data under
// an invalid output cell is unspecified. The output masks are computed
// with word-level operations, separately from the arithmetic.
//
// If the DEM has a mask, every output must have one too, or the call
// panics; allocate outputs with raster.NewFloat32Like(dem). If the DEM
// has no mask (all valid), no erosion is done and an output's mask is
// left as it is except for its border bits, which are cleared.
//
// # Errors
//
// Like package raster, functions panic on programming errors: rasters
// that fail Validate, mismatched dimensions, outputs whose Data overlaps
// another raster's Data (the stencil reads neighbours of cells it has not
// written yet), overlapping mask bits, and invalid options.
//
// # Backends
//
// Row kernels live in internal/stencil. Builds with GOEXPERIMENT=simd on
// amd64 CPUs with AVX2 run vectorized kernels that agree bit-for-bit with
// the scalar ones; other builds run scalar.
package terrain
