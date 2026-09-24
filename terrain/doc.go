// Package terrain computes terrain derivatives of elevation rasters:
// Gradient, Slope, Aspect and Hillshade from Horn's 3×3 gradient,
// profile, plan and mean Curvature from the Zevenbergen–Thorne quadratic
// fitted to the same 3×3 window, and Ruggedness (the terrain ruggedness
// index, topographic position index and roughness) from the window's
// differences, bit-identical to GDAL gdaldem's on the 3×3 window and
// defined the same way over larger ones (RuggednessOptions.Radius), for
// the same measure at several scales. The derivatives take a scale too:
// with FitRadius set they come from Wood's quadratic fitted to a larger
// window (see Multi-scale derivatives). HeatLoad is McCune and Keon's
// heat load index, or their potential direct incident radiation, from the
// same gradient and a latitude. WeightedSlope is Slope multiplied
// cell by cell by a weight raster, in one pass. Surface writes any of
// Gradient, Slope, Aspect and Hillshade at once from one gradient, and
// Features any mix of Slope, Aspect, Hillshade, HeatLoad, Curvature and
// Ruggedness at any radii from one reading of the DEM, each bit for bit
// what the standalone function writes.
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
// raster.Grid. A grid in a geographic CRS has cells in degrees, not ground
// distances: project it first (DESIGN.md §36). A ZFactor multiplies
// elevations before differencing, for elevations in different units from
// the cell size; gdaldem's -s scale corresponds to ZFactor 1/scale.
//
// # Multi-scale derivatives
//
// FitRadius, in GradientOptions, SlopeOptions, AspectOptions,
// HillshadeOptions, HeatLoadOptions, CurvatureOptions and
// SurfaceOptions, replaces the 3×3
// estimate with Wood's (1996) least-squares quadratic
//
//	z = a·x² + b·y² + c·x·y + d·x + e·y + f
//
// fitted to the (2r+1)×(2r+1) window around each cell, r = FitRadius, with
// x = i·CellSize and y = j·CellSizeY for column and row offsets i and j
// from −r to r. It is the method of GRASS r.param.scale and LandSerf
// (unweighted), and smooths over the window, so a larger r measures
// the surface at a coarser scale. The derivatives are p = d and q = e (dx
// and dy in Gradient's conventions), r = 2a, t = 2b and s = c, and the
// products are the same functions of them as of Horn's gradient and ZT's
// derivatives. At FitRadius 1 the fit is Evans's 3×3 method, not Horn's
// or ZT's, so it does not equal FitRadius 0.
//
// On a square window the least-squares equations separate, and with
// K = 2r+1, S = Σ i² = r(r+1)(2r+1)/3, t2(i) = 3i² − r(r+1) and
// Q = Σ t2(i)², each derivative is one weighted sum of the window:
//
//	p = Σ i·z · Z/(K·S·CellSize)
//	q = Σ j·z · Z/(K·S·CellSizeY)
//	r = Σ t2(i)·z · 6Z/(K·Q·CellSize²)
//	t = Σ t2(j)·z · 6Z/(K·Q·CellSizeY²)
//	s = Σ i·j·z · Z/(S²·CellSize·CellSizeY)
//
// with the sums over the window and Z the ZFactor. Each sum is computed
// in float32 as focal.CorrelateSeparable computes one: the 2r+1 rows under
// a cell folded into column sums, top to bottom, then 2r+1 of those
// folded left to right, every tap applied, zeros included. Then it is
// multiplied by its factor, computed in float64 and rounded to float32
// once. The border is r cells wide, and a cell is valid iff its whole
// window is. Every tap is an integer, so the column and row passes run
// focal's SIMD kernels.
//
// # Edges
//
// Every operation needs all eight neighbours, which the one-cell border
// of the raster does not have. Border cells of every output get NaN in Data
// and, if the output has a validity mask, a cleared validity bit. A
// raster narrower or shorter than three cells is all border. Ruggedness
// over a (2r+1)² window needs all of it, so its border is r cells wide.
//
// The edge is the edge of the rasters passed in, even when they are
// windows whose parent has data beyond them.
//
// # Tiled execution
//
// SlopeTiled, AspectTiled, HillshadeTiled, HeatLoadTiled, GradientTiled, CurvatureTiled,
// RuggednessTiled, WeightedSlopeTiled, SurfaceTiled and FeaturesTiled run the same operations in tiles on engine.Options.Workers goroutines (by default
// one per GOMAXPROCS) with a context, and return ctx.Err() if cancelled
// (see package engine). Cells on tile boundaries read their neighbours
// from the DEM, so the result is bit-for-bit the plain function's for
// every tiling and worker count. The plain functions run the same kernels
// as one tile with one worker, on the calling goroutine.
//
// SlopeChunked, AspectChunked, HillshadeChunked, HeatLoadChunked, GradientChunked,
// CurvatureChunked, RuggednessChunked, WeightedSlopeChunked,
// SurfaceChunked and FeaturesChunked read the DEM from an engine.RasterSource and write to engine.RasterSinks a
// tile at a time, so rasters larger than memory, such as raw float32
// files, run in Workers × tile buffers (DESIGN.md §27). They give the
// same bits as the plain functions on the same data, for every tiling and
// worker count; the edge is the raster's edge, not a tile's.
//
// # Validity
//
// Validity is never inferred from Data: every interior cell is computed,
// and a NaN or ±Inf elevation with its bit set is an ordinary value that
// flows through IEEE arithmetic. Values that are defined but degenerate,
// such as the aspect of a flat cell, are ordinary valid values too. If the DEM has a mask, an output cell is
// valid iff it is interior and all nine cells of its 3×3 neighbourhood are
// valid (the centre too, which Curvature and Ruggedness read although
// Horn gives it zero weight), or for Ruggedness at radius r all (2r+1)²
// cells of its window. WeightedSlope's weight is read at the
// cell alone, so it narrows validity by that cell only. Data under
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
// written yet), overlapping mask bits, and invalid options, including
// cell sizes and z-factors so far apart that ZFactor/(8·CellSize) or
// ZFactor/(8·CellSizeY) overflows or underflows float32 (for Curvature,
// ZFactor/(2·CellSize), ZFactor/CellSize², ZFactor/(4·CellSize·CellSizeY)
// and their CellSizeY counterparts). Ruggedness has no cell size or
// ZFactor, and panics only on an unknown RuggednessType. The operand
// checks are shared with the Tiled functions, and their panic messages
// start with "engine:". The Tiled functions return an error only for
// cancellation, and the Chunked functions also for errors of their
// sources and sinks.
//
// # Backends
//
// Row kernels live in internal/stencil. Builds with GOEXPERIMENT=simd on
// amd64 CPUs with AVX2 run vectorized kernels that agree bit-for-bit with
// the scalar ones; other builds run scalar.
package terrain
