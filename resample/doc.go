// Package resample resamples a raster from one grid onto another in the
// same coordinate reference system: a different resolution, a different
// origin, or both, with axis-aligned cells (DESIGN.md §54). It follows
// gdalwarp's conventions for the same case, so that
//
//	gdalwarp -tr <resx> <resy> -te <extent> -r <method>
//
// is its outside reference (acceptance/). Reprojection, which needs CRS
// transformations, is out of scope (DESIGN.md §36).
//
//	resample.Resample(dst, src, resample.Options{Method: resample.Cubic})
//	err := resample.ResampleTiled(ctx, dst, src, opts, engine.Options{})
//	err := resample.ResampleChunked(ctx, sink, dstGrid, source, srcGrid, opts, engine.Options{TileHeight: 256})
//
// # Geometry
//
// Cells are areas (raster.Grid): output cell (c, r) is represented by its
// centre, which maps to source pixel coordinates (u, v), where source
// cell (i, j) covers [i, i+1) × [j, j+1). Resolutions may have any sign on
// either grid; a south-up source resamples onto a north-up grid.
//
//   - Nearest takes source cell (floor(u), floor(v)). A centre on a cell
//     edge takes the cell after it, as gdalwarp does.
//   - Bilinear, Cubic and Lanczos weigh the source cells around (u, v) by
//     a separable kernel of the distance between centres: the tent, Keys'
//     cubic with a = -0.5, and the Lanczos window with a = 3. When an axis
//     downsamples, with s source cells per output cell, the kernel is
//     stretched by s along that axis so that it averages rather than
//     aliases: for Bilinear and Cubic once s > 1/0.95, for Lanczos once
//     s > 1, gdalwarp's thresholds.
//   - Average weighs the source cells by the area of the output cell each
//     one covers.
//
// Weights are normalised to sum to 1 over the source cells inside the
// source, so the edge of the source renormalises rather than reading
// past it. An output cell whose centre lies outside the source is
// invalid, except under Average, which needs only to overlap it.
//
// Cubic follows gdalwarp in one more respect: where neither axis
// downsamples, a cell whose 4×4 neighbourhood loses a cell of non-zero
// weight to the edge of the source, or includes an invalid cell, is
// computed with Bilinear instead.
//
// # Validity
//
// A cell of an input without a mask is valid. With one, an output cell is
// valid when the source cell under its centre is valid (except under
// Average) and the weights of its valid source cells have a positive sum;
// under Lanczos, at least half of the source cells its kernel reaches
// must also be valid, unless its centre lies exactly on a source centre. Its value is then the weighted mean of its
// valid source cells: the same bits as without a mask when every one is
// valid, renormalised over them otherwise. This is gdalwarp's rule, so
// NoData erodes nothing that is still valid under the centre, unlike the
// stencils' validity (DESIGN.md §31).
//
// Invalid cells get NaN in Data and a cleared bit. dst must have a mask
// when src has one, or when part of dst lies outside the source.
//
// # Arithmetic
//
// Bilinear, Cubic, Lanczos and Average run as two passes, one along
// rows and one along columns, over weights computed once per call, in
// float32 with every product rounded separately (no fused multiply-add)
// and every sum taken in increasing source index. Every backend, tiling
// and worker count gives the same bits (DESIGN.md §15, §23). Nearest
// copies bits, NaN payloads included.
//
// # Deviations from gdalwarp
//
// Values agree with gdalwarp -wt Float64 to within float32 rounding,
// except where gdalwarp itself departs from the definitions above:
//
//   - Lanczos downsampling by an odd integer factor, where gdalwarp gives
//     the tap at an output centre about 83 times its weight;
//   - a source one cell wide or high, where gdalwarp's Bilinear degrades
//     to Nearest;
//   - Lanczos for 1 < s < 1.05, where gdalwarp switches between stretched
//     and unstretched kernels;
//   - Average on output cells that extend past the source, by about 1%.
//
// # Tiled and chunked execution
//
// ResampleTiled splits dst into engine.Options tiles and tiles into
// bands of rows, and runs the bands on workers. ResampleChunked works a
// tile at a time over a RasterSource and a RasterSink: each worker holds
// an output tile, the window of the source that tile's cells reach, and
// the intermediate between the passes. With TW×TH tiles and s_x, s_y
// source cells per output cell, that is about
//
//	Workers × 4 B × (TW·TH + (TW·s_x + k)(TH·s_y + k)·(1 + 3·masked) + TW·(TH·s_y + k)·(1 + 2·masked))
//
// where k is the kernel's reach in source cells (2 for Bilinear, 4 for
// Cubic, 6 for Lanczos, each stretched by s when downsampling), plus the
// weight tables, which grow with dst's width and height, not its area.
// Downsampling by a large factor therefore wants small tiles.
//
// # Backends
//
// Builds with GOEXPERIMENT=simd run vectorised passes: AVX2 on amd64 CPUs
// that have it, NEON on arm64. Other builds run the scalar passes, with
// the same results.
//
// # Errors
//
// Programming errors panic with a "resample: " message; Tiled returns
// only ctx.Err(), and Chunked also returns source and sink errors.
package resample
