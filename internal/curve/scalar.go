// This file holds the scalar backend: the correctness reference and
// fallback implementation for every kernel in this package. Any SIMD
// backend added later must produce identical results to these functions,
// including on NaN, infinities, signed zeros, cells that land exactly on
// a break or a knot, and one-entry tables.
//
// Both kernels find a cell's place in the table with the same forward
// scan, stopping at the first entry above the cell. The tables are small
// — four to eight breaks for a risk class, five to ten knots for a factor
// curve — and BenchmarkTableSize measures the scan against the two
// obvious alternatives over that range, on uniformly spread cells, on
// cells clustered in one class, and on cells outside the table entirely.
//
// On spread cells, the case a computed surface actually produces, the
// scan wins at every size that matters: 105 against 83 Mcells/s for a
// binary search at four breaks, 90 against 62 at eight, 72 against 47 at
// sixteen (Zen 2, scalar, ±2%). A binary search only overtakes it past
// about sixteen breaks, and then only on clustered or out-of-range cells,
// whose branches it predicts; on spread cells it is still behind at
// sixty-four. A branchless count of the whole table — the form one
// reaches for expecting mispredictions to dominate — is the slowest of
// the three beyond two breaks: never stopping early costs more than the
// branches it avoids.
//
// The input is not assumed sorted: cells arrive in whatever order the
// raster holds them.
package curve

// scalarReclassFloat32 writes values[k] for each cell, where k is the
// number of breaks at or below it, so values[k] covers
// [breaks[k-1], breaks[k]) and the ends run to the infinities.
//
// The NaN test comes first because the scan alone would get it wrong:
// v < b is false for every break, so a NaN would count as at or above all
// of them and take the top class. -0 and +0 compare equal, so they always
// share a class, which is what a classification of a measured surface
// wants.
func scalarReclassFloat32(dst, src, breaks, values []float32) {
	dst = dst[:len(src)]
	values = values[:len(breaks)+1]
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		k := len(breaks)
		for j, b := range breaks {
			if v < b {
				k = j
				break
			}
		}
		dst[i] = values[k]
	}
}

// scalarLookupFloat32 interpolates each cell along the polyline through
// the knots (xs[j], ys[j]), clamped to the end values outside them. That
// clamp is what makes the curve bounded, and it is why a cell below the
// first knot or above the last needs no separate operation.
//
// A cell sitting exactly on a knot returns that knot's y directly rather
// than interpolating from it. t would be 0 there, and 0·(y1-y0) is 0 for
// an ordinary y1 — but it is NaN when y1 is NaN or infinite, so without
// the short circuit a single NaN in ys would poison the knot below it,
// and a -0 in ys would come back as +0. The curve passes through its
// knots for every table, which is what the relations in fuzz_test.go and
// package transfer's documentation rely on. The branch is taken by a
// vanishing fraction of cells, so it predicts perfectly.
//
// Elsewhere the segment value is y0 + t·(y1-y0) rather than
// (1-t)·y0 + t·y1: t is never 1, because the upper end of a segment
// belongs to the next segment or to the clamp, so the form that is exact
// at t == 0 is the one whose exactness is reachable. The product is
// wrapped in an explicit float32 conversion, so
// the compiler cannot fuse it with the add into an FMA and make the
// canonical result differ between amd64 and arm64
// (docs/adr/0001-simd-backend.md). The quotient needs no conversion: a
// division has nothing to fuse with.
//
// Like Reclass, NaN is tested before the scan, which would otherwise
// place a NaN above every knot and return the last y.
func scalarLookupFloat32(dst, src, xs, ys []float32) {
	dst = dst[:len(src)]
	ys = ys[:len(xs)]
	for i, v := range src {
		if v != v {
			dst[i] = v
			continue
		}
		k := len(xs)
		for j, x := range xs {
			if v < x {
				k = j
				break
			}
		}
		switch {
		case k == 0:
			dst[i] = ys[0]
		case k == len(xs):
			dst[i] = ys[len(ys)-1]
		default:
			x0, y0 := xs[k-1], ys[k-1]
			if v == x0 {
				dst[i] = y0
				continue
			}
			t := (v - x0) / (xs[k] - x0)
			dst[i] = y0 + float32(t*(ys[k]-y0))
		}
	}
}
