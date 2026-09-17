// Package nodata is the STRATA-3 spike: it benchmarks three NoData
// representations for float32 rasters before the raster package fixes
// one (STRATA-4). It is benchmark/spike code, not a public API.
//
// Representations:
//
//   - Sentinel: a reserved value (Sentinel = -9999) marks NoData in Data.
//     Every kernel must compare inputs against it and write it to outputs.
//   - NaN: NoData is any NaN in Data. Arithmetic propagates it for free.
//   - Mask: a separate []uint64 validity bitmap, bit i (LSB-first within
//     word i/64) set means cell i is valid. Data under a cleared bit is
//     unspecified. Output validity is the AND of the input masks (eroded
//     by the 3×3 footprint for neighbourhood kernels).
//
// Workloads are element-wise Add and a 3×3 Horn gradient magnitude
// ("slope"). Each workload has straightforward scalar variants and
// vectorization-friendly variants (internal/vec kernels, or the
// simd/archsimd kernels in simd_amd64.go, plus a mask pass where the
// representation needs one). The vectorized variants need
// GOEXPERIMENT=simd, as internal/vec does (docs/adr/0001-simd-backend.md).
// All variants of one workload must agree on validity and, for valid
// cells, bit-for-bit on values; nodata_test.go checks that.
//
// Rasters here are row-major with stride == width. Slope treats the
// one-cell border as NoData, as a tile without a halo would.
package nodata

import "math"

// Sentinel is the NoData value used by the sentinel representation.
// It is exactly representable in float32 and is the most common GDAL
// fill value for float DEMs.
const Sentinel float32 = -9999

// NaN32 is a canonical quiet float32 NaN.
var NaN32 = float32(math.NaN())

func isNaN32(v float32) bool { return v != v }
