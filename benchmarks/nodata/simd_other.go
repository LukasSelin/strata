//go:build !(goexperiment.simd && amd64)

package nodata

// HaveAVX2 is false without GOEXPERIMENT=simd on amd64. The AVX2 variants
// fall back to the equivalent scalar forms so the package still builds
// and tests pass; the benchmarks skip them.
var HaveAVX2 = false

func AddSentinelAVX2(dst, a, b []float32, nd float32) { AddSentinelSelect(dst, a, b, nd) }

func SlopeNaNAVX2(dst, src []float32, w, h int, cellSize float32) {
	SlopeNaNScalar(dst, src, w, h, cellSize)
}

func SlopeSentinelAVX2(dst, src []float32, w, h int, cellSize, nd float32) {
	SlopeSentinelSelect(dst, src, w, h, cellSize, nd)
}

func SlopeMaskAVX2(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64) {
	SlopeMaskScalar(dst, src, dstValid, valid, w, h, cellSize, scratch)
}

func SlopeMaskAVX2Fill(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64, fill float32) {
	SlopeMaskAVX2(dst, src, dstValid, valid, w, h, cellSize, scratch)
	FillInvalid(dst, dstValid, fill)
}
