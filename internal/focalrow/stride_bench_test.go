package focalrow

import (
	"fmt"
	"testing"
)

// BenchmarkColumnStride times the column passes of Mean (ColumnSum) and
// CorrelateSeparable (ColumnCorrelate) over 64 output rows 16320 cells
// wide, varying only the row stride and the number of rows. It is the
// experiment behind benchmarks/focal/RESULTS.md's "16384² and the row
// stride": 16384, 32768 and 49152 cells put every row in one set of Zen
// 2's L2, 16400 and 16416 put them one and two lines apart modulo 64 KiB,
// 24576 alternates two sets and 20480 four, and 16320 and 16448 spread
// them. Run it pinned, as the focal suite is:
//
//	GOEXPERIMENT=simd go test ./internal/focalrow -run '^$' -bench ColumnStride -count 5
func BenchmarkColumnStride(b *testing.B) {
	const n, h = 16320, 64
	for _, k := range []int{7, 9, 11, 17} {
		for _, stride := range []int{16320, 16384, 16400, 16416, 16448, 20480, 24576, 32768, 49152} {
			src := make([]float32, (h+k)*stride)
			for i := range src {
				src[i] = float32(i % 97)
			}
			dst := make([]float32, n)
			taps := make([]float32, k)
			for i := range taps {
				taps[i] = 1 / float32(k)
			}
			for _, name := range []string{"sum", "correlate"} {
				b.Run(fmt.Sprintf("k=%d/stride=%d/%s", k, stride, name), func(b *testing.B) {
					for b.Loop() {
						for y := range h {
							if name == "sum" {
								ColumnSum(dst, src[y*stride:], stride, k)
							} else {
								ColumnCorrelate(dst, src[y*stride:], stride, taps)
							}
						}
					}
					b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*n*h), "ns/cell")
				})
			}
		}
	}
}
