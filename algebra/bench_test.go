package algebra

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"strata/internal/vec"
	"strata/raster"
)

// Benchmarks for Add and Clamp at 1024² and 4096², with and without
// validity masks, over three paths:
//
//   - whole:   compact rasters, one kernel call and one mask pass;
//   - rows:    the same compact rasters forced through the per-row path,
//     to measure what the whole-raster fast path buys;
//   - strided: rasters with Stride rounded up to the next multiple of 64
//     plus 64, which always take the per-row path.
//
// On wide rasters whole and rows are within noise of each other: a 1024-
// cell row already amortises the call. BenchmarkAddNarrow shows where the
// whole-raster path pays off.
//
// Masked inputs have about 10% of cells cleared; dst has its own mask.
//
//	go test ./algebra -run '^$' -bench . -count 5
//	GOEXPERIMENT=simd go test ./algebra -run '^$' -bench . -count 5

var benchSizes = []int{1024, 4096}

func benchRaster(rng *rand.Rand, size int, strided, masked bool) raster.Float32Raster {
	stride := size
	if strided {
		stride = (size/64 + 1) * 64
	}
	r := raster.NewFloat32Stride(size, size, stride, make([]float32, (size-1)*stride+size))
	for i := range r.Data {
		r.Data[i] = rng.Float32()*200 - 100
	}
	if masked {
		r.Valid = raster.NewMask(len(r.Data))
		for i := range r.Data {
			if rng.IntN(10) == 0 {
				raster.MaskSet(r.Valid, i, false)
			}
		}
	}
	return r
}

var benchCases = []struct {
	path    string
	strided bool
}{
	{"whole", false},
	{"rows", false},
	{"strided", true},
}

func reportCells(b *testing.B, cells int) {
	b.ReportMetric(float64(cells)*float64(b.N)/1e6/b.Elapsed().Seconds(), "Mcells/s")
}

func BenchmarkAdd(b *testing.B) {
	for _, size := range benchSizes {
		for _, masked := range []bool{false, true} {
			for _, c := range benchCases {
				name := fmt.Sprintf("%d/mask=%v/%s", size, masked, c.path)
				b.Run(name, func(b *testing.B) {
					rng := rand.New(rand.NewPCG(1, 1))
					x := benchRaster(rng, size, c.strided, masked)
					y := benchRaster(rng, size, c.strided, masked)
					dst := benchRaster(rng, size, c.strided, masked)
					b.ResetTimer()
					for b.Loop() {
						if c.path == "rows" {
							binaryApply(dst, x, y, vec.Add, false)
						} else {
							Add(dst, x, y)
						}
					}
					reportCells(b, size*size)
				})
			}
		}
	}
}

func BenchmarkClamp(b *testing.B) {
	for _, size := range benchSizes {
		for _, masked := range []bool{false, true} {
			for _, c := range benchCases {
				name := fmt.Sprintf("%d/mask=%v/%s", size, masked, c.path)
				b.Run(name, func(b *testing.B) {
					rng := rand.New(rand.NewPCG(2, 2))
					src := benchRaster(rng, size, c.strided, masked)
					dst := benchRaster(rng, size, c.strided, masked)
					b.ResetTimer()
					for b.Loop() {
						if c.path == "rows" {
							clamp(dst, src, -50, 50, false)
						} else {
							Clamp(dst, src, -50, 50)
						}
					}
					reportCells(b, size*size)
				})
			}
		}
	}
}

// BenchmarkAddNarrow isolates the per-row call overhead that the
// whole-raster path avoids: 16×65536 cells, compact, no masks.
func BenchmarkAddNarrow(b *testing.B) {
	const w, h = 16, 65536
	mk := func() raster.Float32Raster {
		return raster.NewFloat32(w, h, make([]float32, w*h))
	}
	x, y, dst := mk(), mk(), mk()
	for _, c := range benchCases[:2] {
		b.Run(c.path, func(b *testing.B) {
			for b.Loop() {
				binaryApply(dst, x, y, vec.Add, c.path == "whole")
			}
			reportCells(b, w*h)
		})
	}
}
