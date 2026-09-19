package engine_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// BenchmarkBandWidth sweeps the compute tile's width over one full-width
// IO tile, which is the experiment DESIGN.md §53 turns on. Every case
// reads and writes exactly the same cells through exactly the same
// sources; only the rectangle one kernel call covers changes, and with it
// the halo the call reads:
//
//	ComputeWidth = the raster's width   whole-row bands, the old shape
//	ComputeWidth = 1024                 1024×64 bands, the engine's choice
//	ComputeWidth = 256                  256×256 bands, the squarest
//
// Stats.Halo is reported alongside the time, so the traffic the shape
// saves and the time it costs or buys are read off one line. Run it as:
//
//	GOEXPERIMENT=simd go test ./benchmarks/engine -run '^$' -bench BandWidth -count 6
func BenchmarkBandWidth(b *testing.B) {
	defer kernels.UseScalar(false)
	kernels.UseScalar(false)
	ctx := context.Background()
	for _, size := range []int{4096, 16384} {
		f := newFixture(size)
		for _, masked := range []bool{false, true} {
			dst, dem := f.rasters(masked)
			for _, workers := range []int{1, 12} {
				for _, cw := range []int{0, size, 2048, 1024, 512, 256} {
					name := fmt.Sprintf("size=%d/masked=%v/workers=%d/width=%s",
						size, masked, workers, widthName(cw, size))
					b.Run(name, func(b *testing.B) {
						opts := engine.Options{Workers: workers, ComputeWidth: cw}
						var s engine.Stats
						measure := opts
						measure.Stats = &s
						_ = terrain.SlopeTiled(ctx, dst, dem, slopeOpts, measure)
						b.SetBytes(int64(size) * int64(size) * 8)
						b.ResetTimer()
						for b.Loop() {
							_ = terrain.SlopeTiled(ctx, dst, dem, slopeOpts, opts)
						}
						b.StopTimer()
						b.ReportMetric(haloPercent(s), "halo%")
						b.ReportMetric(s.BytesPerCell(), "B/cell")
					})
				}
			}
			_ = raster.Float32Raster(dst)
		}
	}
}

func widthName(cw, size int) string {
	switch cw {
	case 0:
		return "auto"
	case size:
		return "rows"
	}
	return fmt.Sprint(cw)
}

func haloPercent(s engine.Stats) float64 {
	if s.Cells == 0 {
		return 0
	}
	return 100 * float64(s.Halo(1)) / float64(s.Cells*4)
}
