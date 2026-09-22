package resample_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// BenchmarkResample is a quick per-package check of the passes; the
// suite with §28 classification is benchmarks/resample.
func BenchmarkResample(b *testing.B) {
	const n = 1024
	rng := rand.New(rand.NewPCG(1, 1))
	for _, sc := range []float64{0.5, 2} {
		sg := raster.Grid{Width: n, Height: n, ResolutionX: 1, ResolutionY: -1, OriginY: n}
		src := raster.NewDataset(sg, randomRaster(rng, n, n, false))
		dg := raster.Grid{Width: int(n / sc), Height: int(n / sc), ResolutionX: sc, ResolutionY: -sc, OriginY: n}
		dst := raster.NewDataset(dg, raster.NewFloat32(dg.Width, dg.Height, make([]float32, dg.Width*dg.Height)))
		for _, m := range methods[1:] {
			for _, backend := range []string{"scalar", "simd"} {
				b.Run(fmt.Sprintf("%v/scale=%v/backend=%s", m, sc, backend), func(b *testing.B) {
					resamp.UseScalar(backend == "scalar")
					defer resamp.UseScalar(false)
					if backend == "simd" && resamp.Backend() == "scalar" {
						b.Skip("no SIMD backend")
					}
					for b.Loop() {
						resample.Resample(dst, src, resample.Options{Method: m})
					}
					b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(dg.Width*dg.Height), "ns/cell")
				})
			}
		}
	}
}
