package terrain

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"strata/internal/stencil"
	"strata/raster"
)

// benchDEM is rolling synthetic terrain with noise, optionally with a
// mask that has about 1% of cells invalid.
func benchDEM(n int, masked bool) raster.Float32Raster {
	rng := rand.New(rand.NewPCG(uint64(n), 1))
	d := make([]float32, n*n)
	for y := range n {
		for x := range n {
			fx, fy := float64(x)/97, float64(y)/131
			d[y*n+x] = float32(800 + 300*math.Sin(fx)*math.Cos(fy) + 40*math.Sin(3*fx+fy) + rng.NormFloat64())
		}
	}
	dem := raster.NewFloat32(n, n, d)
	if masked {
		dem.Valid = raster.NewMask(n * n)
		for i := range d {
			if rng.IntN(100) == 0 {
				raster.MaskSet(dem.Valid, i, false)
			}
		}
	}
	return dem
}

// BenchmarkSlope reports whole-raster throughput in cells/s, for the
// scalar and SIMD kernels, with and without a mask. Degrees pay for the
// arctangent; percent does not.
//
//	GOEXPERIMENT=simd go test -run - -bench Slope ./terrain
func BenchmarkSlope(b *testing.B) {
	defer stencil.UseScalar(false)
	for _, n := range []int{1024, 4096} {
		for _, masked := range []bool{false, true} {
			dem := benchDEM(n, masked)
			dst := raster.NewFloat32Like(dem)
			for _, backend := range []string{"scalar", "simd"} {
				for _, u := range []struct {
					name  string
					units SlopeUnits
				}{{"degrees", SlopeDegrees}, {"percent", SlopePercent}} {
					mask := "nomask"
					if masked {
						mask = "mask"
					}
					name := fmt.Sprintf("%d/%s/%s/%s", n, mask, backend, u.name)
					b.Run(name, func(b *testing.B) {
						stencil.UseScalar(backend == "scalar")
						if backend == "simd" && stencil.Backend() == "scalar" {
							b.Skip("no SIMD backend in this build")
						}
						opts := SlopeOptions{CellSize: 10, Units: u.units}
						for b.Loop() {
							Slope(dst, dem, opts)
						}
						b.ReportMetric(float64(n*n)*float64(b.N)/b.Elapsed().Seconds(), "cells/s")
						b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
					})
				}
			}
		}
	}
}
