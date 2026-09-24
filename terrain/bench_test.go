package terrain

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
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

// BenchmarkAspect reports whole-raster throughput in cells/s for the
// scalar and SIMD kernels, with and without a mask.
//
//	GOEXPERIMENT=simd go test -run - -bench Aspect ./terrain
func BenchmarkAspect(b *testing.B) {
	benchShade(b, func(dst, dem raster.Float32Raster) { Aspect(dst, dem, AspectOptions{CellSize: 10}) })
}

// BenchmarkHillshade is BenchmarkAspect for Hillshade with the default
// light.
//
//	GOEXPERIMENT=simd go test -run - -bench Hillshade ./terrain
func BenchmarkHillshade(b *testing.B) {
	benchShade(b, func(dst, dem raster.Float32Raster) { Hillshade(dst, dem, HillshadeOptions{CellSize: 10}) })
}

// BenchmarkCurvature is BenchmarkAspect for each kind of Curvature.
//
//	GOEXPERIMENT=simd go test -run - -bench Curvature ./terrain
func BenchmarkCurvature(b *testing.B) {
	for _, ct := range []CurvatureType{CurvatureProfile, CurvaturePlan, CurvatureMean} {
		b.Run(ct.String(), func(b *testing.B) {
			benchShade(b, func(dst, dem raster.Float32Raster) { Curvature(dst, dem, CurvatureOptions{CellSize: 10, Type: ct}) })
		})
	}
}

func benchShade(b *testing.B, op func(dst, dem raster.Float32Raster)) {
	defer stencil.UseScalar(false)
	for _, n := range []int{1024, 4096} {
		for _, masked := range []bool{false, true} {
			dem := benchDEM(n, masked)
			dst := raster.NewFloat32Like(dem)
			for _, backend := range []string{"scalar", "simd"} {
				mask := "nomask"
				if masked {
					mask = "mask"
				}
				b.Run(fmt.Sprintf("%d/%s/%s", n, mask, backend), func(b *testing.B) {
					stencil.UseScalar(backend == "scalar")
					if backend == "simd" && stencil.Backend() == "scalar" {
						b.Skip("no SIMD backend in this build")
					}
					for b.Loop() {
						op(dst, dem)
					}
					b.ReportMetric(float64(n*n)*float64(b.N)/b.Elapsed().Seconds(), "cells/s")
					b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
				})
			}
		}
	}
}

// BenchmarkSurface compares Surface writing slope, aspect and hillshade
// with the three standalone calls it replaces, on one worker and on
// every core, masked. ns/cell is per DEM cell, for all three products.
//
//	GOEXPERIMENT=simd go test -run - -bench Surface ./terrain
func BenchmarkSurface(b *testing.B) {
	for _, n := range []int{1024, 4096} {
		dem := benchDEM(n, true)
		slope, aspect, shade := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
		so := SurfaceOptions{CellSize: 10}
		for _, workers := range []int{1, 0} {
			eo := engine.Options{Workers: workers}
			ctx := context.Background()
			b.Run(fmt.Sprintf("%d/workers=%d/separate", n, workers), func(b *testing.B) {
				for b.Loop() {
					_ = SlopeTiled(ctx, slope, dem, SlopeOptions{CellSize: 10}, eo)
					_ = AspectTiled(ctx, aspect, dem, AspectOptions{CellSize: 10}, eo)
					_ = HillshadeTiled(ctx, shade, dem, HillshadeOptions{CellSize: 10}, eo)
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
			})
			b.Run(fmt.Sprintf("%d/workers=%d/surface", n, workers), func(b *testing.B) {
				out := SurfaceOutputs{Slope: slope, Aspect: aspect, Hillshade: shade}
				for b.Loop() {
					_ = SurfaceTiled(ctx, out, dem, so, eo)
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
			})
		}
	}
}
