package engine_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"strata/algebra"
	"strata/engine"
	"strata/raster"
	"strata/terrain"
)

// The benchmarks compare engine.Process with the public functions on
// whole compact rasters at 1024² and 4096², with and without masks (1% of
// cells invalid):
//
//   - direct:    algebra.Clamp or terrain.Slope;
//   - engine:    engine.Process with zero Options and a background context;
//   - cancel:    the same with a cancellable context;
//   - tiles256:  the same in 256×256 tiles.
//
// terrain.Slope itself runs through the engine with zero Options, so for
// Slope direct and engine differ only by the kernel construction per call;
// compare against a build of the parent commit for the cost of the engine
// over the old row loop.
//
//	go test ./engine -run '^$' -bench . -count 5
//	GOEXPERIMENT=simd go test ./engine -run '^$' -bench . -count 5

func benchDEM(n int, masked bool) raster.Float32Raster {
	rng := rand.New(rand.NewPCG(uint64(n), 1))
	d := make([]float32, n*n)
	for y := range n {
		for x := range n {
			fx, fy := float64(x)/97, float64(y)/131
			d[y*n+x] = float32(800 + 300*math.Sin(fx)*math.Cos(fy) + rng.NormFloat64())
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

func BenchmarkClamp(b *testing.B) {
	benchOp(b, func(dst, src raster.Float32Raster) { algebra.Clamp(dst, src, 700, 900) },
		func() engine.Kernel { return algebra.ClampKernel(700, 900) })
}

func BenchmarkSlope(b *testing.B) {
	opts := terrain.SlopeOptions{CellSize: 10}
	benchOp(b, func(dst, src raster.Float32Raster) { terrain.Slope(dst, src, opts) },
		func() engine.Kernel { return terrain.SlopeKernel(opts) })
}

func benchOp(b *testing.B, direct func(dst, src raster.Float32Raster), kernel func() engine.Kernel) {
	for _, n := range []int{1024, 4096} {
		for _, masked := range []bool{false, true} {
			src := benchDEM(n, masked)
			dst := raster.NewFloat32Like(src)
			k := kernel()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			paths := []struct {
				name string
				run  func()
			}{
				{"direct", func() { direct(dst, src) }},
				{"engine", func() { _ = engine.Process(context.Background(), dst, src, k, engine.Options{}) }},
				{"cancel", func() { _ = engine.Process(ctx, dst, src, k, engine.Options{}) }},
				{"tiles256", func() {
					_ = engine.Process(context.Background(), dst, src, k, engine.Options{TileWidth: 256, TileHeight: 256})
				}},
			}
			for _, p := range paths {
				b.Run(fmt.Sprintf("%d/mask=%v/%s", n, masked, p.name), func(b *testing.B) {
					for b.Loop() {
						p.run()
					}
					b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
				})
			}
		}
	}
}
