package exec_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// The benchmarks compare the tiled entry points with the plain functions on
// whole compact rasters at 1024² and 4096², with and without masks (1% of
// cells invalid), all with one worker:
//
//   - direct:    algebra.Clamp, terrain.Slope or terrain.Hillshade;
//   - engine:    the Tiled function with Workers 1 and a background context;
//   - cancel:    the same with a cancellable context;
//   - tiles256:  the same in 256×256 tiles;
//   - strips256: the same in full-width tiles of 256 rows.
//
// Worker scaling is measured by the suite in benchmarks/terrain.
//
// terrain.Slope itself runs through the engine with zero Options, so for
// Slope direct and engine differ only by the kernel construction per call;
// compare against a build of the parent commit for the cost of the engine
// over the old row loop.
//
//	go test ./internal/exec -run '^$' -bench . -count 5
//	GOEXPERIMENT=simd go test ./internal/exec -run '^$' -bench . -count 5

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
		func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return algebra.ClampTiled(ctx, dst, src, 700, 900, o)
		})
}

func BenchmarkSlope(b *testing.B) {
	opts := terrain.SlopeOptions{CellSize: 10}
	benchOp(b, func(dst, src raster.Float32Raster) { terrain.Slope(dst, src, opts) },
		func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return terrain.SlopeTiled(ctx, dst, src, opts, o)
		})
}

func BenchmarkHillshade(b *testing.B) {
	opts := terrain.HillshadeOptions{CellSize: 10}
	benchOp(b, func(dst, src raster.Float32Raster) { terrain.Hillshade(dst, src, opts) },
		func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return terrain.HillshadeTiled(ctx, dst, src, opts, o)
		})
}

func benchOp(b *testing.B, direct func(dst, src raster.Float32Raster), tiled func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error) {
	for _, n := range []int{1024, 4096} {
		for _, masked := range []bool{false, true} {
			src := benchDEM(n, masked)
			dst := raster.NewFloat32Like(src)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			paths := []struct {
				name string
				run  func()
			}{
				{"direct", func() { direct(dst, src) }},
				{"engine", func() { _ = tiled(context.Background(), dst, src, engine.Options{Workers: 1}) }},
				{"cancel", func() { _ = tiled(ctx, dst, src, engine.Options{Workers: 1}) }},
				{"tiles256", func() {
					_ = tiled(context.Background(), dst, src, engine.Options{TileWidth: 256, TileHeight: 256, Workers: 1})
				}},
				{"strips256", func() {
					_ = tiled(context.Background(), dst, src, engine.Options{TileHeight: 256, Workers: 1})
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

// BenchmarkProduct times a six-input weighted factor product — the shape
// §52 and §29 are written for — three ways over the same rasters:
//
//   - chained:  five algebra.MulTiled calls, which is what a caller
//     wrote before §52. Five passes over the whole raster, 60 B/cell.
//   - staged:   one Pipeline, which loads the tile once but carries each
//     intermediate through a span-sized scratch buffer. 28 B/cell to
//     §51's counter, plus the scratch traffic the counter cannot see.
//   - fused:    the same Pipeline lowered to a vec.Chain, which carries
//     the intermediate in a register and writes it nowhere (§29).
//
// The difference between the last two is the whole of §29, and it is a
// time difference: the counter reads 28 for both.
//
//	go test ./internal/exec -run - -bench Product -count 5
//	GOEXPERIMENT=simd go test ./internal/exec -run - -bench Product -count 5
func BenchmarkProduct(b *testing.B) {
	const inputs = 6
	stages := make([]exec.Stage, inputs-1)
	acc := 0
	for i := range stages {
		stages[i] = exec.Stage{Kernel: opKernel{op: vec.OpMul, inputs: 2}, In: []int{acc, i + 1}}
		acc = inputs + i
	}
	out := acc

	b.Logf("backend: %s", vec.Backend())
	for _, n := range []int{1024, 4096} {
		src := make([]raster.Float32Raster, inputs)
		for i := range src {
			src[i] = benchDEM(n, false)
		}
		dst := raster.NewFloat32Like(src[0])
		work := raster.NewFloat32Like(src[0])

		fused := exec.NewPipeline(inputs, stages, out)
		if !fused.Fused() {
			b.Fatal("the pipeline did not lower to a fused chain")
		}
		restore := exec.SetFusion(false)
		staged := exec.NewPipeline(inputs, stages, out)
		restore()

		for _, workers := range []int{1, 4} {
			opts := engine.Options{Workers: workers}
			paths := []struct {
				name string
				run  func()
			}{
				{"chained", func() {
					_ = algebra.MulTiled(context.Background(), work, src[0], src[1], opts)
					for i := 2; i < inputs; i++ {
						_ = algebra.MulTiled(context.Background(), work, work, src[i], opts)
					}
				}},
				{"staged", func() {
					_ = exec.ProcessN(context.Background(), []raster.Float32Raster{dst}, src, staged, opts)
				}},
				{"fused", func() {
					_ = exec.ProcessN(context.Background(), []raster.Float32Raster{dst}, src, fused, opts)
				}},
			}
			for _, p := range paths {
				b.Run(fmt.Sprintf("%d/workers=%d/%s", n, workers, p.name), func(b *testing.B) {
					for b.Loop() {
						p.run()
					}
					b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n*n)/float64(b.N), "ns/cell")
				})
			}
		}
	}
}
