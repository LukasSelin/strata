package engine_test

import (
	"context"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// Run the suite with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/engine -run '^$' -bench . -count 5 -timeout 3h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

// kernels switches both kernel packages the operations use: Clamp's
// internal/vec and the terrain operations' internal/stencil.
var kernels = suite.Kernels{
	Name:    "stencil",
	Backend: stencil.Backend,
	UseScalar: func(scalar bool) {
		vec.UseScalar(scalar)
		stencil.UseScalar(scalar)
	},
}

var vecKernels = suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, vecKernels, kernels) }

// Tile shapes, the last level of every benchmark name.
const (
	// tilesPlain is the plain function (terrain.Slope, algebra.Clamp),
	// which has no tiles and runs on one goroutine: the reference.
	tilesPlain = "plain"
	// tilesStrips is the Tiled function with default tiles: one tile the
	// width of the raster, split into bands of about 1<<16 cells that
	// the workers share.
	tilesStrips = "strips"
	// tiles256 is the Tiled function with 256×256 tiles.
	tiles256 = "256x256"
)

var tileShapes = []string{tilesPlain, tilesStrips, tiles256}

// sizes are the engine suite's sizes: 256² is a single band, so workers
// cannot help it.
var sizes = []int{1024, 4096, 16384}

// fixture holds a DEM and an output, with masks attached per case. Every
// operation reads the DEM and writes dst, so one fixture serves all.
type fixture struct {
	size               int
	dem, dst           []float32
	demValid, dstValid []uint64
}

// newFixture builds the suite's smooth DEM (suite.FillDEM): elevations
// of 800 ± 300 with gentle slopes, like terrain rather than white noise,
// so scalar branches (Hillshade's clamps) behave as on real data. The
// terrain category uses the same one.
func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, dem: make([]float32, n), dst: make([]float32, n)}
	suite.FillDEM(f.dem, size, 1)
	suite.FillUniform(f.dst, 3, 0, 1) // fault dst's pages in before timing
	f.demValid = suite.RandomMask(n, 4, 0.1)
	f.dstValid = raster.NewMask(n)
	return f
}

func (f *fixture) rasters(masked bool) (dst, dem raster.Float32Raster) {
	dst = raster.NewFloat32(f.size, f.size, f.dst)
	dem = raster.NewFloat32(f.size, f.size, f.dem)
	if masked {
		dst.Valid, dem.Valid = f.dstValid, f.demValid
	}
	return dst, dem
}

// op is one operation with a plain and a tiled form over dst and a DEM.
type op struct {
	plain func(dst, dem raster.Float32Raster)
	tiled func(ctx context.Context, dst, dem raster.Float32Raster, o engine.Options) error
}

var (
	slopeOpts     = terrain.SlopeOptions{CellSize: 10}
	hillshadeOpts = terrain.HillshadeOptions{CellSize: 10}
	curvatureOpts = terrain.CurvatureOptions{CellSize: 10}

	opSlope = op{
		plain: func(dst, dem raster.Float32Raster) { terrain.Slope(dst, dem, slopeOpts) },
		tiled: func(ctx context.Context, dst, dem raster.Float32Raster, o engine.Options) error {
			return terrain.SlopeTiled(ctx, dst, dem, slopeOpts, o)
		},
	}
	opHillshade = op{
		plain: func(dst, dem raster.Float32Raster) { terrain.Hillshade(dst, dem, hillshadeOpts) },
		tiled: func(ctx context.Context, dst, dem raster.Float32Raster, o engine.Options) error {
			return terrain.HillshadeTiled(ctx, dst, dem, hillshadeOpts, o)
		},
	}
	opCurvature = op{
		plain: func(dst, dem raster.Float32Raster) { terrain.Curvature(dst, dem, curvatureOpts) },
		tiled: func(ctx context.Context, dst, dem raster.Float32Raster, o engine.Options) error {
			return terrain.CurvatureTiled(ctx, dst, dem, curvatureOpts, o)
		},
	}
	opClamp = op{
		plain: func(dst, dem raster.Float32Raster) { algebra.Clamp(dst, dem, 700, 900) },
		tiled: func(ctx context.Context, dst, dem raster.Float32Raster, o engine.Options) error {
			return algebra.ClampTiled(ctx, dst, dem, 700, 900, o)
		},
	}
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	dst, dem := f.rasters(c.Masked)
	bytes := 8.0 // two float32 operands
	if c.Masked {
		bytes += 2.0 / 8
	}
	w := suite.Workload{BytesPerCell: bytes}
	if c.Backend == suite.Scalar && c.Workers > 1 {
		return w // §43 compares scalar, SIMD and SIMD with workers
	}
	ctx := context.Background()
	switch c.Tiles {
	case tilesPlain:
		if c.Workers == 1 {
			w.Run = func() { o.plain(dst, dem) }
		}
	case tilesStrips:
		opts := engine.Options{Workers: c.Workers}
		w.Run = func() { _ = o.tiled(ctx, dst, dem, opts) }
	case tiles256:
		opts := engine.Options{TileWidth: 256, TileHeight: 256, Workers: c.Workers}
		w.Run = func() { _ = o.tiled(ctx, dst, dem, opts) }
	}
	return w
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  kernels,
		Fixture:  newFixture,
		Workload: o.workload,
		Sizes:    sizes,
		Workers:  suite.Workers(),
		Tiles:    tileShapes,
	})
}

func BenchmarkSlope(b *testing.B)     { opSlope.bench(b) }
func BenchmarkHillshade(b *testing.B) { opHillshade.bench(b) }
func BenchmarkCurvature(b *testing.B) { opCurvature.bench(b) }
func BenchmarkClamp(b *testing.B)     { opClamp.bench(b) }

// TestAllocs checks the allocations of every case on a small raster: none
// for algebra.Clamp, and for everything that runs through the engine
// (terrain's plain functions too) a bound that does not grow with the
// tile count: the per-call setup, a few slices plus about one allocation
// per worker. It runs without -bench.
func TestAllocs(t *testing.T) {
	defer kernels.UseScalar(false)
	f := newFixture(300)
	ops := map[string]op{"Slope": opSlope, "Hillshade": opHillshade, "Curvature": opCurvature, "Clamp": opClamp}
	direct := map[string]bool{"Clamp": true} // plain function without the engine
	for name, o := range ops {
		for _, masked := range []bool{false, true} {
			for _, backend := range suite.Backends {
				for _, workers := range []int{1, 4} {
					for _, tiles := range tileShapes {
						c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: workers, Tiles: tiles}
						kernels.UseScalar(backend == suite.Scalar)
						w := o.workload(f, c)
						if w.Run == nil {
							continue
						}
						// AllocsPerRun sets GOMAXPROCS to 1, which explicit
						// worker counts do not depend on.
						limit := float64(16 + 2*workers)
						if tiles == tilesPlain && direct[name] {
							limit = 0
						}
						if allocs := testing.AllocsPerRun(5, w.Run); allocs > limit {
							t.Errorf("%s/%s: %v allocs/op, want at most %v", name, c.Name(), allocs, limit)
						}
					}
				}
			}
		}
	}
}
