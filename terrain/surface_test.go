package terrain

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// TestSurfaceIsTheStandaloneProducts checks Surface's promise: each
// product it writes is the bits the standalone function writes, Data
// and validity, for every subset of products, both backends, every
// entry point, tiling and worker count, masked or not, windowed or not.
func TestSurfaceIsTheStandaloneProducts(t *testing.T) {
	const w, h = 67, 29
	opts := SurfaceOptions{CellSize: 12.5, CellSizeY: 9, ZFactor: 1.5, Units: SlopePercent,
		ZeroForFlat: true, Azimuth: 200, Altitude: 30}
	runs := []engine.Options{
		{Workers: 1},
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 5, TileHeight: 3, Workers: 2},
	}
	backends := []bool{true, false}
	defer stencil.UseScalar(false)
	for _, scalar := range backends {
		stencil.UseScalar(scalar)
		for _, masked := range []bool{false, true} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(53, 7))
				dem := weightedOperand(rng, w, h, 800, windowed, masked)
				// Flat patches, so aspect's flat value and zero gradients occur.
				for y := 3; y < 9; y++ {
					for x := 10; x < 20; x++ {
						dem.Data[dem.Index(x, y)] = 812
					}
				}
				newOut := func() raster.Float32Raster {
					out := raster.NewFloat32(w, h, make([]float32, w*h))
					if masked {
						out.Valid = make([]uint64, raster.MaskWords(w*h))
					}
					return out
				}
				var want [surfaceProducts]raster.Float32Raster
				for k := range want {
					want[k] = newOut()
				}
				Gradient(want[surfaceDx], want[surfaceDy], dem, GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor})
				Slope(want[surfaceSlope], dem, SlopeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor, Units: opts.Units})
				Aspect(want[surfaceAspect], dem, AspectOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
					ZeroForFlat: opts.ZeroForFlat, Trigonometric: opts.Trigonometric})
				Hillshade(want[surfaceHillshade], dem, HillshadeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
					Azimuth: opts.Azimuth, Altitude: opts.Altitude})

				for set := 1; set < 1<<surfaceProducts; set++ {
					var got [surfaceProducts]raster.Float32Raster
					outputs := func() SurfaceOutputs {
						for k := range got {
							got[k] = raster.Float32Raster{}
							if set&(1<<k) != 0 {
								got[k] = newOut()
							}
						}
						return SurfaceOutputs{Dx: got[0], Dy: got[1], Slope: got[2], Aspect: got[3], Hillshade: got[4]}
					}
					check := func(how string) {
						t.Helper()
						for k := range got {
							if set&(1<<k) != 0 {
								sameCells(t, fmt.Sprintf("scalar=%v masked=%v windowed=%v set=%05b %s product %d",
									scalar, masked, windowed, set, how, k), got[k], want[k])
							}
						}
					}
					Surface(outputs(), dem, opts)
					check("plain")
					if set != 1<<surfaceProducts-1 && set != 1<<surfaceSlope && set != 0b01100 {
						continue // the whole matrix for a few sets only
					}
					for _, eo := range runs {
						if err := SurfaceTiled(context.Background(), outputs(), dem, opts, eo); err != nil {
							t.Fatal(err)
						}
						check(fmt.Sprintf("tiled %+v", eo))
						o := outputs()
						sinks := SurfaceSinks{}
						for k, r := range []raster.Float32Raster{o.Dx, o.Dy, o.Slope, o.Aspect, o.Hillshade} {
							if r.Data == nil {
								continue
							}
							s := engine.NewMemorySink(r)
							switch k {
							case 0:
								sinks.Dx = s
							case 1:
								sinks.Dy = s
							case 2:
								sinks.Slope = s
							case 3:
								sinks.Aspect = s
							case 4:
								sinks.Hillshade = s
							}
						}
						if err := SurfaceChunked(context.Background(), sinks, engine.NewMemorySource(dem), opts, eo); err != nil {
							t.Fatal(err)
						}
						check(fmt.Sprintf("chunked %+v", eo))
					}
				}
			}
		}
	}
}

func TestSurfacePanics(t *testing.T) {
	r := raster.NewFloat32(6, 6, make([]float32, 36))
	o := func() raster.Float32Raster { return raster.NewFloat32Like(r) }
	mustPanic(t, "no outputs", func() { Surface(SurfaceOutputs{}, r, SurfaceOptions{CellSize: 1}) })
	mustPanic(t, "output is the DEM", func() { Surface(SurfaceOutputs{Slope: r}, r, SurfaceOptions{CellSize: 1}) })
	shared := o()
	mustPanic(t, "outputs share memory", func() {
		Surface(SurfaceOutputs{Slope: shared, Aspect: shared}, r, SurfaceOptions{CellSize: 1})
	})
	mustPanic(t, "size", func() {
		Surface(SurfaceOutputs{Slope: raster.NewFloat32(5, 6, make([]float32, 30))}, r, SurfaceOptions{CellSize: 1})
	})
	mustPanic(t, "cell size", func() { Surface(SurfaceOutputs{Slope: o()}, r, SurfaceOptions{}) })
	mustPanic(t, "altitude", func() { Surface(SurfaceOutputs{Hillshade: o()}, r, SurfaceOptions{CellSize: 1, Altitude: 91}) })
	mustPanic(t, "no sinks", func() {
		_ = SurfaceChunked(context.Background(), SurfaceSinks{}, engine.NewMemorySource(r), SurfaceOptions{CellSize: 1}, engine.Options{})
	})
}
