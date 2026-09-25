package zarr

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
	zarrv3 "github.com/LukasSelin/zarr"
)

// TestUnderEngine checks that slope and focal operations over a Zarr
// source write the bits the same operations write over a MemorySource of
// the same cells, for tiles that do and do not line up with the chunks,
// on one worker and several, with and without a cache.
func TestUnderEngine(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const w, h = 150, 97
	dem := make([]float32, w*h)
	for y := range h {
		for x := range w {
			dem[y*w+x] = float32(100 + 30*math.Sin(float64(x)/9) + 20*math.Cos(float64(y)/7) + rng.Float64())
		}
	}
	for range 60 { // holes of NoData
		cx, cy := rng.IntN(w), rng.IntN(h)
		for y := max(cy-2, 0); y < min(cy+2, h); y++ {
			for x := max(cx-3, 0); x < min(cx+3, w); x++ {
				dem[y*w+x] = -9999
			}
		}
	}
	a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
		Shape: []int{h, w}, ChunkShape: []int{32, 40}, FillValue: -9999,
		Codecs: []zarrv3.Codec{zarrv3.BytesCodec{Endian: zarrv3.Little}, zarrv3.GzipCodec{Level: 1}},
	}, dem)
	mem := engine.NewMemorySource(want[float32](t, a, nil, true))

	ops := []struct {
		name string
		run  func(context.Context, engine.RasterSink, engine.RasterSource, engine.Options) error
	}{
		{"slope", func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, e engine.Options) error {
			return terrain.SlopeChunked(ctx, dst, src, terrain.SlopeOptions{CellSize: 10}, e)
		}},
		{"mean r=2", func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, e engine.Options) error {
			return focal.MeanChunked(ctx, dst, src, focal.BoxOptions{Radius: 2}, e)
		}},
	}
	run := func(op func(context.Context, engine.RasterSink, engine.RasterSource, engine.Options) error,
		src engine.RasterSource, e engine.Options) raster.Float32Raster {
		t.Helper()
		out := raster.NewFloat32(w, h, make([]float32, w*h))
		out.Valid = raster.NewMask(w * h)
		if err := op(context.Background(), engine.NewMemorySink(out), src, e); err != nil {
			t.Fatal(err)
		}
		return out
	}
	for _, op := range ops {
		ref := run(op.run, mem, engine.Options{Workers: 1})
		for _, e := range []engine.Options{
			{Workers: 1},
			{Workers: 4, TileHeight: 16},
			{Workers: 3, TileWidth: 40, TileHeight: 32},
			{Workers: 5, TileWidth: 37, TileHeight: 11},
		} {
			for _, cache := range []int64{0, -1, 3 * 32 * 40 * 4} {
				src, err := NewSource(a, SourceOptions{CacheBytes: cache})
				if err != nil {
					t.Fatal(err)
				}
				got := run(op.run, src, e)
				name := fmt.Sprintf("%s, workers %d, tiles %d×%d, cache %d", op.name, e.Workers, e.TileWidth, e.TileHeight, cache)
				for i := range got.Data {
					gv, rv := raster.MaskGet(got.Valid, i), raster.MaskGet(ref.Valid, i)
					if gv != rv || gv && math.Float32bits(got.Data[i]) != math.Float32bits(ref.Data[i]) {
						t.Fatalf("%s: cell (%d, %d) is %v (valid %v), want %v (valid %v)",
							name, i%w, i/w, got.Data[i], gv, ref.Data[i], rv)
					}
				}
			}
		}
	}
}
