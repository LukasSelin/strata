package graph

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/terrain"
)

// BenchmarkTerrainStack times four products and a summary of one DEM,
// chunked from memory: as a graph, one pass that reads the DEM once, and
// as the separate calls, which read it four times and fold the slope
// again. It measures the planner's own claim, a read saved per product;
// the workflow's external evidence is gdalsuite's (DESIGN.md §55).
func BenchmarkTerrainStack(b *testing.B) {
	const n = 2048
	rng := rand.New(rand.NewPCG(1, 2))
	dem := raster.NewFloat32(n, n, make([]float32, n*n))
	for i := range dem.Data {
		dem.Data[i] = 500 + 100*rng.Float32()
	}
	outs := map[string]raster.Float32Raster{}
	for _, name := range []string{"slope", "aspect", "hillshade", "tri"} {
		outs[name] = raster.NewFloat32(n, n, make([]float32, n*n))
	}
	sl := terrain.SlopeOptions{CellSize: 10}
	as := terrain.AspectOptions{CellSize: 10}
	hs := terrain.HillshadeOptions{CellSize: 10}
	ru := terrain.RuggednessOptions{}
	eo := engine.Options{TileHeight: 256}
	ctx := context.Background()
	src := engine.NewMemorySource(dem)
	sink := func(name string) engine.RasterSink { return engine.NewMemorySink(outs[name]) }

	b.Run("graph", func(b *testing.B) {
		g := New()
		d := g.Input("dem")
		s := Slope(d, sl)
		g.Output("slope", s)
		g.Output("aspect", Aspect(d, as))
		g.Output("hillshade", Hillshade(d, hs))
		g.Output("tri", Ruggedness(d, ru))
		g.Stats("slope", s)
		p := g.Plan(PlanOptions{})
		sinks := map[string]engine.RasterSink{}
		for name := range outs {
			sinks[name] = sink(name)
		}
		b.SetBytes(4 * n * n)
		for b.Loop() {
			if _, err := p.RunChunked(ctx, map[string]engine.RasterSource{"dem": src}, sinks, ChunkedOptions{Engine: eo}); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("separate", func(b *testing.B) {
		b.SetBytes(4 * n * n)
		for b.Loop() {
			must(b, terrain.SlopeChunked(ctx, sink("slope"), src, sl, eo))
			must(b, terrain.AspectChunked(ctx, sink("aspect"), src, as, eo))
			must(b, terrain.HillshadeChunked(ctx, sink("hillshade"), src, hs, eo))
			must(b, terrain.RuggednessChunked(ctx, sink("tri"), src, ru, eo))
			_, err := reduce.StatsChunked(ctx, engine.NewMemorySource(outs["slope"]), eo)
			must(b, err)
		}
	})
}

func must(b *testing.B, err error) {
	b.Helper()
	if err != nil {
		b.Fatal(err)
	}
}
