package resample_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// mosaicGrids returns a destination grid and 0 to 4 source grids of
// mixed resolutions, signs and origins scattered over and around it:
// overlapping each other, off its edges, or missing it entirely.
func mosaicGrids(rng *rand.Rand) (dg raster.Grid, sgs []raster.Grid) {
	dg = raster.Grid{Width: 4 + rng.IntN(40), Height: 4 + rng.IntN(40), ResolutionX: 1, ResolutionY: -1}
	dg.OriginX = rng.Float64()*4 - 2
	dg.OriginY = float64(dg.Height) + rng.Float64()*4 - 2
	scales := []float64{0.25, 0.5, 0.73, 1, 1.37, 2, 3.3}
	for range rng.IntN(5) {
		sx, sy := scales[rng.IntN(len(scales))], scales[rng.IntN(len(scales))]
		sg := raster.Grid{ResolutionX: sx, ResolutionY: -sy}
		// A world extent somewhere over dst, up to past its far edges.
		x0 := rng.Float64()*float64(dg.Width)*1.2 - 0.3*float64(dg.Width)
		y0 := rng.Float64()*float64(dg.Height)*1.2 - 0.3*float64(dg.Height)
		ew := (0.2 + rng.Float64()) * float64(dg.Width)
		eh := (0.2 + rng.Float64()) * float64(dg.Height)
		sg.Width = max(1, int(ew/sx))
		sg.Height = max(1, int(eh/sy))
		sg.OriginX = dg.OriginX + x0
		sg.OriginY = dg.OriginY - y0
		if rng.IntN(4) == 0 {
			// South-up.
			sg.OriginY -= float64(sg.Height) * sy
			sg.ResolutionY = sy
		}
		if rng.IntN(5) == 0 {
			// On dst's lattice, as tiles of one product are.
			sg.ResolutionX, sg.ResolutionY = 1, -1
			sg.OriginX = dg.OriginX + float64(rng.IntN(dg.Width)-2)
			sg.OriginY = dg.OriginY - float64(rng.IntN(dg.Height)-2)
		}
		sgs = append(sgs, sg)
	}
	return dg, sgs
}

// wantMosaic is the reference: Resample of each source onto dg, laid
// over one another in order, the last valid cell winning.
func wantMosaic(dg raster.Grid, srcs []raster.Dataset, m resample.Method) raster.Float32Raster {
	want := newDst(dg).Raster
	for i := range want.Data {
		want.Data[i] = float32(math.NaN())
		raster.MaskSet(want.Valid, i, false)
	}
	for _, s := range srcs {
		one := newDst(dg)
		resample.Resample(one, s, resample.Options{Method: m})
		for y := range dg.Height {
			for x := range dg.Width {
				if one.Raster.IsValid(x, y) {
					want.Data[want.Index(x, y)] = one.Raster.Data[one.Raster.Index(x, y)]
					raster.MaskSet(want.Valid, want.Index(x, y), true)
				}
			}
		}
	}
	return want
}

// TestMosaicIsOverlaidResamples is the mosaic's contract: Mosaic,
// MosaicTiled and MosaicChunked, over every tiling and worker count,
// give the bits of the separate Resample calls overlaid in order, Data
// and validity.
func TestMosaicIsOverlaidResamples(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	for iter := range 150 {
		dg, sgs := mosaicGrids(rng)
		srcs := make([]raster.Dataset, len(sgs))
		sources := make([]engine.RasterSource, len(sgs))
		for i, sg := range sgs {
			srcs[i] = raster.NewDataset(sg, randomRaster(rng, sg.Width, sg.Height, (iter+i)%3 == 0))
			sources[i] = engine.NewMemorySource(srcs[i].Raster)
		}
		for _, m := range methods {
			want := wantMosaic(dg, srcs, m)
			got := newDst(dg)
			resample.Mosaic(got, srcs, resample.Options{Method: m})
			sameBits(t, "Mosaic", m, engine.Options{}, want, got.Raster)
			for _, tile := range [][2]int{{1, 1}, {7, 5}, {64, 64}, {0, 3}} {
				for _, workers := range []int{1, 3, 0} {
					eo := engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers}
					got := newDst(dg)
					if err := resample.MosaicTiled(t.Context(), got, srcs, resample.Options{Method: m}, eo); err != nil {
						t.Fatal(err)
					}
					sameBits(t, "MosaicTiled", m, eo, want, got.Raster)
					got = newDst(dg)
					err := resample.MosaicChunked(t.Context(), engine.NewMemorySink(got.Raster), dg, sources, sgs,
						resample.Options{Method: m}, eo)
					if err != nil {
						t.Fatal(err)
					}
					sameBits(t, "MosaicChunked", m, eo, want, got.Raster)
				}
			}
		}
	}
}

// TestMosaicOfOneIsResample: a mosaic of one source is Resample, bit for
// bit, including a dst without a mask where the source covers it.
func TestMosaicOfOneIsResample(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))
	for range 60 {
		dg, sg := grids(rng)
		src := raster.NewDataset(sg, randomRaster(rng, sg.Width, sg.Height, false))
		for _, m := range methods {
			want := newDst(dg)
			resample.Resample(want, src, resample.Options{Method: m})
			got := newDst(dg)
			resample.Mosaic(got, []raster.Dataset{src}, resample.Options{Method: m})
			sameBits(t, "Mosaic", m, engine.Options{}, want.Raster, got.Raster)
		}
	}
}

// TestMosaicUnmaskedDst: sources without masks that together cover dst,
// none alone, may write a dst without a mask, and give the masked dst's
// values.
func TestMosaicUnmaskedDst(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))
	dg := raster.Grid{Width: 20, Height: 12, ResolutionX: 1, ResolutionY: -1, OriginY: 12}
	// Left half at 0.5 m, right half at 2 m, overlapping in the middle.
	left := raster.Grid{Width: 24, Height: 24, ResolutionX: 0.5, ResolutionY: -0.5, OriginY: 12}
	right := raster.Grid{Width: 6, Height: 6, ResolutionX: 2, ResolutionY: -2, OriginX: 8, OriginY: 12}
	srcs := []raster.Dataset{
		raster.NewDataset(left, randomRaster(rng, left.Width, left.Height, false)),
		raster.NewDataset(right, randomRaster(rng, right.Width, right.Height, false)),
	}
	for _, m := range methods {
		want := wantMosaic(dg, srcs, m)
		got := raster.NewDataset(dg, raster.NewFloat32(dg.Width, dg.Height, make([]float32, dg.Width*dg.Height)))
		resample.Mosaic(got, srcs, resample.Options{Method: m})
		for i, v := range got.Raster.Data {
			if math.Float32bits(v) != math.Float32bits(want.Data[i]) {
				t.Fatalf("%v: cell %d = %v, want %v", m, i, v, want.Data[i])
			}
		}
	}
}

// TestMosaicChunkedSkipsHidden: a source entirely under an unmasked
// source after it is never read, and a read error of a source that
// shows comes back wrapped with its index.
func TestMosaicChunkedSkipsHidden(t *testing.T) {
	rng := rand.New(rand.NewPCG(27, 28))
	dg := raster.Grid{Width: 64, Height: 64, ResolutionX: 1, ResolutionY: -1, OriginY: 64}
	under := raster.Grid{Width: 40, Height: 40, ResolutionX: 1, ResolutionY: -1, OriginX: 10, OriginY: 50}
	over := raster.Grid{Width: 50, Height: 50, ResolutionX: 2, ResolutionY: -2, OriginX: -10, OriginY: 80}
	hidden := &recordingSource{MemorySource: engine.NewMemorySource(randomRaster(rng, 40, 40, true))}
	top := &recordingSource{MemorySource: engine.NewMemorySource(randomRaster(rng, 50, 50, false))}
	for _, m := range methods {
		hidden.reads.Store(0)
		err := resample.MosaicChunked(t.Context(), engine.NewMemorySink(newDst(dg).Raster), dg,
			[]engine.RasterSource{hidden, top}, []raster.Grid{under, over}, resample.Options{Method: m},
			engine.Options{TileWidth: 16, TileHeight: 16, Workers: 2})
		if err != nil {
			t.Fatal(err)
		}
		if n := hidden.reads.Load(); n != 0 {
			t.Fatalf("%v: the hidden source was read %d times", m, n)
		}
	}
	top.failAt = top.reads.Load() + 1
	err := resample.MosaicChunked(t.Context(), engine.NewMemorySink(newDst(dg).Raster), dg,
		[]engine.RasterSource{hidden, top}, []raster.Grid{under, over}, resample.Options{},
		engine.Options{TileWidth: 16, TileHeight: 16, Workers: 1})
	if !errors.Is(err, errRead) || !strings.Contains(err.Error(), "src 1") {
		t.Fatalf("error %v, want errRead naming src 1", err)
	}
}

func TestMosaicPanics(t *testing.T) {
	g := raster.Grid{Width: 4, Height: 4, ResolutionX: 1, ResolutionY: -1, OriginY: 4}
	plain := func(g raster.Grid) raster.Dataset {
		return raster.NewDataset(g, raster.NewFloat32(g.Width, g.Height, make([]float32, g.Width*g.Height)))
	}
	src := plain(g)
	masked := newDst(g)
	half := g
	half.Width = 2
	a, b := g, g
	a.CRS.Code, b.CRS.Code = "EPSG:25833", "EPSG:4326"
	for _, c := range []struct {
		name string
		run  func()
		want string
	}{
		{"crs between sources", func() {
			resample.Mosaic(newDst(g), []raster.Dataset{plain(a), plain(g), plain(b)}, resample.Options{})
		}, `src 0 CRS "EPSG:25833" (https://epsg.io/25833) differs from src 2 CRS "EPSG:4326"`},
		{"masked src", func() { resample.Mosaic(plain(g), []raster.Dataset{src, masked}, resample.Options{}) }, "src 1 has a validity mask"},
		{"gap", func() { resample.Mosaic(plain(g), []raster.Dataset{plain(half)}, resample.Options{}) }, "do not cover dst"},
		{"none", func() { resample.Mosaic(plain(g), nil, resample.Options{}) }, "do not cover dst"},
		{"same memory", func() {
			resample.Mosaic(raster.NewDataset(g, src.Raster), []raster.Dataset{plain(g), src}, resample.Options{})
		}, "src 1 share memory"},
		{"grids", func() {
			_ = resample.MosaicChunked(t.Context(), engine.NewMemorySink(newDst(g).Raster), g,
				[]engine.RasterSource{engine.NewMemorySource(src.Raster)}, nil, resample.Options{}, engine.Options{})
		}, "1 sources but 0 source grids"},
		{"method", func() { resample.Mosaic(newDst(g), []raster.Dataset{src}, resample.Options{Method: 9}) }, "unknown"},
	} {
		func() {
			defer func() {
				v := recover()
				s, _ := v.(string)
				if !strings.HasPrefix(s, "resample: ") || !strings.Contains(s, c.want) {
					t.Errorf("%s: panic %v, want a resample: panic mentioning %q", c.name, v, c.want)
				}
			}()
			c.run()
		}()
	}
}
