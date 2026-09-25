// Package zarrbench times the zarr adapter's decoded-chunk cache: slope
// and a focal mean over a Zarr source with the cache on and off, at chunk
// sides of 256 and 512, on one worker and on all of them, and plain window
// reads of the sizes github.com/LukasSelin/zarr's own benchmarks time.
// Numbers are in RESULTS.md.
//
// The fixture is a 4096 × 4096 float32 DEM, smooth with noise and about 1%
// NoData, in a directory store, gzip level 5, written once per chunk side
// into a temporary directory. Every case reports decodes/chunk: the chunk
// reads the store served over the chunks in the raster, 1.00 being every
// chunk decoded once.
//
//	go test -run '^$' -bench . -count 5 -timeout 2h > testdata/bench.txt
package zarrbench

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/zarr"
	zarrv3 "github.com/LukasSelin/zarr"
)

// side is the raster's width and height.
const side = 4096

var (
	tmp      string
	fixtures = map[int]*fixture{}
)

func TestMain(m *testing.M) {
	var err error
	if tmp, err = os.MkdirTemp("", "zarrbench"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("gomaxprocs: %d\n", runtime.GOMAXPROCS(0))
	code := m.Run()
	_ = os.RemoveAll(tmp)
	os.Exit(code)
}

// counting counts the chunk reads a store serves.
type counting struct {
	zarrv3.Store
	reads atomic.Int64
}

func (c *counting) Get(ctx context.Context, key string) ([]byte, error) {
	if !strings.HasSuffix(key, "zarr.json") {
		c.reads.Add(1)
	}
	return c.Store.Get(ctx, key)
}

type fixture struct {
	store  *counting
	a      *zarrv3.Array
	chunks int
}

// fixtureFor returns the DEM stored in chunks of chunk × chunk cells,
// writing it the first time.
func fixtureFor(b *testing.B, chunk int) *fixture {
	if f, ok := fixtures[chunk]; ok {
		return f
	}
	ctx := context.Background()
	store := &counting{Store: zarrv3.NewDirStore(fmt.Sprintf("%s/dem%d.zarr", tmp, chunk))}
	a, err := zarrv3.CreateArray(ctx, store, "dem", zarrv3.ArrayOptions{
		Shape: []int{side, side}, ChunkShape: []int{chunk, chunk}, DataType: zarrv3.Float32,
		FillValue: -9999,
		Codecs:    []zarrv3.Codec{zarrv3.BytesCodec{Endian: zarrv3.Little}, zarrv3.GzipCodec{Level: 5}},
	})
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, uint64(chunk)))
	dem := make([]float32, side*side)
	for y := range side {
		for x := range side {
			dem[y*side+x] = float32(400 + 150*math.Sin(float64(x)/310)*math.Cos(float64(y)/270) +
				20*math.Sin(float64(x+y)/37) + rng.Float64())
		}
	}
	for range 1600 { // about 1% NoData, in holes
		cx, cy := rng.IntN(side), rng.IntN(side)
		for y := max(cy-5, 0); y < min(cy+5, side); y++ {
			for x := max(cx-5, 0); x < min(cx+5, side); x++ {
				dem[y*side+x] = -9999
			}
		}
	}
	if err := zarrv3.Write(ctx, a, []int{0, 0}, []int{side, side}, dem); err != nil {
		b.Fatal(err)
	}
	n := (side + chunk - 1) / chunk
	f := &fixture{store: store, a: a, chunks: n * n}
	fixtures[chunk] = f
	return f
}

// source returns a source over f with the cache on (the default) or off,
// loading up to conc chunks of a window at once.
func (f *fixture) source(b *testing.B, cache bool, conc int) *zarr.Source {
	opts := zarr.SourceOptions{ReadConcurrency: conc}
	if !cache {
		opts.CacheBytes = -1
	}
	src, err := zarr.NewSource(f.a, opts)
	if err != nil {
		b.Fatal(err)
	}
	return src
}

// concurrencies is the ReadConcurrency values run: one chunk at a time,
// and the default.
var concurrencies = []int{1, zarr.DefaultReadConcurrency}

func workerCounts() []int {
	if n := runtime.GOMAXPROCS(0); n > 1 {
		return []int{1, n}
	}
	return []int{1}
}

type operation func(context.Context, engine.RasterSink, engine.RasterSource, engine.Options) error

func benchOp(b *testing.B, op operation) {
	out := raster.NewFloat32(side, side, make([]float32, side*side))
	out.Valid = raster.NewMask(side * side)
	sink := engine.NewMemorySink(out)
	// The floor: the same operation over the same cells in memory.
	for _, workers := range workerCounts() {
		b.Run(fmt.Sprintf("src=memory/workers=%d", workers), func(b *testing.B) {
			f := fixtureFor(b, 256)
			dem := raster.NewFloat32(side, side, make([]float32, side*side))
			dem.Valid = raster.NewMask(side * side)
			if err := f.source(b, false, 0).ReadWindow(context.Background(), dem, 0, 0); err != nil {
				b.Fatal(err)
			}
			src := engine.NewMemorySource(dem)
			eopts := engine.Options{Workers: workers, TileHeight: 256}
			for b.Loop() {
				if err := op(context.Background(), sink, src, eopts); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(side*side)/1e6*float64(b.N)/b.Elapsed().Seconds(), "Mcells/s")
		})
	}
	for _, chunk := range []int{256, 512} {
		for _, cache := range []bool{true, false} {
			for _, conc := range concurrencies {
				for _, workers := range workerCounts() {
					name := fmt.Sprintf("src=zarr/chunk=%d/cache=%s/conc=%d/workers=%d", chunk, onOff(cache), conc, workers)
					b.Run(name, func(b *testing.B) {
						f := fixtureFor(b, chunk)
						f.store.reads.Store(0)
						eopts := engine.Options{Workers: workers, TileHeight: 256}
						for b.Loop() {
							// A new source each run, so every run starts cold.
							if err := op(context.Background(), sink, f.source(b, cache, conc), eopts); err != nil {
								b.Fatal(err)
							}
						}
						b.ReportMetric(float64(f.store.reads.Load())/float64(b.N*f.chunks), "decodes/chunk")
						b.ReportMetric(float64(side*side)/1e6*float64(b.N)/b.Elapsed().Seconds(), "Mcells/s")
					})
				}
			}
		}
	}
}

func onOff(on bool) string {
	if on {
		return "on"
	}
	return "off"
}

// BenchmarkSlope is terrain.SlopeChunked, 3×3, on 256-row strips.
func BenchmarkSlope(b *testing.B) {
	benchOp(b, func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, e engine.Options) error {
		return terrain.SlopeChunked(ctx, dst, src, terrain.SlopeOptions{CellSize: 10}, e)
	})
}

// BenchmarkMean is focal.MeanChunked, radius 5 (11×11), on 256-row strips.
func BenchmarkMean(b *testing.B) {
	benchOp(b, func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, e engine.Options) error {
		return focal.MeanChunked(ctx, dst, src, focal.BoxOptions{Radius: 5}, e)
	})
}

// BenchmarkWindow reads one window again and again: with the cache off,
// every read decodes the chunks under it, as the library's own Read does;
// with it on, the first read decodes them and the rest copy from the
// cache.
func BenchmarkWindow(b *testing.B) {
	for _, chunk := range []int{256, 512} {
		windows := []struct {
			name       string
			x, y, w, h int
		}{
			{"pixel", 1000, 1000, 1, 1},
			{"3x3-corner", chunk - 1, chunk - 1, 3, 3},
			{"256x256-4chunks", chunk - 128, chunk - 128, 256, 256},
		}
		for _, win := range windows {
			for _, cache := range []bool{false, true} {
				b.Run(fmt.Sprintf("chunk=%d/window=%s/cache=%s", chunk, win.name, onOff(cache)), func(b *testing.B) {
					f := fixtureFor(b, chunk)
					src := f.source(b, cache, 0)
					dst := raster.NewFloat32(win.w, win.h, make([]float32, win.w*win.h))
					dst.Valid = raster.NewMask(win.w * win.h)
					ctx := context.Background()
					if cache {
						_ = src.ReadWindow(ctx, dst, win.x, win.y)
					}
					for b.Loop() {
						if err := src.ReadWindow(ctx, dst, win.x, win.y); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkReadAll reads the whole raster in 256-row strips, each worker
// its own strips, with a halo of one row above and below as a 3×3
// operation's tiles have: nothing is computed.
func BenchmarkReadAll(b *testing.B) {
	for _, chunk := range []int{256, 512} {
		for _, cache := range []bool{true, false} {
			for _, conc := range concurrencies {
				for _, workers := range workerCounts() {
					b.Run(fmt.Sprintf("chunk=%d/cache=%s/conc=%d/workers=%d", chunk, onOff(cache), conc, workers), func(b *testing.B) {
						f := fixtureFor(b, chunk)
						f.store.reads.Store(0)
						for b.Loop() {
							readAll(b, f.source(b, cache, conc), workers)
						}
						b.ReportMetric(float64(f.store.reads.Load())/float64(b.N*f.chunks), "decodes/chunk")
						b.ReportMetric(float64(side*side)/1e6*float64(b.N)/b.Elapsed().Seconds(), "Mcells/s")
					})
				}
			}
		}
	}
}

func readAll(b *testing.B, src *zarr.Source, workers int) {
	const rows = 256
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			buf := raster.NewFloat32(side, rows+2, make([]float32, side*(rows+2)))
			buf.Valid = raster.NewMask(side * (rows + 2))
			for {
				y := int(next.Add(rows)) - rows
				if y >= side {
					return
				}
				y0, y1 := max(y-1, 0), min(y+rows+1, side)
				if err := src.ReadWindow(context.Background(), buf.Window(0, 0, side, y1-y0), 0, y0); err != nil {
					b.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
}
