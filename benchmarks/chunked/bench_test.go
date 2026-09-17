package chunked_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"strata/algebra"
	"strata/benchmarks/internal/suite"
	"strata/engine"
	"strata/internal/stencil"
	"strata/internal/vec"
	"strata/raster"
	"strata/terrain"
)

// Run the suite with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/chunked -run '^$' -bench . -count 5 -timeout 4h > bench.txt
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
	// tilesPlain is the plain function on rasters in memory, on one
	// goroutine: the whole-raster reference, with no IO.
	tilesPlain = "plain"
	// tilesStrips256 is the Chunked function from a raw float32 file to
	// another in full-width tiles of 256 rows.
	tilesStrips256 = "strips256"
	// tiles256 is the same in 256×256 tiles.
	tiles256 = "256x256"
	// tilesMemStrips256 is the Chunked function over a MemorySource and
	// a MemorySink in full-width tiles of 256 rows: the tile buffers
	// without file IO.
	tilesMemStrips256 = "memstrips256"
)

var tileShapes = []string{tilesPlain, tilesStrips256, tiles256, tilesMemStrips256}

// sizes are the category's sizes. 256² is not run: it is a single strip.
var sizes = []int{1024, 4096, 16384}

// fill is the NoData value of the raw files with mask=on.
const fill = -9999

// fixture holds a DEM and an output in memory, with masks attached per
// case, and the same DEM in two raw files: without a fill value, and with
// the fill value under the mask's invalid cells. out is the raw output
// file. The files live in a temporary directory that release removes.
type fixture struct {
	size               int
	dem, dst           []float32
	demValid, dstValid []uint64

	dir                    string
	demFile, fillFile, out *engine.RawFile
}

// newFixture builds the engine suite's DEM (benchmarks/engine), a smooth
// surface of 800 ± 300 with uniform noise of ±1, and writes the files.
// Writing them puts both inputs in the OS file cache and allocates the
// output file at full size, so neither page faults nor file growth are
// timed.
func newFixture(size int) *fixture {
	n := size * size
	f := &fixture{size: size, dem: make([]float32, n), dst: make([]float32, n)}
	suite.FillUniform(f.dem, 1, -1, 1)
	cols := make([]float32, size)
	for x := range cols {
		cols[x] = float32(300 * math.Sin(float64(x)/97))
	}
	for y := range size {
		c := float32(math.Cos(float64(y) / 131))
		row := f.dem[y*size : (y+1)*size]
		for x := range row {
			row[x] += 800 + cols[x]*c
		}
	}
	suite.FillUniform(f.dst, 3, 0, 1)
	f.demValid = suite.RandomMask(n, 4, 0.1)
	f.dstValid = raster.NewMask(n)

	dir, err := os.MkdirTemp("", "strata-chunked-")
	must(err)
	f.dir = dir
	// Every file has one handle per logical CPU, so that workers do not
	// queue on a handle (engine.RawFile).
	create := func(name string) *engine.RawFile {
		file, err := engine.OpenRawFile(filepath.Join(dir, name), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, runtime.NumCPU())
		must(err)
		return file
	}
	f.demFile, f.fillFile, f.out = create("dem.f32"), create("dem-fill.f32"), create("out.f32")
	ctx := context.Background()
	dem, _ := f.rasters(false)
	for _, file := range []*engine.RawFile{f.demFile, f.out} {
		must(engine.NewRawSink(file, size, size, engine.RawOptions{}).WriteWindow(ctx, dem, 0, 0))
	}
	// The fill file, a row at a time: RawSink writes the fill value into
	// the Data of the invalid cells it is given, so give it a copy.
	sink := engine.NewRawSink(f.fillFile, size, size, engine.RawOptions{Fill: fill, HasFill: true})
	row := make([]float32, size)
	for y := range size {
		copy(row, f.dem[y*size:(y+1)*size])
		r := raster.Float32Raster{Data: row, Width: size, Height: 1, Stride: size, Valid: f.demValid, ValidOffset: y * size}
		must(sink.WriteWindow(ctx, r, 0, y))
	}
	return f
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func release(f *fixture) {
	for _, file := range []*engine.RawFile{f.demFile, f.fillFile, f.out} {
		file.Close()
	}
	os.RemoveAll(f.dir)
}

func (f *fixture) rasters(masked bool) (dem, dst raster.Float32Raster) {
	dem = raster.NewFloat32(f.size, f.size, f.dem)
	dst = raster.NewFloat32(f.size, f.size, f.dst)
	if masked {
		dem.Valid, dst.Valid = f.demValid, f.dstValid
	}
	return dem, dst
}

// files returns a source over a DEM file and a sink over the output file,
// with the fill value if masked.
func (f *fixture) files(masked bool) (engine.RasterSource, engine.RasterSink) {
	if masked {
		opts := engine.RawOptions{Fill: fill, HasFill: true}
		return engine.NewRawSource(f.fillFile, f.size, f.size, opts), engine.NewRawSink(f.out, f.size, f.size, opts)
	}
	return engine.NewRawSource(f.demFile, f.size, f.size, engine.RawOptions{}),
		engine.NewRawSink(f.out, f.size, f.size, engine.RawOptions{})
}

// op is one operation with a plain and a chunked form over a DEM.
type op struct {
	plain   func(dst, dem raster.Float32Raster)
	chunked func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error
}

var (
	slopeOpts     = terrain.SlopeOptions{CellSize: 10}
	hillshadeOpts = terrain.HillshadeOptions{CellSize: 10}

	opSlope = op{
		plain: func(dst, dem raster.Float32Raster) { terrain.Slope(dst, dem, slopeOpts) },
		chunked: func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return terrain.SlopeChunked(ctx, dst, dem, slopeOpts, o)
		},
	}
	opHillshade = op{
		plain: func(dst, dem raster.Float32Raster) { terrain.Hillshade(dst, dem, hillshadeOpts) },
		chunked: func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return terrain.HillshadeChunked(ctx, dst, dem, hillshadeOpts, o)
		},
	}
	opClamp = op{
		plain: func(dst, dem raster.Float32Raster) { algebra.Clamp(dst, dem, 700, 900) },
		chunked: func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return algebra.ClampChunked(ctx, dst, dem, 700, 900, o)
		},
	}
)

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	bytes := 8.0 // the DEM and the output
	if c.Masked {
		bytes += 2.0 / 8
	}
	w := suite.Workload{BytesPerCell: bytes}
	if c.Backend == suite.Scalar && c.Workers > 1 {
		return w // §43 compares scalar, SIMD and SIMD with workers
	}
	ctx := context.Background()
	dem, dst := f.rasters(c.Masked)
	chunked := func(sink engine.RasterSink, src engine.RasterSource, opts engine.Options) func() {
		return func() { must(o.chunked(ctx, sink, src, opts)) }
	}
	switch c.Tiles {
	case tilesPlain:
		if c.Workers == 1 {
			w.Run = func() { o.plain(dst, dem) }
		}
	case tilesStrips256:
		src, sink := f.files(c.Masked)
		w.Run = chunked(sink, src, engine.Options{TileHeight: 256, Workers: c.Workers})
	case tiles256:
		src, sink := f.files(c.Masked)
		w.Run = chunked(sink, src, engine.Options{TileWidth: 256, TileHeight: 256, Workers: c.Workers})
	case tilesMemStrips256:
		w.Run = chunked(engine.NewMemorySink(dst), engine.NewMemorySource(dem),
			engine.Options{TileHeight: 256, Workers: c.Workers})
	}
	return w
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  kernels,
		Fixture:  newFixture,
		Release:  release,
		Workload: o.workload,
		Sizes:    sizes,
		Workers:  suite.Workers(),
		Tiles:    tileShapes,
	})
}

func BenchmarkSlope(b *testing.B)     { opSlope.bench(b) }
func BenchmarkHillshade(b *testing.B) { opHillshade.bench(b) }
func BenchmarkClamp(b *testing.B)     { opClamp.bench(b) }

// TestAllocs checks the allocations of every case on a small raster: for
// chunked calls a bound that grows with the worker count and not with the
// tile count, and for the plain functions their own (none for
// algebra.Clamp). It runs without -bench.
func TestAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector allocates in file IO")
	}
	defer kernels.UseScalar(false)
	f := newFixture(600)
	defer release(f)
	ops := map[string]op{"Slope": opSlope, "Hillshade": opHillshade, "Clamp": opClamp}
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
						limit := float64(16 + 12*workers)
						switch {
						case tiles == tilesPlain && name == "Clamp":
							limit = 0
						case tiles == tilesPlain:
							limit = 16
						}
						if allocs := testing.AllocsPerRun(3, w.Run); allocs > limit {
							t.Errorf("%s/%s: %v allocs/op, want at most %v", name, c.Name(), allocs, limit)
						}
					}
				}
			}
		}
	}
}

// TestFilesMatchPlain checks that the raw file shapes write what the
// plain function computes: bit for bit on valid cells, and the fill value
// under invalid ones.
func TestFilesMatchPlain(t *testing.T) {
	f := newFixture(700)
	defer release(f)
	for _, masked := range []bool{false, true} {
		dem, dst := f.rasters(masked)
		opSlope.plain(dst, dem)
		want := append([]float32(nil), dst.Data...)
		for _, tiles := range []string{tilesStrips256, tiles256} {
			c := suite.Case{Size: f.size, Masked: masked, Backend: suite.SIMD, Workers: 3, Tiles: tiles}
			opSlope.workload(f, c).Run()
			got := raster.NewFloat32(f.size, f.size, make([]float32, f.size*f.size))
			must(engine.NewRawSource(f.out, f.size, f.size, engine.RawOptions{}).ReadWindow(context.Background(), got, 0, 0))
			for i, v := range got.Data {
				w := want[i]
				if masked && !raster.MaskGet(dst.Valid, i) {
					w = fill
				}
				if math.Float32bits(v) != math.Float32bits(w) && !(v != v && w != w) {
					t.Fatalf("mask=%v tiles=%s: cell %d = %v, want %v", masked, tiles, i, v, w)
				}
			}
		}
	}
}
