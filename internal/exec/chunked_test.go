package exec_test

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// chunkedFunc runs an operation over sources and sinks.
type chunkedFunc func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error

// chunked holds the Chunked entry point of each adapter, by name.
var chunked = map[string]chunkedFunc{
	"clamp": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.ClampChunked(ctx, dst[0], src[0], 950, 1050, o)
	},
	"add": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.AddChunked(ctx, dst[0], src[0], src[1], o)
	},
	"sub": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.SubChunked(ctx, dst[0], src[0], src[1], o)
	},
	"mul": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.MulChunked(ctx, dst[0], src[0], src[1], o)
	},
	"min": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.MinChunked(ctx, dst[0], src[0], src[1], o)
	},
	"max": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return algebra.MaxChunked(ctx, dst[0], src[0], src[1], o)
	},
	"slope-degrees": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.SlopeChunked(ctx, dst[0], src[0], terrain.SlopeOptions{CellSize: 10, CellSizeY: 12}, o)
	},
	"slope-percent": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.SlopeChunked(ctx, dst[0], src[0], terrain.SlopeOptions{CellSize: 3, ZFactor: 2, Units: terrain.SlopePercent}, o)
	},
	"hillshade": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.HillshadeChunked(ctx, dst[0], src[0], terrain.HillshadeOptions{CellSize: 30, Azimuth: 100, Altitude: 20}, o)
	},
	"aspect": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.AspectChunked(ctx, dst[0], src[0], terrain.AspectOptions{CellSize: 5, Trigonometric: true}, o)
	},
	"curvature-plan": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.CurvatureChunked(ctx, dst[0], src[0], terrain.CurvatureOptions{CellSize: 5, CellSizeY: 4, Type: terrain.CurvaturePlan}, o)
	},
	"gradient": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return terrain.GradientChunked(ctx, dst[0], dst[1], src[0], terrain.GradientOptions{CellSize: 7}, o)
	},
	"box-r2": func(ctx context.Context, dst []engine.RasterSink, src []engine.RasterSource, o engine.Options) error {
		return exec.ProcessChunked(ctx, dst, src, boxKernel{r: 2, inputs: 1, outputs: 1}, o)
	},
}

// memorySinks and memorySources wrap operands' rasters.
func memorySinks(ops []operand) []engine.RasterSink {
	out := make([]engine.RasterSink, len(ops))
	for i, op := range ops {
		out[i] = engine.NewMemorySink(op.r)
	}
	return out
}

func memorySources(ops []operand) []engine.RasterSource {
	out := make([]engine.RasterSource, len(ops))
	for i, op := range ops {
		out[i] = engine.NewMemorySource(op.r)
	}
	return out
}

// TestChunkedMatchesPlain checks every Chunked entry point against its
// plain function on small rasters of many shapes, with the broad
// engineRuns and memory sources and sinks over compact rasters and
// windows.
func TestChunkedMatchesPlain(t *testing.T) {
	for _, a := range adapters {
		f := chunked[a.name]
		if f == nil {
			t.Fatalf("no Chunked function for %s", a.name)
		}
		t.Run(a.name, func(t *testing.T) {
			for _, sz := range adapterSizes {
				for _, lay := range layouts {
					for _, masks := range []struct{ in, out bool }{{false, false}, {false, true}, {true, true}} {
						rng := rand.New(rand.NewPCG(uint64(sz[0]*1000+sz[1]), 3))
						ins, outs := newOperands(rng, a, sz[0], sz[1], lay, masks.in, masks.out, -1)
						want := cloneAll(ins, outs, false)
						a.direct(want.dst(), want.src())
						for _, run := range engineRuns {
							id := fmt.Sprintf("%dx%d %v inMask=%v outMask=%v %v", sz[0], sz[1], lay, masks.in, masks.out, run)
							got := cloneAll(ins, outs, false)
							processWith(t, run, func(ctx context.Context, o engine.Options) error {
								return f(ctx, memorySinks(got.outs), memorySources(got.ins), o)
							})
							requireSameOperands(t, id, got, want)
						}
					}
				}
			}
		})
	}
}

// newOperands builds an adapter's inputs and outputs; with masks, the
// first input always has one and a second input half the time. pad is
// newOperandPad's.
func newOperands(rng *rand.Rand, a adapter, w, h int, lay layout, inMask, outMask bool, pad int) (ins, outs []operand) {
	for i := range a.inputs {
		ins = append(ins, newOperandPad(rng, w, h, lay.in, inMask && (i == 0 || rng.IntN(2) == 0), pad))
	}
	for range a.outputs {
		outs = append(outs, newOperandPad(rng, w, h, lay.out, outMask, pad))
	}
	return ins, outs
}

func requireSameOperands(t *testing.T, id string, got, want operands) {
	t.Helper()
	for i := range want.outs {
		requireSameRoots(t, fmt.Sprintf("%s dst[%d]", id, i), got.outs[i].root, want.outs[i].root)
	}
	for i := range want.ins {
		requireSameRoots(t, fmt.Sprintf("%s src[%d]", id, i), got.ins[i].root, want.ins[i].root)
	}
}

// TestChunkedTilesAndWorkers is the §23 contract for bounded-memory
// execution, TestTilesAndWorkers' matrix through memory sources and
// sinks: tile widths and heights in {1, 7, 64, 256, full, larger than the
// raster} × workers in {1, 2, 3, GOMAXPROCS} × no masks, masks on inputs
// and outputs, masks on outputs only, on an odd compact raster and two
// windows whose Stride is not a multiple of 64. The sinks must hold the
// plain function's bits, and nothing outside them may change.
func TestChunkedTilesAndWorkers(t *testing.T) {
	type shape struct {
		name     string
		w, h     int
		windowed bool
		pad      int
	}
	shapes := []shape{
		{"odd-compact", 61, 37, false, 0},
		{"wide-window", 300, 9, true, 5},
		{"tall-window", 21, 131, true, 37},
	}
	dims := []int{1, 7, 64, 256, 0, 1 << 20}
	workers := []int{1, 2, 3, runtime.GOMAXPROCS(0)}
	if testing.Short() {
		dims = []int{1, 7, 0}
	}
	names := []string{"box-r2", "clamp", "add", "slope-degrees", "aspect", "hillshade", "curvature-plan", "gradient"}
	masks := []struct{ in, out bool }{{false, false}, {true, true}, {false, true}}

	for _, name := range names {
		op := box2Adapter
		if name != op.name {
			op = adapterNamed(t, name)
		}
		f := chunked[name]
		t.Run(name, func(t *testing.T) {
			for _, sh := range shapes {
				for _, m := range masks {
					rng := rand.New(rand.NewPCG(uint64(sh.w*1000+sh.h), 11))
					ins, outs := newOperands(rng, op, sh.w, sh.h, layout{sh.windowed, sh.windowed}, m.in, m.out, sh.pad)
					want := cloneAll(ins, outs, false)
					op.direct(want.dst(), want.src())
					n := 0
					for _, tw := range dims {
						for _, th := range dims {
							for _, wk := range workers {
								run := engineRun{
									opts:      engine.Options{TileWidth: tw, TileHeight: th, Workers: wk},
									bandCells: []int{1, 97, 0}[n%3],
								}
								n++
								id := fmt.Sprintf("%s %dx%d inMask=%v outMask=%v %v", sh.name, sh.w, sh.h, m.in, m.out, run)
								got := cloneAll(ins, outs, false)
								processWith(t, run, func(ctx context.Context, o engine.Options) error {
									return f(ctx, memorySinks(got.outs), memorySources(got.ins), o)
								})
								requireSameOperands(t, id, got, want)
							}
						}
					}
				}
			}
		})
	}
}

// recordingSource wraps a source and records the buffers it is handed.
type recordingSource struct {
	engine.RasterSource
	mu      sync.Mutex
	buffers map[*float32]raster.Float32Raster // by first cell
	regions [][4]int                          // x, y, w, h of each read
	strides []int
}

func (s *recordingSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	s.mu.Lock()
	if s.buffers == nil {
		s.buffers = map[*float32]raster.Float32Raster{}
	}
	s.buffers[&dst.Data[0]] = dst
	s.regions = append(s.regions, [4]int{x, y, dst.Width, dst.Height})
	s.strides = append(s.strides, dst.Stride)
	s.mu.Unlock()
	return s.RasterSource.ReadWindow(ctx, dst, x, y)
}

// TestChunkedBuffers checks the memory model of DESIGN.md §23 and §27:
// each read covers its tile grown by the radius and clipped to the
// raster, into a buffer whose Stride is a multiple of 64 with its mask at
// bit 0 for a masked source and compact for one without, and there are
// no more distinct buffers than workers, whatever the number of tiles.
func TestChunkedBuffers(t *testing.T) {
	const w, h = 70, 45
	rng := rand.New(rand.NewPCG(1, 2))
	for _, masked := range []bool{true, false} {
		testChunkedBuffers(t, newOperand(rng, w, h, true, masked), masked)
	}
}

func testChunkedBuffers(t *testing.T, dem operand, masked bool) {
	const w, h = 70, 45
	for _, o := range []engine.Options{
		{TileWidth: 16, TileHeight: 10, Workers: 3},
		{TileWidth: 1, TileHeight: 1, Workers: 2},
		{TileHeight: 8, Workers: 1},
		{Workers: 4},
	} {
		for _, r := range []int{0, 1, 2} {
			src := &recordingSource{RasterSource: engine.NewMemorySource(dem.r)}
			out := raster.NewFloat32Like(dem.r)
			if err := exec.ProcessChunked(context.Background(), []engine.RasterSink{engine.NewMemorySink(out)},
				[]engine.RasterSource{src}, boxKernel{r: r, inputs: 1}, o); err != nil {
				t.Fatal(err)
			}
			tw, th := o.TileWidth, o.TileHeight
			if tw == 0 {
				tw = w
			}
			if th == 0 {
				th = h
			}
			tiles := ((w + tw - 1) / tw) * ((h + th - 1) / th)
			id := fmt.Sprintf("%+v r=%d masked=%v", o, r, masked)
			if len(src.regions) != tiles {
				t.Fatalf("%s: %d reads, want one per tile, %d", id, len(src.regions), tiles)
			}
			var want [][4]int
			for ty := 0; ty < h; ty += th {
				for tx := 0; tx < w; tx += tw {
					x0, y0 := max(0, tx-r), max(0, ty-r)
					x1, y1 := min(w, tx+tw+r), min(h, ty+th+r)
					want = append(want, [4]int{x0, y0, x1 - x0, y1 - y0})
				}
			}
			got := slices.Clone(src.regions)
			cmp := func(a, b [4]int) int { return slices.Compare(a[:], b[:]) }
			slices.SortFunc(got, cmp)
			slices.SortFunc(want, cmp)
			if !slices.Equal(got, want) {
				t.Fatalf("%s: read regions %v, want %v", id, got, want)
			}
			workers := min(max(o.Workers, 1), tiles)
			if len(src.buffers) > workers {
				t.Fatalf("%s: %d distinct buffers, want at most %d", id, len(src.buffers), workers)
			}
			for _, b := range src.buffers {
				switch {
				case masked && (b.Stride%64 != 0 || b.ValidOffset != 0 || b.Valid == nil):
					t.Fatalf("%s: buffer stride %d, mask offset %d, mask %v; want a multiple of 64, 0 and a mask",
						id, b.Stride, b.ValidOffset, b.Valid != nil)
				case !masked && (b.Stride != min(tw+2*r, w) || b.Valid != nil):
					t.Fatalf("%s: buffer stride %d, mask %v; want compact, %d, and no mask",
						id, b.Stride, b.Valid != nil, min(tw+2*r, w))
				}
				if full := (min(th+2*r, h)-1)*b.Stride + min(tw+2*r, w); cap(b.Data) > full {
					t.Fatalf("%s: buffer view capacity %d, want at most one grown tile, %d", id, cap(b.Data), full)
				}
			}
		}
	}
}

// failingSource fails its nth read (counting from 1) with err, or panics
// with err if panics is set.
type failingSource struct {
	engine.RasterSource
	n      int64
	err    error
	panics bool
	reads  atomic.Int64
}

func (s *failingSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	if s.reads.Add(1) == s.n {
		if s.panics {
			panic(s.err)
		}
		return s.err
	}
	return s.RasterSource.ReadWindow(ctx, dst, x, y)
}

// failingSink fails its nth write after writing its first rows, so the
// failed tile is partly written.
type failingSink struct {
	engine.RasterSink
	n            int64
	err          error
	writes       atomic.Int64
	failX, failY int // the failed write's position
}

func (s *failingSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	if s.writes.Add(1) == s.n {
		s.failX, s.failY = x, y
		if src.Height > 1 {
			part := src.Window(0, 0, src.Width, src.Height/2)
			if err := s.RasterSink.WriteWindow(ctx, part, x, y); err != nil {
				return err
			}
		}
		return s.err
	}
	return s.RasterSink.WriteWindow(ctx, src, x, y)
}

// chunkTiles returns the tile rectangles [x0, y0, x1, y1) of a w×h raster
// in plan order.
func chunkTiles(w, h, tw, th int) [][4]int {
	var out [][4]int
	for y := 0; y < h; y += th {
		for x := 0; x < w; x += tw {
			out = append(out, [4]int{x, y, min(x+tw, w), min(y+th, h)})
		}
	}
	return out
}

// tileState reports whether every cell of a tile is final (holds the
// whole-raster result) and whether every cell is untouched.
func tileState(got, final, orig operand, b [4]int) (isFinal, isOrig bool) {
	g, f, o := got.r, final.r, orig.r
	isFinal, isOrig = true, true
	for y := b[1]; y < b[3]; y++ {
		for x := b[0]; x < b[2]; x++ {
			i := g.Index(x, y)
			if g.IsValid(x, y) != f.IsValid(x, y) || (f.IsValid(x, y) && !sameFloat(g.Data[i], f.Data[i])) {
				isFinal = false
			}
			if g.IsValid(x, y) != o.IsValid(x, y) || math.Float32bits(g.Data[i]) != math.Float32bits(o.Data[i]) {
				isOrig = false
			}
		}
	}
	return isFinal, isOrig
}

// requireTiles checks what a failed or cancelled call leaves: whole
// tiles. Up to the last tile that was touched, every tile is final,
// except at most holes untouched tiles (tiles whose read failed) and the
// tile anyTile (whose write failed), which may hold anything; every later
// tile is untouched; and nothing outside the output changed. It returns
// the number of final tiles.
func requireTiles(t *testing.T, id string, got, final, orig operand, tiles [][4]int, holes, anyTile int) int {
	t.Helper()
	requireOutsideUntouched(t, id, got, orig)
	type state struct{ final, orig bool }
	states := make([]state, len(tiles))
	last := -1
	for i, b := range tiles {
		f, o := tileState(got, final, orig, b)
		states[i] = state{f, o}
		if !o {
			last = i
		}
	}
	written, seen := 0, 0
	for i := 0; i <= last; i++ {
		switch {
		case i == anyTile:
		case states[i].final:
			written++
		case states[i].orig:
			seen++
			if seen > holes {
				t.Fatalf("%s: tile %d %v is untouched but tile %d %v is written", id, i, tiles[i], last, tiles[last])
			}
		default:
			t.Fatalf("%s: tile %d %v is partly written", id, i, tiles[i])
		}
	}
	return written
}

// TestChunkedContextDone checks that a context done before the call
// reads and writes nothing.
func TestChunkedContextDone(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 6))
	dem := newOperand(rng, 20, 20, true, true)
	out := newOperand(rng, 20, 20, true, true)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, workers := range []int{1, 4} {
		got := out.clone()
		src := &failingSource{RasterSource: engine.NewMemorySource(dem.r), n: math.MaxInt64}
		err := exec.ProcessChunked(ctx, []engine.RasterSink{engine.NewMemorySink(got.r)}, []engine.RasterSource{src},
			boxKernel{r: 1, inputs: 1}, engine.Options{TileWidth: 5, TileHeight: 5, Workers: workers})
		if !errors.Is(err, context.Canceled) || src.reads.Load() != 0 {
			t.Fatalf("workers=%d: err = %v after %d reads, want context.Canceled and none", workers, err, src.reads.Load())
		}
		requireSameRoots(t, "untouched", got.root, out.root)
	}
}

// orDim is an Options tile dimension resolved against the raster's.
func orDim(v, d int) int {
	if v == 0 {
		return d
	}
	return v
}

// TestChunkedPanics checks the programming errors ProcessChunked rejects
// and that a panic in a source reaches the caller.
func TestChunkedPanics(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	a := newOperand(rng, 10, 8, true, true)
	b := newOperand(rng, 10, 8, false, false)
	c := newOperand(rng, 9, 8, false, false)
	box := boxKernel{r: 1, inputs: 1}
	sink := func(r raster.Float32Raster) []engine.RasterSink { return []engine.RasterSink{engine.NewMemorySink(r)} }
	source := func(r raster.Float32Raster) []engine.RasterSource {
		return []engine.RasterSource{engine.NewMemorySource(r)}
	}
	ctx := context.Background()
	unmasked := raster.NewFloat32(10, 8, make([]float32, 80))
	mustPanic(t, "nil kernel", func() { _ = exec.ProcessChunked(ctx, sink(b.r), source(a.r), nil, engine.Options{}) })
	mustPanic(t, "takes 1 inputs", func() { _ = exec.ProcessChunked(ctx, sink(b.r), nil, box, engine.Options{}) })
	mustPanic(t, "nil sink", func() {
		_ = exec.ProcessChunked(ctx, []engine.RasterSink{nil}, source(a.r), box, engine.Options{})
	})
	mustPanic(t, "nil source", func() {
		_ = exec.ProcessChunked(ctx, sink(b.r), []engine.RasterSource{nil}, box, engine.Options{})
	})
	mustPanic(t, "src[0] is 9×8", func() { _ = exec.ProcessChunked(ctx, sink(b.r), source(c.r), box, engine.Options{}) })
	mustPanic(t, "is not; its validity would be lost", func() {
		_ = exec.ProcessChunked(ctx, sink(unmasked), source(a.r), box, engine.Options{})
	})
	mustPanic(t, "negative Options", func() {
		_ = exec.ProcessChunked(ctx, sink(b.r), source(a.r), box, engine.Options{Workers: -1})
	})
	mustPanic(t, "shares memory with memory source", func() {
		_ = exec.ProcessChunked(ctx, sink(a.r), source(a.r), boxKernel{r: 0, inputs: 1}, engine.Options{})
	})
	// Side-by-side windows of one masked root share mask words.
	left, right := a.root.Window(0, 0, 5, a.r.Height), a.root.Window(5, 0, 5, a.r.Height)
	mustPanic(t, "memory sinks dst[0] and dst[1] share memory", func() {
		_ = exec.ProcessChunked(ctx, []engine.RasterSink{engine.NewMemorySink(left), engine.NewMemorySink(right)},
			[]engine.RasterSource{engine.NewMemorySource(raster.NewFloat32Like(left))},
			boxKernel{r: 1, inputs: 1, outputs: 2}, engine.Options{})
	})
	errBoom := errors.New("boom")
	for _, workers := range []int{1, 3} {
		src := &failingSource{RasterSource: engine.NewMemorySource(a.r), n: 2, err: errBoom, panics: true}
		got := func() (r any) {
			defer func() { r = recover() }()
			_ = exec.ProcessChunked(ctx, sink(raster.NewFloat32Like(a.r)), []engine.RasterSource{src}, box,
				engine.Options{TileWidth: 3, TileHeight: 3, Workers: workers})
			return nil
		}()
		//nolint:errorlint // the panic value must be the source's own error, not one wrapping it
		if got != errBoom {
			t.Fatalf("workers=%d: recovered %v, want %v", workers, got, errBoom)
		}
	}
}

// TestChunkedAllocs checks that allocations depend on the worker count,
// not on the number of tiles.
func TestChunkedAllocs(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 10))
	dem := newOperand(rng, 128, 96, false, true)
	out := raster.NewFloat32Like(dem.r)
	sinks := []engine.RasterSink{engine.NewMemorySink(out)}
	sources := []engine.RasterSource{engine.NewMemorySource(dem.r)}
	k := boxKernel{r: 1, inputs: 1}
	allocs := func(o engine.Options) float64 {
		return testing.AllocsPerRun(5, func() {
			if err := exec.ProcessChunked(context.Background(), sinks, sources, k, o); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, workers := range []int{1, 4} {
		few := allocs(engine.Options{TileWidth: 64, TileHeight: 48, Workers: workers})
		many := allocs(engine.Options{TileWidth: 4, TileHeight: 3, Workers: workers})
		if many != few || few > float64(8+10*workers) {
			t.Errorf("workers=%d: %v allocs/op with 4 tiles, %v with 1024; want equal and at most %d",
				workers, few, many, 8+10*workers)
		}
	}
}

// byteFile is an in-memory io.ReaderAt and io.WriterAt of a fixed size,
// safe for concurrent use on disjoint ranges.
type byteFile []byte

func (f byteFile) ReadAt(p []byte, off int64) (int, error) { return copy(p, f[off:]), nil }

func (f byteFile) WriteAt(p []byte, off int64) (int, error) { return copy(f[off:], p), nil }

func (f byteFile) cell(i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(f[4*i:]))
}

// TestChunkedRawFiles runs Slope, Hillshade and Clamp from a raw float32
// file to another, without a fill value and with one, against the plain
// functions on the same data in memory: with a fill value the DEM's mask
// comes from the cells holding it, and the output file holds the fill
// value under invalid cells.
func TestChunkedRawFiles(t *testing.T) {
	const w, h = 157, 83
	const fill = -9999
	rng := rand.New(rand.NewPCG(12, 13))
	for _, name := range []string{"slope-degrees", "hillshade", "clamp"} {
		op := adapterNamed(t, name)
		f := chunked[name]
		for _, withFill := range []bool{false, true} {
			dem := raster.NewFloat32(w, h, make([]float32, w*h))
			in := make(byteFile, 4*w*h)
			for i := range dem.Data {
				v := float32(1000 + 50*rng.NormFloat64())
				if withFill && rng.IntN(15) == 0 {
					v = fill
				}
				dem.Data[i] = v
				binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(v))
			}
			opts := engine.RawOptions{}
			want := raster.NewFloat32Like(dem)
			if withFill {
				opts = engine.RawOptions{Fill: fill, HasFill: true}
				dem.Valid = raster.NewMask(w * h)
				want.Valid = raster.NewMask(w * h)
				for i, v := range dem.Data {
					raster.MaskSet(dem.Valid, i, v != fill)
				}
			}
			op.direct([]raster.Float32Raster{want}, []raster.Float32Raster{dem})
			for _, o := range []engine.Options{{TileWidth: 32, TileHeight: 20, Workers: 3}, {TileHeight: 7}, {TileWidth: 1, TileHeight: 50, Workers: 2}} {
				id := fmt.Sprintf("%s fill=%v %+v", name, withFill, o)
				out := make(byteFile, 4*w*h)
				err := f(context.Background(), []engine.RasterSink{engine.NewRawSink(out, w, h, opts)},
					[]engine.RasterSource{engine.NewRawSource(in, w, h, opts)}, o)
				if err != nil {
					t.Fatalf("%s: %v", id, err)
				}
				for i := range w * h {
					got, wantV := out.cell(i), want.Data[i]
					if want.Valid != nil && !raster.MaskGet(want.Valid, i) {
						wantV = fill
					}
					if !sameFloat(got, wantV) || (withFill && got == fill) != (want.Valid != nil && !raster.MaskGet(want.Valid, i)) {
						t.Fatalf("%s: cell %d = %v, want %v", id, i, got, wantV)
					}
				}
			}
		}
	}
}
