package exec_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

func newRaster(w, h int) raster.Float32Raster {
	return raster.NewFloat32(w, h, make([]float32, w*h))
}

// tally is a reduction partial that is exactly order-independent by
// construction: the XOR of a hash of every valid cell's value and raster
// position, and how many cells there were. XOR is associative and
// commutative, so no tiling, worker count or combination order can change
// it — but mixing the position in means a cell read at the wrong place
// does, a cell read twice cancels itself, and an invalid cell that is
// read at all shows up. That makes it a driver test with no arithmetic of
// its own to get wrong.
type tally struct {
	bits  uint64
	count int64
}

type tallyOp struct{ inputs int }

func (o tallyOp) Inputs() int { return o.inputs }

func (tallyOp) Combine(a *tally, b tally) {
	a.bits ^= b.bits
	a.count += b.count
}

func (tallyOp) Fold(p *tally, c exec.Cells) {
	for y := range c.Height {
		for x := range c.Width {
			if c.ValidBits(x, y, 1) == 0 {
				continue
			}
			for i, s := range c.Src {
				p.bits ^= cellHash(i, c.X+x, c.Y+y, s.Data[s.Index(x, y)])
			}
			p.count++
		}
	}
}

func cellHash(operand, x, y int, v float32) uint64 {
	h := uint64(math.Float32bits(v)) | uint64(operand+1)<<32
	h *= 0x9e3779b97f4a7c15
	h ^= uint64(x)*0x100000001b3 + uint64(y)*0xff51afd7ed558ccd
	h *= 0xc4ceb9fe1a85ec53
	return h ^ h>>31
}

// wantTally is the reduction computed cell by cell, with no engine.
func wantTally(src []raster.Float32Raster) tally {
	var w tally
	r0 := src[0]
	for y := range r0.Height {
		for x := range r0.Width {
			valid := true
			for _, s := range src {
				valid = valid && s.IsValid(x, y)
			}
			if !valid {
				continue
			}
			for i, s := range src {
				w.bits ^= cellHash(i, x, y, s.Data[s.Index(x, y)])
			}
			w.count++
		}
	}
	return w
}

// TestReduceTilesAndWorkers is the DESIGN.md §23 contract for folds: for
// every tile width and height in {1, 7, 64, 256, full, larger than the
// raster} and every worker count in {1, 2, 3, GOMAXPROCS}, Reduce gives
// the whole-raster answer exactly. It runs one, two and three inputs,
// without masks and with masks on some or all of them, on an odd-sized
// compact raster and on windows whose Stride exceeds Width and is not a
// multiple of 64. Bands are one row or 97 cells, so workers share many
// bands even on small rasters.
func TestReduceTilesAndWorkers(t *testing.T) {
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

	for _, inputs := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("inputs=%d", inputs), func(t *testing.T) {
			for _, sh := range shapes {
				for _, masked := range []string{"none", "first", "all"} {
					rng := rand.New(rand.NewPCG(uint64(sh.w*1000+sh.h), 11))
					src := make([]raster.Float32Raster, inputs)
					for i := range src {
						m := masked == "all" || (masked == "first" && i == 0)
						src[i] = newOperandPad(rng, sh.w, sh.h, sh.windowed, m, sh.pad).r
					}
					want := wantTally(src)
					n := 0
					for _, tw := range dims {
						for _, th := range dims {
							for _, wk := range workers {
								run := engineRun{
									opts:      engine.Options{TileWidth: tw, TileHeight: th, Workers: wk},
									bandCells: []int{1, 97, 0}[n%3],
								}
								n++
								id := fmt.Sprintf("%s %dx%d masked=%s %v", sh.name, sh.w, sh.h, masked, run)
								got := runReduce(t, run, src, tallyOp{inputs})
								if got != want {
									t.Fatalf("%s: got %+v, want %+v", id, got, want)
								}
							}
						}
					}
				}
			}
		})
	}
}

// TestReduceChunkedTilesAndWorkers is TestReduceTilesAndWorkers through
// memory sources, where each tile is read into a worker's own buffer.
func TestReduceChunkedTilesAndWorkers(t *testing.T) {
	shapes := []struct {
		name     string
		w, h     int
		windowed bool
		pad      int
	}{
		{"odd-compact", 61, 37, false, 0},
		{"wide-window", 300, 9, true, 5},
		{"tall-window", 21, 131, true, 37},
	}
	dims := []int{1, 7, 64, 256, 0, 1 << 20}
	workers := []int{1, 2, 3, runtime.GOMAXPROCS(0)}
	if testing.Short() {
		dims = []int{1, 7, 0}
	}

	for _, inputs := range []int{1, 2} {
		t.Run(fmt.Sprintf("inputs=%d", inputs), func(t *testing.T) {
			for _, sh := range shapes {
				for _, masked := range []string{"none", "first", "all"} {
					rng := rand.New(rand.NewPCG(uint64(sh.w*7+sh.h), 3))
					src := make([]raster.Float32Raster, inputs)
					srcs := make([]engine.RasterSource, inputs)
					for i := range src {
						m := masked == "all" || (masked == "first" && i == 0)
						src[i] = newOperandPad(rng, sh.w, sh.h, sh.windowed, m, sh.pad).r
						srcs[i] = engine.NewMemorySource(src[i])
					}
					want := wantTally(src)
					n := 0
					for _, tw := range dims {
						for _, th := range dims {
							for _, wk := range workers {
								run := engineRun{
									opts:      engine.Options{TileWidth: tw, TileHeight: th, Workers: wk},
									bandCells: []int{1, 97, 0}[n%3],
								}
								n++
								id := fmt.Sprintf("%s %dx%d masked=%s %v", sh.name, sh.w, sh.h, masked, run)
								got := runReduceChunked(t, run, srcs, tallyOp{inputs})
								if got != want {
									t.Fatalf("%s: got %+v, want %+v", id, got, want)
								}
							}
						}
					}
				}
			}
		})
	}
}

// runReduce runs Reduce the way run says and fails the test on an error.
func runReduce(t *testing.T, run engineRun, src []raster.Float32Raster, r exec.Reducer[tally]) tally {
	t.Helper()
	if run.bandCells > 0 {
		defer exec.SetBandCells(run.bandCells)()
	}
	got, err := exec.Reduce(context.Background(), src, r, run.opts)
	if err != nil {
		t.Fatalf("%v: %v", run, err)
	}
	return got
}

func runReduceChunked(t *testing.T, run engineRun, src []engine.RasterSource, r exec.Reducer[tally]) tally {
	t.Helper()
	if run.bandCells > 0 {
		defer exec.SetBandCells(run.bandCells)()
	}
	got, err := exec.ReduceChunked(context.Background(), src, r, run.opts)
	if err != nil {
		t.Fatalf("%v: %v", run, err)
	}
	return got
}

// TestReduceAllInvalid checks the empty reduction: a raster whose every
// cell is invalid folds nothing, whatever the tiling, and the partial
// stays the zero value.
func TestReduceAllInvalid(t *testing.T) {
	w, h := 37, 21
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = make([]uint64, raster.MaskWords(w*h))
	for i := range r.Data {
		r.Data[i] = float32(i)
	}
	ctx := context.Background()
	for _, run := range engineRuns {
		got := runReduce(t, run, []raster.Float32Raster{r}, tallyOp{1})
		if (got != tally{}) {
			t.Fatalf("%v: got %+v, want the zero partial", run, got)
		}
		gotC, err := exec.ReduceChunked(ctx, []engine.RasterSource{engine.NewMemorySource(r)}, tallyOp{1}, run.opts)
		if err != nil || (gotC != tally{}) {
			t.Fatalf("%v chunked: got %+v, %v", run, gotC, err)
		}
	}
}

// TestReducePanics checks that every programming error panics with this
// package's own message, before anything is folded.
func TestReducePanics(t *testing.T) {
	ctx := context.Background()
	a := newRaster(6, 5)
	one := []raster.Float32Raster{a}

	mustPanic(t, "engine: nil reducer", func() {
		_, _ = exec.Reduce[tally](ctx, one, nil, engine.Options{})
	})
	mustPanic(t, "needs at least one input", func() {
		_, _ = exec.Reduce(ctx, nil, tallyOp{0}, engine.Options{})
	})
	mustPanic(t, "reducer takes 2 inputs, got 1", func() {
		_, _ = exec.Reduce(ctx, one, tallyOp{2}, engine.Options{})
	})
	mustPanic(t, "negative Options", func() {
		_, _ = exec.Reduce(ctx, one, tallyOp{1}, engine.Options{TileHeight: -1})
	})
	mustPanic(t, "engine: src[1] is 5×6, src[0] is 6×5", func() {
		_, _ = exec.Reduce(ctx, []raster.Float32Raster{a, newRaster(5, 6)}, tallyOp{2}, engine.Options{})
	})
	mustPanic(t, "engine: src[0]: raster: data has", func() {
		short := a
		short.Data = short.Data[:5]
		_, _ = exec.Reduce(ctx, []raster.Float32Raster{short}, tallyOp{1}, engine.Options{})
	})

	mustPanic(t, "engine: nil reducer", func() {
		_, _ = exec.ReduceChunked[tally](ctx, []engine.RasterSource{engine.NewMemorySource(a)}, nil, engine.Options{})
	})
	mustPanic(t, "engine: src[0] is a nil source", func() {
		_, _ = exec.ReduceChunked(ctx, []engine.RasterSource{nil}, tallyOp{1}, engine.Options{})
	})
	mustPanic(t, "engine: src[1] is 5×6, src[0] is 6×5", func() {
		_, _ = exec.ReduceChunked(ctx, []engine.RasterSource{
			engine.NewMemorySource(a), engine.NewMemorySource(newRaster(5, 6)),
		}, tallyOp{2}, engine.Options{})
	})
}

// TestReduceCancelled is DESIGN.md §49's cancellation rule, the one place
// a fold differs from a map: a cancelled reduction returns ctx.Err() and
// the zero value, never the partial fold of the bands that did run.
func TestReduceCancelled(t *testing.T) {
	w, h := 64, 64
	src := []raster.Float32Raster{newRaster(w, h)}
	srcs := []engine.RasterSource{engine.NewMemorySource(src[0])}
	defer exec.SetBandCells(1)()

	for _, wk := range []int{1, 2, 4} {
		opts := engine.Options{TileWidth: 8, TileHeight: 8, Workers: wk}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		got, err := exec.Reduce(ctx, src, tallyOp{1}, opts)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("workers=%d: err = %v, want context.Canceled", wk, err)
		}
		if (got != tally{}) {
			t.Fatalf("workers=%d: got %+v, want the zero value with an error", wk, got)
		}
		gotC, err := exec.ReduceChunked(ctx, srcs, tallyOp{1}, opts)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("workers=%d chunked: err = %v, want context.Canceled", wk, err)
		}
		if (gotC != tally{}) {
			t.Fatalf("workers=%d chunked: got %+v, want the zero value with an error", wk, gotC)
		}
	}
}

// TestReduceSourceError checks that a source's error reaches the caller
// named with the operand and tile, and that no value comes back with it.
func TestReduceSourceError(t *testing.T) {
	boom := errors.New("boom")
	base := engine.NewMemorySource(newRaster(32, 32))
	src := []engine.RasterSource{&failingSource{RasterSource: base, n: 2, err: boom}}
	got, err := exec.ReduceChunked(context.Background(), src, tallyOp{1},
		engine.Options{TileWidth: 8, TileHeight: 8, Workers: 2})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	if (got != tally{}) {
		t.Fatalf("got %+v, want the zero value with an error", got)
	}
}
