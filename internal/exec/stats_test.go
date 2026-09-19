package exec_test

import (
	"context"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// ramp returns a w×h raster whose cell i in row-major order holds i.
func ramp(w, h int) raster.Float32Raster {
	data := make([]float32, w*h)
	for i := range data {
		data[i] = float32(i)
	}
	return raster.NewFloat32(w, h, data)
}

func rasters(n, w, h int) []raster.Float32Raster {
	out := make([]raster.Float32Raster, n)
	for i := range out {
		out[i] = ramp(w, h)
	}
	return out
}

// TestStatsTiledIsExact pins what a tiled call reports for the simplest
// case there is: one pointwise input, one output, one tile. Every figure
// is forced, so a change to the accounting fails here before it reaches a
// benchmark.
func TestStatsTiledIsExact(t *testing.T) {
	const w, h = 64, 32
	dst, src := rasters(1, w, h), rasters(1, w, h)
	var s engine.Stats
	if err := exec.ProcessN(context.Background(), dst, src, boxKernel{r: 0, inputs: 1, outputs: 1},
		engine.Options{Stats: &s}); err != nil {
		t.Fatalf("ProcessN: %v", err)
	}

	cells := int64(w * h)
	if s.Cells != cells {
		t.Errorf("Cells = %d, want %d", s.Cells, cells)
	}
	if s.KernelRead != cells*4 {
		t.Errorf("KernelRead = %d, want %d", s.KernelRead, cells*4)
	}
	if s.KernelWritten != cells*4 {
		t.Errorf("KernelWritten = %d, want %d", s.KernelWritten, cells*4)
	}
	if s.SourceRead != 0 || s.SinkWritten != 0 {
		t.Errorf("SourceRead = %d, SinkWritten = %d, want 0 and 0 for a tiled call",
			s.SourceRead, s.SinkWritten)
	}
	if s.Tiles != 0 {
		t.Errorf("Tiles = %d, want 0 for a tiled call", s.Tiles)
	}
	if s.Ideal != cells*8 {
		t.Errorf("Ideal = %d, want %d", s.Ideal, cells*8)
	}
	if got := s.Amplification(); got != 1 {
		t.Errorf("Amplification = %v, want exactly 1: a pointwise tiled call moves "+
			"every byte once (%v)", got, s)
	}
	if got := s.Halo(1); got != 0 {
		t.Errorf("Halo = %d, want 0 for a radius-0 kernel", got)
	}
}

// TestStatsKernelWrittenIsExact pins the invariant that makes the
// counters trustworthy: however the plan splits the raster and whatever
// the radius, every output cell is written exactly once, so KernelWritten
// is Cells × 4 × outputs and Cells is the raster. Interior cells are
// counted by interior and edge cells by band, and this is what says the
// two neither overlap nor leave a gap.
func TestStatsKernelWrittenIsExact(t *testing.T) {
	const w, h = 37, 23 // not a multiple of any tile size below
	for _, r := range []int{0, 1, 2} {
		for _, nin := range []int{1, 2} {
			for _, nout := range []int{1, 2} {
				for _, tile := range []int{0, 8, 16} {
					var s engine.Stats
					dst, src := rasters(nout, w, h), rasters(nin, w, h)
					err := exec.ProcessN(context.Background(), dst, src,
						boxKernel{r: r, inputs: nin, outputs: nout},
						engine.Options{TileWidth: tile, TileHeight: tile, Workers: 3, Stats: &s})
					if err != nil {
						t.Fatalf("r=%d in=%d out=%d tile=%d: %v", r, nin, nout, tile, err)
					}
					if want := int64(w * h); s.Cells != want {
						t.Errorf("r=%d in=%d out=%d tile=%d: Cells = %d, want %d",
							r, nin, nout, tile, s.Cells, want)
					}
					if want := s.Cells * 4 * int64(nout); s.KernelWritten != want {
						t.Errorf("r=%d in=%d out=%d tile=%d: KernelWritten = %d, want %d",
							r, nin, nout, tile, s.KernelWritten, want)
					}
					if want := s.Cells * 4 * int64(nin+nout); s.Ideal != want {
						t.Errorf("r=%d in=%d out=%d tile=%d: Ideal = %d, want %d",
							r, nin, nout, tile, s.Ideal, want)
					}
				}
			}
		}
	}
}

// TestStatsRadius0IsSchedulingInvariant checks that a pointwise call
// reports the same traffic whatever the tile size and worker count. The
// engine guarantees that Options never change the result; for radius 0
// they must not change the traffic either, because there is no halo to
// read twice. Anything else that moved would be bookkeeping following the
// plan, which would make two benchmark runs incomparable.
func TestStatsRadius0IsSchedulingInvariant(t *testing.T) {
	const w, h = 64, 64
	var want engine.Stats
	first := true
	for _, workers := range []int{1, 2, 4} {
		for _, tile := range []int{0, 8, 32} {
			var s engine.Stats
			dst, src := rasters(1, w, h), rasters(1, w, h)
			err := exec.ProcessN(context.Background(), dst, src, boxKernel{r: 0, inputs: 1, outputs: 1},
				engine.Options{TileWidth: tile, TileHeight: tile, Workers: workers, Stats: &s})
			if err != nil {
				t.Fatalf("workers=%d tile=%d: %v", workers, tile, err)
			}
			s.Bands = 0 // the one figure that is meant to follow the plan
			if first {
				want, first = s, false
				continue
			}
			if s != want {
				t.Errorf("workers=%d tile=%d: stats %v, want %v", workers, tile, s, want)
			}
		}
	}
}

// TestStatsHaloIsTheCostOfTiling pins the halo for a case small enough to
// count by hand, and checks that splitting the raster more finely reads
// more. On an 8×8 raster with radius 1:
//
//   - One band covers the whole raster. Its interior is the 6×6 centre,
//     whose window is the whole 8×8 raster: 64 cells, read once. No cell
//     is read twice, so the halo is 0.
//   - Eight one-row bands. Rows 0 and 7 have no interior and read
//     nothing. Each of rows 1 to 6 has a 6×1 interior whose window is
//     8×3 = 24 cells, so 144 cells are read for 64 cells of output: a
//     halo of (144 − 64) × 4 = 320 bytes.
func TestStatsHaloIsTheCostOfTiling(t *testing.T) {
	const w, h = 8, 8
	run := func(bandCells int) engine.Stats {
		t.Helper()
		defer exec.SetBandCells(bandCells)()
		var s engine.Stats
		dst, src := rasters(1, w, h), rasters(1, w, h)
		err := exec.ProcessN(context.Background(), dst, src, boxKernel{r: 1, inputs: 1, outputs: 1},
			engine.Options{Workers: 1, Stats: &s})
		if err != nil {
			t.Fatalf("ProcessN: %v", err)
		}
		return s
	}

	whole := run(1 << 16)
	if got := whole.Halo(1); got != 0 {
		t.Errorf("one band: Halo = %d, want 0; KernelRead = %d (%v)", got, whole.KernelRead, whole)
	}
	rows := run(w) // bandRows = 1
	if got := rows.Halo(1); got != 320 {
		t.Errorf("one-row bands: Halo = %d, want 320; KernelRead = %d (%v)",
			got, rows.KernelRead, rows)
	}
	if rows.Amplification() <= whole.Amplification() {
		t.Errorf("finer bands must move more: %v vs %v", rows.Amplification(), whole.Amplification())
	}
}

// TestStatsChunkedAmplificationIsTwo is the measurement the counters were
// added for. A pointwise chunked call moves every cell four times — the
// source fills a buffer, the kernel reads it, the kernel writes an output
// buffer, the sink drains it — where the work strictly needs two. That is
// the fixed per-cell tax behind the chunked path running Slope, Hillshade
// and Clamp at one speed whatever their compute costs
// (benchmarks/chunked/RESULTS.md), and here it is exactly 2.00 rather
// than an inference from three throughputs that happen to coincide.
func TestStatsChunkedAmplificationIsTwo(t *testing.T) {
	const w, h = 64, 64
	dstR, srcR := ramp(w, h), ramp(w, h)
	var s engine.Stats
	err := exec.ProcessChunked(context.Background(),
		[]engine.RasterSink{engine.NewMemorySink(dstR)},
		[]engine.RasterSource{engine.NewMemorySource(srcR)},
		boxKernel{r: 0, inputs: 1, outputs: 1},
		engine.Options{TileHeight: 16, Workers: 2, Stats: &s})
	if err != nil {
		t.Fatalf("ProcessChunked: %v", err)
	}

	cells := int64(w * h)
	for _, c := range []struct {
		name string
		got  int64
	}{
		{"SourceRead", s.SourceRead},
		{"KernelRead", s.KernelRead},
		{"KernelWritten", s.KernelWritten},
		{"SinkWritten", s.SinkWritten},
	} {
		if c.got != cells*4 {
			t.Errorf("%s = %d, want %d (%v)", c.name, c.got, cells*4, s)
		}
	}
	if s.Tiles != 4 {
		t.Errorf("Tiles = %d, want 4", s.Tiles)
	}
	if got := s.Amplification(); got != 2 {
		t.Errorf("Amplification = %v, want exactly 2 (%v)", got, s)
	}
	if got := s.BytesPerCell(); got != 16 {
		t.Errorf("BytesPerCell = %v, want 16 (%v)", got, s)
	}
}

// TestStatsReduce checks the fold paths. A reduction has no outputs, so
// Ideal is its inputs alone: reading every cell once in memory is 1.00,
// and the chunked fold's copy into a tile buffer makes it 2.00 — the same
// tax as the chunked map, with two stages instead of four.
func TestStatsReduce(t *testing.T) {
	const w, h = 64, 64
	cells := int64(w * h)

	var mem engine.Stats
	if _, err := exec.Reduce(context.Background(), rasters(1, w, h), tallyOp{inputs: 1},
		engine.Options{Workers: 2, Stats: &mem}); err != nil {
		t.Fatalf("Reduce: %v", err)
	}
	if mem.KernelWritten != 0 || mem.SourceRead != 0 {
		t.Errorf("Reduce wrote something: %v", mem)
	}
	if mem.KernelRead != cells*4 || mem.Ideal != cells*4 {
		t.Errorf("Reduce: KernelRead = %d, Ideal = %d, want %d for both",
			mem.KernelRead, mem.Ideal, cells*4)
	}
	if got := mem.Amplification(); got != 1 {
		t.Errorf("Reduce: Amplification = %v, want exactly 1 (%v)", got, mem)
	}

	var chunked engine.Stats
	_, err := exec.ReduceChunked(context.Background(),
		[]engine.RasterSource{engine.NewMemorySource(ramp(w, h))}, tallyOp{inputs: 1},
		engine.Options{TileHeight: 16, Workers: 2, Stats: &chunked})
	if err != nil {
		t.Fatalf("ReduceChunked: %v", err)
	}
	if chunked.SourceRead != cells*4 || chunked.KernelRead != cells*4 {
		t.Errorf("ReduceChunked: SourceRead = %d, KernelRead = %d, want %d for both",
			chunked.SourceRead, chunked.KernelRead, cells*4)
	}
	if got := chunked.Amplification(); got != 2 {
		t.Errorf("ReduceChunked: Amplification = %v, want exactly 2 (%v)", got, chunked)
	}
}

// TestStatsAccumulate checks that one Stats can total a pipeline. Two
// calls of the same shape must double every figure and leave
// Amplification where one call left it, which is what lets a caller
// measure a chain of operations with one Stats.
func TestStatsAccumulate(t *testing.T) {
	const w, h = 32, 32
	k := boxKernel{r: 1, inputs: 1, outputs: 1}
	call := func(into *engine.Stats) {
		t.Helper()
		dst, src := rasters(1, w, h), rasters(1, w, h)
		if err := exec.ProcessN(context.Background(), dst, src, k,
			engine.Options{Stats: into}); err != nil {
			t.Fatalf("ProcessN: %v", err)
		}
	}
	var one, two engine.Stats
	call(&one)
	call(&two)
	call(&two)

	if two.Cells != 2*one.Cells || two.Total() != 2*one.Total() || two.Ideal != 2*one.Ideal {
		t.Errorf("two calls = %v, want twice %v", two, one)
	}
	if two.Amplification() != one.Amplification() {
		t.Errorf("Amplification changed under accumulation: %v then %v",
			one.Amplification(), two.Amplification())
	}
}

// TestStatsZeroValue checks the methods on an empty Stats, which a caller
// meets whenever a call wrote nothing.
func TestStatsZeroValue(t *testing.T) {
	var s engine.Stats
	if s.Total() != 0 || s.Amplification() != 0 || s.BytesPerCell() != 0 || s.Halo(1) != 0 {
		t.Errorf("zero Stats: %v", s)
	}
	if s.Halo(0) != 0 {
		t.Errorf("Halo(0) = %d, want 0", s.Halo(0))
	}
}
