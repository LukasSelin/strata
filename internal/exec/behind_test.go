package exec_test

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// panickySink panics with val on its nth write.
type panickySink struct {
	engine.RasterSink
	n      int64
	val    any
	writes atomic.Int64
}

func (s *panickySink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	if s.writes.Add(1) == s.n {
		panic(s.val)
	}
	return s.RasterSink.WriteWindow(ctx, src, x, y)
}

// TestChunkedSinkPanics checks that a sink's panic, raised on a worker's
// writer goroutine, is re-raised on the calling goroutine with the same
// value, whichever tile it hits, the last included, once every
// goroutine has stopped (TestMain's goleak check).
func TestChunkedSinkPanics(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 11))
	dem := newOperand(rng, 20, 18, false, true)
	box := boxKernel{r: 1, inputs: 1}
	errBoom := errors.New("boom")
	for _, workers := range []int{1, 3} {
		for _, n := range []int64{1, 5, 30} { // 30 is the last of 5×6 tiles
			sink := &panickySink{RasterSink: engine.NewMemorySink(raster.NewFloat32Like(dem.r)), n: n, val: errBoom}
			got := func() (r any) {
				defer func() { r = recover() }()
				_ = exec.ProcessChunked(context.Background(), []engine.RasterSink{sink},
					[]engine.RasterSource{engine.NewMemorySource(dem.r)}, box,
					engine.Options{TileWidth: 4, TileHeight: 3, Workers: workers})
				return nil
			}()
			//nolint:errorlint // the panic value must be the sink's own error, not one wrapping it
			if got != errBoom {
				t.Fatalf("workers=%d n=%d: recovered %v, want %v", workers, n, got, errBoom)
			}
		}
	}
}

// TestChunkedLastWriteFails fails the write of the last tile, which is
// claimed after every other: no worker is left to notice it but the
// call itself, which must still return it.
func TestChunkedLastWriteFails(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 12))
	dem := newOperand(rng, 20, 18, false, true)
	errIO := errors.New("last write lost")
	for _, workers := range []int{1, 4} {
		sink := &failingSink{RasterSink: engine.NewMemorySink(raster.NewFloat32Like(dem.r)), n: 5 * 6, err: errIO}
		err := exec.ProcessChunked(context.Background(), []engine.RasterSink{sink},
			[]engine.RasterSource{engine.NewMemorySource(dem.r)}, boxKernel{r: 1, inputs: 1},
			engine.Options{TileWidth: 4, TileHeight: 3, Workers: workers})
		if !errors.Is(err, errIO) {
			t.Fatalf("workers=%d: err = %v, want %v", workers, err, errIO)
		}
		// A cancelled call whose write failed reports the write, as a
		// failed tile comes before cancellation.
		ctx, cancel := context.WithCancel(context.Background())
		sink = &failingSink{RasterSink: engine.NewMemorySink(raster.NewFloat32Like(dem.r)), n: 1, err: errIO}
		cancelling := &cancelSink{RasterSink: sink, cancel: cancel}
		err = exec.ProcessChunked(ctx, []engine.RasterSink{cancelling},
			[]engine.RasterSource{engine.NewMemorySource(dem.r)}, boxKernel{r: 1, inputs: 1},
			engine.Options{TileWidth: 4, TileHeight: 3, Workers: workers})
		if !errors.Is(err, errIO) {
			t.Fatalf("workers=%d, cancelled: err = %v, want %v", workers, err, errIO)
		}
	}
}

// cancelSink cancels a context before its first write.
type cancelSink struct {
	engine.RasterSink
	cancel context.CancelFunc
}

func (s *cancelSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	s.cancel()
	return s.RasterSink.WriteWindow(ctx, src, x, y)
}

// slowSource reads a tile in a tick of virtual time.
type slowSource struct{ engine.RasterSource }

func (s slowSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	time.Sleep(tick)
	return s.RasterSource.ReadWindow(ctx, dst, x, y)
}

// slowSink writes a tile in a tick of virtual time.
type slowSink struct{ engine.RasterSink }

func (s slowSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	time.Sleep(tick)
	return s.RasterSink.WriteWindow(ctx, src, x, y)
}

// TestChunkedWritesBehind checks that a worker reads the next tile while
// the last is written: with reads and writes a tick each, n tiles on one
// worker take n+1 ticks, not 2n. The result is still the plain one.
func TestChunkedWritesBehind(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 13))
	dem := newOperand(rng, 16, 12, false, true)
	want := raster.NewFloat32Like(dem.r)
	naiveBox(want, []raster.Float32Raster{dem.r}, 1, float32(math.NaN()))
	for _, workers := range []int{1, 2} {
		synctest.Test(t, func(t *testing.T) {
			got := raster.NewFloat32Like(dem.r)
			start := time.Now()
			err := exec.ProcessChunked(context.Background(),
				[]engine.RasterSink{slowSink{engine.NewMemorySink(got)}},
				[]engine.RasterSource{slowSource{engine.NewMemorySource(dem.r)}},
				boxKernel{r: 1, inputs: 1}, engine.Options{TileHeight: 1, Workers: workers})
			if err != nil {
				t.Fatal(err)
			}
			rounds := 12 / workers
			if took, want := time.Since(start), time.Duration(rounds+1)*tick; took != want {
				t.Fatalf("workers=%d: %v for %d rounds of tiles, want %v", workers, took, rounds, want)
			}
			for i := range want.Data {
				if math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) {
					t.Fatalf("workers=%d: cell %d is %v, want %v", workers, i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}
