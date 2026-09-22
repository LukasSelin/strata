package resample_test

import (
	"context"
	"errors"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// TestBackendsMatch runs every method through the public API on both
// backends and requires the same bits, masks and all: the passes are
// checked lane by lane in internal/resamp, and this checks the band
// driver feeds them the same way on either.
func TestBackendsMatch(t *testing.T) {
	if resamp.Backend() == "scalar" {
		t.Skip("no SIMD backend in this build")
	}
	defer resamp.UseScalar(false)
	rng := rand.New(rand.NewPCG(13, 17))
	for iter := range 80 {
		dg, sg := grids(rng)
		src := raster.NewDataset(sg, randomRaster(rng, sg.Width, sg.Height, iter%2 == 0))
		for _, m := range methods {
			resamp.UseScalar(true)
			want := newDst(dg)
			resample.Resample(want, src, resample.Options{Method: m})
			resamp.UseScalar(false)
			got := newDst(dg)
			resample.Resample(got, src, resample.Options{Method: m})
			sameBits(t, resamp.Backend(), m, engine.Options{}, want.Raster, got.Raster)
		}
	}
}

// recordingSource is a MemorySource that records the windows it reads
// and fails its failAt-th read (1-based; 0 never).
type recordingSource struct {
	*engine.MemorySource
	mu     sync.Mutex
	maxW   int
	maxH   int
	reads  atomic.Int64
	failAt int64
}

var errRead = errors.New("disk on fire")

func (s *recordingSource) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	if n := s.reads.Add(1); n == s.failAt {
		return errRead
	}
	s.mu.Lock()
	s.maxW, s.maxH = max(s.maxW, dst.Width), max(s.maxH, dst.Height)
	s.mu.Unlock()
	return s.MemorySource.ReadWindow(ctx, dst, x, y)
}

// TestChunkedReadsFootprints: a Chunked call reads, for each tile, only
// the source window its cells reach, so that downsampling by 4 with 32×32
// tiles reads windows of about 128 cells plus the kernel's reach, not the
// whole source; and a source error comes back wrapped, with a prefix of
// whole tiles written.
func TestChunkedReadsFootprints(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 3))
	sg := raster.Grid{Width: 512, Height: 512, ResolutionX: 1, ResolutionY: -1, OriginY: 512}
	src := randomRaster(rng, 512, 512, true)
	dg := raster.Grid{Width: 128, Height: 128, ResolutionX: 4, ResolutionY: -4, OriginY: 512}
	for _, m := range methods {
		rs := &recordingSource{MemorySource: engine.NewMemorySource(src)}
		dst := newDst(dg)
		var st engine.Stats
		err := resample.ResampleChunked(t.Context(), engine.NewMemorySink(dst.Raster), dg, rs, sg, resample.Options{Method: m},
			engine.Options{TileWidth: 32, TileHeight: 32, Workers: 3, Stats: &st})
		if err != nil {
			t.Fatal(err)
		}
		// Lanczos at 4× reaches 12 source cells either side.
		if rs.maxW > 32*4+26 || rs.maxH > 32*4+26 {
			t.Errorf("%v: read windows up to %d×%d for 32×32 tiles", m, rs.maxW, rs.maxH)
		}
		if st.Tiles != 16 || st.Cells != 128*128 || st.SourceRead == 0 || st.SinkWritten != 128*128*4 {
			t.Errorf("%v: stats %+v", m, st)
		}
		want := newDst(dg)
		resample.Resample(want, raster.NewDataset(sg, src), resample.Options{Method: m})
		sameBits(t, "Chunked", m, engine.Options{}, want.Raster, dst.Raster)

		rs = &recordingSource{MemorySource: engine.NewMemorySource(src), failAt: 5}
		err = resample.ResampleChunked(t.Context(), engine.NewMemorySink(newDst(dg).Raster), dg, rs, sg, resample.Options{Method: m},
			engine.Options{TileWidth: 32, TileHeight: 32, Workers: 2})
		if !errors.Is(err, errRead) {
			t.Errorf("%v: err = %v, want the source's", m, err)
		}
	}
}

// TestCancel: a cancelled call returns ctx.Err() and a Tiled call stops
// claiming bands.
func TestCancel(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	sg := raster.Grid{Width: 300, Height: 300, ResolutionX: 1, ResolutionY: -1, OriginY: 300}
	src := raster.NewDataset(sg, randomRaster(rng, 300, 300, false))
	dg := raster.Grid{Width: 600, Height: 600, ResolutionX: 0.5, ResolutionY: -0.5, OriginY: 300}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	var st engine.Stats
	err := resample.ResampleTiled(ctx, newDst(dg), src, resample.Options{Method: resample.Cubic}, engine.Options{Workers: 2, Stats: &st})
	if !errors.Is(err, context.Canceled) || st.Cells != 0 {
		t.Fatalf("Tiled: err %v, %d cells written", err, st.Cells)
	}
	err = resample.ResampleChunked(ctx, engine.NewMemorySink(newDst(dg).Raster), dg, engine.NewMemorySource(src.Raster), sg,
		resample.Options{Method: resample.Cubic}, engine.Options{TileHeight: 50})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Chunked: err %v", err)
	}
}

// TestStatsTiled: a Tiled call counts every output cell once and reads
// at least the source cells its footprint covers.
func TestStatsTiled(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 5))
	sg := raster.Grid{Width: 200, Height: 150, ResolutionX: 1, ResolutionY: -1, OriginY: 150}
	src := raster.NewDataset(sg, randomRaster(rng, 200, 150, false))
	dg := raster.Grid{Width: 100, Height: 75, ResolutionX: 2, ResolutionY: -2, OriginY: 150}
	var st engine.Stats
	if err := resample.ResampleTiled(t.Context(), newDst(dg), src, resample.Options{Method: resample.Bilinear},
		engine.Options{TileWidth: 30, Workers: 3, Stats: &st}); err != nil {
		t.Fatal(err)
	}
	if st.Cells != 100*75 || st.KernelWritten != st.Cells*4 || st.KernelRead < 200*150*4 || st.Bands == 0 {
		t.Fatalf("stats %+v", st)
	}
}
