package exec_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// clusterMask gives op a mask whose invalid cells are only those inside
// the w×h block at (x, y); every other cell of its window is valid. Bits
// outside the window stay random.
func clusterMask(rng *rand.Rand, op operand, x, y, w, h int) {
	r := op.r
	for yy := range r.Height {
		for xx := range r.Width {
			in := xx >= x && xx < x+w && yy >= y && yy < y+h
			raster.MaskSet(r.Valid, r.ValidOffset+yy*r.Stride+xx, !in || rng.IntN(3) != 0)
		}
	}
}

// TestChunkedAllValidTiles runs chunked operations over masked sources
// whose invalid cells sit in one small block, so that most tiles, halo
// included, are all valid and the engine drops their masks, and the
// tiles over the block, or whose halo reaches it, keep them. A second
// input, where the operation has one, is masked and all valid. The sinks
// must hold the plain function's bits: dropping an all-valid mask may
// change the work, never the output.
func TestChunkedAllValidTiles(t *testing.T) {
	names := []string{"box-r2", "clamp", "add", "mul", "slope-degrees", "gradient", "focal-max-r4", "focal-separable-r2"}
	shapes := []struct {
		name     string
		w, h     int
		windowed bool
	}{
		{"compact", 150, 70, false},
		{"window", 131, 67, true},
	}
	runs := []engine.Options{
		{TileWidth: 16, TileHeight: 16, Workers: 1},
		{TileWidth: 64, TileHeight: 9, Workers: 3},
		{TileHeight: 20, Workers: 2},
		{Workers: 1},
	}
	for _, name := range names {
		op := box2Adapter
		if name != op.name {
			op = adapterNamed(t, name)
		}
		f := chunked[name]
		t.Run(name, func(t *testing.T) {
			for _, sh := range shapes {
				rng := rand.New(rand.NewPCG(uint64(sh.w), 17))
				ins, outs := newOperands(rng, op, sh.w, sh.h, layout{sh.windowed, sh.windowed}, true, true, -1)
				clusterMask(rng, ins[0], 40, 30, 5, 4)
				for _, in := range ins[1:] {
					if in.r.Valid != nil {
						clusterMask(rng, in, 0, 0, 0, 0)
					}
				}
				want := cloneAll(ins, outs, false)
				op.direct(want.dst(), want.src())
				for _, o := range runs {
					id := fmt.Sprintf("%s %+v", sh.name, o)
					got := cloneAll(ins, outs, false)
					processWith(t, engineRun{opts: o}, func(ctx context.Context, o engine.Options) error {
						return f(ctx, memorySinks(got.outs), memorySources(got.ins), o)
					})
					requireSameOperands(t, id, got, want)
				}
			}
		})
	}
}

// maskedFolds is tallyOp that also counts the folds made without masks.
type maskedFolds struct {
	tallyOp
	unmasked *int64
}

func (m maskedFolds) Fold(p *tally, c exec.Cells) {
	if !c.Masked {
		*m.unmasked++
	}
	m.tallyOp.Fold(p, c)
}

// TestReduceChunkedAllValidTiles is TestChunkedAllValidTiles for folds:
// the result must be the cell-by-cell one, and tiles clear of the invalid
// block must fold unmasked, which is what the fast path is for.
func TestReduceChunkedAllValidTiles(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	for _, windowed := range []bool{false, true} {
		a := newOperandPad(rng, 150, 70, windowed, true, -1)
		b := newOperandPad(rng, 150, 70, windowed, true, -1)
		clusterMask(rng, a, 100, 50, 3, 3)
		clusterMask(rng, b, 0, 0, 0, 0)
		src := []raster.Float32Raster{a.r, b.r}
		want := wantTally(src)
		for _, o := range []engine.Options{{TileWidth: 32, TileHeight: 32, Workers: 1}, {TileHeight: 10, Workers: 1}} {
			var unmasked int64
			got := runReduceChunked(t, engineRun{opts: o},
				[]engine.RasterSource{engine.NewMemorySource(a.r), engine.NewMemorySource(b.r)},
				maskedFolds{tallyOp{2}, &unmasked})
			if got != want {
				t.Fatalf("windowed=%v %+v: got %+v, want %+v", windowed, o, got, want)
			}
			if unmasked == 0 {
				t.Fatalf("windowed=%v %+v: every fold was masked; the all-valid tiles kept their masks", windowed, o)
			}
		}
	}
}
