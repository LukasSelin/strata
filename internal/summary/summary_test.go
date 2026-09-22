package summary

import (
	"context"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// collect is a reducer whose partial is every cell Runs hands over, so a
// test can see exactly what reached the fold.
type collect struct{}

type cells struct {
	got  []float32
	runs Runs
	// longest is the longest run handed over, to check packing.
	longest int
}

func (collect) Inputs() int { return 1 }
func (collect) Combine(a *cells, b cells) {
	a.got = append(a.got, b.got...)
	a.longest = max(a.longest, b.longest)
}
func (collect) Fold(p *cells, c exec.Cells) {
	p.runs.Each(c, func(run []float32) {
		p.got = append(p.got, run...)
		p.longest = max(p.longest, len(run))
	})
}

// TestRunsEachValidCellOnce gives every cell a distinct value and every
// invalid cell a negative one, and requires Runs to hand over each valid
// cell exactly once and no invalid cell at all, for masks from sparse to
// dense and tilings that cut words in different places.
func TestRunsEachValidCellOnce(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	for _, invalid := range []int{0, 1, 3, 50, 97, 100} { // percent
		for _, w := range []int{1, 63, 64, 65, 200, 301} {
			h := 1 + rng.IntN(20)
			r := raster.NewFloat32(w, h, make([]float32, w*h))
			r.Valid = raster.NewMask(w * h)
			var want []float32
			for i := range r.Data {
				ok := rng.IntN(100) >= invalid
				raster.MaskSet(r.Valid, i, ok)
				if ok {
					r.Data[i] = float32(i)
					want = append(want, float32(i))
				} else {
					r.Data[i] = -1
				}
			}
			for _, opts := range []engine.Options{{Workers: 1}, {TileWidth: 7, TileHeight: 3, Workers: 3}, {TileWidth: 64, Workers: 2}} {
				p, err := exec.Reduce(context.Background(), []raster.Float32Raster{r}, collect{}, opts)
				if err != nil {
					t.Fatal(err)
				}
				slices.Sort(p.got)
				if !slices.Equal(p.got, want) {
					t.Fatalf("invalid %d%% %d×%d %+v: got %d cells, want %d", invalid, w, h, opts, len(p.got), len(want))
				}
				if p.longest > max(w*h, runCells) {
					t.Fatalf("a run of %d cells", p.longest)
				}
			}
		}
	}
}

// TestRunsPacks checks that a sparse mask still reaches the fold in runs
// of close to the buffer's size, which is what makes the vector kernels
// apply to it.
func TestRunsPacks(t *testing.T) {
	w, h := 1024, 16
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	for i := range r.Data {
		raster.MaskSet(r.Valid, i, i%10 != 0) // no word is wholly valid
	}
	p, err := exec.Reduce(context.Background(), []raster.Float32Raster{r}, collect{}, engine.Options{Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	if p.longest <= runCells-wordBits {
		t.Fatalf("longest run %d, want more than %d", p.longest, runCells-wordBits)
	}
}
