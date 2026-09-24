package terrain

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// weightedOperand returns a w×h raster of values around base, some of
// them NaN or ±Inf, as a window of a larger root when windowed, with
// about 1 cell in 12 invalid when masked.
func weightedOperand(rng *rand.Rand, w, h int, base float64, windowed, masked bool) raster.Float32Raster {
	x, y, rw, rh := 0, 0, w, h
	if windowed {
		x, y, rw, rh = 2, 1, w+5, h+3
	}
	root := raster.NewFloat32(rw, rh, make([]float32, rw*rh))
	hazards := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0}
	for i := range root.Data {
		root.Data[i] = float32(base + base/10*rng.NormFloat64())
		if rng.IntN(60) == 0 {
			root.Data[i] = hazards[rng.IntN(len(hazards))]
		}
	}
	if masked {
		root.Valid = make([]uint64, raster.MaskWords(len(root.Data)))
		for i := range root.Data {
			raster.MaskSet(root.Valid, i, rng.IntN(12) != 0)
		}
	}
	return root.Window(x, y, w, h)
}

// sameCells fails unless got and want hold the same Data bits (any NaN
// matching any NaN) and the same validity, cell by cell.
func sameCells(t *testing.T, id string, got, want raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			g, w := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]
			if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
				t.Fatalf("%s: Data at (%d, %d) = %v, want %v", id, x, y, g, w)
			}
			if want.Valid != nil && got.IsValid(x, y) != want.IsValid(x, y) {
				t.Fatalf("%s: validity at (%d, %d) = %v, want %v", id, x, y, got.IsValid(x, y), want.IsValid(x, y))
			}
		}
	}
}

// TestWeightedSlopeIsSlopeTimesWeight checks the promise WeightedSlope
// makes: the bits of Slope followed by algebra.Mul, Data and validity,
// for every entry point, tiling and worker count.
func TestWeightedSlopeIsSlopeTimesWeight(t *testing.T) {
	const w, h = 70, 41
	opts := SlopeOptions{CellSize: 12.5, CellSizeY: 10, ZFactor: 2}
	runs := []engine.Options{
		{Workers: 1},
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 7, TileHeight: 3, Workers: 2},
		{TileWidth: 64, TileHeight: 1},
	}
	for _, units := range []SlopeUnits{SlopeDegrees, SlopePercent} {
		opts.Units = units
		for _, masked := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
			for _, windowed := range []bool{false, true} {
				rng := rand.New(rand.NewPCG(uint64(units), 41))
				dem := weightedOperand(rng, w, h, 800, windowed, masked[0])
				weight := weightedOperand(rng, w, h, 1, !windowed, masked[1])
				anyMask := masked[0] || masked[1]

				newOut := func() raster.Float32Raster {
					out := raster.NewFloat32(w, h, make([]float32, w*h))
					if anyMask {
						out.Valid = make([]uint64, raster.MaskWords(w*h))
					}
					return out
				}
				slope, want := newOut(), newOut()
				Slope(slope, dem, opts)
				algebra.Mul(want, slope, weight)

				id := fmt.Sprintf("units=%d masked=%v windowed=%v", units, masked, windowed)
				got := newOut()
				WeightedSlope(got, dem, weight, opts)
				sameCells(t, id+" plain", got, want)
				for _, eo := range runs {
					got := newOut()
					if err := WeightedSlopeTiled(context.Background(), got, dem, weight, opts, eo); err != nil {
						t.Fatal(err)
					}
					sameCells(t, fmt.Sprintf("%s tiled %+v", id, eo), got, want)

					got = newOut()
					err := WeightedSlopeChunked(context.Background(), engine.NewMemorySink(got),
						engine.NewMemorySource(dem), engine.NewMemorySource(weight), opts, eo)
					if err != nil {
						t.Fatal(err)
					}
					sameCells(t, fmt.Sprintf("%s chunked %+v", id, eo), got, want)
				}
			}
		}
	}
}

// TestWeightedSlopeValidity checks the validity rule against a per-cell
// reference: the 3×3 of dem and the cell itself of weight. An invalid
// weight must not invalidate its neighbours, which is the rule a single
// erosion over both inputs would break.
func TestWeightedSlopeValidity(t *testing.T) {
	const w, h = 66, 9
	rng := rand.New(rand.NewPCG(3, 3))
	dem := weightedOperand(rng, w, h, 500, true, true)
	weight := weightedOperand(rng, w, h, 1, false, true)
	out := raster.NewFloat32(w, h, make([]float32, w*h))
	out.Valid = make([]uint64, raster.MaskWords(w*h))
	WeightedSlope(out, dem, weight, SlopeOptions{CellSize: 1})
	for y := range h {
		for x := range w {
			want := !isBorder(out, x, y) && weight.IsValid(x, y)
			for dy := -1; want && dy <= 1; dy++ {
				for dx := -1; want && dx <= 1; dx++ {
					want = dem.IsValid(x+dx, y+dy)
				}
			}
			if got := out.IsValid(x, y); got != want {
				t.Fatalf("cell (%d, %d) valid=%v, want %v", x, y, got, want)
			}
		}
	}
	checkBorderNaN(t, "WeightedSlope", out)
}

func TestWeightedSlopePanics(t *testing.T) {
	r := raster.NewFloat32(5, 5, make([]float32, 25))
	small := raster.NewFloat32(4, 5, make([]float32, 20))
	masked := raster.NewFloat32(5, 5, make([]float32, 25))
	masked.Valid = make([]uint64, 1)
	mustPanic(t, "sizes", func() { WeightedSlope(raster.NewFloat32Like(r), r, small, SlopeOptions{CellSize: 1}) })
	mustPanic(t, "overlap", func() { WeightedSlope(r, r, raster.NewFloat32Like(r), SlopeOptions{CellSize: 1}) })
	mustPanic(t, "mask", func() { WeightedSlope(raster.NewFloat32Like(r), r, masked, SlopeOptions{CellSize: 1}) })
	mustPanic(t, "cell size", func() { WeightedSlope(raster.NewFloat32Like(r), r, raster.NewFloat32Like(r), SlopeOptions{}) })
}

// TestWeightedSlopeChunkedAllValidTiles runs WeightedSlopeChunked over a
// dem and a weight whose invalid cells sit in two separate blocks, in
// tiles small enough that some tiles drop the dem's mask only, some the
// weight's, some both and some neither (the chunked driver drops the
// mask of an all-valid tile buffer). The inputs have different reaches,
// 1 and 0, so each tile's validity rule has to be derived from the masks
// it kept. The result must be the plain function's, bit for bit.
func TestWeightedSlopeChunkedAllValidTiles(t *testing.T) {
	const w, h = 96, 40
	rng := rand.New(rand.NewPCG(9, 9))
	dem := weightedOperand(rng, w, h, 800, false, true)
	weight := weightedOperand(rng, w, h, 1, true, true)
	cluster := func(r raster.Float32Raster, x0, y0, bw, bh int) {
		for y := range r.Height {
			for x := range r.Width {
				in := x >= x0 && x < x0+bw && y >= y0 && y < y0+bh
				raster.MaskSet(r.Valid, r.ValidOffset+y*r.Stride+x, !in || rng.IntN(3) != 0)
			}
		}
	}
	cluster(dem, 10, 5, 6, 4)
	cluster(weight, 60, 25, 5, 5)
	opts := SlopeOptions{CellSize: 12.5}
	newOut := func() raster.Float32Raster {
		out := raster.NewFloat32(w, h, make([]float32, w*h))
		out.Valid = make([]uint64, raster.MaskWords(w*h))
		return out
	}
	want := newOut()
	WeightedSlope(want, dem, weight, opts)
	for _, eo := range []engine.Options{
		{TileWidth: 16, TileHeight: 8, Workers: 1},
		{TileWidth: 24, TileHeight: 10, Workers: 3},
		{TileHeight: 4, Workers: 2},
	} {
		got := newOut()
		if err := WeightedSlopeChunked(context.Background(), engine.NewMemorySink(got),
			engine.NewMemorySource(dem), engine.NewMemorySource(weight), opts, eo); err != nil {
			t.Fatal(err)
		}
		sameCells(t, fmt.Sprintf("chunked %+v", eo), got, want)
	}
}
