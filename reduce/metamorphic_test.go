package reduce_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// FuzzReduceRelations checks relations between reductions of related
// rasters, each side run through its own execution path, layout, tiling
// and worker count:
//
//   - a permutation of the cells reduces to the same bits, checked with
//     the eight symmetries of the square grid;
//   - Data under an invalid cell never reaches the result;
//   - splitting a raster into two halves and combining their reductions
//     gives the reduction of the whole;
//   - adding an all-invalid border changes nothing;
//   - Count is the popcount of the mask, whatever the tiling;
//   - doubling every cell doubles Sum, Mean and StdDev exactly, since a
//     power-of-two scale commutes with exact arithmetic and with
//     rounding, and doubles Min and Max.
//
// Each relation checks Stats and Sum beside MinMax and Count, except the
// split, whose halves' rounded sums cannot be combined exactly.
//
// Unlike a comparison against a reference these say what a reduction
// means rather than how it is computed, so they fail for an engine that
// visits a cell twice, skips one, reads the wrong one, or lets the answer
// depend on the tiling — the thing DESIGN.md §49 promises it does not.
func FuzzReduceRelations(f *testing.F) {
	for rel := range 6 {
		f.Add([]byte{byte(rel), 1, 1, 20, 6})
		f.Add([]byte{byte(rel), 3, 2, 66, 9, 1})
		f.Add([]byte{byte(rel), 0, 3, 9, 3, 0, 2})
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		reduceRelations(t, fuzzdata.New(data))
	})
}

// result is what a reduction of one raster gives: the extremes and the
// count, and the Summary, as one comparable value.
type result struct {
	mn, mx float32
	count  int64
	s      reduce.Summary
}

func (r result) String() string {
	return fmt.Sprintf("min %v (%#08x) max %v (%#08x) count %d stats %s",
		r.mn, math.Float32bits(r.mn), r.mx, math.Float32bits(r.mx), r.count, fmtSummary(r.s))
}

// equal is bit equality, with two NaNs equal because both are the
// canonical NaN by then (TestNaNIsCanonical pins that separately).
func (r result) equal(o result) bool {
	return r.extremesEqual(o) && sameSummary(r.s, o.s)
}

// extremesEqual compares only what combine can rebuild from two halves.
func (r result) extremesEqual(o result) bool {
	return same(r.mn, o.mn) && same(r.mx, o.mx) && r.count == o.count
}

// reduceRelations decodes a raster and a relation from d and checks it.
// FuzzReduceRelations drives it with a fuzz input, TestReduceRelations
// with rapid.
func reduceRelations(t rastertest.TB, d fuzzdata.Source) {
	rel := d.IntN(6)
	w, h := d.Range(1, 70), d.Range(1, 9)
	masked := rel == 1 || rel == 4 || d.Bool()

	in := raster.NewFloat32(w, h, make([]float32, w*h))
	for j := range in.Data {
		in.Data[j] = d.Float32()
	}
	if masked {
		in.Valid = raster.NewMask(w * h)
		for j := range in.Data {
			raster.MaskSet(in.Valid, j, d.IntN(10) != 0)
		}
	}

	// run reduces a raster through a layout, tiling, worker count and
	// execution path drawn from d, so the two sides of a relation almost
	// never take the same route to their answer.
	run := func(r raster.Float32Raster) result {
		t.Helper()
		placed := rastertest.Place(d, r)
		opts := engine.Options{
			TileWidth:  d.Range(0, r.Width+2),
			TileHeight: d.Range(0, r.Height+2),
			Workers:    d.Range(0, 3),
		}
		var (
			mn, mx float32
			n      int64
			s      reduce.Summary
			sum    float64
			sn     int64
			err    error
			errS   error
			errSum error
		)
		ctx := context.Background()
		switch d.IntN(3) {
		case 0:
			mn, mx, n = reduce.MinMax(placed)
			s = reduce.Stats(placed)
			sum, sn = reduce.Sum(placed)
		case 1:
			mn, mx, n, err = reduce.MinMaxTiled(ctx, placed, opts)
			s, errS = reduce.StatsTiled(ctx, placed, opts)
			sum, sn, errSum = reduce.SumTiled(ctx, placed, opts)
		default:
			src := engine.NewMemorySource(placed)
			mn, mx, n, err = reduce.MinMaxChunked(ctx, src, opts)
			s, errS = reduce.StatsChunked(ctx, src, opts)
			sum, sn, errSum = reduce.SumChunked(ctx, src, opts)
		}
		for _, e := range []error{err, errS, errSum} {
			if e != nil {
				t.Fatal(e)
			}
		}
		// Stats and Sum run reductions of their own and must agree with
		// MinMax about the extremes and with each other about the sum.
		if !same(s.Min, mn) || !same(s.Max, mx) || s.Count != n || !sameF64(sum, s.Sum) || sn != n {
			t.Fatalf("Stats %s and Sum %v %d disagree with MinMax %v %v %d", fmtSummary(s), sum, sn, mn, mx, n)
		}
		// Count runs its own reduction and must agree about how many
		// cells took part.
		c, err := reduce.CountTiled(context.Background(), placed, opts)
		if err != nil {
			t.Fatal(err)
		}
		if c != n {
			t.Fatalf("Count = %d but MinMax counted %d", c, n)
		}
		return result{mn, mx, n, s}
	}

	id := fmt.Sprintf("relation %d %d×%d masked %v", rel, w, h, masked)

	switch rel {
	case 0:
		// A permutation of the cells reduces to the same bits: the
		// guarantee stated directly, since a tiling is one more way of
		// visiting the same cells in a different order.
		want := run(rastertest.Compact(in))
		for _, sym := range rastertest.All() {
			got := run(sym.Apply(in))
			if !got.equal(want) {
				t.Fatalf("%s: %+v gives %v, want %v", id, sym, got, want)
			}
		}
	case 1:
		// Data under an invalid cell is unspecified (DESIGN.md §31), so
		// overwriting every one of them cannot move the answer.
		want := run(rastertest.Compact(in))
		scrambled := rastertest.Compact(in)
		rastertest.ScrambleInvalid(scrambled, d)
		if got := run(scrambled); !got.equal(want) {
			t.Fatalf("%s: scrambling invalid cells gives %v, want %v", id, got, want)
		}
	case 2:
		// Reducing two halves and combining gives the whole, which is
		// what a tiling does and what Combine must therefore be.
		if h < 2 {
			return
		}
		cut := d.Range(1, h-1)
		want := run(rastertest.Compact(in))
		top := run(rastertest.Compact(in.Window(0, 0, w, cut)))
		bottom := run(rastertest.Compact(in.Window(0, cut, w, h-cut)))
		got := combine(top, bottom)
		if !got.extremesEqual(want) {
			t.Fatalf("%s: cut at row %d gives %v, want %v", id, cut, got, want)
		}
		// and the other way round, since Combine is commutative
		if other := combine(bottom, top); !other.extremesEqual(want) {
			t.Fatalf("%s: cut at row %d combined in reverse gives %v, want %v", id, cut, other, want)
		}
	case 3:
		// An all-invalid border adds cells that take no part, so the
		// extremes do not move and the count does not change.
		want := run(rastertest.Compact(in))
		if want.count == 0 {
			return
		}
		bordered := withInvalidBorder(rastertest.Compact(in), d)
		if got := run(bordered); !got.equal(want) {
			t.Fatalf("%s: an all-invalid border gives %v, want %v", id, got, want)
		}
	case 4:
		// Count is the popcount of the mask, computed here without the
		// engine, for every tiling.
		var want int64
		for y := range h {
			for x := range w {
				if in.IsValid(x, y) {
					want++
				}
			}
		}
		got := run(rastertest.Compact(in))
		if got.count != want {
			t.Fatalf("%s: counted %d valid cells, want %d", id, got.count, want)
		}
	case 5:
		// Doubling every cell is exact in float32 unless a finite cell
		// overflows, and scales the exact sum, mean and variance by
		// powers of two, which rounding commutes with.
		doubled := rastertest.Compact(in)
		for y := range h {
			for x := range w {
				v := doubled.Data[doubled.Index(x, y)]
				if in.IsValid(x, y) && !math.IsInf(float64(v), 0) && math.Abs(float64(v)) > math.MaxFloat32/2 {
					return
				}
				doubled.Data[doubled.Index(x, y)] = 2 * v
			}
		}
		want := run(rastertest.Compact(in))
		got := run(doubled)
		w2 := want.s
		scaled := reduce.Summary{
			Count: w2.Count, Sum: 2 * w2.Sum, Mean: 2 * w2.Mean, StdDev: 2 * w2.StdDev,
			Min: 2 * w2.Min, Max: 2 * w2.Max,
		}
		if !sameSummary(got.s, scaled) {
			t.Fatalf("%s: doubling gives %s, want %s", id, fmtSummary(got.s), fmtSummary(scaled))
		}
	}
}

// combine folds two results as the engine folds two partials.
func combine(a, b result) result {
	switch {
	case a.count == 0:
		return b
	case b.count == 0:
		return a
	}
	return result{mn: min(a.mn, b.mn), mx: max(a.mx, b.mx), count: a.count + b.count}
}

// withInvalidBorder returns r in the middle of a raster one cell larger
// on every side, whose border cells are invalid and hold values from d.
func withInvalidBorder(r raster.Float32Raster, d fuzzdata.Source) raster.Float32Raster {
	w, h := r.Width+2, r.Height+2
	out := raster.NewFloat32(w, h, make([]float32, w*h))
	out.Valid = raster.NewMask(w * h)
	for i := range out.Data {
		out.Data[i] = d.Float32()
	}
	for y := range h {
		for x := range w {
			out.SetValid(x, y, false)
		}
	}
	inner := out.Window(1, 1, r.Width, r.Height)
	for y := range r.Height {
		copy(inner.Row(y), r.Row(y))
		for x := range r.Width {
			inner.SetValid(x, y, r.IsValid(x, y))
		}
	}
	return out
}
