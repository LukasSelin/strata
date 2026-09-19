package transfer_test

import (
	"context"
	"math"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/transfer"
)

// The relations here hold between results rather than pinning values, so
// they need no reference implementation and survive changes to the
// kernels. Each side runs through the plain, Tiled or Chunked form as
// the input chooses, which is how the guarantee that the three agree
// gets fuzzed rather than only tabulated.

// apply runs one operation on src placed in a fresh layout, through a
// path the input chooses, and returns a compact copy of the result.
func apply(t rastertest.TB, d fuzzdata.Source, src raster.Float32Raster, run relOp) raster.Float32Raster {
	placed := rastertest.Place(d, src)
	w, h := src.Width, src.Height
	masked := src.Valid != nil
	dst := rastertest.Output(d, w, h, masked, true)
	opts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 3)}
	ctx := context.Background()

	var err error
	switch d.IntN(3) {
	case 0:
		run.plain(dst, placed)
	case 1:
		err = run.tiled(ctx, dst, placed, opts)
	default:
		// The chunked path writes whole tiles into a sink of its own.
		dst = raster.NewFloat32Like(placed)
		sink := engine.NewMemorySink(dst)
		err = run.chunked(ctx, sink, engine.NewMemorySource(placed), opts)
	}
	if err != nil {
		t.Fatalf("%s: %v", run.name, err)
	}
	return rastertest.Compact(dst)
}

type relOp struct {
	name    string
	plain   func(dst, src raster.Float32Raster)
	tiled   func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error
}

func reclassOp(breaks, values []float32) relOp {
	return relOp{
		name:  "Reclass",
		plain: func(dst, src raster.Float32Raster) { transfer.Reclass(dst, src, breaks, values) },
		tiled: func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.ReclassTiled(c, dst, src, breaks, values, o)
		},
		chunked: func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.ReclassChunked(c, dst, src, breaks, values, o)
		},
	}
}

func lookupOp(xs, ys []float32) relOp {
	return relOp{
		name:  "Lookup",
		plain: func(dst, src raster.Float32Raster) { transfer.Lookup(dst, src, xs, ys) },
		tiled: func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.LookupTiled(c, dst, src, xs, ys, o)
		},
		chunked: func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.LookupChunked(c, dst, src, xs, ys, o)
		},
	}
}

func rescaleOp(a, b float32) relOp {
	return relOp{
		name:  "Rescale",
		plain: func(dst, src raster.Float32Raster) { transfer.Rescale(dst, src, a, b) },
		tiled: func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.RescaleTiled(c, dst, src, a, b, o)
		},
		chunked: func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.RescaleChunked(c, dst, src, a, b, o)
		},
	}
}

func equalRasters(t rastertest.TB, what string, got, want raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			g, w := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]
			if !sameFloat(g, w) {
				t.Fatalf("%s: cell (%d, %d) = %v (%#08x), want %v (%#08x)",
					what, x, y, g, math.Float32bits(g), w, math.Float32bits(w))
			}
			if gv, wv := got.IsValid(x, y), want.IsValid(x, y); gv != wv {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", what, x, y, gv, wv)
			}
		}
	}
}

func transferRelations(t rastertest.TB, d fuzzdata.Source) {
	w, h := d.Range(1, 30), d.Range(1, 5)
	masked := d.Bool()
	src := rastertest.Output(d, w, h, masked, true)
	for i := range src.Data {
		src.Data[i] = d.Float32()
	}
	tab := drawTable(d)
	xs, ys := tab.xs, tab.ys
	breaks, values := xs[:len(xs)-1], ys

	// 1. Every class is one of the values, or the cell's own NaN. The
	// operation cannot invent a value, which is the whole contract of a
	// reclassification.
	got := apply(t, d, src, reclassOp(breaks, values))
	for y := range h {
		for x := range w {
			v := got.Data[got.Index(x, y)]
			if v != v {
				continue // NaN in, NaN out
			}
			found := false
			for _, want := range values {
				if sameFloat(v, want) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("Reclass produced %v (%#08x), which is not in values %v", v, math.Float32bits(v), values)
			}
		}
	}

	// 2. A curve with a monotone ys is bounded by its own knots: no cell
	// can leave the range the table spans, however the cells fall.
	// Sorting a copy of ys makes the relation checkable without
	// constraining the table the other relations use.
	//
	// The guard is not squeamishness. The segment value is
	// y0 + t*(y1-y0), so a rise that overflows float32 gives an infinity
	// between two finite knots -- FuzzTransferRelations found it with a
	// table spanning -1.79e38 to 3.40e38. Requiring the whole span to be
	// finite bounds every adjacent rise, since ys is sorted here.
	// Every y must be finite before the sort, not after: a NaN compares
	// false against everything, so an insertion sort leaves the slice
	// unsorted around it and the "range" read off its ends is nonsense.
	// TestTransferRelations caught exactly that, reporting a range of
	// [1000, 0].
	sorted := append([]float32(nil), ys...)
	allFinite := true
	for _, v := range sorted {
		if v-v != 0 {
			allFinite = false
		}
	}
	if allFinite {
		for i := 1; i < len(sorted); i++ {
			for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
				sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
			}
		}
	}
	lo, hi := sorted[0], sorted[len(sorted)-1]
	if allFinite && hi-lo-(hi-lo) == 0 {
		bounded := apply(t, d, src, lookupOp(xs, sorted))
		for y := range h {
			for x := range w {
				v := bounded.Data[bounded.Index(x, y)]
				if v != v {
					continue
				}
				if v < lo || v > hi {
					t.Fatalf("Lookup produced %v outside the table's range [%v, %v]", v, lo, hi)
				}
			}
		}
	}

	// 3. Both operations are pointwise, so they commute with every grid
	// symmetry: transforming the input and then computing is the same as
	// computing and then transforming. This is what catches an indexing
	// mistake that a per-cell reference would share.
	tr := rastertest.All()[d.IntN(len(rastertest.All()))]
	for _, op := range []relOp{reclassOp(breaks, values), lookupOp(xs, ys), rescaleOp(d.Float32(), d.Float32())} {
		first := rastertest.Compact(tr.Apply(apply(t, d, src, op)))
		second := apply(t, d, rastertest.Compact(tr.Apply(src)), op)
		equalRasters(t, op.name+" under "+symmetryName(tr), second, first)
	}

	// 4. The two operations meet on the knots. A curve whose ys are its
	// knot indices returns i at xs[i]; Reclass over the same breaks
	// counts the breaks at or below the cell, so it returns values[i+1],
	// and setting values to {0} followed by the indices makes the two
	// agree at every knot and beyond both ends. Between knots they
	// differ, which is the difference between interpolating and not.
	index := make([]float32, len(xs))
	for i := range index {
		index[i] = float32(i)
	}
	onKnots := raster.NewFloat32(len(xs)+2, 1,
		append(append([]float32{xs[0] - 1}, xs...), xs[len(xs)-1]+1))
	viaLookup := apply(t, d, onKnots, lookupOp(xs, index))
	viaReclass := apply(t, d, onKnots, reclassOp(xs, append([]float32{0}, index...)))
	for i := range onKnots.Data {
		l, r := viaLookup.Data[i], viaReclass.Data[i]
		if !sameFloat(l, r) {
			t.Fatalf("at %v: Lookup gave index %v but Reclass gave class %v", onKnots.Data[i], l, r)
		}
	}

	// 5. Rescale with a unit scale and a negative zero offset is the bit
	// identity, which pins that the addition is real arithmetic rather
	// than an elided no-op.
	negZeroOffset := apply(t, d, src, rescaleOp(1, negZero))
	equalRasters(t, "Rescale by 1 and -0", negZeroOffset, rastertest.Compact(src))
}

func symmetryName(t rastertest.Dihedral) string {
	s := ""
	if t.Swap {
		s += "swap"
	}
	if t.FlipX {
		s += "flipX"
	}
	if t.FlipY {
		s += "flipY"
	}
	if s == "" {
		s = "identity"
	}
	return s
}

// FuzzTransferRelations drives the relations from a fuzz corpus.
func FuzzTransferRelations(f *testing.F) {
	f.Add([]byte{9, 2, 0, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	f.Add([]byte{1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9})
	f.Add([]byte{17, 4, 1, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	f.Fuzz(func(t *testing.T, data []byte) {
		transferRelations(t, fuzzdata.New(data))
	})
}
