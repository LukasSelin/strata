package reduce_test

import (
	"context"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// FuzzReduce compares every execution path against the naive reference —
// a plain loop over the cells with Go's builtins — over arbitrary values,
// masks, layouts, tilings and worker counts. The reference is the
// definition of the answer, so this catches a reduction that is wrong in
// the same way twice, which the relations of metamorphic_test.go cannot.
func FuzzReduce(f *testing.F) {
	f.Add([]byte{1, 1, 0, 0, 0})
	f.Add([]byte{9, 3, 1, 2, 7, 4})
	f.Add([]byte{64, 5, 0, 200, 17, 3, 1})
	f.Add([]byte{17, 1, 1, 9, 9, 9, 9, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		w, h := d.Range(1, 70), d.Range(1, 9)
		masked := d.Bool()

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
		// Data under an invalid cell must never reach the result, so
		// fill it with something a correct reduction cannot use.
		rastertest.ScrambleInvalid(in, d)
		placed := rastertest.Place(d, in)

		wantMn, wantMx, wantN := ref(placed)
		opts := engine.Options{
			TileWidth:  d.Range(0, w+2),
			TileHeight: d.Range(0, h+2),
			Workers:    d.Range(0, 3),
		}
		ctx := context.Background()
		src := engine.NewMemorySource(placed)

		check := func(path string, mn, mx float32, n int64, err error) {
			t.Helper()
			if err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			if !same(mn, wantMn) || !same(mx, wantMx) || n != wantN {
				t.Fatalf("%s %+v: %v %v %d, want %v %v %d",
					path, opts, mn, mx, n, wantMn, wantMx, wantN)
			}
		}
		mn, mx, n := reduce.MinMax(placed)
		check("plain", mn, mx, n, nil)
		mn, mx, n, err := reduce.MinMaxTiled(ctx, placed, opts)
		check("tiled", mn, mx, n, err)
		mn, mx, n, err = reduce.MinMaxChunked(ctx, src, opts)
		check("chunked", mn, mx, n, err)

		if c := reduce.Count(placed); c != wantN {
			t.Fatalf("Count = %d, want %d", c, wantN)
		}
		c, err := reduce.CountTiled(ctx, placed, opts)
		if err != nil || c != wantN {
			t.Fatalf("CountTiled %+v = %d, %v, want %d", opts, c, err, wantN)
		}
		c, err = reduce.CountChunked(ctx, src, opts)
		if err != nil || c != wantN {
			t.Fatalf("CountChunked %+v = %d, %v, want %d", opts, c, err, wantN)
		}
	})
}
