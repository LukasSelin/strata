package transfer_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/transfer"
)

// fuzzTable is a table drawn from the fuzz input: xs strictly increasing
// and finite, ys arbitrary. Reclass uses one fewer x than y, Lookup uses
// them in pairs, so one draw serves both.
type fuzzTable struct{ xs, ys []float32 }

func drawTable(d fuzzdata.Source) fuzzTable {
	n := d.Range(1, 8)
	t := fuzzTable{xs: make([]float32, n), ys: make([]float32, n)}
	x := float32(d.Range(-500, 500)) / float32(d.Range(1, 8))
	for i := range t.xs {
		t.xs[i] = x
		next := x + float32(d.Range(1, 500))/float32(d.Range(1, 64))
		if !(next > x) {
			next = math.Float32frombits(math.Float32bits(x) + 1)
		}
		x = next
		t.ys[i] = d.Float32()
	}
	return t
}

// drawPath picks which of the three forms of an operation to run, so
// that agreement between them is fuzzed rather than only tabulated.
// They must write the same bits, which is the package's central claim.
func drawPath(d fuzzdata.Source) int { return d.IntN(3) }

func runPath(t *testing.T, path int, dst, src raster.Float32Raster, run func(dst, src raster.Float32Raster) error,
	tiled func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error,
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error,
	opts engine.Options) {
	t.Helper()
	ctx := context.Background()
	switch path {
	case 0:
		if err := run(dst, src); err != nil {
			t.Fatalf("plain: %v", err)
		}
	case 1:
		if err := tiled(ctx, dst, src, opts); err != nil {
			t.Fatalf("tiled: %v", err)
		}
	default:
		sink := engine.NewMemorySink(dst)
		if err := chunked(ctx, sink, engine.NewMemorySource(src), opts); err != nil {
			t.Fatalf("chunked: %v", err)
		}
	}
}

// FuzzTransfer runs each operation over arbitrary rasters, layouts,
// masks and tables, on a path the input chooses, and requires every cell
// to match the per-cell reference and every cell outside dst to be
// untouched.
func FuzzTransfer(f *testing.F) {
	f.Add([]byte{0, 0, 9, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Add([]byte{1, 1, 17, 2, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3})
	f.Add([]byte{2, 2, 64, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	f.Add([]byte{3, 1, 1, 1, 9, 9, 9, 9, 9, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		op := d.IntN(4)
		path := drawPath(d)
		w, h := d.Range(1, 40), d.Range(1, 6)
		masked := d.Bool()

		src := rastertest.Place(d, rastertest.Output(d, w, h, masked, d.Bool()))
		dst := rastertest.Output(d, w, h, masked, true)
		if path == 2 {
			// A memory sink cannot share memory with a memory source, and
			// the chunked path writes whole tiles, so dst is compact here.
			dst = raster.NewFloat32Like(src)
		}
		for i := range src.Data {
			src.Data[i] = d.Float32()
		}

		tab := drawTable(d)
		opts := engine.Options{TileWidth: d.Range(0, 9), TileHeight: d.Range(0, 5), Workers: d.Range(0, 4)}

		var ref func(float32) float32
		switch op {
		case 0:
			breaks, values := tab.xs[:len(tab.xs)-1], tab.ys
			ref = func(v float32) float32 { return refReclass(v, breaks, values) }
			runPath(t, path, dst, src,
				func(dst, src raster.Float32Raster) error { transfer.Reclass(dst, src, breaks, values); return nil },
				func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
					return transfer.ReclassTiled(c, dst, src, breaks, values, o)
				},
				func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
					return transfer.ReclassChunked(c, dst, src, breaks, values, o)
				}, opts)
		case 1:
			xs, ys := tab.xs, tab.ys
			ref = func(v float32) float32 { return refLookup(v, xs, ys) }
			runPath(t, path, dst, src,
				func(dst, src raster.Float32Raster) error { transfer.Lookup(dst, src, xs, ys); return nil },
				func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
					return transfer.LookupTiled(c, dst, src, xs, ys, o)
				},
				func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
					return transfer.LookupChunked(c, dst, src, xs, ys, o)
				}, opts)
		case 2:
			a, b := d.Float32(), d.Float32()
			ref = func(v float32) float32 { return float32(v*a) + b }
			runPath(t, path, dst, src,
				func(dst, src raster.Float32Raster) error { transfer.Rescale(dst, src, a, b); return nil },
				func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
					return transfer.RescaleTiled(c, dst, src, a, b, o)
				},
				func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
					return transfer.RescaleChunked(c, dst, src, a, b, o)
				}, opts)
		default:
			// Two distinct finite input bounds and two finite output ones,
			// which is what RescaleRange documents as acceptable.
			inLo, inHi := tab.xs[0], tab.xs[len(tab.xs)-1]
			if inLo == inHi {
				return
			}
			outLo, outHi := float32(d.Range(-1000, 1000)), float32(d.Range(-1000, 1000))
			a, b := refCoeffs(inLo, inHi, outLo, outHi)
			if a-a != 0 || b-b != 0 {
				return // the coefficients overflow; the operation panics, which TestPanics covers
			}
			ref = func(v float32) float32 { return float32(v*a) + b }
			runPath(t, path, dst, src,
				func(dst, src raster.Float32Raster) error {
					transfer.RescaleRange(dst, src, inLo, inHi, outLo, outHi)
					return nil
				},
				func(c context.Context, dst, src raster.Float32Raster, o engine.Options) error {
					return transfer.RescaleRangeTiled(c, dst, src, inLo, inHi, outLo, outHi, o)
				},
				func(c context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
					return transfer.RescaleRangeChunked(c, dst, src, inLo, inHi, outLo, outHi, o)
				}, opts)
		}

		for y := range dst.Height {
			for x := range dst.Width {
				want := ref(src.Data[src.Index(x, y)])
				if got := dst.Data[dst.Index(x, y)]; !sameFloat(got, want) {
					t.Fatalf("op=%d path=%d %dx%d cell (%d, %d) from %v = %v (%#08x), want %v (%#08x)",
						op, path, w, h, x, y, src.Data[src.Index(x, y)],
						got, math.Float32bits(got), want, math.Float32bits(want))
				}
				if v := dst.IsValid(x, y); v != src.IsValid(x, y) {
					t.Fatalf("op=%d path=%d cell (%d, %d) valid = %v, want %v", op, path, x, y, v, src.IsValid(x, y))
				}
			}
		}
	})
}

// FuzzTransferPanics requires every rejected table to panic with this
// package's own message, on every path, before any cell is written. A
// runtime error such as an index out of range would fail here, which is
// what keeps the table checks ahead of the kernels.
func FuzzTransferPanics(f *testing.F) {
	f.Add([]byte{0, 3, 1, 2, 3, 4, 5})
	f.Add([]byte{1, 2, 9, 9, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		n := d.Range(0, 6)
		xs := make([]float32, n)
		ys := make([]float32, d.Range(0, 7))
		for i := range xs {
			xs[i] = d.Float32()
		}
		for i := range ys {
			ys[i] = d.Float32()
		}

		const w, h = 8, 2
		src := raster.NewFloat32(w, h, make([]float32, w*h))
		dst := raster.NewFloat32(w, h, make([]float32, w*h))
		for i := range dst.Data {
			dst.Data[i] = 12345
		}

		lookup := d.Bool()
		ok := validTable(xs, ys, lookup)
		err := recovered(func() {
			if lookup {
				transfer.Lookup(dst, src, xs, ys)
			} else {
				transfer.Reclass(dst, src, xs, ys)
			}
		})
		switch {
		case ok && err != "":
			t.Fatalf("lookup=%v xs=%v ys=%v: valid table panicked: %s", lookup, xs, ys, err)
		case !ok && err == "":
			t.Fatalf("lookup=%v xs=%v ys=%v: invalid table did not panic", lookup, xs, ys)
		case !ok:
			if !strings.HasPrefix(err, "transfer.") {
				t.Fatalf("panic %q, want a transfer. message", err)
			}
			for i, v := range dst.Data {
				if v != 12345 {
					t.Fatalf("cell %d was written before the table was rejected: %v", i, v)
				}
			}
		}
	})
}

// validTable restates the documented table rules, independently of the
// checks in the package.
func validTable(xs, ys []float32, lookup bool) bool {
	if lookup {
		if len(xs) != len(ys) || len(xs) == 0 {
			return false
		}
	} else if len(ys) != len(xs)+1 {
		return false
	}
	for _, x := range xs {
		if x != x {
			return false
		}
		if lookup && x-x != 0 {
			return false
		}
	}
	for i := 1; i < len(xs); i++ {
		if !(xs[i-1] < xs[i]) {
			return false
		}
	}
	return true
}

// recovered runs f and returns the panic message, or "" if it did not
// panic. A non-string panic is returned with its value, so the caller's
// prefix check fails on it.
func recovered(f func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			s, ok := r.(string)
			if !ok {
				s = "non-string panic: " + fmtValue(r)
			}
			msg = s
		}
	}()
	f()
	return ""
}

func fmtValue(v any) string {
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return "unknown"
}
