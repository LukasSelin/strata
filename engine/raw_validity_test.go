package engine_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// TestRawValidityWide reads rows wider than a mask word through
// RawSource, on the scalar and the SIMD kernels, into compact buffers
// (whole words written straight from the kernel) and into windows whose
// bits start mid-word (a head and a tail through scratch), and writes
// masked rows back through RawSink. Validity must follow RawOptions.Fill
// cell by cell: == as floats, so -0 matches 0, and any NaN for a NaN
// fill. Nothing outside a window may change, and the sink must store the
// fill under exactly the invalid cells.
func TestRawValidityWide(t *testing.T) {
	defer vec.UseScalar(false)
	rng := rand.New(rand.NewPCG(8, 21))
	const w, h = 301, 5
	for _, fill := range []float32{-9999, 0, float32(math.Copysign(0, -1)), float32(math.NaN()),
		math.Float32frombits(0xffc0_0123), float32(math.Inf(1))} {
		data := raster.NewFloat32(w, h, make([]float32, w*h))
		for i := range data.Data {
			switch rng.IntN(10) {
			case 0:
				data.Data[i] = fill
			case 1:
				data.Data[i] = -fill
			case 2:
				data.Data[i] = math.Float32frombits(0x7fc0_0000 | rng.Uint32()&0x3f_ffff) // a NaN
			case 3:
				data.Data[i] = float32(math.Copysign(0, float64(rng.IntN(2)*2-1)))
			default:
				data.Data[i] = rng.Float32()*2000 - 1000
			}
		}
		if rng.IntN(3) == 0 {
			// A stretch with no fill at all, several words long.
			for i := 64; i < 256; i++ {
				data.Data[i] = 1
			}
		}
		f := &memFile{b: rawBytes(data)}
		opts := engine.RawOptions{Fill: fill, HasFill: true}
		src := engine.NewRawSource(f, w, h, opts)
		wantValid := func(v float32) bool { return v != fill && !(fill != fill && v != v) }

		for _, scalar := range []bool{true, false} {
			vec.UseScalar(scalar)
			id := fmt.Sprintf("fill %#08x on %s", math.Float32bits(fill), vec.Backend())

			// Compact, as the chunked engine's buffers of whole rows are.
			got := raster.NewFloat32(w, h, make([]float32, w*h))
			got.Valid = make([]uint64, raster.MaskWords(w*h))
			for i := range got.Valid {
				got.Valid[i] = rng.Uint64() // every word must be written
			}
			if err := src.ReadWindow(context.Background(), got, 0, 0); err != nil {
				t.Fatal(err)
			}
			for i, v := range data.Data {
				if raster.MaskGet(got.Valid, i) != wantValid(v) {
					t.Fatalf("%s, compact: cell %d = %#08x valid %v", id, i, math.Float32bits(v), !wantValid(v))
				}
			}

			for _, reg := range [][4]int{{0, 0, w, h}, {1, 1, 200, 3}, {63, 0, 130, 5}, {100, 4, 65, 1}, {0, 2, 64, 2}} {
				rid := fmt.Sprintf("%s, region %v", id, reg)
				dst, root := window(rng, reg[2], reg[3], true)
				orig := clone(root)
				if err := src.ReadWindow(context.Background(), dst, reg[0], reg[1]); err != nil {
					t.Fatal(err)
				}
				for y := range reg[3] {
					for x := range reg[2] {
						v := data.Data[data.Index(reg[0]+x, reg[1]+y)]
						if dst.IsValid(x, y) != wantValid(v) || !sameBits(dst.Data[dst.Index(x, y)], v) {
							t.Fatalf("%s: cell (%d, %d) = %#08x valid %v, want %#08x valid %v", rid, x, y,
								math.Float32bits(dst.Data[dst.Index(x, y)]), dst.IsValid(x, y), math.Float32bits(v), wantValid(v))
						}
					}
				}
				requireOutside(t, rid, root, orig, 0, 0, reg[2], reg[3])
			}

			// Back out through a sink, from random bits: a window's, which
			// start mid-word, and a compact raster's, word-aligned.
			win, _ := window(rng, w, h, true)
			if rng.IntN(2) == 0 {
				win = raster.NewFloat32(w, h, make([]float32, w*h))
				win.Valid = make([]uint64, raster.MaskWords(w*h))
				for i := range win.Valid {
					win.Valid[i] = rng.Uint64() | rng.Uint64() // mostly valid
				}
				win.Valid[1] = ^uint64(0)
			}
			for y := range h {
				copy(win.Row(y), data.Row(y))
			}
			out := &memFile{}
			if err := engine.NewRawSink(out, w, h, opts).WriteWindow(context.Background(), win, 0, 0); err != nil {
				t.Fatal(err)
			}
			back := engine.NewRawSource(out, w, h, engine.RawOptions{})
			stored := raster.NewFloat32(w, h, make([]float32, w*h))
			if err := back.ReadWindow(context.Background(), stored, 0, 0); err != nil {
				t.Fatal(err)
			}
			for y := range h {
				for x := range w {
					want := data.Data[data.Index(x, y)]
					if !win.IsValid(x, y) {
						want = fill
					}
					if got := stored.Data[stored.Index(x, y)]; !sameBits(got, want) {
						t.Fatalf("%s, sink: cell (%d, %d) = %#08x, want %#08x", id, x, y, math.Float32bits(got), math.Float32bits(want))
					}
				}
			}
		}
	}
}
