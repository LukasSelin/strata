package resample_test

import (
	"math"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// fuzzGrids decodes a source grid and a destination grid from d: sane
// ones most of the time, and now and then a resolution or origin that is
// zero, huge, tiny, negative or not finite.
func fuzzGrids(d fuzzdata.Source) (dg, sg raster.Grid) {
	pick := func(sane float64) float64 {
		switch d.IntN(24) {
		case 0:
			return 0
		case 1:
			return math.Inf(1)
		case 2:
			return math.NaN()
		case 3:
			return d.Float64() * 1e12
		case 4:
			return -sane
		}
		return sane
	}
	sg = raster.Grid{Width: d.Range(1, 40), Height: d.Range(1, 40)}
	sg.ResolutionX = pick(math.Ldexp(1, d.Range(-3, 3)))
	sg.ResolutionY = pick(-math.Ldexp(1, d.Range(-3, 3)))
	sg.OriginX = pick(float64(d.Range(-5, 5)))
	sg.OriginY = pick(float64(d.Range(-5, 45)))
	scale := []float64{0.25, 0.5, 0.73, 1, 1.37, 2, 3, 4.5}
	dg = raster.Grid{Width: d.Range(1, 60), Height: d.Range(1, 60)}
	dg.ResolutionX = pick(sg.ResolutionX * scale[d.IntN(len(scale))])
	dg.ResolutionY = pick(sg.ResolutionY * scale[d.IntN(len(scale))])
	dg.OriginX = pick(sg.OriginX + d.Float64()*4 - 2)
	dg.OriginY = pick(sg.OriginY + d.Float64()*4 - 2)
	if d.IntN(16) == 0 {
		dg.CRS.Code, sg.CRS.Code = "EPSG:25833", "EPSG:32633"
	}
	return dg, sg
}

// FuzzResample runs every method on grids, rasters, windows, strides and
// masks decoded from the input, with arbitrary values. Invalid grids and
// operands must panic with a "resample:" message before anything is
// written. Otherwise the plain, Tiled and Chunked forms must write the
// same bits for any tiles and workers, and nothing outside dst's cells;
// invalid cells hold NaN, and valid cells of a masked source never see
// the Data under its cleared bits.
func FuzzResample(f *testing.F) {
	f.Add([]byte{1, 10, 5, 0})
	f.Add([]byte{3, 40, 4, 1, 1, 0, 0, 1, 2, 3})
	f.Add([]byte{2, 3, 3, 2, 0, 1, 7, 9, 200, 13, 0, 0, 5})
	f.Fuzz(func(t *testing.T, b []byte) {
		d := fuzzdata.New(b)
		dg, sg := fuzzGrids(d)
		m := resample.Method(d.IntN(6))
		masked := d.Bool()
		src := raster.NewFloat32(sg.Width, sg.Height, make([]float32, sg.Width*sg.Height))
		for i := range src.Data {
			src.Data[i] = d.Float32()
		}
		if masked {
			src.Valid = raster.NewMask(len(src.Data))
			for i := range src.Data {
				raster.MaskSet(src.Valid, i, d.IntN(5) != 0)
			}
		}
		src = rastertest.Place(d, src)
		dstMasked := masked || d.IntN(4) != 0
		out := func() raster.Float32Raster { return rastertest.Output(d, dg.Width, dg.Height, dstMasked, true) }

		plain := out()
		root := snapshot(plain)
		if msg, ok := panics(func() {
			resample.Resample(raster.Dataset{Grid: dg, Raster: plain}, raster.Dataset{Grid: sg, Raster: src}, resample.Options{Method: m})
		}); ok {
			if !strings.HasPrefix(msg, "resample: ") {
				t.Fatalf("panic %q is not a resample: panic", msg)
			}
			if !root.same(plain) {
				t.Fatal("a rejected call wrote to dst")
			}
			return
		}
		if !root.sameOutside(plain) {
			t.Fatal("Resample wrote outside dst's cells")
		}
		for y := range plain.Height {
			for x := range plain.Width {
				v := plain.Data[plain.Index(x, y)]
				if plain.Valid != nil && !plain.IsValid(x, y) && v == v {
					t.Fatalf("invalid cell (%d, %d) holds %v, want NaN", x, y, v)
				}
			}
		}

		// Scrambling the Data under the source's cleared bits changes no bit.
		scrambled := rastertest.Compact(src)
		rastertest.ScrambleInvalid(scrambled, d)
		again := out()
		resample.Resample(raster.Dataset{Grid: dg, Raster: again}, raster.Dataset{Grid: sg, Raster: scrambled}, resample.Options{Method: m})
		requireSame(t, "scrambled", plain, again)

		eo := engine.Options{TileWidth: d.IntN(dg.Width + 3), TileHeight: d.IntN(dg.Height + 3), Workers: d.IntN(4)}
		tiled := out()
		if err := resample.ResampleTiled(t.Context(), raster.Dataset{Grid: dg, Raster: tiled}, raster.Dataset{Grid: sg, Raster: src},
			resample.Options{Method: m}, eo); err != nil {
			t.Fatal(err)
		}
		requireSame(t, "Tiled", plain, tiled)

		chunked := rastertest.Compact(out())
		if err := resample.ResampleChunked(t.Context(), engine.NewMemorySink(chunked), dg, engine.NewMemorySource(src), sg,
			resample.Options{Method: m}, eo); err != nil {
			t.Fatal(err)
		}
		requireSame(t, "Chunked", plain, chunked)
	})
}

func panics(f func()) (msg string, ok bool) {
	defer func() {
		if v := recover(); v != nil {
			msg, ok = strings.TrimSpace(toString(v)), true
		}
	}()
	f()
	return "", false
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return "non-string panic"
}

// rootCopy is a copy of the whole memory behind a raster view, to check
// what a call wrote.
type rootCopy struct {
	data []float32
	bits []uint64
	r    raster.Float32Raster
}

func snapshot(r raster.Float32Raster) rootCopy {
	full := r.Data[:cap(r.Data)]
	c := rootCopy{data: append([]float32(nil), full...), r: r}
	if r.Valid != nil {
		c.bits = append([]uint64(nil), r.Valid...)
	}
	return c
}

func (c rootCopy) same(r raster.Float32Raster) bool {
	full := r.Data[:cap(r.Data)]
	for i := range full {
		if math.Float32bits(full[i]) != math.Float32bits(c.data[i]) {
			return false
		}
	}
	for i := range c.bits {
		if c.bits[i] != r.Valid[i] {
			return false
		}
	}
	return true
}

// sameOutside reports whether nothing but r's cells and their bits
// changed.
func (c rootCopy) sameOutside(r raster.Float32Raster) bool {
	full := r.Data[:cap(r.Data)]
	inside := func(i int) bool {
		return i < len(r.Data) && i%r.Stride < r.Width
	}
	for i := range full {
		if !inside(i) && math.Float32bits(full[i]) != math.Float32bits(c.data[i]) {
			return false
		}
	}
	for w := range c.bits {
		for b := range 64 {
			bit := w*64 + b
			i := bit - r.ValidOffset
			if (i < 0 || !inside(i)) && (c.bits[w]>>b)&1 != (r.Valid[w]>>b)&1 {
				return false
			}
		}
	}
	return true
}

func requireSame(t *testing.T, form string, want, got raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			if want.Valid != nil && want.IsValid(x, y) != got.IsValid(x, y) {
				t.Fatalf("%s: cell (%d, %d) validity differs", form, x, y)
			}
			if !rastertest.SameFloat(want.Data[want.Index(x, y)], got.Data[got.Index(x, y)]) {
				t.Fatalf("%s: cell (%d, %d) = %v, plain %v", form, x, y, got.Data[got.Index(x, y)], want.Data[want.Index(x, y)])
			}
		}
	}
}
