package focal_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
)

// FuzzFocal decodes an operation, its options, an input and engine
// options, and checks that:
//
//   - invalid options (a radius out of range, weights or taps of the
//     wrong length or not finite) panic with a "focal:" message before
//     anything is written, on every path;
//   - otherwise the plain function writes the per-cell definition, bit
//     for bit, with the documented edges and validity;
//   - the Tiled and Chunked forms write the plain function's bits;
//   - the scalar backend writes them too.
func FuzzFocal(f *testing.F) {
	f.Add([]byte{0, 1, 12, 5})
	f.Add([]byte{2, 2, 40, 9, 1})
	f.Add([]byte{3, 5, 33, 13, 0, 1})
	f.Add([]byte{5, 1, 1, 1})
	f.Add([]byte{1, 0, 9, 9, 7, 7, 7})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		kind, r := d.IntN(numKinds), d.Range(1, 3)
		if d.IntN(8) == 0 {
			r = d.Range(4, focal.MaxRadius)
		}
		s := newSpec(d, kind, r, weightValues(d.IntN(2)))
		w, h := d.Range(1, 48), d.Range(1, 2*r+8)
		masked := d.Bool()
		src := newSrc(d, w, h, cellValues(d.IntN(3)), masked)
		path := d.IntN(3)
		eopts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 4)}
		id := fmt.Sprintf("%v %d×%d masked %v path %d %+v", s, w, h, masked, path, eopts)

		// One decoded mistake in the options, or none.
		bad := s
		broken := true
		switch d.IntN(8) {
		case 0:
			bad.r = []int{0, -1, focal.MaxRadius + 1}[d.IntN(3)]
		case 1:
			switch kind {
			case kCorrelate, kConvolve:
				bad.w = bad.w[:len(bad.w)-1-d.IntN(len(bad.w))]
			case kSeparable:
				bad.col = append(bad.col, 1, 1)
			default:
				broken = false
			}
		case 2:
			nonFinite := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}[d.IntN(3)]
			switch kind {
			case kCorrelate, kConvolve:
				bad.w = append([]float32(nil), bad.w...)
				bad.w[d.IntN(len(bad.w))] = nonFinite
			case kSeparable:
				bad.row = append([]float32(nil), bad.row...)
				bad.row[d.IntN(len(bad.row))] = nonFinite
			default:
				broken = false
			}
		default:
			broken = false
		}
		if broken {
			dst := rastertest.Output(d, w, h, masked, true)
			before := rastertest.Compact(dst)
			mustPanic(t, id, "focal: ", func() { _ = bad.run(path, eopts, dst, src) })
			for y := range h {
				for x := range w {
					if math.Float32bits(dst.Data[dst.Index(x, y)]) != math.Float32bits(before.Data[before.Index(x, y)]) ||
						(dst.Valid != nil && dst.IsValid(x, y) != before.IsValid(x, y)) {
						t.Fatalf("%s: wrote (%d, %d) before panicking", id, x, y)
					}
				}
			}
			return
		}

		plain := rastertest.Output(d, w, h, masked || d.Bool(), true)
		if err := s.run(0, engine.Options{}, plain, src); err != nil {
			t.Fatal(err)
		}
		requireNaive(t, id, s, plain, src)

		got := rastertest.Output(d, w, h, plain.Valid != nil, true)
		if err := s.run(path, eopts, got, src); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		requireSame(t, id, got, plain)

		if focalrow.Backend() != "scalar" {
			focalrow.UseScalar(true)
			defer focalrow.UseScalar(false)
			scalar := raster.NewFloat32Like(plain)
			if err := s.run(0, engine.Options{}, scalar, src); err != nil {
				t.Fatal(err)
			}
			requireSame(t, id+" scalar", scalar, plain)
		}
	})
}
