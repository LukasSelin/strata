package exec_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"strata/engine"
	"strata/internal/exec"
	"strata/internal/fuzzdata"
	"strata/internal/rastertest"
	"strata/raster"
)

// FuzzProcessRelations checks relations between runs of box kernels of
// radius 0 to 3 with one to three inputs and outputs, each run through
// ProcessN or ProcessChunked with its own tiles, workers, band size and
// memory layouts, over arbitrary values and masks:
//
//   - Data under invalid input cells never reaches a valid output cell;
//   - invalidating one input cell invalidates exactly the output cells
//     within radius r of it (in every output) and changes nothing else;
//   - changing one input value changes no output cell farther than r;
//   - the interior of a run on windows of the inputs is the run on the
//     whole inputs at the same offset.
//
// Unlike a comparison with the naive reference, these hold for any kernel
// whose cells depend only on their neighbourhood, and fail when tiles,
// bands or halos read or write the wrong cells.
func FuzzProcessRelations(f *testing.F) {
	for rel := range 4 {
		f.Add([]byte{byte(rel), 1, 1, 1, 20, 6})
		f.Add([]byte{byte(rel), 3, 2, 2, 66, 9, 1})
		f.Add([]byte{byte(rel), 0, 3, 1, 9, 3, 0, 2})
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		processRelations(t, fuzzdata.New(data))
	})
}

// processRelations decodes a kernel, its operands and a relation from d
// and checks that relation. FuzzProcessRelations drives it with a fuzz
// input, TestProcessRelations with rapid.
func processRelations(t rastertest.TB, d fuzzdata.Source) {
	rel, r := d.IntN(4), d.Range(0, 3)
	nin, nout := d.Range(1, 3), d.Range(1, 3)
	w, h := d.Range(1, 70), d.Range(1, 9)
	masked := rel == 0 || rel == 1 || d.Bool()
	k := boxKernel{r: r, inputs: nin, outputs: nout}

	inputs := make([]raster.Float32Raster, nin)
	for i := range inputs {
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
		inputs[i] = in
	}
	run := func(ins []raster.Float32Raster) []raster.Float32Raster {
		rw, rh := ins[0].Width, ins[0].Height
		src := make([]raster.Float32Raster, len(ins))
		for i, in := range ins {
			src[i] = rastertest.Place(d, in)
		}
		dst := make([]raster.Float32Raster, nout)
		for i := range dst {
			dst[i] = rastertest.Output(d, rw, rh, masked, true)
		}
		opts := engine.Options{TileWidth: d.Range(0, rw+2), TileHeight: d.Range(0, rh+2), Workers: d.Range(0, 3)}
		if cells := d.Range(0, 3*rw); cells > 0 {
			defer exec.SetBandCells(cells)()
		}
		var err error
		if d.Bool() {
			err = exec.ProcessChunked(context.Background(), memorySinksOf(dst), memorySourcesOf(src), k, opts)
		} else {
			err = exec.ProcessN(context.Background(), dst, src, k, opts)
		}
		if err != nil {
			t.Fatal(err)
		}
		for i, out := range dst {
			dst[i] = rastertest.Compact(out)
		}
		return dst
	}
	clone := func() []raster.Float32Raster {
		out := make([]raster.Float32Raster, nin)
		for i, in := range inputs {
			out[i] = rastertest.Compact(in)
		}
		return out
	}
	id := fmt.Sprintf("relation %d r %d inputs %d outputs %d %d×%d masked %v", rel, r, nin, nout, w, h, masked)
	sameCell := func(cid string, got raster.Float32Raster, x, y int, want raster.Float32Raster, wx, wy int) {
		t.Helper()
		if gv, wv := got.IsValid(x, y), want.IsValid(wx, wy); gv != wv {
			t.Fatalf("%s: cell (%d, %d) valid = %v, want %v as cell (%d, %d)", cid, x, y, gv, wv, wx, wy)
		}
		if g, wv := got.Data[got.Index(x, y)], want.Data[want.Index(wx, wy)]; got.IsValid(x, y) && !rastertest.SameFloat(g, wv) {
			t.Fatalf("%s: cell (%d, %d) = %v (%#08x), want %v (%#08x) as cell (%d, %d)",
				cid, x, y, g, math.Float32bits(g), wv, math.Float32bits(wv), wx, wy)
		}
	}

	switch rel {
	case 0:
		scrambled := clone()
		for _, in := range scrambled {
			rastertest.ScrambleInvalid(in, d)
		}
		a, b := run(inputs), run(scrambled)
		for o := range a {
			for y := range h {
				for x := range w {
					sameCell(fmt.Sprintf("%s: output %d with scrambled invalid cells", id, o), b[o], x, y, a[o], x, y)
				}
			}
		}
	case 1:
		j, px, py := d.IntN(nin), d.IntN(w), d.IntN(h)
		holed := clone()
		holed[j].SetValid(px, py, false)
		a, b := run(inputs), run(holed)
		for o := range a {
			cid := fmt.Sprintf("%s: output %d with input %d cell (%d, %d) invalid", id, o, j, px, py)
			for y := range h {
				for x := range w {
					want := a[o].IsValid(x, y) && !rastertest.Near(x, y, px, py, r)
					if got := b[o].IsValid(x, y); got != want {
						t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", cid, x, y, got, want)
					}
					if want {
						sameCell(cid, b[o], x, y, a[o], x, y)
					}
				}
			}
		}
	case 2:
		j, px, py := d.IntN(nin), d.IntN(w), d.IntN(h)
		changed := clone()
		changed[j].Data[changed[j].Index(px, py)] = d.Float32()
		a, b := run(inputs), run(changed)
		for o := range a {
			for y := range h {
				for x := range w {
					if !rastertest.Near(x, y, px, py, r) {
						sameCell(fmt.Sprintf("%s: output %d with input %d cell (%d, %d) changed", id, o, j, px, py), b[o], x, y, a[o], x, y)
					}
				}
			}
		}
	case 3:
		ww, hh := d.Range(1, w), d.Range(1, h)
		x0, y0 := d.Range(0, w-ww), d.Range(0, h-hh)
		windows := make([]raster.Float32Raster, nin)
		for i, in := range inputs {
			windows[i] = in.Window(x0, y0, ww, hh)
		}
		full, part := run(inputs), run(windows)
		for o := range full {
			for y := r; y < hh-r; y++ {
				for x := r; x < ww-r; x++ {
					sameCell(fmt.Sprintf("%s: output %d of the %d×%d windows at (%d, %d)", id, o, ww, hh, x0, y0),
						part[o], x, y, full[o], x0+x, y0+y)
				}
			}
		}
	}
}
func memorySinksOf(rs []raster.Float32Raster) []engine.RasterSink {
	out := make([]engine.RasterSink, len(rs))
	for i, r := range rs {
		out[i] = engine.NewMemorySink(r)
	}
	return out
}

func memorySourcesOf(rs []raster.Float32Raster) []engine.RasterSource {
	out := make([]engine.RasterSource, len(rs))
	for i, r := range rs {
		out[i] = engine.NewMemorySource(r)
	}
	return out
}
