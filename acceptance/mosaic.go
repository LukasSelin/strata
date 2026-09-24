package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// The mosaic cases: four sources at mixed resolutions and origins laid
// onto one 10 m grid, with every method, in all three forms, written to
// mosaic.json for gdalwarp_mosaic.py to judge against gdalwarp given the
// same sources in the same order.
//
// The layout, in metres, over a destination of 60×45 cells from
// (1000, 2000):
//
//   - A, 10 m, 36×30 cells of the hill, offset by 0.3 of a cell;
//   - B, 20 m, 16×14 cells of the noisy surface through its NoData
//     stripe, over A and past the destination's right edge;
//   - C, 4 m, 80×60 cells of the noisy surface through its NoData disc,
//     over A and B, on the destination's cell edges;
//   - D, 10 m south-up, 20×20 cells of the hill offset by half a cell,
//     over B and C and past the right edge.
//
// So the seams cross every pairing of upsampling, downsampling and 1:1,
// NoData holes in later sources show earlier ones through them, and part
// of the destination is covered by no source at all.

type mosaicSource struct {
	Src     string   `json:"src"`
	SrcMask string   `json:"src_mask,omitempty"`
	SrcGrid gridJSON `json:"src_grid"`
}

type mosaicCase struct {
	Name    string         `json:"name"`
	Method  string         `json:"method"`
	Form    string         `json:"form"`
	Sources []mosaicSource `json:"sources"`
	DstGrid gridJSON       `json:"dst_grid"`
	Out     string         `json:"out"`
	OutMask string         `json:"out_mask"`
}

type mosaicManifest struct {
	Fill  float64      `json:"fill"`
	Cases []mosaicCase `json:"cases"`
}

func runMosaic() error {
	hr, nr := hill(), noisy()
	type layer struct {
		name string
		r    raster.Float32Raster
		g    raster.Grid
	}
	layers := []layer{
		{"ms-a", hr.Window(90, 60, 36, 30),
			raster.Grid{Width: 36, Height: 30, ResolutionX: 10, ResolutionY: -10, OriginX: 1003, OriginY: 1997}},
		{"ms-b", nr.Window(170, 40, 16, 14),
			raster.Grid{Width: 16, Height: 14, ResolutionX: 20, ResolutionY: -20, OriginX: 1290, OriginY: 1905}},
		{"ms-c", nr.Window(40, 25, 80, 60),
			raster.Grid{Width: 80, Height: 60, ResolutionX: 4, ResolutionY: -4, OriginX: 1240, OriginY: 1800}},
		{"ms-d", hr.Window(20, 100, 20, 20),
			raster.Grid{Width: 20, Height: 20, ResolutionX: 10, ResolutionY: 10, OriginX: 1455, OriginY: 1605}},
	}
	dg := raster.Grid{Width: 60, Height: 45, ResolutionX: 10, ResolutionY: -10, OriginX: 1000, OriginY: 2000}

	var srcs []raster.Dataset
	var entries []mosaicSource
	for _, l := range layers {
		// A compact copy, so the file is the raster and nothing else.
		src := raster.NewFloat32Like(l.r)
		for y := range l.r.Height {
			copy(src.Row(y), l.r.Row(y))
			if l.r.Valid != nil {
				for x := range l.r.Width {
					src.SetValid(x, y, l.r.IsValid(x, y))
				}
			}
		}
		e := mosaicSource{Src: l.name + ".f32", SrcGrid: toJSON(l.g)}
		if err := writeRaster(filepath.Join(*dir, e.Src), src); err != nil {
			return err
		}
		if src.Valid != nil {
			e.SrcMask = l.name + ".mask.u8"
			if err := writeMask(filepath.Join(*dir, e.SrcMask), src); err != nil {
				return err
			}
		}
		srcs = append(srcs, raster.NewDataset(l.g, src))
		entries = append(entries, e)
	}

	var m mosaicManifest
	m.Fill = fill
	for _, meth := range []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic, resample.Lanczos, resample.Average} {
		opts := resample.Options{Method: meth}
		outs := map[string]raster.Float32Raster{}

		plain := newMasked(dg.Width, dg.Height)
		resample.Mosaic(raster.NewDataset(dg, plain), srcs, opts)
		outs["plain"] = plain

		tiled := newMasked(dg.Width, dg.Height)
		if err := resample.MosaicTiled(context.Background(), raster.NewDataset(dg, tiled), srcs, opts, resampleTiled); err != nil {
			return err
		}
		outs["tiled"] = tiled

		chunked, err := mosaicRaw(entries, srcs, dg, opts)
		if err != nil {
			return err
		}
		outs["chunked"] = chunked

		for _, form := range []string{"plain", "tiled", "chunked"} {
			c := mosaicCase{
				Name: fmt.Sprintf("mosaic-%s-%s", meth, form), Method: meth.String(), Form: form,
				Sources: entries, DstGrid: toJSON(dg),
			}
			c.Out, c.OutMask = c.Name+".f32", c.Name+".mask.u8"
			if err := writeRaster(filepath.Join(*dir, c.Out), outs[form]); err != nil {
				return err
			}
			if err := writeMask(filepath.Join(*dir, c.OutMask), outs[form]); err != nil {
				return err
			}
			m.Cases = append(m.Cases, c)
		}
	}
	f, err := os.Create(filepath.Join(*dir, "mosaic.json"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		f.Close()
		return err
	}
	fmt.Printf("wrote %d mosaic cases to %s\n", len(m.Cases), *dir)
	return f.Close()
}

// mosaicRaw runs MosaicChunked from the sources' raw files into a raw
// file, with the fill value standing for NoData, and reads the result
// back.
func mosaicRaw(entries []mosaicSource, srcs []raster.Dataset, dg raster.Grid, opts resample.Options) (raster.Float32Raster, error) {
	var zero raster.Float32Raster
	var sources []engine.RasterSource
	var grids []raster.Grid
	for i, e := range entries {
		in, err := engine.OpenRawFile(filepath.Join(*dir, e.Src), os.O_RDONLY, 0, 4)
		if err != nil {
			return zero, err
		}
		defer in.Close()
		g := srcs[i].Grid
		sources = append(sources, engine.NewRawSource(in, g.Width, g.Height, engine.RawOptions{Fill: fill, HasFill: e.SrcMask != ""}))
		grids = append(grids, g)
	}
	tmp := filepath.Join(*dir, "mosaic.tmp")
	out, err := engine.CreateRawFile(tmp, 4*int64(dg.Width)*int64(dg.Height), 0o644, 4)
	if err != nil {
		return zero, err
	}
	ro := engine.RawOptions{Fill: fill, HasFill: true}
	if err := resample.MosaicChunked(context.Background(), engine.NewRawSink(out, dg.Width, dg.Height, ro), dg, sources, grids, opts, resampleChunked); err != nil {
		out.Close()
		return zero, err
	}
	back := newMasked(dg.Width, dg.Height)
	if err := engine.NewRawSource(out, dg.Width, dg.Height, ro).ReadWindow(context.Background(), back, 0, 0); err != nil {
		out.Close()
		return zero, err
	}
	if err := out.Close(); err != nil {
		return zero, err
	}
	return back, os.Remove(tmp)
}
