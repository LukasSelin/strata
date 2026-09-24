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

// The resampling cases: every method at five scale factors (output cells
// 0.5, 2, 4, 1.37 and 1.25 source cells wide), over a clean and a masked
// source and onto a south-up grid, in all three forms, written with their
// grids to resample.json for check_resample.py (numerical reference) and
// gdalwarp_resample.py (gdalwarp, the outside oracle) to judge.
//
// The sources are 80×60 windows of the hill and of the noisy surface (the
// latter through the edge of its NoData disc), small enough for a
// reference in plain Python. Most destination grids are offset from the
// source by a fraction of a cell and run one cell past its far edge, so
// that edges, partial cells and uncovered cells all occur; the 1.25 grid
// lies inside the source on its cell edges instead (see below).

type gridJSON struct {
	Width  int     `json:"width"`
	Height int     `json:"height"`
	OX     float64 `json:"ox"`
	OY     float64 `json:"oy"`
	RX     float64 `json:"rx"`
	RY     float64 `json:"ry"`
}

type resampleCase struct {
	Name    string   `json:"name"`
	Surface string   `json:"surface"`
	Method  string   `json:"method"`
	Scale   float64  `json:"scale"`
	Form    string   `json:"form"`
	Src     string   `json:"src"`
	SrcMask string   `json:"src_mask,omitempty"`
	SrcGrid gridJSON `json:"src_grid"`
	DstGrid gridJSON `json:"dst_grid"`
	Out     string   `json:"out"`
	OutMask string   `json:"out_mask"`
}

type resampleManifest struct {
	Fill  float64        `json:"fill"`
	Cases []resampleCase `json:"cases"`
}

var (
	resampleTiled   = engine.Options{TileWidth: 7, TileHeight: 5, Workers: 3}
	resampleChunked = engine.Options{TileWidth: 23, TileHeight: 16, Workers: 4}
)

func toJSON(g raster.Grid) gridJSON {
	return gridJSON{g.Width, g.Height, g.OriginX, g.OriginY, g.ResolutionX, g.ResolutionY}
}

func runResample() error {
	const sw, sh = 80, 60
	type source struct {
		name, surface string
		r             raster.Float32Raster
		southUp       bool
	}
	hr, nr := hill(), noisy()
	sources := []source{
		{"rs-hill", "hill", hr.Window(90, 60, sw, sh), false},
		{"rs-noisy", "noisy", nr.Window(40, 25, sw, sh), false},
		// The hill again, resampled onto a south-up destination.
		{"rs-hill", "hill-southup", hr.Window(90, 60, sw, sh), true},
	}
	sg := raster.Grid{Width: sw, Height: sh, ResolutionX: 10, ResolutionY: -10, OriginX: 1000, OriginY: 2000}
	methods := []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic, resample.Lanczos, resample.Average}
	var m resampleManifest
	m.Fill = fill
	written := map[string]bool{}
	for _, s := range sources {
		src := raster.NewFloat32Like(s.r)
		for y := range sh {
			copy(src.Row(y), s.r.Row(y))
			if s.r.Valid != nil {
				for x := range sw {
					src.SetValid(x, y, s.r.IsValid(x, y))
				}
			}
		}
		if !written[s.name] {
			written[s.name] = true
			if err := writeRaster(filepath.Join(*dir, s.name+".f32"), src); err != nil {
				return err
			}
			if src.Valid != nil {
				if err := writeMask(filepath.Join(*dir, s.name+".mask.u8"), src); err != nil {
					return err
				}
			}
		}
		for _, f := range []float64{0.5, 2, 4, 1.37, 1.25} {
			res := 10 * f
			dg := raster.Grid{
				Width:       int((float64(sw)*10-3)/res) + 1,
				Height:      int((float64(sh)*10-3)/res) + 1,
				ResolutionX: res, ResolutionY: -res,
				OriginX: 1003, OriginY: 1997,
			}
			if f == 1.25 {
				// Inside the source and on its cell edges, 48×32 cells over
				// 60×40 source cells: the one grid here on which gdalwarp's
				// stretch, taken from pixel counts, equals the resolution
				// ratio for a non-integer factor (DESIGN.md §54).
				dg = raster.Grid{Width: 48, Height: 32, ResolutionX: res, ResolutionY: -res, OriginX: 1100, OriginY: 1900}
			}
			if s.southUp {
				dg.ResolutionY, dg.OriginY = res, dg.OriginY-float64(dg.Height)*res
			}
			for _, meth := range methods {
				base := fmt.Sprintf("%s-%s-%s", s.surface, meth, fmtScale(f))
				c := resampleCase{
					Surface: s.surface, Method: meth.String(), Scale: f,
					Src: s.name + ".f32", SrcGrid: toJSON(sg), DstGrid: toJSON(dg),
				}
				if src.Valid != nil {
					c.SrcMask = s.name + ".mask.u8"
				}
				opts := resample.Options{Method: meth}
				outs := map[string]raster.Float32Raster{}

				plain := newMasked(dg.Width, dg.Height)
				resample.Resample(raster.NewDataset(dg, plain), raster.NewDataset(sg, src), opts)
				outs["plain"] = plain

				tiled := newMasked(dg.Width, dg.Height)
				if err := resample.ResampleTiled(context.Background(), raster.NewDataset(dg, tiled), raster.NewDataset(sg, src), opts, resampleTiled); err != nil {
					return err
				}
				outs["tiled"] = tiled

				chunked, err := resampleRaw(s.name, src.Valid != nil, sg, dg, opts)
				if err != nil {
					return err
				}
				outs["chunked"] = chunked

				for _, form := range []string{"plain", "tiled", "chunked"} {
					cc := c
					cc.Name = base + "-" + form
					cc.Form = form
					cc.Out = cc.Name + ".f32"
					cc.OutMask = cc.Name + ".mask.u8"
					if err := writeRaster(filepath.Join(*dir, cc.Out), outs[form]); err != nil {
						return err
					}
					if err := writeMask(filepath.Join(*dir, cc.OutMask), outs[form]); err != nil {
						return err
					}
					m.Cases = append(m.Cases, cc)
				}
			}
		}
	}
	f, err := os.Create(filepath.Join(*dir, "resample.json"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		f.Close()
		return err
	}
	fmt.Printf("wrote %d resampling cases to %s\n", len(m.Cases), *dir)
	return f.Close()
}

func fmtScale(f float64) string {
	return fmt.Sprintf("x%g", f)
}

func newMasked(w, h int) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	return r
}

// resampleRaw runs ResampleChunked from the source's raw file into a raw
// file, both with the fill value standing for NoData, and reads the
// result back.
func resampleRaw(name string, masked bool, sg, dg raster.Grid, opts resample.Options) (raster.Float32Raster, error) {
	var zero raster.Float32Raster
	in, err := engine.OpenRawFile(filepath.Join(*dir, name+".f32"), os.O_RDONLY, 0, 4)
	if err != nil {
		return zero, err
	}
	defer in.Close()
	tmp := filepath.Join(*dir, name+".resample.tmp")
	out, err := engine.CreateRawFile(tmp, 4*int64(dg.Width)*int64(dg.Height), 0o644, 4)
	if err != nil {
		return zero, err
	}
	src := engine.NewRawSource(in, sg.Width, sg.Height, engine.RawOptions{Fill: fill, HasFill: masked})
	// The destination always needs validity: part of it lies past the
	// source.
	ro := engine.RawOptions{Fill: fill, HasFill: true}
	if err := resample.ResampleChunked(context.Background(), engine.NewRawSink(out, dg.Width, dg.Height, ro), dg, src, sg, opts, resampleChunked); err != nil {
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
