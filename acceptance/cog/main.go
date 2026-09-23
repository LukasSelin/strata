// Command cogread reads the GeoTIFFs that cogmake.py wrote through
// strata's cog package, so that cogcompare.py can difference them
// against GDAL's reading of the same files.
//
//	go run ./cog -dir out-cog
//
// For every file in manifest.json, every band and every level, it writes
// <case>.b<band>.l<level>.strata.f32 (little-endian float32) and
// .strata.mask (255 valid, 0 invalid), and records the levels, grids and
// CRS it found in strata.json. It reads through Source.ReadWindow in
// windows of 100×77 cells, a size that lines up with none of the files'
// blocks, so blocks are cut at every offset and shared through the cache.
// A file cog refuses is recorded with its error, not skipped silently.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LukasSelin/strata/cog"
	"github.com/LukasSelin/strata/raster"
)

var dir = flag.String("dir", "out-cog", "directory holding manifest.json and the files")

type manifest struct {
	Files []struct {
		Name string `json:"name"`
	} `json:"files"`
}

type level struct {
	W, H int
	GT   [6]float64
}

type result struct {
	Name   string  `json:"name"`
	Error  string  `json:"error,omitempty"`
	Bands  int     `json:"bands"`
	Levels []level `json:"levels"`
	EPSG   string  `json:"epsg"`
	Masked bool    `json:"masked"`
}

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cogread:", err)
		os.Exit(1)
	}
}

func run() error {
	b, err := os.ReadFile(filepath.Join(*dir, "manifest.json"))
	if err != nil {
		return err
	}
	var m manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	var results []result
	for _, f := range m.Files {
		r := readFile(f.Name)
		if r.Error != "" {
			fmt.Printf("  %-48s refused: %s\n", f.Name, r.Error)
		}
		results = append(results, r)
	}
	out, err := json.MarshalIndent(results, "", " ")
	if err != nil {
		return err
	}
	fmt.Printf("strata read %d files\n", len(results))
	return os.WriteFile(filepath.Join(*dir, "strata.json"), out, 0o600)
}

func readFile(name string) result {
	r := result{Name: name}
	fail := func(err error) result {
		r.Error = err.Error()
		return r
	}
	fh, err := os.Open(filepath.Join(*dir, name+".tif"))
	if err != nil {
		return fail(err)
	}
	defer fh.Close()
	file, err := cog.Open(fh)
	if err != nil {
		return fail(err)
	}
	r.Bands = file.Bands()
	for lvl := range file.Levels() {
		g := file.Grid(lvl)
		r.Levels = append(r.Levels, level{W: g.Width, H: g.Height,
			GT: [6]float64{g.OriginX, g.ResolutionX, 0, g.OriginY, 0, g.ResolutionY}})
		r.EPSG = g.CRS.Code
		for band := range file.Bands() {
			src, err := file.Source(cog.SourceOptions{Band: band, Level: lvl})
			if err != nil {
				return fail(err)
			}
			r.Masked = src.Masked()
			vals, mask, err := readAll(src)
			if err != nil {
				return fail(err)
			}
			base := filepath.Join(*dir, fmt.Sprintf("%s.b%d.l%d.strata", name, band, lvl))
			if err := os.WriteFile(base+".f32", vals, 0o600); err != nil {
				return fail(err)
			}
			if err := os.WriteFile(base+".mask", mask, 0o600); err != nil {
				return fail(err)
			}
		}
	}
	return r
}

// readAll reads the whole source in 100×77 windows and returns its cells
// as little-endian float32 bytes and its validity as one byte per cell.
func readAll(src *cog.Source) (vals, mask []byte, err error) {
	w, h := src.Size()
	vals, mask = make([]byte, 4*w*h), make([]byte, w*h)
	const ww, wh = 100, 77
	buf := raster.NewFloat32(ww, wh, make([]float32, ww*wh))
	buf.Valid = raster.NewMask(ww * wh)
	for y := 0; y < h; y += wh {
		for x := 0; x < w; x += ww {
			win := buf.Window(0, 0, min(ww, w-x), min(wh, h-y))
			if err := src.ReadWindow(context.Background(), win, x, y); err != nil {
				return nil, nil, err
			}
			for j := range win.Height {
				for i := range win.Width {
					k := (y+j)*w + x + i
					bits := math.Float32bits(win.Data[win.Index(i, j)])
					vals[4*k], vals[4*k+1], vals[4*k+2], vals[4*k+3] =
						byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
					if win.IsValid(i, j) {
						mask[k] = 255
					}
				}
			}
		}
	}
	return vals, mask, nil
}
