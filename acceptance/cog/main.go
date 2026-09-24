// Command cogread reads the GeoTIFFs that cogmake.py wrote, or the
// corpus cogcorpus.py recorded, through strata's cog package, so that
// cogcompare.py can difference them against GDAL's reading of the same
// files.
//
//	go run ./cog -dir out-cog
//	go run ./cog -dir out-corpus -src corpus/files
//
// For every file in manifest.json, every band and every level, it writes
// <case>.b<band>.l<level>.strata.f32 (little-endian float32) and
// .strata.mask (255 valid, 0 invalid), and records the levels, grids and
// CRS it found in strata.json. It reads through Source.ReadWindow in
// windows of 100×77 cells, a size that lines up with none of the files'
// blocks, so blocks are cut at every offset and shared through the cache.
// A level over 2²⁴ cells is read over one window of it, and at most 16
// bands, by the same rule as cogtruth.py. A file cog refuses is recorded
// with its error, not skipped silently, and so is a panic.
//
//	go run ./cog -dir out-cog -url http://127.0.0.1:8080 -workers 8
//
// reads every file through cog.HTTPReaderAt instead, from <url>/<case>.tif,
// with -workers goroutines reading windows at once, and records in
// strata.json the HTTP requests and bytes each file took
// (coghttpcheck.sh).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/LukasSelin/strata/cog"
	"github.com/LukasSelin/strata/raster"
)

var (
	dir     = flag.String("dir", "out-cog", "directory holding manifest.json and the results")
	src     = flag.String("src", "", "directory the manifest's paths are relative to (default: -dir, as <name>.tif)")
	baseURL = flag.String("url", "", "read <url>/<case>.tif through cog.HTTPReaderAt instead of the local files")
	workers = flag.Int("workers", 1, "goroutines reading windows of one source at once")
)

// The window rule of cogtruth.py: a level over maxCells cells is compared
// over at most windowSide² cells starting a third of the way in, and at
// most maxBands bands are compared.
const (
	maxCells   = 1 << 24
	windowSide = 4096
	maxBands   = 16
)

func levelWindow(w, h int) [4]int {
	if w*h <= maxCells {
		return [4]int{0, 0, w, h}
	}
	x, y := w/3, h/3
	return [4]int{x, y, min(w-x, windowSide), min(h-y, windowSide)}
}

type manifest struct {
	Files []struct {
		Name string `json:"name"`
		Path string `json:"path"`
	} `json:"files"`
}

type level struct {
	W, H int
	GT   [6]float64
	Win  [4]int
}

type result struct {
	Name   string  `json:"name"`
	Error  string  `json:"error,omitempty"`
	Bands  int     `json:"bands"`
	Levels []level `json:"levels"`
	EPSG   string  `json:"epsg"`
	Masked bool    `json:"masked"`
	// HTTP is what reading the file over HTTP took, with -url.
	HTTP *httpStats `json:"http,omitempty"`
}

type httpStats struct {
	Size     int64 `json:"size"`
	Prefetch int64 `json:"prefetch"`
	Requests int64 `json:"requests"`
	Bytes    int64 `json:"bytes"`
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
		path := filepath.Join(*dir, f.Name+".tif")
		if f.Path != "" {
			path = filepath.Join(*src, filepath.FromSlash(f.Path))
		}
		r := readFile(f.Name, path)
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

func readFile(name, path string) (r result) {
	r.Name = name
	fail := func(err error) result {
		r.Error = err.Error()
		return r
	}
	defer func() {
		if p := recover(); p != nil {
			r.Error = fmt.Sprintf("panic: %v", p)
		}
	}()
	var ra io.ReaderAt
	if *baseURL != "" {
		hr, err := cog.NewHTTPReaderAt(context.Background(), strings.TrimSuffix(*baseURL, "/")+"/"+name+".tif", cog.HTTPOptions{})
		if err != nil {
			return fail(err)
		}
		defer hr.Close()
		defer func() {
			st := hr.Stats()
			r.HTTP = &httpStats{Size: hr.Size(), Prefetch: cog.DefaultPrefetchBytes, Requests: st.Requests, Bytes: st.Bytes}
		}()
		ra = hr
	} else {
		fh, err := os.Open(path)
		if err != nil {
			return fail(err)
		}
		defer fh.Close()
		ra = fh
	}
	file, err := cog.Open(ra)
	if err != nil {
		return fail(err)
	}
	r.Bands = file.Bands()
	for lvl := range file.Levels() {
		g := file.Grid(lvl)
		win := levelWindow(g.Width, g.Height)
		r.Levels = append(r.Levels, level{W: g.Width, H: g.Height,
			GT:  [6]float64{g.OriginX, g.ResolutionX, 0, g.OriginY, 0, g.ResolutionY},
			Win: win})
		r.EPSG = g.CRS.Code
		for band := range min(file.Bands(), maxBands) {
			src, err := file.Source(cog.SourceOptions{Band: band, Level: lvl})
			if err != nil {
				return fail(err)
			}
			if lvl == 0 && band == 0 {
				r.Masked = src.Masked()
			}
			vals, mask, err := readAll(src, win)
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

// readAll reads the region area (x, y, width, height) of the source in
// 100×77 windows, -workers of them at once, and returns its cells as
// little-endian float32 bytes and its validity as one byte per cell.
func readAll(src *cog.Source, area [4]int) (vals, mask []byte, err error) {
	x0, y0, w, h := area[0], area[1], area[2], area[3]
	vals, mask = make([]byte, 4*w*h), make([]byte, w*h)
	const ww, wh = 100, 77
	type window struct{ x, y int }
	todo := make(chan window)
	var (
		wg    sync.WaitGroup
		once  sync.Once
		first error
	)
	for range max(*workers, 1) {
		wg.Go(func() {
			buf := raster.NewFloat32(ww, wh, make([]float32, ww*wh))
			buf.Valid = raster.NewMask(ww * wh)
			for wd := range todo {
				win := buf.Window(0, 0, min(ww, w-wd.x), min(wh, h-wd.y))
				if err := src.ReadWindow(context.Background(), win, x0+wd.x, y0+wd.y); err != nil {
					once.Do(func() { first = err })
					continue
				}
				// Windows do not overlap, so the workers write disjoint cells.
				for j := range win.Height {
					for i := range win.Width {
						k := (wd.y+j)*w + wd.x + i
						bits := math.Float32bits(win.Data[win.Index(i, j)])
						vals[4*k], vals[4*k+1], vals[4*k+2], vals[4*k+3] =
							byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
						if win.IsValid(i, j) {
							mask[k] = 255
						}
					}
				}
			}
		})
	}
	for y := 0; y < h; y += wh {
		for x := 0; x < w; x += ww {
			todo <- window{x, y}
		}
	}
	close(todo)
	wg.Wait()
	if first != nil {
		return nil, nil, first
	}
	return vals, mask, nil
}
