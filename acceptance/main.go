// Command strata-acceptance drives strata through its public API only and
// writes every input and output to plain files, so that an independent
// program can judge the results without reading strata's source.
//
//	go run .              # in this directory
//	go run . -dir out     # somewhere else
//
// It builds three small DEMs, runs the terrain, algebra and reduce
// operations on each in all three forms (plain, Tiled, Chunked), resamples
// windows of two of them (resample.go, judged by check_resample.py and
// gdalwarp_resample.py), and writes:
//
//	<dem>.f32              the elevation, raw little-endian float32
//	<dem>.mask.u8          1 per valid cell, 0 per NoData cell (masked DEMs)
//	<dem>-<op>-<form>.f32  one result
//	manifest.json          what every file is, and the scalar results
//
// Nothing here checks anything. check.py does the judging.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
	"github.com/LukasSelin/strata/terrain"
)

// Deliberately not square, not a power of two, and not a multiple of the
// tile sizes below: tiles and worker bands then land on ragged edges.
const (
	width  = 257
	height = 193
)

// The NoData value written into the raw files under invalid cells.
const fill = -9999

// Tiled and Chunked runs use these, so that every result is produced by
// more than one tiling and worker count.
var (
	tiledOpts   = engine.Options{TileWidth: 37, TileHeight: 23, Workers: 3}
	chunkedOpts = engine.Options{TileHeight: 16, Workers: 4}
)

type rasterCase struct {
	Name      string  `json:"name"`
	Surface   string  `json:"surface"`
	Op        string  `json:"op"`
	Form      string  `json:"form"`
	DEM       string  `json:"dem"`
	DEMMask   string  `json:"dem_mask,omitempty"`
	Out       string  `json:"out"`
	OutMask   string  `json:"out_mask,omitempty"`
	CellSize  float64 `json:"cell_size"`
	CellSizeY float64 `json:"cell_size_y"`
	Azimuth   float64 `json:"azimuth,omitempty"`
	Altitude  float64 `json:"altitude,omitempty"`
}

type scalarCase struct {
	Name    string `json:"name"`
	Op      string `json:"op"`
	Form    string `json:"form"`
	DEM     string `json:"dem"`
	DEMMask string `json:"dem_mask,omitempty"`
	Min     string `json:"min,omitempty"`
	Max     string `json:"max,omitempty"`
	Count   int64  `json:"count"`
}

type manifest struct {
	Width   int          `json:"width"`
	Height  int          `json:"height"`
	Fill    float64      `json:"fill"`
	Rasters []rasterCase `json:"rasters"`
	Scalars []scalarCase `json:"scalars"`
}

var dir = flag.String("dir", "out", "directory for the generated files")

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "strata-acceptance:", err)
		os.Exit(1)
	}
}

type dem struct {
	name    string
	surface string
	r       raster.Float32Raster
	cellX   float64
	cellY   float64
}

func run() error {
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		return err
	}
	m := manifest{Width: width, Height: height, Fill: fill}

	dems := []dem{
		// A tilted plane: every interior cell has the same slope and
		// aspect, and both follow from the corner elevations alone.
		{name: "plane", surface: "plane", r: plane(0.3, -0.7), cellX: 10, cellY: 10},
		// A smooth hill on rectangular cells, so that the two cell
		// sizes have to be used separately, and every aspect occurs.
		{name: "hill", surface: "hill", r: hill(), cellX: 10, cellY: 25},
		// Noise over the hill, with two NoData regions, so validity
		// has to propagate through a 3x3 neighbourhood.
		{name: "noisy", surface: "noisy", r: noisy(), cellX: 30, cellY: 30},
	}

	for _, d := range dems {
		if err := writeRaster(filepath.Join(*dir, d.name+".f32"), d.r); err != nil {
			return err
		}
		if d.r.Valid != nil {
			if err := writeMask(filepath.Join(*dir, d.name+".mask.u8"), d.r); err != nil {
				return err
			}
		}
		if err := d.runTerrain(&m); err != nil {
			return err
		}
		if err := d.runReduce(&m); err != nil {
			return err
		}
	}
	if err := runAlgebra(&m, dems[0], dems[1]); err != nil {
		return err
	}
	if err := runResample(); err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(*dir, "manifest.json"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("wrote %d rasters and %d scalars to %s\n", len(m.Rasters), len(m.Scalars), *dir)
	return nil
}

// op is one operation in its three forms, with its options already bound.
type op struct {
	name     string
	azimuth  float64
	altitude float64
	plain    func(dst, dem raster.Float32Raster)
	tiled    func(ctx context.Context, dst, dem raster.Float32Raster, eo engine.Options) error
	chunked  func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error
}

func (d dem) ops() []op {
	ao := terrain.AspectOptions{CellSize: d.cellX, CellSizeY: d.cellY}
	gopts := terrain.GradientOptions{CellSize: d.cellX, CellSizeY: d.cellY}
	ho := terrain.HillshadeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Azimuth: 315, Altitude: 45}

	slope := func(u terrain.SlopeUnits) op {
		o := terrain.SlopeOptions{CellSize: d.cellX, CellSizeY: d.cellY, Units: u}
		return op{
			name:  "slope_" + unitName(u),
			plain: func(dst, dm raster.Float32Raster) { terrain.Slope(dst, dm, o) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.SlopeTiled(ctx, dst, dm, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.SlopeChunked(ctx, dst, src, o, eo)
			},
		}
	}
	// Gradient writes two rasters; each component is exported as its own
	// case, with the other one thrown away.
	grad := func(wantDX bool) op {
		name := "gradient_dy"
		if wantDX {
			name = "gradient_dx"
		}
		pick := func(dst, other raster.Float32Raster) (dx, dy raster.Float32Raster) {
			if wantDX {
				return dst, other
			}
			return other, dst
		}
		return op{
			name: name,
			plain: func(dst, dm raster.Float32Raster) {
				dx, dy := pick(dst, raster.NewFloat32Like(dm))
				terrain.Gradient(dx, dy, dm, gopts)
			},
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				dx, dy := pick(dst, raster.NewFloat32Like(dm))
				return terrain.GradientTiled(ctx, dx, dy, dm, gopts, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				w, h := dst.Size()
				spare := engine.NewMemorySink(maskedLike(dst.Masked(), w, h))
				if wantDX {
					return terrain.GradientChunked(ctx, dst, spare, src, gopts, eo)
				}
				return terrain.GradientChunked(ctx, spare, dst, src, gopts, eo)
			},
		}
	}

	curvature := func(ct terrain.CurvatureType, name string) op {
		o := terrain.CurvatureOptions{CellSize: d.cellX, CellSizeY: d.cellY, Type: ct}
		return op{
			name:  "curvature_" + name,
			plain: func(dst, dm raster.Float32Raster) { terrain.Curvature(dst, dm, o) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.CurvatureTiled(ctx, dst, dm, o, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.CurvatureChunked(ctx, dst, src, o, eo)
			},
		}
	}

	return []op{
		slope(terrain.SlopeDegrees),
		slope(terrain.SlopeRadians),
		slope(terrain.SlopePercent),
		grad(true),
		grad(false),
		{
			name:  "aspect",
			plain: func(dst, dm raster.Float32Raster) { terrain.Aspect(dst, dm, ao) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.AspectTiled(ctx, dst, dm, ao, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.AspectChunked(ctx, dst, src, ao, eo)
			},
		},
		{
			// Not a terrain operation, but it needs a whole DEM and all
			// three forms, which is what this list runs.
			name:  "normalize",
			plain: func(dst, dm raster.Float32Raster) { algebra.Normalize(dst, dm) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				_, _, err := algebra.NormalizeTiled(ctx, dst, dm, eo)
				return err
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				_, _, err := algebra.NormalizeChunked(ctx, dst, src, eo)
				return err
			},
		},
		{
			name:     "hillshade",
			azimuth:  ho.Azimuth,
			altitude: ho.Altitude,
			plain:    func(dst, dm raster.Float32Raster) { terrain.Hillshade(dst, dm, ho) },
			tiled: func(ctx context.Context, dst, dm raster.Float32Raster, eo engine.Options) error {
				return terrain.HillshadeTiled(ctx, dst, dm, ho, eo)
			},
			chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, eo engine.Options) error {
				return terrain.HillshadeChunked(ctx, dst, src, ho, eo)
			},
		},
		curvature(terrain.CurvatureProfile, "profile"),
		curvature(terrain.CurvaturePlan, "plan"),
		curvature(terrain.CurvatureMean, "mean"),
	}
}

func (d dem) runTerrain(m *manifest) error {
	ctx := context.Background()
	for _, o := range d.ops() {
		// Plain, in memory.
		plain := raster.NewFloat32Like(d.r)
		o.plain(plain, d.r)
		if err := d.emit(m, o, "plain", plain); err != nil {
			return err
		}

		// Tiled, in memory, on several workers with ragged tiles.
		tiled := raster.NewFloat32Like(d.r)
		if err := o.tiled(ctx, tiled, d.r, tiledOpts); err != nil {
			return err
		}
		if err := d.emit(m, o, "tiled", tiled); err != nil {
			return err
		}

		// Chunked, reading and writing raw float32 files.
		chunked, err := d.runChunked(ctx, o)
		if err != nil {
			return err
		}
		if err := d.emit(m, o, "chunked", chunked); err != nil {
			return err
		}
	}
	return nil
}

// runChunked runs o's Chunked form from the DEM file on disk into a
// temporary raw file, and reads the result back.
func (d dem) runChunked(ctx context.Context, o op) (raster.Float32Raster, error) {
	var zero raster.Float32Raster
	ro := engine.RawOptions{Fill: fill, HasFill: d.r.Valid != nil}

	in, err := engine.OpenRawFile(filepath.Join(*dir, d.name+".f32"), os.O_RDONLY, 0, 4)
	if err != nil {
		return zero, err
	}
	defer in.Close()

	tmp := filepath.Join(*dir, d.name+"-"+o.name+".chunked.tmp")
	out, err := engine.OpenRawFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, 4)
	if err != nil {
		return zero, err
	}
	src := engine.NewRawSource(in, width, height, ro)
	sink := engine.NewRawSink(out, width, height, ro)
	if err := o.chunked(ctx, sink, src, chunkedOpts); err != nil {
		out.Close()
		return zero, err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return zero, err
	}
	// Read it back through the same file interface, one whole-raster
	// window, so what lands in the manifest is what is on disk.
	back := raster.NewFloat32Like(d.r)
	if err := engine.NewRawSource(out, width, height, ro).ReadWindow(ctx, back, 0, 0); err != nil {
		out.Close()
		return zero, err
	}
	if err := out.Close(); err != nil {
		return zero, err
	}
	if err := os.Remove(tmp); err != nil {
		return zero, err
	}
	return back, nil
}

func (d dem) emit(m *manifest, o op, form string, r raster.Float32Raster) error {
	base := d.name + "-" + o.name + "-" + form
	c := rasterCase{
		Name:      base,
		Surface:   d.surface,
		Op:        o.name,
		Form:      form,
		DEM:       d.name + ".f32",
		Out:       base + ".f32",
		CellSize:  d.cellX,
		CellSizeY: d.cellY,
		Azimuth:   o.azimuth,
		Altitude:  o.altitude,
	}
	if d.r.Valid != nil {
		c.DEMMask = d.name + ".mask.u8"
		c.OutMask = base + ".mask.u8"
		if err := writeMask(filepath.Join(*dir, c.OutMask), r); err != nil {
			return err
		}
	}
	if err := writeRaster(filepath.Join(*dir, c.Out), r); err != nil {
		return err
	}
	m.Rasters = append(m.Rasters, c)
	return nil
}

func (d dem) runReduce(m *manifest) error {
	ctx := context.Background()
	add := func(o, form string, mn, mx float32, n int64, hasMinMax bool) {
		s := scalarCase{Name: d.name + "-" + o + "-" + form, Op: o, Form: form, DEM: d.name + ".f32", Count: n}
		if d.r.Valid != nil {
			s.DEMMask = d.name + ".mask.u8"
		}
		if hasMinMax {
			s.Min = f32(mn)
			s.Max = f32(mx)
		}
		m.Scalars = append(m.Scalars, s)
	}

	add("count", "plain", 0, 0, reduce.Count(d.r), false)
	n, err := reduce.CountTiled(ctx, d.r, tiledOpts)
	if err != nil {
		return err
	}
	add("count", "tiled", 0, 0, n, false)

	mn, mx, n := reduce.MinMax(d.r)
	add("minmax", "plain", mn, mx, n, true)
	mn, mx, n, err = reduce.MinMaxTiled(ctx, d.r, tiledOpts)
	if err != nil {
		return err
	}
	add("minmax", "tiled", mn, mx, n, true)

	ro := engine.RawOptions{Fill: fill, HasFill: d.r.Valid != nil}
	in, err := engine.OpenRawFile(filepath.Join(*dir, d.name+".f32"), os.O_RDONLY, 0, 4)
	if err != nil {
		return err
	}
	defer in.Close()
	n, err = reduce.CountChunked(ctx, engine.NewRawSource(in, width, height, ro), chunkedOpts)
	if err != nil {
		return err
	}
	add("count", "chunked", 0, 0, n, false)
	mn, mx, n, err = reduce.MinMaxChunked(ctx, engine.NewRawSource(in, width, height, ro), chunkedOpts)
	if err != nil {
		return err
	}
	add("minmax", "chunked", mn, mx, n, true)
	return nil
}

// runAlgebra exports the pointwise operations on two of the DEMs.
func runAlgebra(m *manifest, a, b dem) error {
	ctx := context.Background()
	cases := []struct {
		name  string
		plain func(dst raster.Float32Raster)
		tiled func(ctx context.Context, dst raster.Float32Raster) error
	}{
		{"add", func(dst raster.Float32Raster) { algebra.Add(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.AddTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"sub", func(dst raster.Float32Raster) { algebra.Sub(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.SubTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"mul", func(dst raster.Float32Raster) { algebra.Mul(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MulTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"min", func(dst raster.Float32Raster) { algebra.Min(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MinTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"max", func(dst raster.Float32Raster) { algebra.Max(dst, a.r, b.r) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.MaxTiled(ctx, dst, a.r, b.r, tiledOpts)
			}},
		{"clamp", func(dst raster.Float32Raster) { algebra.Clamp(dst, b.r, 50, 300) },
			func(ctx context.Context, dst raster.Float32Raster) error {
				return algebra.ClampTiled(ctx, dst, b.r, 50, 300, tiledOpts)
			}},
	}
	for _, c := range cases {
		for _, form := range []string{"plain", "tiled"} {
			dst := raster.NewFloat32Like(a.r)
			if form == "plain" {
				c.plain(dst)
			} else if err := c.tiled(ctx, dst); err != nil {
				return err
			}
			base := "algebra-" + c.name + "-" + form
			if err := writeRaster(filepath.Join(*dir, base+".f32"), dst); err != nil {
				return err
			}
			m.Rasters = append(m.Rasters, rasterCase{
				Name: base, Surface: "algebra", Op: "algebra_" + c.name, Form: form,
				DEM: a.name + ".f32", Out: base + ".f32",
			})
		}
	}
	return nil
}

// --- surfaces -------------------------------------------------------

func plane(a, b float64) raster.Float32Raster {
	data := make([]float32, width*height)
	for y := range height {
		for x := range width {
			data[y*width+x] = float32(a*float64(x) + b*float64(y) + 100)
		}
	}
	return raster.NewFloat32(width, height, data)
}

func hill() raster.Float32Raster {
	data := make([]float32, width*height)
	cx, cy := float64(width)/2, float64(height)/2
	for y := range height {
		for x := range width {
			dx, dy := float64(x)-cx, float64(y)-cy
			data[y*width+x] = float32(500 * math.Exp(-(dx*dx+dy*dy)/(2*40*40)))
		}
	}
	return raster.NewFloat32(width, height, data)
}

func noisy() raster.Float32Raster {
	r := hill()
	// A fixed 64-bit LCG, so the surface is the same on every machine.
	state := uint64(0x2545F4914F6CDD1D)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>11) / float64(uint64(1)<<53)
	}
	for i := range r.Data {
		r.Data[i] += float32(40 * (next() - 0.5))
	}
	r.Valid = raster.NewMask(width * height)
	raster.MaskFillRange(r.Valid, 0, width*height, true)
	// A disc of NoData, and a vertical stripe that crosses tile edges.
	for y := range height {
		for x := range width {
			dx, dy := float64(x)-70, float64(y)-60
			if dx*dx+dy*dy < 25*25 || (x >= 180 && x < 186) {
				r.SetValid(x, y, false)
				r.Data[y*width+x] = fill
			}
		}
	}
	return r
}

func maskedLike(masked bool, w, h int) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	if masked {
		r.Valid = raster.NewMask(w * h)
		raster.MaskFillRange(r.Valid, 0, w*h, true)
	}
	return r
}

// --- output ---------------------------------------------------------

func writeRaster(path string, r raster.Float32Raster) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<16)
	buf := make([]byte, 4*r.Width)
	for y := range r.Height {
		row := r.Row(y)
		for x, v := range row[:r.Width] {
			binary.LittleEndian.PutUint32(buf[4*x:], math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func writeMask(path string, r raster.Float32Raster) error {
	buf := make([]byte, r.Width*r.Height)
	for y := range r.Height {
		for x := range r.Width {
			if r.IsValid(x, y) {
				buf[y*r.Width+x] = 1
			}
		}
	}
	return os.WriteFile(path, buf, 0o644)
}

func f32(v float32) string { return strconv.FormatFloat(float64(v), 'g', -1, 32) }

func unitName(u terrain.SlopeUnits) string {
	switch u {
	case terrain.SlopeDegrees:
		return "deg"
	case terrain.SlopeRadians:
		return "rad"
	default:
		return "pct"
	}
}
