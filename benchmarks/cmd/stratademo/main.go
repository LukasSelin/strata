// Command stratademo runs the first validation target (DESIGN.md §43): a
// large DEM stored as a raw float32 file, processed with bounded memory,
// in tiles, on several workers and with SIMD, checked against the
// whole-raster result, with peak memory measured against the §27 bound.
//
//	GOEXPERIMENT=simd go run ./benchmarks/cmd/stratademo -dir D:\strata-demo > demo.txt
//	go run ./benchmarks/cmd/stratademo -render demo.txt
//
// In -dir it writes dem-<size>.f32, a DEM of size² cells (1.49 GiB at
// 20000²) generated a row at a time, unless it exists, and for each
// operation a reference result ref-<op>-<size>.f32 computed by the plain
// function with the whole raster in memory. Then, for every operation,
// tile shape, backend and worker count, and -count times, it runs the
// operation's Chunked entry point from the DEM file to out-<size>.f32 and
// compares that file with the reference, cell for cell (any NaN matching
// any NaN).
//
// Every run and every reference is a separate child process, so that its
// peak memory is its own: the child reports the operating system's peak
// private bytes and peak working set, and its private bytes just before
// the call. The files are opened before, and the call alone is timed:
// for the reference, the plain function on a DEM already read and an
// output whose pages are already touched. Files are opened as
// engine.RawFile with one handle per worker. The
// output file is then flushed to disk (Sync), timed separately, so that
// the next run does not compete with the OS writing it out.
//
// The output is one "run: {json}" line per run, then the tables that
// -render prints from those lines.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"strata/algebra"
	"strata/benchmarks/internal/suite"
	"strata/engine"
	"strata/internal/stencil"
	"strata/internal/vec"
	"strata/raster"
	"strata/terrain"
)

var (
	dirFlag     = flag.String("dir", "", "directory for the DEM, reference and output files (required)")
	sizesFlag   = flag.String("sizes", "20000", "comma-separated square DEM sizes")
	opsFlag     = flag.String("ops", "slope,hillshade,clamp", "comma-separated operations: slope, hillshade, clamp")
	tilesFlag   = flag.String("tiles", "strips256,1024x1024", "comma-separated tile shapes: strips<rows> (full width) or <w>x<h>")
	workersFlag = flag.String("workers", "", "comma-separated SIMD worker counts; default 1, physical cores, logical CPUs")
	countFlag   = flag.Int("count", 3, "runs per case")
	scalarFlag  = flag.Bool("scalar", true, "also run the scalar backend with one worker, in the first tile shape")
	renderFlag  = flag.String("render", "", "print the tables for the run lines in this file and exit")

	childFlag = flag.String("child", "", "internal: run one measurement (reference or run) and print its JSON")
	opFlag    = flag.String("op", "", "internal: the child's operation")
	sizeFlag  = flag.Int("size", 0, "internal: the child's DEM size")
	tileFlag  = flag.String("tile", "", "internal: the child's tile shape")
	wFlag     = flag.Int("w", 1, "internal: the child's worker count")
	backend   = flag.String("backend", "simd", "internal: the child's backend, scalar or simd")
)

func main() {
	flag.Parse()
	var err error
	switch {
	case *renderFlag != "":
		err = renderFile(os.Stdout, *renderFlag)
	case *childFlag != "":
		err = child()
	default:
		err = parent()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "stratademo:", err)
		os.Exit(1)
	}
}

// result is one measured run or reference.
type result struct {
	Kind    string  `json:"kind"` // "reference" or "run"
	Op      string  `json:"op"`
	Size    int     `json:"size"`
	Tiles   string  `json:"tiles"` // "plain" for the reference
	TileW   int     `json:"tileW"`
	TileH   int     `json:"tileH"`
	Workers int     `json:"workers"` // requested
	Used    int     `json:"used"`    // min(workers, tiles)
	Backend string  `json:"backend"`
	Kernels string  `json:"kernels"`
	Handles int     `json:"handles"` // file handles per file (engine.RawFile)
	Radius  int     `json:"radius"`
	Seconds float64 `json:"seconds"`
	SyncSec float64 `json:"syncSeconds"`
	// Memory, in bytes: private bytes before the call, peak private bytes
	// and peak working set of the process, and the §27 bound.
	BasePrivate    uint64 `json:"basePrivate"`
	PeakPrivate    uint64 `json:"peakPrivate"`
	PeakWorkingSet uint64 `json:"peakWorkingSet"`
	Bound          uint64 `json:"bound"`
	// Identical is set by the parent after comparing with the reference.
	Identical *bool `json:"identical,omitempty"`
}

// ops are the operations, with their radius, plain and chunked forms.
var ops = map[string]struct {
	radius  int
	plain   func(dst, dem raster.Float32Raster)
	chunked func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error
}{
	"slope": {1,
		func(dst, dem raster.Float32Raster) { terrain.Slope(dst, dem, terrain.SlopeOptions{CellSize: 10}) },
		func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return terrain.SlopeChunked(ctx, dst, dem, terrain.SlopeOptions{CellSize: 10}, o)
		}},
	"hillshade": {1,
		func(dst, dem raster.Float32Raster) {
			terrain.Hillshade(dst, dem, terrain.HillshadeOptions{CellSize: 10})
		},
		func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return terrain.HillshadeChunked(ctx, dst, dem, terrain.HillshadeOptions{CellSize: 10}, o)
		}},
	"clamp": {0,
		func(dst, dem raster.Float32Raster) { algebra.Clamp(dst, dem, 700, 900) },
		func(ctx context.Context, dst engine.RasterSink, dem engine.RasterSource, o engine.Options) error {
			return algebra.ClampChunked(ctx, dst, dem, 700, 900, o)
		}},
}

func demPath(dir string, size int) string { return filepath.Join(dir, fmt.Sprintf("dem-%d.f32", size)) }
func refPath(dir, op string, size int) string {
	return filepath.Join(dir, fmt.Sprintf("ref-%s-%d.f32", op, size))
}
func outPath(dir string, size int) string { return filepath.Join(dir, fmt.Sprintf("out-%d.f32", size)) }

// parseTiles resolves a tile shape for a size² raster. Both dimensions
// must be positive: 0 would divide by zero when counting tiles, and a
// negative one is not an engine.Options value.
func parseTiles(shape string, size int) (w, h int, err error) {
	if rows, ok := strings.CutPrefix(shape, "strips"); ok {
		w = size
		h, err = strconv.Atoi(rows)
	} else if ws, hs, ok := strings.Cut(shape, "x"); !ok {
		return 0, 0, fmt.Errorf("bad tile shape %q", shape)
	} else if w, err = strconv.Atoi(ws); err == nil {
		h, err = strconv.Atoi(hs)
	}
	if err != nil {
		return 0, 0, err
	}
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("bad tile shape %q: dimensions must be positive", shape)
	}
	return w, h, nil
}

func ints(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("bad count %q in %q", f, s)
		}
		out = append(out, v)
	}
	return out, nil
}

func parent() error {
	if *dirFlag == "" {
		return errors.New("-dir is required")
	}
	if err := os.MkdirAll(*dirFlag, 0o750); err != nil {
		return err
	}
	sizes, err := ints(*sizesFlag)
	if err != nil {
		return err
	}
	workers := suite.Workers()
	if *workersFlag != "" {
		if workers, err = ints(*workersFlag); err != nil {
			return err
		}
	}
	opNames := strings.Split(*opsFlag, ",")
	shapes := strings.Split(*tilesFlag, ",")
	for _, op := range opNames {
		if _, ok := ops[op]; !ok {
			return fmt.Errorf("unknown operation %q", op)
		}
	}

	suite.PrintConfig(os.Stdout, suite.Kernels{Name: "vec", Backend: vec.Backend, UseScalar: vec.UseScalar},
		suite.Kernels{Name: "stencil", Backend: stencil.Backend, UseScalar: stencil.UseScalar})
	var results []result
	emit := func(r result) {
		line, _ := json.Marshal(r)
		fmt.Printf("run: %s\n", line)
		results = append(results, r)
	}
	for _, size := range sizes {
		if err := generate(demPath(*dirFlag, size), size); err != nil {
			return err
		}
		if err := copyFile(outPath(*dirFlag, size), demPath(*dirFlag, size)); err != nil {
			return err
		}
		for _, op := range opNames {
			ref, err := runChild("reference", op, size, "plain", 1, "simd")
			if err != nil {
				return err
			}
			emit(ref)
			for si, shape := range shapes {
				type run struct {
					workers int
					backend string
				}
				var runs []run
				if *scalarFlag && si == 0 {
					runs = append(runs, run{1, "scalar"})
				}
				for _, w := range workers {
					runs = append(runs, run{w, "simd"})
				}
				for _, rn := range runs {
					for range *countFlag {
						r, err := runChild("run", op, size, shape, rn.workers, rn.backend)
						if err != nil {
							return err
						}
						same, err := sameFiles(outPath(*dirFlag, size), refPath(*dirFlag, op, size))
						if err != nil {
							return err
						}
						r.Identical = &same
						emit(r)
						if !same {
							return fmt.Errorf("%s %d² %s workers=%d %s: output differs from the whole-raster result",
								op, size, shape, rn.workers, rn.backend)
						}
					}
				}
			}
		}
	}
	fmt.Println()
	render(os.Stdout, results)
	return nil
}

// runChild runs one measurement in a child process and parses its JSON.
func runChild(kind, op string, size int, tiles string, workers int, backend string) (result, error) {
	self, err := os.Executable()
	if err != nil {
		return result{}, err
	}
	cmd := exec.Command(self, // #nosec G204 -- this binary, rerun as a child with its own flags
		"-child", kind, "-dir", *dirFlag, "-op", op, "-size", strconv.Itoa(size),
		"-tile", tiles, "-w", strconv.Itoa(workers), "-backend", backend)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return result{}, fmt.Errorf("child %s %s %d %s %d %s: %w", kind, op, size, tiles, workers, backend, err)
	}
	var r result
	if err := json.Unmarshal(out, &r); err != nil {
		return result{}, fmt.Errorf("child output %q: %w", out, err)
	}
	return r, nil
}

func child() error {
	op, ok := ops[*opFlag]
	if !ok {
		return fmt.Errorf("unknown operation %q", *opFlag)
	}
	scalar := *backend == "scalar"
	vec.UseScalar(scalar)
	stencil.UseScalar(scalar)
	size := *sizeFlag
	r := result{Kind: *childFlag, Op: *opFlag, Size: size, Tiles: *tileFlag, Workers: *wFlag, Backend: *backend,
		Kernels: stencil.Backend() + "/" + vec.Backend(), Radius: op.radius}
	ctx := context.Background()

	// One handle per worker, so that workers do not queue on a handle.
	r.Handles = max(1, *wFlag)
	in, err := engine.OpenRawFile(demPath(*dirFlag, size), os.O_RDONLY, 0, r.Handles)
	if err != nil {
		return err
	}
	defer in.Close()
	var outFile string
	switch *childFlag {
	case "reference":
		outFile = refPath(*dirFlag, *opFlag, size)
	case "run":
		outFile = outPath(*dirFlag, size)
	default:
		return fmt.Errorf("unknown child %q", *childFlag)
	}
	out, err := engine.OpenRawFile(outFile, os.O_RDWR|os.O_CREATE, 0o644, r.Handles)
	if err != nil {
		return err
	}
	defer out.Close()
	src := engine.NewRawSource(in, size, size, engine.RawOptions{})
	sink := engine.NewRawSink(out, size, size, engine.RawOptions{})

	var elapsed time.Duration
	switch *childFlag {
	case "reference":
		// The whole raster in memory: read, compute (timed), write.
		dem := raster.NewFloat32(size, size, make([]float32, size*size))
		dst := raster.NewFloat32Like(dem)
		base, _ := suite.ProcessMemory()
		r.BasePrivate = base.Private
		if err := src.ReadWindow(ctx, dem, 0, 0); err != nil {
			return err
		}
		// Fault dst's pages in first, as the chunked runs' output file is
		// in the OS cache: the reference times the computation only.
		for i := range dst.Data {
			dst.Data[i] = 1
		}
		start := time.Now()
		op.plain(dst, dem)
		elapsed = time.Since(start)
		if err := sink.WriteWindow(ctx, dst, 0, 0); err != nil {
			return err
		}
		r.TileW, r.TileH, r.Used = size, size, 1
	case "run":
		tw, th, err := parseTiles(*tileFlag, size)
		if err != nil {
			return err
		}
		tw, th = min(tw, size), min(th, size)
		tiles := ((size + tw - 1) / tw) * ((size + th - 1) / th)
		r.TileW, r.TileH, r.Used = tw, th, min(*wFlag, tiles)
		// §27: Workers × (TileW+2r) × (TileH+2r) × Σ bytes per cell, one
		// float32 input and one output.
		// #nosec G115 -- tile sizes are positive.
		cells := uint64(min(tw+2*op.radius, size)) * uint64(min(th+2*op.radius, size))
		// #nosec G115 -- the worker count is positive.
		r.Bound = uint64(r.Used) * cells * 8
		base, _ := suite.ProcessMemory()
		r.BasePrivate = base.Private
		start := time.Now()
		if err := op.chunked(ctx, sink, src, engine.Options{TileWidth: tw, TileHeight: th, Workers: *wFlag}); err != nil {
			return err
		}
		elapsed = time.Since(start)
	}
	r.Seconds = elapsed.Seconds()
	start := time.Now()
	if err := out.Sync(); err != nil {
		return err
	}
	r.SyncSec = time.Since(start).Seconds()
	peak, ok := suite.ProcessMemory()
	if !ok {
		return errors.New("peak memory is not available on this platform")
	}
	r.PeakPrivate, r.PeakWorkingSet = peak.PeakPrivate, peak.PeakWorkingSet
	return json.NewEncoder(os.Stdout).Encode(r)
}

// generate writes a size² DEM to path, unless the file exists with the
// right length: the benchmarks/engine surface of 800 ± 300 with gentle
// slopes plus uniform noise of ±1, a row at a time.
func generate(path string, size int) error {
	if st, err := os.Stat(path); err == nil && st.Size() == 4*int64(size)*int64(size) {
		return nil
	}
	f, err := os.Create(path) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<22)
	cols := make([]float32, size)
	for x := range cols {
		cols[x] = float32(300 * math.Sin(float64(x)/97))
	}
	row := make([]float32, size)
	buf := make([]byte, 4*size)
	for y := range size {
		suite.FillUniform(row, uint64(y)+1, -1, 1)
		c := float32(math.Cos(float64(y) / 131))
		for x := range row {
			row[x] += 800 + cols[x]*c
			binary.LittleEndian.PutUint32(buf[4*x:], math.Float32bits(row[x]))
		}
		if _, err := w.Write(buf); err != nil {
			_ = f.Close() // the write error matters more
			return err
		}
	}
	if err := w.Flush(); err != nil {
		_ = f.Close() // the flush error matters more
		return err
	}
	return f.Close()
}

// copyFile copies src to dst, so that the output file exists at full size
// before the first run.
func copyFile(dst, src string) error {
	if st, err := os.Stat(dst); err == nil {
		if ss, err := os.Stat(src); err == nil && st.Size() == ss.Size() {
			return nil
		}
	}
	in, err := os.Open(src) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close() // the copy error matters more
		return err
	}
	return out.Close()
}

// sameFiles compares two raw float32 files cell for cell, any NaN
// matching any NaN, reading 16 MiB at a time.
func sameFiles(a, b string) (bool, error) {
	fa, err := os.Open(a) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return false, err
	}
	defer fa.Close()
	fb, err := os.Open(b) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return false, err
	}
	defer fb.Close()
	ba, bb := make([]byte, 1<<24), make([]byte, 1<<24)
	for {
		na, ea := io.ReadFull(fa, ba)
		nb, eb := io.ReadFull(fb, bb)
		if na != nb || na%4 != 0 {
			return false, nil
		}
		for i := 0; i < na; i += 4 {
			x, y := binary.LittleEndian.Uint32(ba[i:]), binary.LittleEndian.Uint32(bb[i:])
			if x != y {
				fx, fy := math.Float32frombits(x), math.Float32frombits(y)
				if fx == fx || fy == fy {
					return false, nil
				}
			}
		}
		if ea == io.EOF || ea == io.ErrUnexpectedEOF {
			return eb == ea, nil
		}
		if ea != nil {
			return false, ea
		}
		if eb != nil {
			return false, eb
		}
	}
}

func renderFile(w io.Writer, path string) error {
	f, err := os.Open(path) // #nosec G304 -- a file in the -dir the user chose
	if err != nil {
		return err
	}
	defer f.Close()
	results, err := parseRuns(f)
	if err != nil {
		return err
	}
	render(w, results)
	return nil
}

func parseRuns(r io.Reader) ([]result, error) {
	var out []result
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "run: ")
		if !ok {
			continue
		}
		var res result
		if err := json.Unmarshal([]byte(line), &res); err != nil {
			return nil, fmt.Errorf("%q: %w", line, err)
		}
		out = append(out, res)
	}
	return out, sc.Err()
}
