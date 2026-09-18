package engine_test

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"go.uber.org/goleak"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// window returns a w×h window with Stride not a multiple of 64 and an odd
// mask offset into a larger root filled with random values and bits, and
// the root.
func window(rng *rand.Rand, w, h int, masked bool) (win, root raster.Float32Raster) {
	rootW, rootH, stride := w+5, h+3, w+5+rng.IntN(9)
	if stride%64 == 0 {
		stride++
	}
	root = raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, (rootH-1)*stride+rootW))
	for i := range root.Data {
		root.Data[i] = rng.Float32()*2000 - 1000
	}
	if masked {
		root.ValidOffset = 1 + 2*rng.IntN(40)
		root.Valid = make([]uint64, raster.MaskWords(root.ValidOffset+len(root.Data))+1)
		for i := range root.Valid {
			root.Valid[i] = rng.Uint64()
		}
	}
	return root.Window(2, 1, w, h), root
}

func clone(r raster.Float32Raster) raster.Float32Raster {
	r.Data = append([]float32(nil), r.Data...)
	if r.Valid != nil {
		r.Valid = append([]uint64(nil), r.Valid...)
	}
	return r
}

func sameBits(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }

// requireCells compares got's w×h cells at (gx, gy) with want's at (wx,
// wy): Data bits on valid cells, and validity if withValid.
func requireCells(t *testing.T, id string, got raster.Float32Raster, gx, gy int, want raster.Float32Raster, wx, wy, w, h int, withValid bool) {
	t.Helper()
	for y := range h {
		for x := range w {
			gv, wv := got.IsValid(gx+x, gy+y), want.IsValid(wx+x, wy+y)
			if withValid && gv != wv {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", id, x, y, gv, wv)
			}
			g, v := got.Data[got.Index(gx+x, gy+y)], want.Data[want.Index(wx+x, wy+y)]
			if wv && !sameBits(g, v) {
				t.Fatalf("%s: cell (%d, %d) = %v, want %v", id, x, y, g, v)
			}
		}
	}
}

// requireOutside checks that got, a root made by window, equals orig
// except in the w×h cells at (x, y) of its window.
func requireOutside(t *testing.T, id string, got, orig raster.Float32Raster, x, y, w, h int) {
	t.Helper()
	inside := func(i int) bool {
		ry, rx := i/got.Stride-1-y, i%got.Stride-2-x
		return i >= 0 && i < len(got.Data) && ry >= 0 && ry < h && rx >= 0 && rx < w
	}
	for i := range got.Data {
		if !inside(i) && !sameBits(got.Data[i], orig.Data[i]) {
			t.Fatalf("%s: root Data[%d] changed outside the region", id, i)
		}
	}
	for b := range len(got.Valid) * 64 {
		if !inside(b-got.ValidOffset) && raster.MaskGet(got.Valid, b) != raster.MaskGet(orig.Valid, b) {
			t.Fatalf("%s: root mask bit %d changed outside the region", id, b)
		}
	}
}

func TestMemorySourceReadWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	for _, masked := range []bool{false, true} {
		src, _ := window(rng, 23, 17, masked)
		s := engine.NewMemorySource(src)
		if w, h := s.Size(); w != 23 || h != 17 || s.Masked() != masked {
			t.Fatalf("Size %d×%d Masked %v", w, h, s.Masked())
		}
		for _, dstMasked := range []bool{false, true} {
			if masked && !dstMasked {
				continue
			}
			for _, reg := range [][4]int{{0, 0, 23, 17}, {3, 4, 7, 5}, {22, 16, 1, 1}, {0, 5, 23, 1}} {
				// dst is a window of a root with random bits, like src.
				dst, root := window(rng, reg[2], reg[3], true)
				if !dstMasked {
					dst.Valid, root.Valid = nil, nil
					dst.ValidOffset, root.ValidOffset = 0, 0
				}
				orig := clone(root)
				id := fmt.Sprintf("masked=%v dstMasked=%v region=%v", masked, dstMasked, reg)
				if err := s.ReadWindow(context.Background(), dst, reg[0], reg[1]); err != nil {
					t.Fatal(err)
				}
				requireCells(t, id, dst, 0, 0, src, reg[0], reg[1], reg[2], reg[3], dstMasked)
				if dstMasked && !masked {
					for y := range reg[3] {
						for x := range reg[2] {
							if !dst.IsValid(x, y) {
								t.Fatalf("%s: cell (%d, %d) invalid from a source without a mask", id, x, y)
							}
						}
					}
				}
				requireOutside(t, id, root, orig, 0, 0, reg[2], reg[3])
			}
		}
	}
}

func TestMemorySinkWriteWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	for _, masked := range []bool{false, true} {
		for _, srcMasked := range []bool{false, true} {
			if srcMasked && !masked {
				continue
			}
			for _, reg := range [][4]int{{0, 0, 23, 17}, {3, 4, 7, 5}, {22, 16, 1, 1}} {
				dst, root := window(rng, 23, 17, masked)
				orig := clone(root)
				src, _ := window(rng, reg[2], reg[3], srcMasked)
				before := clone(src)
				id := fmt.Sprintf("masked=%v srcMasked=%v region=%v", masked, srcMasked, reg)
				s := engine.NewMemorySink(dst)
				if err := s.WriteWindow(context.Background(), src, reg[0], reg[1]); err != nil {
					t.Fatal(err)
				}
				requireCells(t, id, dst, reg[0], reg[1], src, 0, 0, reg[2], reg[3], masked)
				requireOutside(t, id, root, orig, reg[0], reg[1], reg[2], reg[3])
				requireCells(t, id+" src", src, 0, 0, before, 0, 0, reg[2], reg[3], true)
			}
		}
	}
}

// TestMemorySinkConcurrent writes one-cell regions of a masked sink from
// many goroutines, whose mask words are shared; run with -race.
func TestMemorySinkConcurrent(t *testing.T) {
	defer goleak.VerifyNone(t)
	const w, h = 67, 13
	dst := raster.NewFloat32(w, h, make([]float32, w*h))
	dst.Valid = raster.NewMask(w * h)
	s := engine.NewMemorySink(dst)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			for i := g; i < w*h; i += 8 {
				c := raster.Float32Raster{Data: []float32{float32(i)}, Width: 1, Height: 1, Stride: 1,
					Valid: []uint64{uint64(i % 3 % 2)}}
				if err := s.WriteWindow(context.Background(), c, i%w, i/w); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	for i := range w * h {
		if dst.Data[i] != float32(i) || dst.IsValid(i%w, i/w) != (i%3%2 == 1) {
			t.Fatalf("cell %d = %v valid %v", i, dst.Data[i], dst.IsValid(i%w, i/w))
		}
	}
}

// memFile is an in-memory io.ReaderAt and io.WriterAt, safe for
// concurrent use, that can fail.
type memFile struct {
	mu      sync.Mutex
	b       []byte
	failAt  int64 // offset from which reads and writes fail, if failErr is set
	failErr error
}

func (f *memFile) ReadAt(p []byte, off int64) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failErr != nil && off+int64(len(p)) > f.failAt {
		return 0, f.failErr
	}
	if off >= int64(len(f.b)) {
		return 0, io.EOF
	}
	n := copy(p, f.b[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (f *memFile) WriteAt(p []byte, off int64) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failErr != nil && off+int64(len(p)) > f.failAt {
		return 0, f.failErr
	}
	if end := off + int64(len(p)); end > int64(len(f.b)) {
		f.b = append(f.b, make([]byte, end-int64(len(f.b)))...)
	}
	return copy(f.b[off:], p), nil
}

// rawBytes encodes cells row-major as little-endian float32.
func rawBytes(r raster.Float32Raster) []byte {
	var out []byte
	for y := range r.Height {
		for _, c := range r.Row(y) {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(c))
		}
	}
	return out
}

func TestRawSourceReadWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	const w, h = 29, 11
	data, _ := window(rng, w, h, false)
	// Hazards: NaNs with payloads, infinities, signed zeros.
	data.Data[data.Index(1, 0)] = math.Float32frombits(0x7fc0_0001)
	data.Data[data.Index(2, 0)] = float32(math.Inf(-1))
	data.Data[data.Index(3, 0)] = float32(math.Copysign(0, -1))
	data.Data[data.Index(4, 1)] = -9999
	data.Data[data.Index(5, 2)] = 0
	f := &memFile{b: rawBytes(data)}
	for _, reg := range [][4]int{{0, 0, w, h}, {1, 0, 5, 3}, {28, 10, 1, 1}, {0, 4, w, 2}} {
		dst, root := window(rng, reg[2], reg[3], true)
		orig := clone(root)
		id := fmt.Sprintf("region=%v", reg)
		s := engine.NewRawSource(f, w, h, engine.RawOptions{})
		if s.Masked() {
			t.Fatal("RawSource without a fill value is Masked")
		}
		if err := s.ReadWindow(context.Background(), dst, reg[0], reg[1]); err != nil {
			t.Fatal(err)
		}
		requireCells(t, id, dst, 0, 0, data, reg[0], reg[1], reg[2], reg[3], true)
		requireOutside(t, id, root, orig, 0, 0, reg[2], reg[3])

		for _, fill := range []float32{-9999, 0, float32(math.NaN())} {
			id := fmt.Sprintf("%s fill=%v", id, fill)
			s := engine.NewRawSource(f, w, h, engine.RawOptions{Fill: fill, HasFill: true})
			if !s.Masked() {
				t.Fatal("RawSource with a fill value is not Masked")
			}
			got := clone(dst)
			if err := s.ReadWindow(context.Background(), got, reg[0], reg[1]); err != nil {
				t.Fatal(err)
			}
			for y := range reg[3] {
				for x := range reg[2] {
					v := data.Data[data.Index(reg[0]+x, reg[1]+y)]
					wantValid := v != fill && !(fill != fill && v != v)
					if got.IsValid(x, y) != wantValid || !sameBits(got.Data[got.Index(x, y)], v) {
						t.Fatalf("%s: cell (%d, %d) = %v valid %v, want %v valid %v", id, x, y,
							got.Data[got.Index(x, y)], got.IsValid(x, y), v, wantValid)
					}
				}
			}
		}
	}
}

func TestRawSinkWriteWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	const w, h = 29, 11
	for _, fill := range []float32{-9999, float32(math.NaN())} {
		for _, reg := range [][4]int{{0, 0, w, h}, {1, 2, 5, 3}, {28, 10, 1, 1}} {
			id := fmt.Sprintf("fill=%v region=%v", fill, reg)
			f := &memFile{}
			src, _ := window(rng, reg[2], reg[3], true)
			before := clone(src)
			s := engine.NewRawSink(f, w, h, engine.RawOptions{Fill: fill, HasFill: true})
			if !s.Masked() {
				t.Fatal("RawSink with a fill value is not Masked")
			}
			if err := s.WriteWindow(context.Background(), src, reg[0], reg[1]); err != nil {
				t.Fatal(err)
			}
			if want := 4 * ((reg[1]+reg[3]-1)*w + reg[0] + reg[2]); len(f.b) != want {
				t.Fatalf("%s: file is %d bytes, want %d", id, len(f.b), want)
			}
			for y := range reg[3] {
				for x := range reg[2] {
					off := 4 * ((reg[1]+y)*w + reg[0] + x)
					got := math.Float32frombits(binary.LittleEndian.Uint32(f.b[off:]))
					want := before.Data[before.Index(x, y)]
					if !before.IsValid(x, y) {
						want = fill
					}
					if !sameBits(got, want) {
						t.Fatalf("%s: file cell (%d, %d) = %v, want %v", id, x, y, got, want)
					}
					// Only invalid cells of src may change, to the fill value.
					if before.IsValid(x, y) && !sameBits(src.Data[src.Index(x, y)], want) {
						t.Fatalf("%s: valid src cell (%d, %d) changed", id, x, y)
					}
				}
			}
			for i := range src.Valid {
				if src.Valid[i] != before.Valid[i] {
					t.Fatalf("%s: src mask changed", id)
				}
			}

			// Read back through a source with the same fill value.
			back, _ := window(rng, reg[2], reg[3], true)
			rs := engine.NewRawSource(f, w, h, engine.RawOptions{Fill: fill, HasFill: true})
			if err := rs.ReadWindow(context.Background(), back, reg[0], reg[1]); err != nil {
				t.Fatal(err)
			}
			requireCells(t, id+" read back", back, 0, 0, before, 0, 0, reg[2], reg[3], true)
		}
	}

	// Without a fill value, Data is written as is, invalid cells included.
	f := &memFile{}
	src, _ := window(rng, w, h, false)
	if err := engine.NewRawSink(f, w, h, engine.RawOptions{}).WriteWindow(context.Background(), src, 0, 0); err != nil {
		t.Fatal(err)
	}
	if string(f.b) != string(rawBytes(src)) {
		t.Fatal("unmasked write differs from the raw encoding")
	}
}

// TestRawFileOS round-trips a raster through a real file with concurrent
// writes of disjoint windows, and checks that os.File errors surface.
func TestRawFileOS(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	const w, h = 300, 97
	path := filepath.Join(t.TempDir(), "dem.f32")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	src, _ := window(rng, w, h, false)
	sink := engine.NewRawSink(file, w, h, engine.RawOptions{})
	var wg sync.WaitGroup
	for ty := 0; ty < h; ty += 10 {
		for tx := 0; tx < w; tx += 64 {
			wg.Go(func() {
				tile := src.Window(tx, ty, min(64, w-tx), min(10, h-ty))
				if err := sink.WriteWindow(context.Background(), tile, tx, ty); err != nil {
					t.Error(err)
				}
			})
		}
	}
	wg.Wait()
	back := raster.NewFloat32(w, h, make([]float32, w*h))
	if err := engine.NewRawSource(file, w, h, engine.RawOptions{}).ReadWindow(context.Background(), back, 0, 0); err != nil {
		t.Fatal(err)
	}
	requireCells(t, "os file", back, 0, 0, src, 0, 0, w, h, true)

	// A source larger than the file fails with io.ErrUnexpectedEOF.
	big := engine.NewRawSource(file, w, h+1, engine.RawOptions{})
	row := raster.NewFloat32(w, 1, make([]float32, w))
	if err := big.ReadWindow(context.Background(), row, 0, h); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("read past the end: err = %v, want io.ErrUnexpectedEOF", err)
	}
	// A closed file fails with os.ErrClosed.
	file.Close()
	if err := sink.WriteWindow(context.Background(), row, 0, 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("write to a closed file: err = %v, want os.ErrClosed", err)
	}
}

func TestRawErrors(t *testing.T) {
	defer goleak.VerifyNone(t)
	defer engine.SetRawCallBytes(4 * 10 * 2)() // two rows per call
	errDisk := errors.New("disk failed")
	f := &memFile{b: make([]byte, 4*10*10), failAt: 4 * 35, failErr: errDisk}
	dst := raster.NewFloat32(10, 10, make([]float32, 100))
	err := engine.NewRawSource(f, 10, 10, engine.RawOptions{}).ReadWindow(context.Background(), dst, 0, 0)
	if !errors.Is(err, errDisk) || !strings.Contains(err.Error(), "rows 2 to 3") {
		t.Fatalf("read: err = %v, want %v at rows 2 to 3", err, errDisk)
	}
	err = engine.NewRawSink(f, 10, 10, engine.RawOptions{}).WriteWindow(context.Background(), dst, 0, 0)
	if !errors.Is(err, errDisk) || !strings.Contains(err.Error(), "rows 2 to 3") {
		t.Fatalf("write: err = %v, want %v at rows 2 to 3", err, errDisk)
	}
	row := dst.Window(0, 0, 10, 10).Window(1, 3, 9, 1)
	err = engine.NewRawSource(f, 10, 10, engine.RawOptions{}).ReadWindow(context.Background(), row, 1, 3)
	if !errors.Is(err, errDisk) || !strings.Contains(err.Error(), "row 3:") {
		t.Fatalf("narrow read: err = %v, want %v at row 3", err, errDisk)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ok := &memFile{b: make([]byte, 400)}
	if err := engine.NewRawSource(ok, 10, 10, engine.RawOptions{}).ReadWindow(ctx, dst, 0, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled read: err = %v", err)
	}
	if err := engine.NewRawSink(ok, 10, 10, engine.RawOptions{}).WriteWindow(ctx, dst, 0, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled write: err = %v", err)
	}
}

func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("no panic, want %q", want)
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, want) {
			t.Fatalf("panic %q, want %q", msg, want)
		}
	}()
	f()
}

func TestSourceSinkPanics(t *testing.T) {
	ctx := context.Background()
	r := raster.NewFloat32(4, 3, make([]float32, 12))
	masked := raster.NewFloat32Like(raster.Float32Raster{Width: 4, Height: 3, Stride: 4, Valid: []uint64{0}})
	one := raster.NewFloat32(2, 2, make([]float32, 4))
	f := &memFile{b: make([]byte, 48)}

	mustPanic(t, "NewMemorySource", func() { engine.NewMemorySource(raster.Float32Raster{}) })
	mustPanic(t, "NewMemorySink", func() { engine.NewMemorySink(raster.Float32Raster{}) })
	mustPanic(t, "outside 4×3", func() { _ = engine.NewMemorySource(r).ReadWindow(ctx, one, 3, 0) })
	mustPanic(t, "outside 4×3", func() { _ = engine.NewMemorySink(r).WriteWindow(ctx, one, 0, -1) })
	mustPanic(t, "dst has none", func() { _ = engine.NewMemorySource(masked).ReadWindow(ctx, one, 0, 0) })
	maskedOne := raster.NewFloat32Like(raster.Float32Raster{Width: 2, Height: 2, Stride: 2, Valid: []uint64{0}})
	mustPanic(t, "the sink has none", func() { _ = engine.NewMemorySink(r).WriteWindow(ctx, maskedOne, 0, 0) })
	mustPanic(t, "dst: raster", func() { _ = engine.NewMemorySource(r).ReadWindow(ctx, raster.Float32Raster{}, 0, 0) })

	mustPanic(t, "nil file", func() { engine.NewRawSource(nil, 4, 3, engine.RawOptions{}) })
	mustPanic(t, "nil file", func() { engine.NewRawSink(nil, 4, 3, engine.RawOptions{}) })
	mustPanic(t, "dimensions must be positive", func() { engine.NewRawSource(f, 0, 3, engine.RawOptions{}) })
	mustPanic(t, "do not fit in a file", func() { engine.NewRawSink(f, 1<<40, 1<<40, engine.RawOptions{}) })
	mustPanic(t, "outside 4×3", func() {
		_ = engine.NewRawSource(f, 4, 3, engine.RawOptions{}).ReadWindow(ctx, one, 0, 2)
	})
	mustPanic(t, "dst has no validity mask", func() {
		_ = engine.NewRawSource(f, 4, 3, engine.RawOptions{HasFill: true}).ReadWindow(ctx, one, 0, 0)
	})
	mustPanic(t, "the sink has no fill value", func() {
		_ = engine.NewRawSink(f, 4, 3, engine.RawOptions{}).WriteWindow(ctx, maskedOne, 0, 0)
	})
}

// countingFile is a memFile that counts calls.
type countingFile struct {
	memFile
	reads, writes int
}

func (f *countingFile) ReadAt(p []byte, off int64) (int, error) {
	f.reads++
	return f.memFile.ReadAt(p, off)
}

func (f *countingFile) WriteAt(p []byte, off int64) (int, error) {
	f.writes++
	return f.memFile.WriteAt(p, off)
}

// TestRawGroupedCalls checks that full-width windows without row padding
// are read and written in calls of several rows, bit for bit and with
// fill values, and that every other window takes a call per row.
func TestRawGroupedCalls(t *testing.T) {
	const w, h = 13, 23
	defer engine.SetRawCallBytes(4 * w * 5)() // five rows per call
	rng := rand.New(rand.NewPCG(6, 6))
	ctx := context.Background()
	for _, withFill := range []bool{false, true} {
		opts := engine.RawOptions{}
		if withFill {
			opts = engine.RawOptions{Fill: -1, HasFill: true}
		}
		data := raster.NewFloat32(w, h, make([]float32, w*h))
		if withFill {
			data.Valid = raster.NewMask(w * h)
		}
		for i := range data.Data {
			data.Data[i] = rng.Float32()
			if withFill && rng.IntN(4) == 0 {
				raster.MaskSet(data.Valid, i, false)
			}
		}
		for _, tc := range []struct {
			name         string
			x, y, rw, rh int
			calls        int
		}{
			{"whole", 0, 0, w, h, 5},
			{"full width", 0, 4, w, 11, 3},
			{"one row", 0, 7, w, 1, 1},
			{"narrow", 1, 2, w - 1, 6, 6},
		} {
			id := fmt.Sprintf("fill=%v %s", withFill, tc.name)
			f := &countingFile{memFile: memFile{b: make([]byte, 4*w*h)}}
			src := data.Window(tc.x, tc.y, tc.rw, tc.rh)
			if tc.x > 0 {
				// A compact copy: only its width keeps it from grouping.
				c := raster.NewFloat32Like(src)
				must(t, engine.NewMemorySource(src).ReadWindow(ctx, c, 0, 0))
				src = c
			} else {
				src = clone(src)
			}
			before := clone(src)
			must(t, engine.NewRawSink(f, w, h, opts).WriteWindow(ctx, src, tc.x, tc.y))
			if f.writes != tc.calls {
				t.Fatalf("%s: %d writes, want %d", id, f.writes, tc.calls)
			}
			back := raster.NewFloat32Like(before)
			must(t, engine.NewRawSource(f, w, h, opts).ReadWindow(ctx, back, tc.x, tc.y))
			if f.reads != tc.calls {
				t.Fatalf("%s: %d reads, want %d", id, f.reads, tc.calls)
			}
			requireCells(t, id, back, 0, 0, before, 0, 0, tc.rw, tc.rh, true)
		}
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// TestRawFileHandles writes and reads a raster through a RawFile with
// several handles from many goroutines.
func TestRawFileHandles(t *testing.T) {
	defer goleak.VerifyNone(t)
	const w, h = 257, 131
	rng := rand.New(rand.NewPCG(7, 7))
	path := filepath.Join(t.TempDir(), "raw.f32")
	if _, err := engine.OpenRawFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644, 2); err == nil {
		t.Fatal("OpenRawFile accepted O_APPEND")
	}
	f, err := engine.OpenRawFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644, 5)
	if err != nil {
		t.Fatal(err)
	}
	if f.Handles() != 5 {
		t.Fatalf("%d handles, want 5", f.Handles())
	}
	src, _ := window(rng, w, h, false)
	sink := engine.NewRawSink(f, w, h, engine.RawOptions{})
	var wg sync.WaitGroup
	for ty := 0; ty < h; ty += 9 {
		wg.Go(func() {
			tile := src.Window(0, ty, w, min(9, h-ty))
			if err := sink.WriteWindow(context.Background(), tile, 0, ty); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	must(t, f.Sync())
	back := raster.NewFloat32(w, h, make([]float32, w*h))
	must(t, engine.NewRawSource(f, w, h, engine.RawOptions{}).ReadWindow(context.Background(), back, 0, 0))
	requireCells(t, "handles", back, 0, 0, src, 0, 0, w, h, true)
	must(t, f.Close())
	if _, err := f.ReadAt(make([]byte, 4), 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("read after Close: err = %v, want os.ErrClosed", err)
	}
	if _, err := engine.OpenRawFile(filepath.Join(t.TempDir(), "missing", "x"), os.O_RDONLY, 0, 3); err == nil {
		t.Fatal("OpenRawFile opened a missing file")
	}
}
