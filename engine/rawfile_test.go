package engine_test

import (
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// mapsFiles is whether CreateRawFile maps files on this system.
var mapsFiles = runtime.GOOS == "linux" || runtime.GOOS == "windows"

// TestCreateRawFile writes a raster into a created file from many
// goroutines in any order, and checks that the file, read back through
// the RawFile and from disk after Close, holds its raw encoding.
func TestCreateRawFile(t *testing.T) {
	const w, h = 257, 131
	rng := rand.New(rand.NewPCG(8, 8))
	path := filepath.Join(t.TempDir(), "out.f32")
	// A file in the way is truncated.
	must(t, os.WriteFile(path, bytes.Repeat([]byte{0xAB}, 4*w*h+999), 0o644))
	f, err := engine.CreateRawFile(path, 4*w*h, 0o644, 3)
	must(t, err)
	if f.Mapped() != mapsFiles {
		t.Fatalf("Mapped() = %v on %s", f.Mapped(), runtime.GOOS)
	}
	if fi, err := os.Stat(path); err != nil || fi.Size() != 4*w*h {
		t.Fatalf("size after create: %v, %v; want %d", fi, err, 4*w*h)
	}
	src, _ := window(rng, w, h, false)
	sink := engine.NewRawSink(f, w, h, engine.RawOptions{})
	var wg sync.WaitGroup
	for _, ty := range rng.Perm(h/7 + 1) {
		for tx := 0; tx < w; tx += 100 {
			wg.Go(func() {
				tile := src.Window(tx, 7*ty, min(100, w-tx), min(7, h-7*ty))
				if err := sink.WriteWindow(context.Background(), tile, tx, 7*ty); err != nil {
					t.Error(err)
				}
			})
		}
	}
	wg.Wait()
	must(t, f.Sync())
	back := raster.NewFloat32(w, h, make([]float32, w*h))
	must(t, engine.NewRawSource(f, w, h, engine.RawOptions{}).ReadWindow(context.Background(), back, 0, 0))
	requireCells(t, "through the file", back, 0, 0, src, 0, 0, w, h, true)
	must(t, f.Close())
	got, err := os.ReadFile(path)
	must(t, err)
	if !bytes.Equal(got, rawBytes(src)) {
		t.Fatal("file on disk after Close differs from the raw encoding")
	}
	if _, err := f.WriteAt([]byte{1}, 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("write after Close: err = %v, want os.ErrClosed", err)
	}
	if _, err := f.ReadAt(make([]byte, 1), 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("read after Close: err = %v, want os.ErrClosed", err)
	}
}

// TestCreateRawFileEdges checks calls that are not inside the mapping:
// past it, straddling its end, and a file of size 0.
func TestCreateRawFileEdges(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edge")
	f, err := engine.CreateRawFile(path, 10, 0o644, 2)
	must(t, err)
	// Straddling the end: 4 bytes in the mapping, 4 through a handle.
	if n, err := f.WriteAt([]byte("abcdefgh"), 6); n != 8 || err != nil {
		t.Fatalf("straddling write: %d, %v", n, err)
	}
	// Past the end: through a handle, growing the file.
	if n, err := f.WriteAt([]byte("XY"), 20); n != 2 || err != nil {
		t.Fatalf("write past the end: %d, %v", n, err)
	}
	buf := make([]byte, 22)
	if n, err := f.ReadAt(buf, 0); n != 22 || err != nil {
		t.Fatalf("read across the end: %d, %v", n, err)
	}
	want := append(make([]byte, 6), "abcdefgh\x00\x00\x00\x00\x00\x00XY"...)
	if !bytes.Equal(buf, want) {
		t.Fatalf("read back %q, want %q", buf, want)
	}
	must(t, f.Close())
	got, err := os.ReadFile(path)
	must(t, err)
	if !bytes.Equal(got, want) {
		t.Fatalf("on disk %q, want %q", got, want)
	}

	empty, err := engine.CreateRawFile(filepath.Join(dir, "empty"), 0, 0o644, 1)
	must(t, err)
	if empty.Mapped() {
		t.Fatal("a file of size 0 is mapped")
	}
	must(t, empty.Close())
	if _, err := engine.CreateRawFile(filepath.Join(dir, "neg"), -1, 0o644, 1); err == nil {
		t.Fatal("CreateRawFile accepted a negative size")
	}
	if _, err := engine.CreateRawFile(filepath.Join(dir, "missing", "x"), 8, 0o644, 1); err == nil {
		t.Fatal("CreateRawFile created a file in a missing directory")
	}
}

// TestCreateRawFileFault checks that a memory fault in the mapping, the
// way a full disk or tmpfs reports itself to a mapped write, is an error
// and not a crash. Truncating the file under the mapping makes its pages
// unbackable, which raises the same SIGBUS on Linux. Windows refuses to
// truncate a mapped file, so there is nothing to provoke there.
func TestCreateRawFileFault(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("provoking a fault needs a file truncated under its mapping, which only Linux allows")
	}
	const w, h = 1024, 16
	path := filepath.Join(t.TempDir(), "fault")
	f, err := engine.CreateRawFile(path, 4*w*h, 0o644, 2)
	must(t, err)
	defer f.Close()
	must(t, os.Truncate(path, 0))
	src := raster.NewFloat32(w, h, make([]float32, w*h))
	err = engine.NewRawSink(f, w, h, engine.RawOptions{}).WriteWindow(context.Background(), src, 0, 0)
	if !errors.Is(err, engine.ErrMappedFault) {
		t.Fatalf("write to a truncated mapping: err = %v, want ErrMappedFault", err)
	}
	err = engine.NewRawSource(f, w, h, engine.RawOptions{}).ReadWindow(context.Background(), src, 0, 0)
	if !errors.Is(err, engine.ErrMappedFault) {
		t.Fatalf("read from a truncated mapping: err = %v, want ErrMappedFault", err)
	}
	// A fault is not sticky: the goroutine's fault setting is restored
	// and the file works again once it is long enough.
	must(t, os.Truncate(path, 4*w*h))
	must(t, engine.NewRawSink(f, w, h, engine.RawOptions{}).WriteWindow(context.Background(), src, 0, 0))
}

// TestReuseRawFile checks that ReuseRawFile keeps an existing file's
// bytes where nothing is written, cuts a longer file to size, grows a
// shorter one, creates a missing one, and writes like CreateRawFile.
func TestReuseRawFile(t *testing.T) {
	dir := t.TempDir()
	for _, old := range []int{-1, 8, 16, 40} { // -1: no file
		path := filepath.Join(dir, "reuse")
		_ = os.Remove(path)
		if old >= 0 {
			must(t, os.WriteFile(path, bytes.Repeat([]byte{0xAB}, old), 0o644))
		}
		f, err := engine.ReuseRawFile(path, 16, 0o644, 3)
		must(t, err)
		if f.Mapped() != mapsFiles {
			t.Fatalf("old=%d: Mapped() = %v on %s", old, f.Mapped(), runtime.GOOS)
		}
		if n, err := f.WriteAt([]byte("wxyz"), 4); n != 4 || err != nil {
			t.Fatalf("old=%d: write %d, %v", old, n, err)
		}
		must(t, f.Close())
		got, err := os.ReadFile(path)
		must(t, err)
		want := make([]byte, 16)
		for i := range min(old, 16) {
			want[i] = 0xAB
		}
		copy(want[4:], "wxyz")
		if !bytes.Equal(got, want) {
			t.Fatalf("old=%d: file is %x, want %x", old, got, want)
		}
	}
}
