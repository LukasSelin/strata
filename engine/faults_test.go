package engine_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"testing/iotest"

	"strata/engine"
	"strata/internal/faultio"
	"strata/raster"
)

// The raw source and sink read and write through io.ReaderAt and
// io.WriterAt, which the operating system implements by reading or
// writing until the buffer is full. A file behind a network or a pipe
// gives short reads instead, one buffer's worth at a time, and fails or
// times out partway through a call. internal/faultio puts
// testing/iotest's wrappers under a ReaderAt, so these tests run the same
// reads over a file that answers in pieces and check that the cells come
// out identical, and over one that fails and check that the error names
// the rows and reaches the caller whole.

// dstPair returns a function giving a fresh w×h destination with a
// validity mask, and the root that owns its memory. Both hold random
// bits, so a read that leaves a cell or a mask bit alone shows up.
// windowed gives the destination row padding and an odd mask offset, so
// every row takes its own call; a compact destination lets consecutive
// full-width rows group into one.
func dstPair(rng *rand.Rand, w, h int, windowed bool) func() (dst, root raster.Float32Raster) {
	x, y, stride, off := 0, 0, w, 0
	if windowed {
		x, y, off = 2, 1, 1+2*rng.IntN(20)
		stride = w + 2*x + 1 + rng.IntN(7)
		if stride%64 == 0 {
			stride++
		}
	}
	rootW, rootH := w+2*x, h+2*y
	root := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, (rootH-1)*stride+rootW))
	for i := range root.Data {
		root.Data[i] = math.Float32frombits(rng.Uint32())
	}
	root.ValidOffset = off
	root.Valid = make([]uint64, raster.MaskWords(off+len(root.Data))+1)
	for i := range root.Valid {
		root.Valid[i] = rng.Uint64()
	}
	return func() (raster.Float32Raster, raster.Float32Raster) {
		c := clone(root)
		return c.Window(x, y, w, h), c
	}
}

// requireSameRoot fails unless two roots hold identical bits, cells and
// mask words alike, so a read that wrote a different value, or wrote
// outside its window, is caught.
func requireSameRoot(t *testing.T, id string, got, want raster.Float32Raster) {
	t.Helper()
	for i := range want.Valid {
		if got.Valid[i] != want.Valid[i] {
			t.Fatalf("%s: mask word %d = %#016x, want %#016x", id, i, got.Valid[i], want.Valid[i])
		}
	}
	for i := range want.Data {
		if !sameBits(got.Data[i], want.Data[i]) {
			t.Fatalf("%s: cell %d = %#08x, want %#08x", id,
				i, math.Float32bits(got.Data[i]), math.Float32bits(want.Data[i]))
		}
	}
}

// everyRead applies wrap to every call of a faultio.File.
func everyRead(wrap func(io.Reader) io.Reader) func(int64, io.Reader) io.Reader {
	return func(_ int64, r io.Reader) io.Reader { return wrap(r) }
}

// readWrappers shorten reads without failing them.
var readWrappers = []struct {
	name string
	wrap func(io.Reader) io.Reader
}{
	{"half", iotest.HalfReader},
	{"one byte", iotest.OneByteReader},
	{"data with EOF", iotest.DataErrReader},
	{"one byte at a time, half as many", func(r io.Reader) io.Reader {
		return iotest.HalfReader(iotest.OneByteReader(r))
	}},
}

// TestRawSourceShortReads checks that a file answering in pieces gives
// the same cells as one answering whole, in the same number of ReadAt
// calls, for grouped and per-row reads, with and without a fill value.
func TestRawSourceShortReads(t *testing.T) {
	const w, h = 12, 9
	defer engine.SetRawCallBytes(4 * w * 2)() // two full rows per call
	rng := rand.New(rand.NewPCG(21, 22))
	data, _ := window(rng, w, h, false)
	// Hazards: a NaN with a payload, an infinity, a signed zero and the
	// fill value, none of which may change on the way through.
	data.Data[data.Index(1, 0)] = math.Float32frombits(0x7fc0_0001)
	data.Data[data.Index(2, 3)] = float32(math.Inf(-1))
	data.Data[data.Index(3, 5)] = float32(math.Copysign(0, -1))
	data.Data[data.Index(4, 7)] = -9999
	b := rawBytes(data)

	for _, opts := range []engine.RawOptions{{}, {Fill: -9999, HasFill: true}, {Fill: float32(math.NaN()), HasFill: true}} {
		for _, reg := range [][4]int{{0, 0, w, h}, {0, 3, w, 4}, {2, 1, 5, 4}, {11, 8, 1, 1}} {
			for _, windowed := range []bool{false, true} {
				mk := dstPair(rng, reg[2], reg[3], windowed)
				id := fmt.Sprintf("fill=%v region=%v windowed=%v", opts.HasFill, reg, windowed)

				want, wantRoot := mk()
				plain := faultio.New(b)
				if err := engine.NewRawSource(plain, w, h, opts).ReadWindow(context.Background(), want, reg[0], reg[1]); err != nil {
					t.Fatalf("%s: plain read: %v", id, err)
				}
				if !opts.HasFill {
					requireCells(t, id, want, 0, 0, data, reg[0], reg[1], reg[2], reg[3], false)
				}

				for _, wr := range readWrappers {
					got, gotRoot := mk()
					f := faultio.New(b)
					f.Read = everyRead(wr.wrap)
					if err := engine.NewRawSource(f, w, h, opts).ReadWindow(context.Background(), got, reg[0], reg[1]); err != nil {
						t.Fatalf("%s %s: %v", id, wr.name, err)
					}
					requireSameRoot(t, id+" "+wr.name, gotRoot, wantRoot)
					if f.Reads() != plain.Reads() {
						t.Fatalf("%s %s: %d ReadAt calls, want %d", id, wr.name, f.Reads(), plain.Reads())
					}
				}
			}
		}
	}
}

// TestRawSourceReadFaults checks that a read that fails, times out or
// stops short reaches the caller as that error, named with the rows it
// was reading.
func TestRawSourceReadFaults(t *testing.T) {
	const w, h = 10, 8
	defer engine.SetRawCallBytes(4 * w * 2)() // two full rows per call
	rng := rand.New(rand.NewPCG(23, 24))
	data, _ := window(rng, w, h, false)
	b := rawBytes(data)
	errDisk := errors.New("disk failed")
	row := func(y int) int64 { return rowAt(w, y) }

	for _, tc := range []struct {
		name string
		read func(off int64, r io.Reader) io.Reader
		want error
		rows string
	}{
		{"failing", func(off int64, r io.Reader) io.Reader {
			if off >= row(4) {
				return iotest.ErrReader(errDisk)
			}
			return r
		}, errDisk, "rows 4 to 5"},
		{"timing out partway", func(off int64, r io.Reader) io.Reader {
			if off >= row(2) {
				// The timeout lands on the second read of the call,
				// after half the rows have arrived.
				return iotest.TimeoutReader(iotest.HalfReader(r))
			}
			return r
		}, iotest.ErrTimeout, "rows 2 to 3"},
		{"truncated", func(off int64, r io.Reader) io.Reader {
			if off >= row(6) {
				return io.LimitReader(r, 4) // one cell, then end of file
			}
			return r
		}, io.ErrUnexpectedEOF, "rows 6 to 7"},
	} {
		f := faultio.New(b)
		f.Read = tc.read
		dst := raster.NewFloat32(w, h, make([]float32, w*h))
		err := engine.NewRawSource(f, w, h, engine.RawOptions{}).ReadWindow(context.Background(), dst, 0, 0)
		if !errors.Is(err, tc.want) || !strings.Contains(err.Error(), tc.rows) {
			t.Fatalf("%s: err = %v, want %v at %s", tc.name, err, tc.want, tc.rows)
		}
	}

	// A window narrower than the file is read a row at a time, and the
	// error names the one row.
	f := faultio.New(b)
	f.Read = func(off int64, r io.Reader) io.Reader {
		if off == row(3)+4 {
			return iotest.TimeoutReader(iotest.OneByteReader(r))
		}
		return r
	}
	dst := raster.NewFloat32(5, 4, make([]float32, 20))
	err := engine.NewRawSource(f, w, h, engine.RawOptions{}).ReadWindow(context.Background(), dst, 1, 2)
	if !errors.Is(err, iotest.ErrTimeout) || !strings.Contains(err.Error(), "row 3:") {
		t.Fatalf("narrow read: err = %v, want %v at row 3", err, iotest.ErrTimeout)
	}
}

// shortReaderAt breaks io.ReaderAt's contract: it hands back half the
// bytes asked for and no error, as a ReaderAt written over a stream
// without a loop does. faultio keeps the contract instead, which is why
// this one is written out here.
type shortReaderAt struct{ b []byte }

func (r shortReaderAt) ReadAt(p []byte, off int64) (int, error) {
	return copy(p[:(len(p)+1)/2], r.b[off:]), nil
}

// TestRawSourceShortReaderAt checks that a file returning a short count
// without an error fails the read, rather than leaving the cells it
// never read as they were.
func TestRawSourceShortReaderAt(t *testing.T) {
	const w, h = 10, 8
	defer engine.SetRawCallBytes(4 * w * 2)()
	rng := rand.New(rand.NewPCG(27, 28))
	data, _ := window(rng, w, h, false)
	dst := raster.NewFloat32(w, h, make([]float32, w*h))
	s := engine.NewRawSource(shortReaderAt{b: rawBytes(data)}, w, h, engine.RawOptions{})
	err := s.ReadWindow(context.Background(), dst, 0, 0)
	if !errors.Is(err, io.ErrUnexpectedEOF) || !strings.Contains(err.Error(), "rows 0 to 1") {
		t.Fatalf("err = %v, want %v at rows 0 to 1", err, io.ErrUnexpectedEOF)
	}
}

// TestRawSinkShortWrite checks that a file which drops part of a write
// and reports success, as iotest.TruncateWriter does, fails the write
// instead of losing the rows silently.
func TestRawSinkShortWrite(t *testing.T) {
	const w, h = 10, 8
	defer engine.SetRawCallBytes(4 * w * 2)() // two full rows per call
	rng := rand.New(rand.NewPCG(25, 26))
	// Compact, so that consecutive rows are written in one call.
	src := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range src.Data {
		src.Data[i] = rng.Float32()*2000 - 1000
	}
	want := rawBytes(src)

	for _, tc := range []struct {
		name  string
		keep  int64 // bytes the truncated call accepts
		at    int64
		rows  string
		bytes int
	}{
		{"half a call", 4 * w, rowAt(w, 4), "rows 4 to 5", 4 * w * 5},
		{"nothing at all", 0, rowAt(w, 2), "rows 2 to 3", 4 * w * 2},
	} {
		f := faultio.New(nil)
		f.Write = func(off int64, wr io.Writer) io.Writer {
			if off == tc.at {
				return iotest.TruncateWriter(wr, tc.keep)
			}
			return wr
		}
		err := engine.NewRawSink(f, w, h, engine.RawOptions{}).WriteWindow(context.Background(), src, 0, 0)
		if !errors.Is(err, io.ErrShortWrite) || !strings.Contains(err.Error(), tc.rows) {
			t.Fatalf("%s: err = %v, want %v at %s", tc.name, err, io.ErrShortWrite, tc.rows)
		}
		// The rows before the failed call are in the file; the call's
		// own rows are unspecified, as the sink documents.
		if got := f.Bytes(); !bytes.Equal(got[:tc.bytes], want[:tc.bytes]) {
			t.Fatalf("%s: the rows written before the failure differ", tc.name)
		}
	}
}

// rowAt is the file offset of row y of a w cell wide raw file.
func rowAt(w, y int) int64 { return int64(4 * w * y) }
