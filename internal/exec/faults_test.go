package exec_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"testing"
	"testing/iotest"

	"strata/engine"
	"strata/internal/exec"
	"strata/internal/faultio"
	"strata/raster"
)

// The chunked path reads every tile and its halo from a source and
// writes the result to a sink, both of which can fail partway through a
// call on a real file. TestChunkedRawFaults runs it over raw files whose
// reads come back in pieces, fail or time out, and whose writes lose
// their tail (internal/faultio, over testing/iotest), on top of the
// per-tile injection of failingSource.

// rawCells encodes r row-major as little-endian float32, with fill under
// its invalid cells, the layout a raw file holds (engine/raw.go).
func rawCells(r raster.Float32Raster, fill float32) []byte {
	b := make([]byte, 0, 4*r.Width*r.Height)
	for y := range r.Height {
		for x := range r.Width {
			v := r.Data[r.Index(x, y)]
			if !r.IsValid(x, y) {
				v = fill
			}
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
	}
	return b
}

func TestChunkedRawFaults(t *testing.T) {
	const w, h = 13, 9
	const fill float32 = -9999
	rng := rand.New(rand.NewPCG(31, 32))
	dem := newOperand(rng, w, h, true, true)
	cells := rawCells(dem.r, fill)
	opts := engine.RawOptions{Fill: fill, HasFill: true}
	box := boxKernel{r: 1, inputs: 1, outputs: 1}

	// run processes the DEM from a raw file into a raw file, through the
	// given faults, and returns the file written and the error.
	run := func(t *testing.T, id string, workers int,
		read func(off int64, r io.Reader) io.Reader,
		write func(off int64, w io.Writer) io.Writer,
		wrap func(engine.RasterSource) engine.RasterSource) ([]byte, error) {
		t.Helper()
		in := faultio.New(bytes.Clone(cells))
		in.Read = read
		out := faultio.New(nil)
		out.Write = write
		var src engine.RasterSource = engine.NewRawSource(in, w, h, opts)
		if wrap != nil {
			src = wrap(src)
		}
		err := exec.ProcessChunked(context.Background(),
			[]engine.RasterSink{engine.NewRawSink(out, w, h, opts)},
			[]engine.RasterSource{src}, box,
			engine.Options{TileWidth: 4, TileHeight: 3, Workers: workers})
		requireNoLeaks(t, id)
		return out.Bytes(), err
	}

	// A file that answers in pieces gives the same result as one that
	// answers whole, for every worker count.
	for _, workers := range []int{1, 3} {
		want, err := run(t, "plain", workers, nil, nil, nil)
		if err != nil {
			t.Fatalf("workers=%d: plain run: %v", workers, err)
		}
		if len(want) != 4*w*h {
			t.Fatalf("workers=%d: the sink wrote %d bytes, want %d", workers, len(want), 4*w*h)
		}
		for _, wr := range []struct {
			name string
			wrap func(io.Reader) io.Reader
		}{
			{"half", iotest.HalfReader},
			{"one byte", iotest.OneByteReader},
			{"data with EOF", iotest.DataErrReader},
		} {
			id := fmt.Sprintf("workers=%d %s", workers, wr.name)
			got, err := run(t, id, workers, func(_ int64, r io.Reader) io.Reader { return wr.wrap(r) }, nil, nil)
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s: the result differs from the plain run", id)
			}
		}
	}

	// A read or write that fails reaches the caller, whichever worker
	// was running it, and leaves no goroutine behind.
	errDisk := errors.New("disk failed")
	errBoom := errors.New("boom")
	from := int64(4 * w * 5) // the offset of row 5
	for _, workers := range []int{1, 3} {
		for _, tc := range []struct {
			name  string
			read  func(off int64, r io.Reader) io.Reader
			write func(off int64, w io.Writer) io.Writer
			wrap  func(engine.RasterSource) engine.RasterSource
			want  error
		}{
			{name: "read fails", want: errDisk, read: func(off int64, r io.Reader) io.Reader {
				if off >= from {
					return iotest.ErrReader(errDisk)
				}
				return r
			}},
			{name: "read times out", want: iotest.ErrTimeout, read: func(off int64, r io.Reader) io.Reader {
				if off >= from {
					return iotest.TimeoutReader(iotest.OneByteReader(r))
				}
				return r
			}},
			{name: "file ends early", want: io.ErrUnexpectedEOF, read: func(off int64, r io.Reader) io.Reader {
				if off >= from {
					return io.LimitReader(r, 4)
				}
				return r
			}},
			{name: "write loses its tail", want: io.ErrShortWrite, write: func(off int64, w io.Writer) io.Writer {
				if off >= from {
					return iotest.TruncateWriter(w, 0)
				}
				return w
			}},
			// The per-tile injection still works on top of the byte-level
			// faults: the source's own error wins over short reads.
			{name: "a tile fails while reads are short", want: errBoom,
				read: func(_ int64, r io.Reader) io.Reader { return iotest.HalfReader(r) },
				wrap: func(s engine.RasterSource) engine.RasterSource {
					return &failingSource{RasterSource: s, n: 2, err: errBoom}
				}},
		} {
			id := fmt.Sprintf("workers=%d %s", workers, tc.name)
			if _, err := run(t, id, workers, tc.read, tc.write, tc.wrap); !errors.Is(err, tc.want) {
				t.Fatalf("%s: err = %v, want %v", id, err, tc.want)
			}
		}
	}
}
