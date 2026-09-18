package faultio_test

import (
	"bytes"
	"errors"
	"io"
	"testing"
	"testing/iotest"

	"github.com/LukasSelin/strata/internal/faultio"
)

// The engine tests read a file through these adapters and compare the
// cells with a faultless read, so the adapters themselves must behave:
// one call however short the wrapper's reads are, the file's own bytes,
// and an error with every short count.

func TestReadAt(t *testing.T) {
	b := []byte("0123456789")
	for _, tc := range []struct {
		name string
		read func(off int64, r io.Reader) io.Reader
		want error
	}{
		{name: "plain"},
		{name: "half", read: func(_ int64, r io.Reader) io.Reader { return iotest.HalfReader(r) }},
		{name: "one byte", read: func(_ int64, r io.Reader) io.Reader { return iotest.OneByteReader(r) }},
		{name: "data with EOF", read: func(_ int64, r io.Reader) io.Reader { return iotest.DataErrReader(r) }},
	} {
		f := faultio.New(b)
		f.Read = tc.read
		p := make([]byte, 4)
		n, err := f.ReadAt(p, 3)
		if n != 4 || err != nil || !bytes.Equal(p, []byte("3456")) {
			t.Fatalf("%s: ReadAt = %q, %d, %v, want %q, 4, nil", tc.name, p, n, err, "3456")
		}
		if f.Reads() != 1 {
			t.Fatalf("%s: %d calls, want 1", tc.name, f.Reads())
		}
	}

	// Past the end of the file the count is short and the error says so,
	// as os.File.ReadAt does.
	f := faultio.New(b)
	p := make([]byte, 4)
	if n, err := f.ReadAt(p, 8); n != 2 || !errors.Is(err, io.EOF) {
		t.Fatalf("read past the end: %d, %v, want 2, io.EOF", n, err)
	}
	if n, err := f.ReadAt(p, 20); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("read beyond the file: %d, %v, want 0, io.EOF", n, err)
	}
	if _, err := f.ReadAt(p, -1); err == nil {
		t.Fatal("a negative offset did not fail")
	}

	errDisk := errors.New("disk failed")
	f = faultio.New(b)
	f.Read = func(off int64, r io.Reader) io.Reader {
		if off >= 4 {
			return iotest.ErrReader(errDisk)
		}
		return r
	}
	if n, err := f.ReadAt(p, 0); n != 4 || err != nil {
		t.Fatalf("read before the fault: %d, %v, want 4, nil", n, err)
	}
	if n, err := f.ReadAt(p, 4); n != 0 || !errors.Is(err, errDisk) {
		t.Fatalf("read at the fault: %d, %v, want 0, %v", n, err, errDisk)
	}
}

func TestWriteAt(t *testing.T) {
	f := faultio.New(nil)
	if n, err := f.WriteAt([]byte("abcd"), 2); n != 4 || err != nil {
		t.Fatalf("WriteAt = %d, %v, want 4, nil", n, err)
	}
	if got := f.Bytes(); !bytes.Equal(got, []byte("\x00\x00abcd")) {
		t.Fatalf("file = %q, want %q", got, "\x00\x00abcd")
	}
	if f.Writes() != 1 {
		t.Fatalf("%d calls, want 1", f.Writes())
	}

	// A truncating writer reports success; the count is what reached the
	// file, which is the short count without an error that io.WriterAt
	// forbids and a caller must catch.
	f = faultio.New(nil)
	f.Write = func(_ int64, w io.Writer) io.Writer { return iotest.TruncateWriter(w, 2) }
	n, err := f.WriteAt([]byte("abcd"), 0)
	if n != 2 || err != nil {
		t.Fatalf("truncated WriteAt = %d, %v, want 2, nil", n, err)
	}
	if got := f.Bytes(); !bytes.Equal(got, []byte("ab\x00\x00")) {
		t.Fatalf("truncated file = %q, want %q", got, "ab\x00\x00")
	}
	if _, err := f.WriteAt([]byte("a"), -1); err == nil {
		t.Fatal("a negative offset did not fail")
	}
}
