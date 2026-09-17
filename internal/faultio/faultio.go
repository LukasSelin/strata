// Package faultio adapts the fault injection of testing/iotest, which
// wraps io.Reader and io.Writer, to the io.ReaderAt and io.WriterAt that
// engine.RawSource and engine.RawSink work through (DESIGN.md §39).
//
// A File serves every call from a stream of its own over the bytes at
// that offset, so a test can make reads arrive in pieces
// (iotest.HalfReader, iotest.OneByteReader, iotest.DataErrReader), fail
// (iotest.ErrReader), time out (iotest.TimeoutReader), or a write lose
// its tail silently (iotest.TruncateWriter), and can choose per offset
// which calls suffer. It is a package rather than a test helper so that
// the engine tests and the chunked tests of internal/exec share it.
package faultio

import (
	"bytes"
	"errors"
	"io"
	"sync"
	"sync/atomic"
)

// File is an io.ReaderAt and io.WriterAt over a byte slice. It is safe
// for concurrent use: one call runs at a time, and each gets a stream of
// its own, so a wrapper's state never crosses calls. Read and Write must
// be set before the first call.
type File struct {
	// Read and Write wrap the stream of a single call, which starts at
	// file offset off, or are nil to leave it alone. They are called
	// once per ReadAt or WriteAt.
	Read  func(off int64, r io.Reader) io.Reader
	Write func(off int64, w io.Writer) io.Writer

	mu sync.Mutex
	b  []byte

	reads, writes atomic.Int64
}

// New returns a File over b, which it keeps: writes change b in place
// until one grows the file.
func New(b []byte) *File { return &File{b: b} }

// Bytes returns a copy of the file's contents.
func (f *File) Bytes() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return bytes.Clone(f.b)
}

// Reads returns the number of ReadAt calls made so far.
func (f *File) Reads() int64 { return f.reads.Load() }

// Writes returns the number of WriteAt calls made so far.
func (f *File) Writes() int64 { return f.writes.Load() }

// ReadAt reads the bytes at off through Read. It keeps io.ReaderAt's
// contract whatever the wrapper does: it reads until p is full, so a
// wrapper that shortens reads only costs more Read calls and returns the
// same bytes, and a short count always comes with an error, io.EOF past
// the end of the file as os.File gives, or the wrapper's own.
func (f *File) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("faultio: ReadAt: negative offset")
	}
	f.reads.Add(1)
	f.mu.Lock()
	defer f.mu.Unlock()
	var avail []byte
	if off < int64(len(f.b)) {
		avail = f.b[off:]
	}
	var r io.Reader = bytes.NewReader(avail)
	if f.Read != nil {
		r = f.Read(off, r)
	}
	n, err := io.ReadFull(r, p)
	if errors.Is(err, io.ErrUnexpectedEOF) {
		err = io.EOF
	}
	return n, err
}

// WriteAt writes p at off through Write, growing the file to fit. It
// reports the bytes that reached the file, not the count the wrapper
// claimed: iotest.TruncateWriter drops the tail of a write and reports
// success, which then reaches the caller as the short count without an
// error that io.WriterAt forbids, the file a WriteAt caller must not
// trust.
func (f *File) WriteAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("faultio: WriteAt: negative offset")
	}
	f.writes.Add(1)
	f.mu.Lock()
	defer f.mu.Unlock()
	if end := off + int64(len(p)); end > int64(len(f.b)) {
		f.b = append(f.b, make([]byte, end-int64(len(f.b)))...)
	}
	span := &spanWriter{b: f.b[off:]}
	var w io.Writer = span
	if f.Write != nil {
		w = f.Write(off, w)
	}
	n, err := w.Write(p)
	return min(n, span.n), err
}

// spanWriter writes consecutively into a span of the file and counts the
// bytes that land there.
type spanWriter struct {
	b []byte
	n int
}

func (w *spanWriter) Write(p []byte) (int, error) {
	n := copy(w.b[w.n:], p)
	w.n += n
	if n < len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}
