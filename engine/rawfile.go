package engine

import (
	"errors"
	"os"
	"runtime"
	"sync/atomic"
)

// RawFile is one file opened through several handles, for RawSource and
// RawSink under concurrent use. Calls on one *os.File queue behind each
// other: Go serialises ReadAt and WriteAt on a handle on Windows, and the
// operating system may serialise IO on a handle too. RawFile spreads its
// ReadAt and WriteAt calls over its handles in turn, so workers reading
// and writing tiles at the same time mostly use different handles. On the
// benchmarks/chunked machine (Windows 11), one handle per worker made a
// 20000² Slope with 12 workers 1.6× faster than one shared handle.
//
// A RawFile is safe for concurrent use. Close closes every handle.
type RawFile struct {
	files []*os.File
	next  atomic.Uint64
}

// OpenRawFile opens name with handles handles, each as os.OpenFile(name,
// flag, perm) would, except that only the first handle creates or
// truncates the file (os.O_CREATE, os.O_EXCL and os.O_TRUNC). handles ≤ 0
// means runtime.GOMAXPROCS(0). os.O_APPEND is rejected, because WriteAt
// cannot be used with it. On error, the handles opened so far are closed.
func OpenRawFile(name string, flag int, perm os.FileMode, handles int) (*RawFile, error) {
	if flag&os.O_APPEND != 0 {
		return nil, &os.PathError{Op: "open", Path: name, Err: errors.New("engine: OpenRawFile: O_APPEND does not allow WriteAt")}
	}
	if handles <= 0 {
		handles = runtime.GOMAXPROCS(0)
	}
	f := &RawFile{}
	for i := range handles {
		if i == 1 {
			flag &^= os.O_CREATE | os.O_EXCL | os.O_TRUNC
		}
		h, err := os.OpenFile(name, flag, perm)
		if err != nil {
			f.Close()
			return nil, err
		}
		f.files = append(f.files, h)
	}
	return f, nil
}

func (f *RawFile) handle() *os.File {
	return f.files[f.next.Add(1)%uint64(len(f.files))]
}

// ReadAt reads from the file through the next handle. See os.File.ReadAt.
func (f *RawFile) ReadAt(p []byte, off int64) (int, error) { return f.handle().ReadAt(p, off) }

// WriteAt writes to the file through the next handle. See os.File.WriteAt.
func (f *RawFile) WriteAt(p []byte, off int64) (int, error) { return f.handle().WriteAt(p, off) }

// Handles returns the number of handles.
func (f *RawFile) Handles() int { return len(f.files) }

// Sync commits the file's contents to stable storage. See os.File.Sync.
func (f *RawFile) Sync() error { return f.files[0].Sync() }

// Close closes every handle and returns the first error.
func (f *RawFile) Close() error {
	var first error
	for _, h := range f.files {
		if err := h.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
