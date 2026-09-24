package engine

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
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
// A RawFile made by CreateRawFile may also be mapped into memory; see
// there. A RawFile is safe for concurrent use. Close closes every handle.
type RawFile struct {
	files []*os.File
	next  atomic.Uint64

	// m is the file's first len(m) bytes mapped into memory, shared and
	// writable, or nil. mu guards it against Close: calls that copy
	// through it hold mu for reading, and Close unmaps it holding mu.
	mu    sync.RWMutex
	m     []byte
	unmap func() error
	flush func() error // nil where Sync covers the mapping
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
		h, err := os.OpenFile(name, flag, perm) // #nosec G304 -- opening the caller's file is RawFile's purpose
		if err != nil {
			_ = f.Close() // the open error matters more
			return nil, err
		}
		f.files = append(f.files, h)
	}
	return f, nil
}

// CreateRawFile creates the file name for an output of size bytes, such
// as 4·width·height for a RawSink, truncating it if it exists, and opens
// it for reading and writing with handles handles, as OpenRawFile(name,
// os.O_RDWR|os.O_CREATE|os.O_TRUNC, perm, handles) would. handles ≤ 0
// means runtime.GOMAXPROCS(0).
//
// It gives the file its size at once, allocating its blocks where that
// is cheap (fallocate on Linux, except on tmpfs, where allocating is the
// cost of writing; SetEndOfFile on Windows), so that a full disk fails
// here rather than halfway through a run. With more than one handle,
// meant for concurrent writers, it also maps the file into memory on
// Linux and Windows, so that WriteAt and ReadAt within the first size
// bytes copy to and from the operating system's cache without a system
// call. Writes into one file through system calls are serialised by the
// operating system (Linux takes the file's lock for every write), and
// copies into a mapping are not: with 12 workers, a mapped output (and
// the engine's write-behind) made a Slope from a raw file 1.2–2.0×
// faster on tmpfs, ext4 and NTFS (benchmarks/rawio/RESULTS.md). A single
// writer is faster through system calls, since a mapping takes a page
// fault per 4 KiB, so one handle maps nothing. Calls outside the first
// size bytes go through the handles, and so does everything if the file
// could not be mapped; Mapped reports which.
//
// A mapped write cannot return the operating system's error as a system
// call does. A copy into a page the system cannot back, because a tmpfs
// is full or the file was truncated under the mapping, raises a memory
// fault, which WriteAt turns into an error wrapping ErrMappedFault
// instead of crashing the program. Errors writing the cache back to the
// disk surface, as they do for any file, only in Sync. Close unmaps the
// file; the data stays in the cache and reaches the disk as written data
// does.
func CreateRawFile(name string, size int64, perm os.FileMode, handles int) (*RawFile, error) {
	return createRawFile("CreateRawFile", name, size, perm, handles, os.O_TRUNC)
}

// ReuseRawFile is CreateRawFile for an output that overwrites a file
// already there, as a rerun does: it does not truncate the file, only
// sets its size to size, so the operating system keeps the pages and
// blocks it has instead of freeing them now and allocating them again
// as they are written. On tmpfs, freeing and reallocating a 508 MB
// output cost more than the rest of writing it (benchmarks/rawio).
//
// The price is what a failed call leaves: a chunked call that fails or
// is cancelled writes a prefix of tiles (see the package documentation),
// and the cells it did not reach keep whatever the file held before,
// which may look like a plausible result, where CreateRawFile leaves
// zeros. Use it when a failed run's output is discarded anyway. A file
// that does not exist is created, as by CreateRawFile.
func ReuseRawFile(name string, size int64, perm os.FileMode, handles int) (*RawFile, error) {
	return createRawFile("ReuseRawFile", name, size, perm, handles, 0)
}

func createRawFile(fn, name string, size int64, perm os.FileMode, handles, trunc int) (*RawFile, error) {
	if size < 0 {
		return nil, &os.PathError{Op: "create", Path: name, Err: fmt.Errorf("engine: %s: negative size %d", fn, size)}
	}
	f, err := OpenRawFile(name, os.O_RDWR|os.O_CREATE|trunc, perm, handles)
	if err != nil {
		return nil, err
	}
	if trunc == 0 {
		// A file being reused may be longer than the output: fallocate
		// never shrinks one.
		if err := f.files[0].Truncate(size); err != nil {
			_ = f.Close() // the size error matters more
			return nil, &os.PathError{Op: "create", Path: name, Err: err}
		}
	}
	if err := preallocate(f.files[0], size); err != nil {
		if trunc != 0 {
			// Give back what a failed fallocate did allocate; the file
			// was emptied already. A reused file is left as it is.
			_ = f.files[0].Truncate(0)
		}
		_ = f.Close() // the size error matters more
		return nil, &os.PathError{Op: "create", Path: name, Err: err}
	}
	if len(f.files) > 1 && size > 0 && uint64(size) <= uint64(maxMapBytes) {
		// A file that cannot be mapped still works through its handles.
		f.m, f.unmap, f.flush, _ = mapFile(f.files[0], int(size))
	}
	return f, nil
}

// maxMapBytes is the largest mapping CreateRawFile asks for: a quarter
// of the address space, which leaves 32-bit systems their handles.
const maxMapBytes = int(^uint(0) >> 2)

// ErrMappedFault is wrapped by the error of a ReadAt or WriteAt whose
// copy through a RawFile's mapping raised a memory fault: the operating
// system could not back a page of the file, because the disk was full,
// the file was truncated, or an IO error occurred.
var ErrMappedFault = errors.New("engine: memory fault in a mapped file")

// Mapped reports whether the file is mapped into memory (CreateRawFile).
func (f *RawFile) Mapped() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.m != nil
}

func (f *RawFile) handle() *os.File {
	return f.files[f.next.Add(1)%uint64(len(f.files))]
}

// ReadAt reads from the file through its mapping, or through the next
// handle. See os.File.ReadAt.
func (f *RawFile) ReadAt(p []byte, off int64) (int, error) {
	n, err := f.mapped(p, off, false)
	if err != nil || n == len(p) {
		return n, err
	}
	m, err := f.handle().ReadAt(p[n:], off+int64(n))
	return n + m, err
}

// WriteAt writes to the file through its mapping, or through the next
// handle. See os.File.WriteAt.
func (f *RawFile) WriteAt(p []byte, off int64) (int, error) {
	n, err := f.mapped(p, off, true)
	if err != nil || n == len(p) {
		return n, err
	}
	m, err := f.handle().WriteAt(p[n:], off+int64(n))
	return n + m, err
}

// mapped copies between p and the mapping the part of p that starts at
// off and lies inside the mapping, and returns its length: 0 when the
// file is not mapped or off is outside the mapping.
func (f *RawFile) mapped(p []byte, off int64, write bool) (n int, err error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.m == nil || off < 0 || off >= int64(len(f.m)) || len(p) == 0 {
		return 0, nil
	}
	return copyMapped(f.m[off:], p, off, write)
}

// copyMapped copies between p and the mapped bytes m, which start at file
// offset off, turning a memory fault into an error: the SIGBUS of a page
// Linux cannot back and the EXCEPTION_IN_PAGE_ERROR of Windows are
// otherwise fatal to the program. debug.SetPanicOnFault exists for this.
func copyMapped(m, p []byte, off int64, write bool) (n int, err error) {
	defer func() {
		if v := recover(); v != nil {
			re, ok := v.(runtime.Error)
			if !ok || !isFault(re) {
				panic(v)
			}
			// The runtime describes every fault as a nil dereference;
			// the address says more.
			n, err = 0, fmt.Errorf("%w at offset %d (address %#x): the system could not back the page; is the disk full?",
				ErrMappedFault, off, re.(interface{ Addr() uintptr }).Addr())
		}
	}()
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	if write {
		return copy(m, p), nil
	}
	return copy(p, m), nil
}

// isFault reports whether a runtime error is a memory fault, which
// carries the faulting address, rather than, say, an index out of range.
func isFault(e runtime.Error) bool {
	_, ok := e.(interface{ Addr() uintptr })
	return ok
}

// Handles returns the number of handles.
func (f *RawFile) Handles() int { return len(f.files) }

// Sync commits the file's contents to stable storage, those written
// through the mapping included. See os.File.Sync.
func (f *RawFile) Sync() error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.flush != nil {
		if err := f.flush(); err != nil {
			return err
		}
	}
	return f.files[0].Sync()
}

// Close unmaps the file, closes every handle and returns the first error.
// Data written through the mapping stays in the operating system's cache
// and reaches the file as data written through a handle does.
func (f *RawFile) Close() error {
	var first error
	f.mu.Lock()
	if f.unmap != nil {
		first = f.unmap()
		f.m, f.unmap, f.flush = nil, nil, nil
	}
	f.mu.Unlock()
	for _, h := range f.files {
		if err := h.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
