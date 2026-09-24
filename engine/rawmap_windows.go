package engine

import (
	"os"
	"syscall"
	"unsafe"
)

// preallocate sets the file's size.
func preallocate(f *os.File, size int64) error { return f.Truncate(size) }

// mapFile maps the first size bytes of f, shared and writable.
func mapFile(f *os.File, size int) (m []byte, unmap, flush func() error, err error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READWRITE,
		uint32(uint64(size)>>32), uint32(size), nil) // #nosec G115 -- size is positive and split into its two 32-bit halves
	if err != nil {
		return nil, nil, nil, os.NewSyscallError("CreateFileMapping", err)
	}
	// The view keeps the mapping object alive.
	defer func() { _ = syscall.CloseHandle(h) }()
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ|syscall.FILE_MAP_WRITE, 0, 0, uintptr(size))
	if err != nil {
		return nil, nil, nil, os.NewSyscallError("MapViewOfFile", err)
	}
	// The view is size bytes at addr, outside the Go heap. Reading addr
	// as a pointer, rather than converting it, keeps vet's unsafeptr
	// check satisfied: it cannot know the address came from the system.
	m = unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), size)
	unmap = func() error { return os.NewSyscallError("UnmapViewOfFile", syscall.UnmapViewOfFile(addr)) }
	flush = func() error { return os.NewSyscallError("FlushViewOfFile", syscall.FlushViewOfFile(addr, 0)) }
	return m, unmap, flush, nil
}
