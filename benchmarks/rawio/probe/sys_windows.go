package main

import (
	"os"
	"syscall"
	"unsafe"
)

// fallocate sets the size: SetEndOfFile allocates the clusters on NTFS.
func fallocate(f *os.File, size int64) error { return f.Truncate(size) }

func mapFile(f *os.File, size int64) ([]byte, error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READWRITE,
		uint32(uint64(size)>>32), uint32(size), nil) // #nosec G115 -- size is positive and split into its two 32-bit halves
	if err != nil {
		return nil, err
	}
	defer func() { _ = syscall.CloseHandle(h) }()
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(size)) // #nosec G115 -- a positive file size, mapped on 64-bit Windows
	if err != nil {
		return nil, err
	}
	return unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), size), nil
}

func unmap(m []byte) error { return syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&m[0]))) }
