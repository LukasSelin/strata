package main

import (
	"os"
	"syscall"
)

func fallocate(f *os.File, size int64) error { return syscall.Fallocate(int(f.Fd()), 0, 0, size) }

func mapFile(f *os.File, size int64) ([]byte, error) {
	return syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
}

func unmap(m []byte) error { return syscall.Munmap(m) }
