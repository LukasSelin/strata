//go:build !linux && !windows

package main

import (
	"errors"
	"os"
)

func fallocate(f *os.File, size int64) error  { return f.Truncate(size) }
func mapFile(*os.File, int64) ([]byte, error) { return nil, errors.ErrUnsupported }
func unmap([]byte) error                      { return nil }
