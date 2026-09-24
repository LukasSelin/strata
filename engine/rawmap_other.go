//go:build !linux && !windows

package engine

import (
	"errors"
	"os"
)

// preallocate sets the file's size.
func preallocate(f *os.File, size int64) error { return f.Truncate(size) }

// mapFile maps nothing: mapping is only measured, and so only used, on
// Linux and Windows. RawFile works through its handles instead.
func mapFile(*os.File, int) (m []byte, unmap, flush func() error, err error) {
	return nil, nil, nil, errors.ErrUnsupported
}
