package engine

import (
	"errors"
	"os"
	"syscall"
)

// tmpfsMagic is the f_type statfs reports for tmpfs.
const tmpfsMagic = 0x01021994

// preallocate sets the file's size to size. On a disk it allocates the
// blocks too, which is cheap (they are only marked unwritten) and made
// later writes faster, mapped or not, besides reporting a full disk now.
// On tmpfs allocating means zeroing every page, which cost more than
// writing the file (0.19 s for 508 MB, on one core), so there the size
// is only set, and a full tmpfs faults a mapped write instead.
func preallocate(f *os.File, size int64) error {
	if size == 0 {
		return f.Truncate(0)
	}
	var st syscall.Statfs_t
	if err := syscall.Fstatfs(int(f.Fd()), &st); err == nil && st.Type != tmpfsMagic {
		err := syscall.Fallocate(int(f.Fd()), 0, 0, size)
		switch {
		case err == nil:
			return nil
		case errors.Is(err, syscall.EOPNOTSUPP), errors.Is(err, syscall.ENOSYS), errors.Is(err, syscall.EINVAL):
			// The file system cannot allocate ahead: set the size.
		default:
			return os.NewSyscallError("fallocate", err)
		}
	}
	return f.Truncate(size)
}

// mapFile maps the first size bytes of f, shared and writable.
func mapFile(f *os.File, size int) (m []byte, unmap, flush func() error, err error) {
	m, err = syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, nil, os.NewSyscallError("mmap", err)
	}
	// Linux's fsync writes back pages dirtied through a mapping too, so
	// Sync needs no msync.
	return m, func() error { return os.NewSyscallError("munmap", syscall.Munmap(m)) }, nil, nil
}
