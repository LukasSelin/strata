// Command probe times writing one width×height float32 file in strips of
// rows by n goroutines, through system calls or a shared memory mapping,
// with or without allocating the file first: what the engine's output
// writing costs with the computation taken away (benchmarks/rawio).
//
//	probe -f /work/out.raw -mode pwrite|mmap -pre none|truncate|fallocate -n 12
//
// Each run removes the file first (unless -keep, which overwrites the
// last run's), and prints the time to preallocate, to write, and in
// total, which includes unmapping and closing.
package main

import (
	"flag"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

var (
	path     = flag.String("f", "out.raw", "the file")
	width    = flag.Int("w", 11264, "width in cells")
	height   = flag.Int("h", 11264, "height in cells")
	rows     = flag.Int("rows", 256, "rows per strip")
	workers  = flag.Int("n", 12, "writers")
	mode     = flag.String("mode", "pwrite", "pwrite (WriteAt, 1 MiB calls, a handle per writer) or mmap")
	prealloc = flag.String("pre", "none", "none, truncate or fallocate (Linux; truncate elsewhere)")
	reps     = flag.Int("r", 5, "runs")
	keep     = flag.Bool("keep", false, "overwrite the last run's file instead of removing it")
)

func main() {
	flag.Parse()
	for range *reps {
		if err := run(); err != nil {
			fmt.Fprintln(os.Stderr, "probe:", err)
			os.Exit(1)
		}
	}
}

func run() error {
	size := int64(*width) * int64(*height) * 4
	strip := *rows * *width * 4
	strips := (*height + *rows - 1) / *rows
	if !*keep {
		_ = os.Remove(*path) // it may not exist
	}
	t0 := time.Now()
	f, err := os.OpenFile(*path, os.O_RDWR|os.O_CREATE, 0o644) // #nosec G302 G304 -- an ordinary output file at the path the user named
	if err != nil {
		return err
	}
	defer f.Close()
	switch *prealloc {
	case "truncate":
		err = f.Truncate(size)
	case "fallocate":
		err = fallocate(f, size)
	}
	if err != nil {
		return err
	}
	tPre := time.Since(t0)
	var m []byte
	if *mode == "mmap" {
		if *prealloc == "none" {
			if err := f.Truncate(size); err != nil {
				return err
			}
		}
		if m, err = mapFile(f, size); err != nil {
			return err
		}
	}
	handles := make([]*os.File, *workers)
	for i := range handles {
		if handles[i], err = os.OpenFile(*path, os.O_RDWR, 0); err != nil {
			return err
		}
		defer handles[i].Close()
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	errs := make([]error, *workers)
	for k := range *workers {
		wg.Go(func() {
			buf := make([]byte, strip)
			for i := range buf {
				buf[i] = byte(i*7 + k) // #nosec G115 -- truncation is the point: any bytes will do
			}
			for {
				s := int(next.Add(1) - 1)
				if s >= strips {
					return
				}
				off := int64(s) * int64(strip)
				b := buf[:min(int64(strip), size-off)]
				if m != nil {
					copy(m[off:], b)
					continue
				}
				for len(b) > 0 {
					c := min(len(b), 1<<20)
					if _, err := handles[k].WriteAt(b[:c], off); err != nil {
						errs[k] = err
						return
					}
					b, off = b[c:], off+int64(c)
				}
			}
		})
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	tWrite := time.Since(t0)
	if m != nil {
		if err := unmap(m); err != nil {
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("mode=%s pre=%s n=%d keep=%v prealloc=%.3f write=%.3f total=%.3f\n",
		*mode, *prealloc, *workers, *keep, tPre.Seconds(), tWrite.Seconds(), time.Since(t0).Seconds())
	return nil
}
