package exec

import "context"

// RunUnits runs do(w, i) for every unit of work i in [0, n) on workers
// w in [0, workers), for packages that plan their own units instead of
// running a Kernel, such as package resample, whose output tiles read
// source footprints of their own shape rather than a halo (DESIGN.md
// §54). It is the scheduler every entry point here uses, with the same
// guarantees:
//
//   - workers check ctx, then claim the next unit in order, and always
//     finish a unit they have claimed, so the units that ran are a prefix
//     of [0, n) whenever RunUnits returns;
//   - it returns nil if every unit ran and returned nil; otherwise the
//     first error a unit returned, or ctx.Err() if units were left
//     unclaimed;
//   - a panic in do is re-raised on the calling goroutine once every
//     worker has stopped;
//   - the calling goroutine is worker 0, and with one worker nothing is
//     started (DESIGN.md §26).
//
// It panics if workers is not positive or n is negative.
func RunUnits(ctx context.Context, workers, n int, do func(w, i int) error) error {
	if workers < 1 || n < 0 {
		panic("exec: RunUnits needs workers >= 1 and n >= 0")
	}
	return runWorkers(ctx, workers, n, do)
}
