package exec_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/LukasSelin/strata/internal/exec"
)

// TestRunUnits checks the scheduler contract RunUnits exports: every unit
// runs exactly once on a valid worker index, errors and cancellation stop
// the claims with the claimed units a prefix, and panics reach the caller.
func TestRunUnits(t *testing.T) {
	for _, workers := range []int{1, 2, 3, 8} {
		const n = 1000
		var seen [n]atomic.Int32
		err := exec.RunUnits(t.Context(), workers, n, func(w, i int) error {
			if w < 0 || w >= workers {
				t.Errorf("worker %d out of [0, %d)", w, workers)
			}
			seen[i].Add(1)
			return nil
		})
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		for i := range seen {
			if got := seen[i].Load(); got != 1 {
				t.Fatalf("workers=%d: unit %d ran %d times", workers, i, got)
			}
		}

		boom := errors.New("boom")
		var ran atomic.Int32
		err = exec.RunUnits(t.Context(), workers, n, func(_, i int) error {
			ran.Add(1)
			if i == 10 {
				return boom
			}
			return nil
		})
		if !errors.Is(err, boom) {
			t.Fatalf("workers=%d: err = %v, want boom", workers, err)
		}
		if ran.Load() == n {
			t.Errorf("workers=%d: an error did not stop the claims", workers)
		}

		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if err := exec.RunUnits(ctx, workers, n, func(int, int) error { return nil }); !errors.Is(err, context.Canceled) {
			t.Fatalf("workers=%d: cancelled err = %v", workers, err)
		}

		func() {
			defer func() {
				if recover() != "kaboom" {
					t.Errorf("workers=%d: panic not re-raised", workers)
				}
			}()
			_ = exec.RunUnits(t.Context(), workers, n, func(_, i int) error {
				if i == 3 {
					panic("kaboom")
				}
				return nil
			})
		}()
	}
	if err := exec.RunUnits(t.Context(), 2, 0, func(int, int) error { t.Error("ran a unit of none"); return nil }); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][2]int{{0, 1}, {1, -1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("RunUnits(%d, %d) did not panic", bad[0], bad[1])
				}
			}()
			_ = exec.RunUnits(t.Context(), bad[0], bad[1], func(int, int) error { return nil })
		}()
	}
}
