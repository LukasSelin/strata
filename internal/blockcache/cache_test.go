package blockcache

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func size(int) int64 { return 10 }

func value(v int) func() (int, error) { return func() (int, error) { return v, nil } }

// TestHitsAndEviction checks the counters of a cache of two values as
// values come and go.
func TestHitsAndEviction(t *testing.T) {
	ctx := context.Background()
	c := New[string](20, size)
	for _, k := range []string{"a", "b", "a", "c", "b"} {
		if _, err := c.Get(ctx, k, value(len(k))); err != nil {
			t.Fatal(err)
		}
	}
	// a, b loaded; a hit; c loaded, evicting b (a was used after it);
	// b loaded again, evicting a.
	want := Stats{Hits: 1, Loads: 4, Evictions: 2, Bytes: 20, Len: 2}
	if got := c.Stats(); got != want {
		t.Errorf("stats %+v, want %+v", got, want)
	}
}

// TestPeek checks that Peek answers only with values held, and counts
// those as hits.
func TestPeek(t *testing.T) {
	c := New[int](100, size)
	if _, ok := c.Peek(1); ok {
		t.Error("Peek found a value never loaded")
	}
	release := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = c.Get(context.Background(), 1, func() (int, error) { <-release; return 5, nil })
	}()
	for c.Stats().Loads == 0 {
		runtime.Gosched()
	}
	if _, ok := c.Peek(1); ok {
		t.Error("Peek found a value still loading")
	}
	close(release)
	<-done
	if v, ok := c.Peek(1); !ok || v != 5 {
		t.Errorf("Peek: %d, %v; want 5, true", v, ok)
	}
	if st := c.Stats(); st.Hits != 1 || st.Loads != 1 {
		t.Errorf("stats %+v, want 1 hit and 1 load", st)
	}
}

// TestKeepsOneOverLimit checks that a value larger than the limit is kept
// until the next one arrives.
func TestKeepsOneOverLimit(t *testing.T) {
	ctx := context.Background()
	c := New[int](5, size)
	_, _ = c.Get(ctx, 1, value(1))
	_, _ = c.Get(ctx, 1, value(1))
	_, _ = c.Get(ctx, 2, value(2))
	if got := c.Stats(); got.Hits != 1 || got.Len != 1 || got.Evictions != 1 {
		t.Errorf("stats %+v, want 1 hit, 1 held, 1 eviction", got)
	}
}

// TestFailedLoadRetried checks that an error is returned to the reader
// whose load failed and not kept.
func TestFailedLoadRetried(t *testing.T) {
	ctx := context.Background()
	c := New[int](100, size)
	boom := errors.New("boom")
	if _, err := c.Get(ctx, 1, func() (int, error) { return 0, boom }); !errors.Is(err, boom) {
		t.Fatalf("error %v, want %v", err, boom)
	}
	v, err := c.Get(ctx, 1, value(7))
	if err != nil || v != 7 {
		t.Fatalf("second Get: %d, %v; want 7", v, err)
	}
	if got := c.Stats(); got.Loads != 2 || got.Failed != 1 || got.Len != 1 {
		t.Errorf("stats %+v, want 2 loads, 1 failed, 1 held", got)
	}
}

// TestSharedLoad checks that readers asking for a value while it loads
// wait for that load rather than start their own; run it with -race.
func TestSharedLoad(t *testing.T) {
	ctx := context.Background()
	c := New[int](100, size)
	release := make(chan struct{})
	started := make(chan struct{})
	var loads atomic.Int64
	load := func() (int, error) {
		if loads.Add(1) == 1 {
			close(started)
		}
		<-release
		return 42, nil
	}
	const readers = 16
	var wg sync.WaitGroup
	got := make([]int, readers)
	waiting := make(chan struct{}, readers)
	c.onWait = func() { waiting <- struct{}{} }
	wg.Go(func() { got[0], _ = c.Get(ctx, 1, load) })
	<-started
	for i := 1; i < readers; i++ {
		wg.Go(func() { got[i], _ = c.Get(ctx, 1, load) })
	}
	for range readers - 1 {
		<-waiting
	}
	close(release)
	wg.Wait()
	for i, v := range got {
		if v != 42 {
			t.Errorf("reader %d got %d", i, v)
		}
	}
	st := c.Stats()
	if loads.Load() != 1 || st.Loads != 1 || st.Shared != readers-1 {
		t.Errorf("%d loads, stats %+v; want 1 load, shared %d times", loads.Load(), st, readers-1)
	}
}

// TestWaiterOutlivesLoader checks that a reader waiting for a load that
// fails because its loader's context ended loads the value itself, and
// that a waiter whose own context ends gets its error.
func TestWaiterOutlivesLoader(t *testing.T) {
	c := New[int](100, size)
	waiting := make(chan struct{}, 2)
	c.onWait = func() { waiting <- struct{}{} }
	loaderCtx, cancelLoader := context.WithCancel(context.Background())
	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		_, err := c.Get(loaderCtx, 1, func() (int, error) {
			close(started)
			<-loaderCtx.Done()
			return 0, loaderCtx.Err()
		})
		done <- err
	}()
	<-started

	gone, cancelGone := context.WithCancel(context.Background())
	cancelGone()
	if _, err := c.Get(gone, 1, value(0)); !errors.Is(err, context.Canceled) {
		t.Errorf("a waiter whose context is done: %v, want context.Canceled", err)
	}

	result := make(chan int, 1)
	go func() {
		v, err := c.Get(context.Background(), 1, value(9))
		if err != nil {
			t.Error(err)
		}
		result <- v
	}()
	<-waiting // the first waiter, whose context was done
	<-waiting // this one
	cancelLoader()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Errorf("the loader: %v, want context.Canceled", err)
	}
	if v := <-result; v != 9 {
		t.Errorf("the waiter got %d, want 9", v)
	}
}

// TestHolds checks that a counted value is held by the cache while it
// keeps it and by each reader, and dropped by the cache on eviction.
func TestHolds(t *testing.T) {
	ctx := context.Background()
	refs := map[int]int{}
	var mu sync.Mutex
	hold := func(v int) { mu.Lock(); refs[v]++; mu.Unlock() }
	drop := func(v int) { mu.Lock(); refs[v]--; mu.Unlock() }
	c := New[int](10, size).WithHolds(hold, drop)
	load := func(v int) func() (int, error) {
		return func() (int, error) { hold(v); return v, nil }
	}
	a, _ := c.Get(ctx, 1, load(1)) // the reader's and the cache's
	b, _ := c.Get(ctx, 1, load(1)) // and another reader's
	if refs[1] != 3 {
		t.Fatalf("value 1 held %d times, want 3", refs[1])
	}
	drop(a)
	drop(b)
	v, _ := c.Get(ctx, 2, load(2)) // evicts 1
	drop(v)
	if refs[1] != 0 || refs[2] != 1 {
		t.Errorf("references %v, want 1: 0 and 2: 1", refs)
	}
	if c.Limit() != 10 {
		t.Errorf("Limit %d, want 10", c.Limit())
	}
}
