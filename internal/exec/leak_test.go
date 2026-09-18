package exec_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"go.uber.org/goleak"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// TestMain fails the package if any test, including a fuzz target's seed
// corpus, leaves a goroutine behind: ProcessN and ProcessChunked are the
// only code in the module that starts goroutines (DESIGN.md §26), and they
// must join every one before returning, whatever happens.
func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// requireNoLeaks fails unless every goroutine other than the test's own
// has exited. goleak retries for a while, since workers that have called
// Done on the WaitGroup may still be exiting when the call returns.
func requireNoLeaks(t *testing.T, id string) {
	t.Helper()
	if err := goleak.Find(); err != nil {
		t.Fatalf("%s: %v", id, err)
	}
}

// panickingSink panics with value on its nth write.
type panickingSink struct {
	engine.RasterSink
	n      int64
	value  any
	mu     sync.Mutex
	writes int64
}

func (s *panickingSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	s.mu.Lock()
	s.writes++
	fail := s.writes == s.n
	s.mu.Unlock()
	if fail {
		panic(s.value)
	}
	return s.RasterSink.WriteWindow(ctx, src, x, y)
}

// blockingKernel blocks every call after the first until release is
// closed, so that other workers are mid-call when one fails.
type blockingKernel struct {
	exec.Kernel
	release <-chan struct{}
	started *sync.Once
	first   chan<- struct{}
}

func (k blockingKernel) Process(dst exec.Span, src exec.Window) {
	isFirst := false
	k.started.Do(func() { isFirst = true; close(k.first) })
	if !isFirst {
		<-k.release
	}
	k.Kernel.Process(dst, src)
}

// TestNoLeaksOnFailure runs ProcessN and ProcessChunked into every way a
// call can end early, with several workers, and requires the documented
// outcome (the error, or the panic re-raised on the caller) and no
// goroutine left behind: kernel, source and sink panics, source and sink
// errors, cancellation from inside a kernel and from another goroutine
// while workers are mid-call, and a context done before the call.
func TestNoLeaksOnFailure(t *testing.T) {
	defer exec.SetBandCells(1)()
	// Many more one-row bands than workers, so a cancellation always
	// leaves bands and tiles unclaimed: claimed ones always finish, and a
	// call whose every band was claimed returns nil.
	const w, h = 24, 200
	rng := rand.New(rand.NewPCG(11, 12))
	dem := newOperand(rng, w, h, true, true)
	box := boxKernel{r: 1, inputs: 1}
	errBoom := errors.New("boom")
	chunks := engine.Options{TileWidth: 5, TileHeight: 2}

	// run calls f, recovering a panic, and returns the panic value or error.
	run := func(f func() error) (panicked any, err error) {
		defer func() { panicked = recover() }()
		return nil, f()
	}
	chunked := func(ctx context.Context, sink engine.RasterSink, src engine.RasterSource, k exec.Kernel, o engine.Options) func() error {
		return func() error {
			return exec.ProcessChunked(ctx, []engine.RasterSink{sink}, []engine.RasterSource{src}, k, o)
		}
	}
	out := func() raster.Float32Raster { return raster.NewFloat32Like(dem.r) }

	for _, workers := range []int{2, 4, runtime.GOMAXPROCS(0)} {
		o := chunks
		o.Workers = workers
		cases := []struct {
			name      string
			call      func() func() error
			wantErr   error
			wantPanic any
		}{
			{"ProcessN kernel panic", func() func() error {
				k := panicAt{box, h / 2, errBoom, new(atomic.Int64)}
				return func() error {
					return exec.Process(context.Background(), out(), dem.r, k, engine.Options{Workers: workers})
				}
			}, nil, errBoom},
			{"chunked kernel panic", func() func() error {
				k := panicAt{box, h / 2, errBoom, new(atomic.Int64)}
				return chunked(context.Background(), engine.NewMemorySink(out()), engine.NewMemorySource(dem.r), k, o)
			}, nil, errBoom},
			{"chunked source panic", func() func() error {
				src := &failingSource{RasterSource: engine.NewMemorySource(dem.r), n: 3, err: errBoom, panics: true}
				return chunked(context.Background(), engine.NewMemorySink(out()), src, box, o)
			}, nil, errBoom},
			{"chunked sink panic", func() func() error {
				sink := &panickingSink{RasterSink: engine.NewMemorySink(out()), n: 3, value: errBoom}
				return chunked(context.Background(), sink, engine.NewMemorySource(dem.r), box, o)
			}, nil, errBoom},
			{"chunked source error", func() func() error {
				src := &failingSource{RasterSource: engine.NewMemorySource(dem.r), n: 3, err: errBoom}
				return chunked(context.Background(), engine.NewMemorySink(out()), src, box, o)
			}, errBoom, nil},
			{"chunked sink error", func() func() error {
				sink := &failingSink{RasterSink: engine.NewMemorySink(out()), n: 3, err: errBoom}
				return chunked(context.Background(), sink, engine.NewMemorySource(dem.r), box, o)
			}, errBoom, nil},
			{"ProcessN cancelled from a kernel", func() func() error {
				ctx, cancel := context.WithCancel(context.Background())
				k := countCalls{box, 5, new(atomic.Int64), cancel}
				return func() error { return exec.Process(ctx, out(), dem.r, k, engine.Options{Workers: workers}) }
			}, context.Canceled, nil},
			{"ProcessN cancelled while workers are mid-call", func() func() error {
				return midCallCancel(func(ctx context.Context, k exec.Kernel) error {
					return exec.Process(ctx, out(), dem.r, k, engine.Options{Workers: workers})
				}, box)
			}, context.Canceled, nil},
			{"chunked cancelled while workers are mid-call", func() func() error {
				return midCallCancel(func(ctx context.Context, k exec.Kernel) error {
					return chunked(ctx, engine.NewMemorySink(out()), engine.NewMemorySource(dem.r), k, o)()
				}, box)
			}, context.Canceled, nil},
			{"chunked with a done context", func() func() error {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return chunked(ctx, engine.NewMemorySink(out()), engine.NewMemorySource(dem.r), box, o)
			}, context.Canceled, nil},
		}
		for _, tc := range cases {
			id := fmt.Sprintf("%s, workers=%d", tc.name, workers)
			panicked, err := run(tc.call())
			switch {
			case tc.wantPanic != nil && panicked != tc.wantPanic:
				t.Fatalf("%s: recovered %v, want %v", id, panicked, tc.wantPanic)
			case tc.wantPanic == nil && panicked != nil:
				t.Fatalf("%s: panic %v", id, panicked)
			case tc.wantErr != nil && !errors.Is(err, tc.wantErr):
				t.Fatalf("%s: err = %v, want %v", id, err, tc.wantErr)
			}
			requireNoLeaks(t, id)
		}
	}
}

// midCallCancel returns a call of f that cancels its context from another
// goroutine once one kernel call is running and the others are blocked
// mid-call, and releases them only after the cancellation, so that the
// call must still wait for every worker's claimed band or tile.
func midCallCancel(f func(ctx context.Context, k exec.Kernel) error, inner exec.Kernel) func() error {
	return func() error {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		release, first := make(chan struct{}), make(chan struct{})
		k := blockingKernel{inner, release, new(sync.Once), first}
		go func() {
			<-first
			cancel()
			close(release)
		}()
		return f(ctx, k)
	}
}

// panickingReducer panics with value on its nth Fold.
type panickingReducer struct {
	n     int64
	value any
	folds atomic.Int64
}

func (*panickingReducer) Inputs() int { return 1 }

func (*panickingReducer) Combine(a *tally, b tally) { tallyOp{1}.Combine(a, b) }

func (r *panickingReducer) Fold(p *tally, c exec.Cells) {
	if r.folds.Add(1) == r.n {
		panic(r.value)
	}
	tallyOp{1}.Fold(p, c)
}

// TestNoLeaksOnReduceFailure is TestNoLeaksOnFailure for the fold driver:
// every way a reduction can end early, with several workers, must give
// the documented outcome and leave no goroutine behind. It also checks
// what DESIGN.md §49 adds on top — that no value comes back with an
// error, ever — which the map path has no equivalent of.
func TestNoLeaksOnReduceFailure(t *testing.T) {
	defer exec.SetBandCells(1)()
	const w, h = 24, 200
	rng := rand.New(rand.NewPCG(11, 12))
	dem := newOperand(rng, w, h, true, true)
	errBoom := errors.New("boom")
	o := engine.Options{TileWidth: 5, TileHeight: 2}
	op := tallyOp{1}

	run := func(f func() (tally, error)) (panicked any, got tally, err error) {
		defer func() { panicked = recover() }()
		got, err = f()
		return nil, got, err
	}
	chunkedOf := func(ctx context.Context, src engine.RasterSource, r exec.Reducer[tally], opts engine.Options) func() (tally, error) {
		return func() (tally, error) {
			return exec.ReduceChunked(ctx, []engine.RasterSource{src}, r, opts)
		}
	}

	for _, workers := range []int{2, 4, runtime.GOMAXPROCS(0)} {
		opts := o
		opts.Workers = workers
		cases := []struct {
			name      string
			call      func() func() (tally, error)
			wantErr   error
			wantPanic any
		}{
			{"Reduce reducer panic", func() func() (tally, error) {
				r := &panickingReducer{n: 5, value: errBoom}
				return func() (tally, error) {
					return exec.Reduce(context.Background(), []raster.Float32Raster{dem.r}, r,
						engine.Options{Workers: workers})
				}
			}, nil, errBoom},
			{"ReduceChunked reducer panic", func() func() (tally, error) {
				r := &panickingReducer{n: 5, value: errBoom}
				return chunkedOf(context.Background(), engine.NewMemorySource(dem.r), r, opts)
			}, nil, errBoom},
			{"ReduceChunked source panic", func() func() (tally, error) {
				src := &failingSource{RasterSource: engine.NewMemorySource(dem.r), n: 3, err: errBoom, panics: true}
				return chunkedOf(context.Background(), src, op, opts)
			}, nil, errBoom},
			{"ReduceChunked source error", func() func() (tally, error) {
				src := &failingSource{RasterSource: engine.NewMemorySource(dem.r), n: 3, err: errBoom}
				return chunkedOf(context.Background(), src, op, opts)
			}, errBoom, nil},
			{"ReduceChunked with a done context", func() func() (tally, error) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return chunkedOf(ctx, engine.NewMemorySource(dem.r), op, opts)
			}, context.Canceled, nil},
			{"Reduce with a done context", func() func() (tally, error) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return func() (tally, error) {
					return exec.Reduce(ctx, []raster.Float32Raster{dem.r}, op, engine.Options{Workers: workers})
				}
			}, context.Canceled, nil},
		}
		for _, tc := range cases {
			id := fmt.Sprintf("%s, workers=%d", tc.name, workers)
			panicked, got, err := run(tc.call())
			switch {
			case tc.wantPanic != nil && panicked != tc.wantPanic:
				t.Fatalf("%s: recovered %v, want %v", id, panicked, tc.wantPanic)
			case tc.wantPanic == nil && panicked != nil:
				t.Fatalf("%s: panic %v", id, panicked)
			case tc.wantErr != nil && !errors.Is(err, tc.wantErr):
				t.Fatalf("%s: err = %v, want %v", id, err, tc.wantErr)
			}
			if err != nil && got != (tally{}) {
				t.Fatalf("%s: got %+v with error %v, want the zero value", id, got, err)
			}
			requireNoLeaks(t, id)
		}
	}
}
