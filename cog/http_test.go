package cog

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// object is what a test server serves, and can be replaced while a
// reader has it open.
type object struct {
	mu   sync.Mutex
	data []byte
	etag string // "" for none
}

func (o *object) set(data []byte, etag string) {
	o.mu.Lock()
	o.data, o.etag = data, etag
	o.mu.Unlock()
}

// server serves an object with http.ServeContent. fault, if set, sees
// every request first, numbered from 1, and handles it by returning
// true.
type server struct {
	*httptest.Server
	obj      object
	requests atomic.Int64
	fault    func(n int64, w http.ResponseWriter, r *http.Request) bool
}

func newServer(t *testing.T, data []byte, fault func(n int64, w http.ResponseWriter, r *http.Request) bool) *server {
	t.Helper()
	s := &server{fault: fault}
	s.obj.set(data, `"v1"`)
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := s.requests.Add(1)
		if s.fault != nil && s.fault(n, w, r) {
			return
		}
		s.serve(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *server) serve(w http.ResponseWriter, r *http.Request) {
	s.obj.mu.Lock()
	data, etag := s.obj.data, s.obj.etag
	s.obj.mu.Unlock()
	if etag != "" {
		w.Header().Set("ETag", etag)
	}
	http.ServeContent(w, r, "f.tif", time.Time{}, bytes.NewReader(data))
}

// truncated serves the response s would, cut to half its body, with the
// headers of the whole: a connection dropped mid-body.
func (s *server) truncated(w http.ResponseWriter, r *http.Request) {
	rec := httptest.NewRecorder()
	s.serve(rec, r)
	for k, v := range rec.Header() {
		w.Header()[k] = v
	}
	w.WriteHeader(rec.Code)
	_, _ = w.Write(rec.Body.Bytes()[:rec.Body.Len()/2])
}

func randomBytes(n int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, 1))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	return b
}

// fastRetries retries twice, without real waits.
var fastRetries = HTTPOptions{MaxRetries: 2, RetryBackoff: time.Microsecond}

func mustOpenHTTP(t *testing.T, url string, opts HTTPOptions) *HTTPReaderAt {
	t.Helper()
	r, err := NewHTTPReaderAt(context.Background(), url, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	return r
}

// TestHTTPReadAt compares ReadAt with bytes.Reader's at offsets inside
// the prefetch, across its end, past it, at the end of the object and
// beyond it, and counts the requests.
func TestHTTPReadAt(t *testing.T) {
	const size = 200_000
	data := randomBytes(size, 1)
	s := newServer(t, data, nil)
	r := mustOpenHTTP(t, s.URL+"/dem.tif", HTTPOptions{})
	if r.Size() != size {
		t.Fatalf("Size %d, want %d", r.Size(), size)
	}
	if got := r.Stats(); got.Requests != 1 || got.Bytes != DefaultPrefetchBytes {
		t.Fatalf("after the prefetch: %+v, want 1 request of %d bytes", got, DefaultPrefetchBytes)
	}

	want := bytes.NewReader(data)
	requests := int64(1)
	check := func(off int64, n int, fetches bool) {
		t.Helper()
		p, q := make([]byte, n), make([]byte, n)
		gn, gerr := r.ReadAt(p, off)
		wn, werr := want.ReadAt(q, off)
		if gn != wn || !errors.Is(gerr, werr) || !bytes.Equal(p[:gn], q[:wn]) {
			t.Fatalf("ReadAt(%d bytes at %d) = %d, %v; want %d, %v", n, off, gn, gerr, wn, werr)
		}
		if fetches {
			requests++
		}
		if got := r.Stats().Requests; got != requests {
			t.Fatalf("ReadAt(%d bytes at %d): %d requests in all, want %d", n, off, got, requests)
		}
	}
	check(0, 8, false)                        // the TIFF header
	check(100, 5000, false)                   // an IFD
	check(DefaultPrefetchBytes-10, 10, false) // the prefetch's last bytes
	check(DefaultPrefetchBytes-10, 100, true) // across its end
	check(DefaultPrefetchBytes, 1, true)      // just past it
	check(size-100, 100, true)                // the object's last bytes
	check(size-100, 1000, true)               // a short final range: io.EOF
	check(size-1, 2, true)                    // one byte, then io.EOF
	check(size, 10, false)                    // at the end: io.EOF, no request
	check(size+1000, 10, false)               // beyond it
	check(1000, 0, false)                     // nothing
	check(0, size+10, true)                   // everything, and more
	rng := rand.New(rand.NewPCG(2, 3))
	for range 100 {
		off := rng.Int64N(size + 100)
		n := rng.IntN(20_000)
		check(off, n, n > 0 && off < size && off+int64(n) > DefaultPrefetchBytes)
	}
	if _, err := r.ReadAt(make([]byte, 1), -1); err == nil {
		t.Error("ReadAt at -1 succeeded")
	}
}

// TestHTTPSize covers how the size is found: from the prefetch's
// Content-Range, for an object shorter than the prefetch and an empty
// one, and from HEAD without a prefetch or when the server does not know
// the total.
func TestHTTPSize(t *testing.T) {
	t.Run("shorter than the prefetch", func(t *testing.T) {
		data := randomBytes(1000, 4)
		s := newServer(t, data, nil)
		r := mustOpenHTTP(t, s.URL, HTTPOptions{})
		got := make([]byte, 1000)
		if n, err := r.ReadAt(got, 0); n != 1000 || err != nil || !bytes.Equal(got, data) {
			t.Fatalf("ReadAt = %d, %v", n, err)
		}
		if r.Size() != 1000 || s.requests.Load() != 1 {
			t.Errorf("Size %d after %d requests, want 1000 after 1", r.Size(), s.requests.Load())
		}
	})
	t.Run("empty", func(t *testing.T) {
		s := newServer(t, nil, nil)
		r := mustOpenHTTP(t, s.URL, HTTPOptions{})
		if n, err := r.ReadAt(make([]byte, 4), 0); r.Size() != 0 || n != 0 || !errors.Is(err, io.EOF) {
			t.Errorf("Size %d, ReadAt = %d, %v; want 0, 0, io.EOF", r.Size(), n, err)
		}
		if _, err := Open(r); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("Open of an empty object: %v, want io.ErrUnexpectedEOF", err)
		}
	})
	t.Run("empty, answered 416", func(t *testing.T) {
		s := newServer(t, nil, func(_ int64, w http.ResponseWriter, _ *http.Request) bool {
			w.Header().Set("Content-Range", "bytes */0")
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return true
		})
		if r := mustOpenHTTP(t, s.URL, HTTPOptions{}); r.Size() != 0 {
			t.Errorf("Size %d, want 0", r.Size())
		}
	})
	t.Run("416 for a non-empty object", func(t *testing.T) {
		s := newServer(t, nil, func(_ int64, w http.ResponseWriter, _ *http.Request) bool {
			w.Header().Set("Content-Range", "bytes */10")
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return true
		})
		var se *HTTPStatusError
		if _, err := NewHTTPReaderAt(context.Background(), s.URL, HTTPOptions{}); !errors.As(err, &se) || se.StatusCode != 416 {
			t.Errorf("NewHTTPReaderAt: %v, want status 416", err)
		}
	})
	t.Run("HEAD without a prefetch", func(t *testing.T) {
		data := randomBytes(5000, 5)
		var methods []string
		var mu sync.Mutex
		s := newServer(t, data, func(_ int64, _ http.ResponseWriter, r *http.Request) bool {
			mu.Lock()
			methods = append(methods, r.Method)
			mu.Unlock()
			return false
		})
		r := mustOpenHTTP(t, s.URL, HTTPOptions{PrefetchBytes: -1})
		got := make([]byte, 10)
		if _, err := r.ReadAt(got, 100); err != nil || !bytes.Equal(got, data[100:110]) {
			t.Fatal(err)
		}
		if r.Size() != 5000 || fmt.Sprint(methods) != "[HEAD GET]" {
			t.Errorf("Size %d, requests %v; want 5000, [HEAD GET]", r.Size(), methods)
		}
	})
	t.Run("unknown total", func(t *testing.T) {
		data := randomBytes(5000, 6)
		s := newServer(t, data, nil)
		s.fault = func(n int64, w http.ResponseWriter, _ *http.Request) bool {
			if n != 1 {
				return false
			}
			w.Header().Set("Content-Range", "bytes 0-99/*")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(data[:100])
			return true
		}
		r := mustOpenHTTP(t, s.URL, HTTPOptions{PrefetchBytes: 100})
		if r.Size() != 5000 || len(r.head) != 100 || s.requests.Load() != 2 {
			t.Errorf("Size %d, prefetch %d bytes, %d requests; want 5000, 100, 2", r.Size(), len(r.head), s.requests.Load())
		}
	})
}

// TestHTTPOpen reads a tiled GeoTIFF through the HTTP reader, from many
// goroutines at once (run it with -race), with a prefetch too small to
// hold the tiles, and checks every window against the values written.
func TestHTTPOpen(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const w, h = 64, 48
	st := sampleType{"int16", sampleInt, 2}
	vals := make([]float64, w*h)
	for i := range vals {
		vals[i] = randomValue(rng, st)
	}
	sp := imageSpec{w: w, h: h, bands: 1, format: st.format, size: st.size, tiled: true, blockW: 16, blockH: 16,
		planar: planarChunky, compression: compressionZSTD, predictor: predictorHorizontal, vals: [][]float64{vals}}
	data := fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()
	s := newServer(t, data, nil)
	r := mustOpenHTTP(t, s.URL, HTTPOptions{PrefetchBytes: 64, MaxConcurrent: 3})
	f, err := Open(r)
	if err != nil {
		t.Fatal(err)
	}
	src, _ := f.Source(SourceOptions{CacheBytes: 3000})
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			rng := rand.New(rand.NewPCG(uint64(g), 9))
			for range 30 {
				x, y := rng.IntN(w), rng.IntN(h)
				checkWindow(t, "http", src, vals, w, x, y, 1+rng.IntN(w-x), 1+rng.IntN(h-y), false, 0)
			}
		})
	}
	wg.Wait()
}

// TestHTTPFaults serves a range request wrongly in each way a server or
// network can, and checks that the reader retries what is transient,
// refuses the rest, and says which bytes of which URL failed.
func TestHTTPFaults(t *testing.T) {
	const size = 100_000
	const off, n = 70_000, 100
	cases := []struct {
		name     string
		fault    func(s *server, n int64, w http.ResponseWriter, r *http.Request) bool
		want     error // nil: the read succeeds
		status   int   // the *HTTPStatusError wanted, if any
		requests int64 // the prefetch included
	}{
		{"200 instead of 206", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k == 1 {
				return false
			}
			r.Header.Del("Range")
			s.serve(w, r)
			return true
		}, ErrRangeIgnored, 0, 2},
		{"another range", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				r.Header.Set("Range", "bytes=0-99")
			}
			return false
		}, errAny, 0, 2},
		{"truncated once", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k == 2 {
				s.truncated(w, r)
				return true
			}
			return false
		}, nil, 0, 3},
		{"truncated always", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				s.truncated(w, r)
				return true
			}
			return false
		}, io.ErrUnexpectedEOF, 0, 4},
		{"500 twice", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k == 2 || k == 3 {
				http.Error(w, "busy", http.StatusInternalServerError)
				return true
			}
			return false
		}, nil, 0, 4},
		{"500 always", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				http.Error(w, "down", http.StatusServiceUnavailable)
				return true
			}
			return false
		}, nil, http.StatusServiceUnavailable, 4},
		{"429 with Retry-After", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k == 2 {
				w.Header().Set("Retry-After", "0")
				http.Error(w, "slow down", http.StatusTooManyRequests)
				return true
			}
			return false
		}, nil, 0, 3},
		{"connection dropped", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k == 2 {
				conn, _, err := http.NewResponseController(w).Hijack()
				if err == nil {
					_ = conn.Close()
				}
				return true
			}
			return false
		}, nil, 0, 3},
		{"404, not retried", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				http.NotFound(w, r)
				return true
			}
			return false
		}, nil, http.StatusNotFound, 2},
		{"replaced: another ETag", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				s.obj.set(s.obj.data, `"v2"`) // If-Match fails: 412
			}
			return false
		}, ErrObjectChanged, http.StatusPreconditionFailed, 2},
		{"replaced: shorter, no ETag", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				s.obj.set(randomBytes(off-1, 9), "") // the range is past the end: 416
			}
			return false
		}, ErrObjectChanged, http.StatusRequestedRangeNotSatisfiable, 2},
		{"replaced: longer, no ETag", func(s *server, k int64, w http.ResponseWriter, r *http.Request) bool {
			if k > 1 {
				s.obj.set(randomBytes(2*size, 9), "") // Content-Range gives another total
			}
			return false
		}, ErrObjectChanged, 0, 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			data := randomBytes(size, 10)
			s := newServer(t, data, nil)
			s.fault = func(k int64, w http.ResponseWriter, r *http.Request) bool {
				if k == 1 && strings.Contains(c.name, "no ETag") {
					s.obj.set(data, "")
				}
				return c.fault(s, k, w, r)
			}
			r := mustOpenHTTP(t, s.URL+"/data/dem.tif?X-Amz-Signature=secret", fastRetries)
			p := make([]byte, n)
			got, err := r.ReadAt(p, off)
			if s.requests.Load() != c.requests {
				t.Errorf("%d requests, want %d", s.requests.Load(), c.requests)
			}
			if c.want == nil && c.status == 0 {
				if err != nil || got != n || !bytes.Equal(p, data[off:off+n]) {
					t.Fatalf("ReadAt = %d, %v; want the bytes", got, err)
				}
				return
			}
			if err == nil {
				t.Fatal("ReadAt succeeded")
			}
			if c.want != nil && !errors.Is(c.want, errAny) && !errors.Is(err, c.want) {
				t.Errorf("error %v, want %v", err, c.want)
			}
			var se *HTTPStatusError
			if c.status != 0 && (!errors.As(err, &se) || se.StatusCode != c.status) {
				t.Errorf("error %v, want status %d", err, c.status)
			}
			var he *HTTPError
			if !errors.As(err, &he) || he.Off != off || he.End != off+n || he.URL != s.URL+"/data/dem.tif?..." {
				t.Errorf("error %#v, want an *HTTPError for bytes %d-%d of the redacted URL", err, off, off+n-1)
			}
			msg := err.Error()
			if !strings.Contains(msg, fmt.Sprintf("bytes %d-%d", off, off+n-1)) || strings.Contains(msg, "secret") {
				t.Errorf("error %q: want the byte range, and no query string", msg)
			}
		})
	}
}

// errAny stands for an error whose kind a case does not check.
var errAny = errors.New("any error")

// TestHTTPOpenFaults: a server that ignores ranges, or one that is not
// there, fails NewHTTPReaderAt without leaking the query string.
func TestHTTPOpenFaults(t *testing.T) {
	s := newServer(t, randomBytes(1000, 11), func(_ int64, w http.ResponseWriter, _ *http.Request) bool {
		_, _ = w.Write([]byte("the whole thing")) // 200
		return true
	})
	_, err := NewHTTPReaderAt(context.Background(), s.URL+"?sig=secret", fastRetries)
	if !errors.Is(err, ErrRangeIgnored) || s.requests.Load() != 1 {
		t.Errorf("a server ignoring ranges: %v after %d requests, want ErrRangeIgnored after 1", err, s.requests.Load())
	}

	gone := httptest.NewServer(http.NotFoundHandler())
	gone.Close()
	_, err = NewHTTPReaderAt(context.Background(), gone.URL+"/x.tif?sig=secret", fastRetries)
	var he *HTTPError
	if !errors.As(err, &he) || strings.Contains(err.Error(), "secret") || !strings.Contains(err.Error(), "/x.tif") {
		t.Errorf("a closed server: %v, want an *HTTPError with the path and no query string", err)
	}

	if _, err := NewHTTPReaderAt(context.Background(), "file:///tmp/x.tif", HTTPOptions{}); err == nil {
		t.Error("a file URL was accepted")
	}
}

// TestHTTPHeaders checks what every range request carries.
func TestHTTPHeaders(t *testing.T) {
	var bad atomic.Value
	s := newServer(t, randomBytes(100_000, 12), func(n int64, _ http.ResponseWriter, r *http.Request) bool {
		switch {
		case r.Header.Get("Authorization") != "Bearer token":
			bad.Store(fmt.Sprintf("request %d: Authorization %q", n, r.Header.Get("Authorization")))
		case r.Header.Get("Accept-Encoding") != "identity":
			bad.Store(fmt.Sprintf("request %d: Accept-Encoding %q", n, r.Header.Get("Accept-Encoding")))
		case n > 1 && r.Header.Get("If-Match") != `"v1"`:
			bad.Store(fmt.Sprintf("request %d: If-Match %q", n, r.Header.Get("If-Match")))
		}
		return false
	})
	r := mustOpenHTTP(t, s.URL, HTTPOptions{Header: http.Header{"Authorization": {"Bearer token"}}})
	if _, err := r.ReadAt(make([]byte, 10), 90_000); err != nil {
		t.Fatal(err)
	}
	if b := bad.Load(); b != nil {
		t.Error(b)
	}
}

// TestHTTPConcurrencyCap reads from many goroutines at once and checks
// that no more than MaxConcurrent requests are ever in flight.
func TestHTTPConcurrencyCap(t *testing.T) {
	const limit = 3
	data := randomBytes(200_000, 13)
	var inFlight, peak atomic.Int64
	s := newServer(t, data, func(_ int64, _ http.ResponseWriter, _ *http.Request) bool {
		k := inFlight.Add(1)
		for p := peak.Load(); k > p && !peak.CompareAndSwap(p, k); p = peak.Load() {
		}
		time.Sleep(10 * time.Millisecond)
		inFlight.Add(-1)
		return false
	})
	r := mustOpenHTTP(t, s.URL, HTTPOptions{MaxConcurrent: limit})
	var wg sync.WaitGroup
	for g := range 24 {
		wg.Go(func() {
			off := int64(DefaultPrefetchBytes + g*1000)
			p := make([]byte, 1000)
			if _, err := r.ReadAt(p, off); err != nil || !bytes.Equal(p, data[off:off+1000]) {
				t.Errorf("goroutine %d: %v", g, err)
			}
		})
	}
	wg.Wait()
	if peak.Load() != limit {
		t.Errorf("%d requests in flight at most, want %d", peak.Load(), limit)
	}
}

// TestHTTPCancel cancels the reader's context while a read waits for a
// response, for a retry, and for a free request slot.
func TestHTTPCancel(t *testing.T) {
	data := randomBytes(200_000, 14)
	hang := func(k int64, _ http.ResponseWriter, r *http.Request) bool {
		if k > 1 {
			<-r.Context().Done() // until the client gives up
			return true
		}
		return false
	}
	fail := func(k int64, w http.ResponseWriter, _ *http.Request) bool {
		if k > 1 {
			http.Error(w, "down", http.StatusInternalServerError)
			return true
		}
		return false
	}
	cases := []struct {
		name    string
		fault   func(int64, http.ResponseWriter, *http.Request) bool
		opts    HTTPOptions
		readers int
	}{
		{"waiting for a response", hang, HTTPOptions{}, 1},
		{"waiting to retry", fail, HTTPOptions{RetryBackoff: time.Hour}, 1},
		{"waiting for a request slot", hang, HTTPOptions{MaxConcurrent: 1}, 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := newServer(t, data, c.fault)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r, err := NewHTTPReaderAt(ctx, s.URL, c.opts)
			if err != nil {
				t.Fatal(err)
			}
			errs := make(chan error, c.readers)
			for i := range c.readers {
				go func() {
					_, err := r.ReadAt(make([]byte, 100), int64(100_000+1000*i))
					errs <- err
				}()
			}
			time.Sleep(50 * time.Millisecond)
			cancel()
			for range c.readers {
				select {
				case err := <-errs:
					var he *HTTPError
					if !errors.Is(err, context.Canceled) || !errors.As(err, &he) {
						t.Errorf("ReadAt: %v, want context.Canceled in an *HTTPError", err)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("ReadAt did not return after the context was cancelled")
				}
			}
			if _, err := r.ReadAt(make([]byte, 100), 150_000); !errors.Is(err, context.Canceled) {
				t.Errorf("a read after the cancel: %v, want context.Canceled", err)
			}
		})
	}
	t.Run("while opening", func(t *testing.T) {
		s := newServer(t, data, func(_ int64, _ http.ResponseWriter, r *http.Request) bool {
			<-r.Context().Done()
			return true
		})
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		if _, err := NewHTTPReaderAt(ctx, s.URL, HTTPOptions{}); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("NewHTTPReaderAt: %v, want context.DeadlineExceeded", err)
		}
	})
}

func TestParseContentRange(t *testing.T) {
	cases := []struct {
		s                  string
		first, last, total int64
		ok                 bool
	}{
		{"bytes 0-99/1000", 0, 99, 1000, true},
		{"bytes 5-5/6", 5, 5, 6, true},
		{"bytes 0-99/*", 0, 99, -1, true},
		{"bytes */1000", -1, -1, 1000, true},
		{"bytes */0", -1, -1, 0, true},
		{"", 0, 0, 0, false},
		{"bytes 0-99", 0, 0, 0, false},
		{"bytes 99-0/1000", 0, 0, 0, false},
		{"bytes -1-5/10", 0, 0, 0, false},
		{"items 0-9/10", 0, 0, 0, false},
		{"bytes 0-x/10", 0, 0, 0, false},
		{"bytes 0-9/-10", 0, 0, 0, false},
	}
	for _, c := range cases {
		first, last, total, err := parseContentRange(c.s)
		if (err == nil) != c.ok || c.ok && (first != c.first || last != c.last || total != c.total) {
			t.Errorf("parseContentRange(%q) = %d, %d, %d, %v", c.s, first, last, total, err)
		}
	}
}
