package cog

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Defaults of HTTPOptions' zero values.
const (
	// DefaultPrefetchBytes is the header block NewHTTPReaderAt fetches:
	// 64 KiB. GDAL writes a COG's IFDs, and the tile offsets and byte
	// counts they point at, ahead of the first tile, so for any but very
	// large COGs Open is served from it without another request.
	DefaultPrefetchBytes = 64 << 10
	// DefaultMaxConcurrent is the number of requests one reader has in
	// flight at most.
	DefaultMaxConcurrent = 8
	// DefaultMaxRetries is the number of times a request is retried after
	// a transient failure.
	DefaultMaxRetries = 4
	// DefaultRetryBackoff is the wait before the first retry. It doubles
	// with each retry, up to maxRetryWait.
	DefaultRetryBackoff = 100 * time.Millisecond
	// maxRetryWait bounds one wait between attempts, including one a
	// server asks for with Retry-After.
	maxRetryWait = 10 * time.Second
)

// HTTPOptions configures an HTTPReaderAt. The zero value is ready to
// use.
type HTTPOptions struct {
	// Client sends the requests. nil means a client of the reader's own,
	// which keeps MaxConcurrent idle connections to the host.
	Client *http.Client
	// Header is added to every request, for example an Authorization
	// header. Range, If-Match and Accept-Encoding are the reader's own.
	Header http.Header
	// PrefetchBytes is the size of the header block fetched when the
	// reader is created, and from which reads inside it are served: 0
	// means DefaultPrefetchBytes. A negative value fetches none, and the
	// size is then taken from a HEAD request.
	PrefetchBytes int64
	// MaxConcurrent bounds the requests in flight: 0 means
	// DefaultMaxConcurrent.
	MaxConcurrent int
	// MaxRetries bounds the retries of one request after a transient
	// failure: 0 means DefaultMaxRetries, a negative value none.
	MaxRetries int
	// RetryBackoff is the wait before the first retry: 0 means
	// DefaultRetryBackoff.
	RetryBackoff time.Duration
}

var (
	// ErrRangeIgnored is the error when a server answers a range request
	// with the whole object (status 200) instead of the range (206).
	ErrRangeIgnored = errors.New("the server ignored the Range header: status 200, want 206 Partial Content")
	// ErrObjectChanged is the error when the object is no longer the one
	// the reader opened: its size or ETag changed (status 412 or 416, or
	// another size in Content-Range). Mixing blocks of two versions of a
	// file would read as neither, so the reader stops.
	ErrObjectChanged = errors.New("the object changed since the reader opened it")
)

// HTTPStatusError is an unexpected HTTP status.
type HTTPStatusError struct {
	StatusCode int
	Status     string // as the response gives it, e.g. "404 Not Found"
}

func (e *HTTPStatusError) Error() string { return "status " + e.Status }

// HTTPError is a failed request of an HTTPReaderAt: the request, the
// bytes it asked for, and why it failed, after any retries. Err may be
// an *HTTPStatusError, ErrRangeIgnored, ErrObjectChanged, a context error
// or a transport error, and errors.Is and errors.As see through to it.
type HTTPError struct {
	Method string
	// URL is the object's URL without its query string and user
	// information, which in a presigned URL are credentials.
	URL string
	// Off and End are the bytes requested, [Off, End). Both are 0 for a
	// HEAD request.
	Off, End int64
	Err      error
}

func (e *HTTPError) Error() string {
	if e.End > e.Off {
		return fmt.Sprintf("cog: %s %s bytes %d-%d: %v", e.Method, e.URL, e.Off, e.End-1, e.Err)
	}
	return fmt.Sprintf("cog: %s %s: %v", e.Method, e.URL, e.Err)
}

func (e *HTTPError) Unwrap() error { return e.Err }

// HTTPStats counts what an HTTPReaderAt has fetched.
type HTTPStats struct {
	// Requests is the number of HTTP requests sent, the prefetch and
	// every retry included.
	Requests int64
	// Bytes is the number of body bytes received.
	Bytes int64
}

// HTTPReaderAt is an io.ReaderAt over an object served by HTTP range
// requests: a COG on S3, GCS or any HTTPS server, public or through a
// presigned URL. Pass it to Open. It is safe for concurrent ReadAt
// calls, and sends at most HTTPOptions.MaxConcurrent requests at once.
//
// Each ReadAt outside the prefetched header is one GET with a Range
// header, which must be answered with 206 Partial Content and exactly
// the range asked for. Transport errors, truncated bodies, 429 and 5xx
// are retried with exponential backoff, a bounded number of times. If
// the server gives a strong ETag, every request carries it in If-Match,
// so a read of an object replaced since fails with ErrObjectChanged
// instead of returning bytes of the new one.
type HTTPReaderAt struct {
	ctx        context.Context
	url        string
	safeURL    string
	client     *http.Client
	own        *http.Transport // the client's transport, if the reader made it
	header     http.Header
	sem        chan struct{}
	maxRetries int
	backoff    time.Duration

	size int64
	head []byte // the object's first len(head) bytes
	etag string // a strong ETag, sent in If-Match, or ""

	requests, bytes atomic.Int64
}

var _ io.ReaderAt = (*HTTPReaderAt)(nil)

// NewHTTPReaderAt opens the object at rawURL for reading by ranges. It
// fetches the first opts.PrefetchBytes bytes, which gives the object's
// size in Content-Range, or, without a prefetch, sends a HEAD request.
//
// ctx bounds every request the reader makes, for as long as it is used:
// cancelling it makes reads in flight, and every later read, fail with
// its error. It is not a per-read deadline; use the client's Timeout for
// that. A Source's ReadWindow also checks its own ctx, between blocks.
//
// It returns an *HTTPError if the object cannot be read by ranges.
func NewHTTPReaderAt(ctx context.Context, rawURL string, opts HTTPOptions) (*HTTPReaderAt, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("cog: URL scheme %q, want http or https", u.Scheme)
	}
	r := &HTTPReaderAt{
		ctx:        ctx,
		url:        rawURL,
		safeURL:    redact(u),
		client:     opts.Client,
		header:     opts.Header,
		maxRetries: opts.MaxRetries,
		backoff:    opts.RetryBackoff,
	}
	conc := opts.MaxConcurrent
	if conc <= 0 {
		conc = DefaultMaxConcurrent
	}
	r.sem = make(chan struct{}, conc)
	switch {
	case r.maxRetries == 0:
		r.maxRetries = DefaultMaxRetries
	case r.maxRetries < 0:
		r.maxRetries = 0
	}
	if r.backoff <= 0 {
		r.backoff = DefaultRetryBackoff
	}
	if r.client == nil {
		r.client = http.DefaultClient
		// http.DefaultTransport keeps 2 idle connections per host, so
		// most of MaxConcurrent requests would open a new one each.
		if dt, ok := http.DefaultTransport.(*http.Transport); ok {
			r.own = dt.Clone()
			r.own.MaxIdleConnsPerHost = conc
			r.client = &http.Client{Transport: r.own}
		}
	}

	prefetch := opts.PrefetchBytes
	if prefetch == 0 {
		prefetch = DefaultPrefetchBytes
	}
	if prefetch > 0 {
		err = r.prefetch(prefetch)
	} else {
		err = r.headSize()
	}
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	return r, nil
}

// Size returns the object's size in bytes.
func (r *HTTPReaderAt) Size() int64 { return r.size }

// Stats returns the requests the reader has sent and the bytes it has
// received so far.
func (r *HTTPReaderAt) Stats() HTTPStats {
	return HTTPStats{Requests: r.requests.Load(), Bytes: r.bytes.Load()}
}

// Close closes the idle connections of the reader's own client. It does
// nothing to a client passed in HTTPOptions. It always returns nil.
func (r *HTTPReaderAt) Close() error {
	if r.own != nil {
		r.own.CloseIdleConnections()
	}
	return nil
}

// ReadAt reads len(p) bytes at off, as io.ReaderAt specifies: it returns
// io.EOF with fewer bytes when the object ends first. Bytes inside the
// prefetched header are copied from it; the rest is one range request.
// Errors from the server are *HTTPError.
func (r *HTTPReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, &HTTPError{Method: http.MethodGet, URL: r.safeURL, Off: off, End: off + int64(len(p)),
			Err: errors.New("negative offset")}
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off >= r.size {
		return 0, io.EOF
	}
	var eof error
	if int64(len(p)) > r.size-off {
		p, eof = p[:r.size-off], io.EOF
	}
	n := 0
	if off < int64(len(r.head)) {
		n = copy(p, r.head[off:])
	}
	if n < len(p) {
		if err := r.fetch(p[n:], off+int64(n)); err != nil {
			return n, err
		}
	}
	return len(p), eof
}

// fetch reads exactly len(p) bytes at off, which all lie inside the
// object, with one range request and its retries.
func (r *HTTPReaderAt) fetch(p []byte, off int64) error {
	end := off + int64(len(p))
	select {
	case r.sem <- struct{}{}:
	case <-r.ctx.Done():
		return &HTTPError{Method: http.MethodGet, URL: r.safeURL, Off: off, End: end, Err: r.ctx.Err()}
	}
	defer func() { <-r.sem }()
	return r.do(http.MethodGet, off, end, func(resp *http.Response) error {
		if err := r.checkPartial(resp, off, end); err != nil {
			return err
		}
		return r.readBody(resp.Body, p)
	})
}

// checkPartial accepts a 206 response carrying bytes [off, end) of an
// object of the reader's size.
func (r *HTTPReaderAt) checkPartial(resp *http.Response, off, end int64) error {
	switch resp.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		return ErrRangeIgnored
	case http.StatusPreconditionFailed, http.StatusRequestedRangeNotSatisfiable:
		return fmt.Errorf("%w: %w", statusError(resp), ErrObjectChanged)
	default:
		return statusError(resp)
	}
	start, last, total, err := parseContentRange(resp.Header.Get("Content-Range"))
	if err != nil {
		return err
	}
	if total >= 0 && total != r.size {
		return fmt.Errorf("%w: Content-Range %q, and the object had %d bytes",
			ErrObjectChanged, resp.Header.Get("Content-Range"), r.size)
	}
	if start != off || last != end-1 {
		return fmt.Errorf("Content-Range %q, want bytes %d-%d", resp.Header.Get("Content-Range"), off, end-1)
	}
	return nil
}

// prefetch fetches the first n bytes, or the whole object if it is
// shorter, and takes the object's size and ETag from the response.
func (r *HTTPReaderAt) prefetch(n int64) error {
	unknownSize := false
	err := r.do(http.MethodGet, 0, n, func(resp *http.Response) error {
		switch resp.StatusCode {
		case http.StatusPartialContent:
		case http.StatusOK:
			if resp.ContentLength == 0 {
				// An empty object: Go's ServeContent answers so, and
				// some object stores do.
				r.size, r.head = 0, nil
				return nil
			}
			return ErrRangeIgnored
		case http.StatusRequestedRangeNotSatisfiable:
			// Only an empty object has no byte 0.
			_, _, total, err := parseContentRange(resp.Header.Get("Content-Range"))
			if err != nil || total != 0 {
				return statusError(resp)
			}
			r.size, r.head = 0, nil
			return nil
		default:
			return statusError(resp)
		}
		start, last, total, err := parseContentRange(resp.Header.Get("Content-Range"))
		if err != nil {
			return err
		}
		if start != 0 || last < 0 || last >= n || (total >= 0 && last >= total) {
			return fmt.Errorf("Content-Range %q, want bytes 0-%d", resp.Header.Get("Content-Range"), n-1)
		}
		head := make([]byte, last+1)
		if err := r.readBody(resp.Body, head); err != nil {
			return err
		}
		r.head, r.size, unknownSize = head, total, total < 0
		r.setETag(resp)
		return nil
	})
	if err != nil {
		return err
	}
	if unknownSize {
		// "bytes 0-65535/*": the server does not know the size.
		return r.headSize()
	}
	return nil
}

// headSize takes the object's size and ETag from a HEAD request.
func (r *HTTPReaderAt) headSize() error {
	return r.do(http.MethodHead, 0, 0, func(resp *http.Response) error {
		if resp.StatusCode != http.StatusOK {
			return statusError(resp)
		}
		if resp.ContentLength < 0 {
			return errors.New("the response to HEAD has no Content-Length")
		}
		if r.head != nil && resp.ContentLength < int64(len(r.head)) {
			return fmt.Errorf("%w: HEAD gives %d bytes, and a range of %d was read",
				ErrObjectChanged, resp.ContentLength, len(r.head))
		}
		r.size = resp.ContentLength
		if r.etag == "" {
			r.setETag(resp)
		}
		return nil
	})
}

// setETag keeps the response's ETag, if it is a strong one: If-Match
// compares ETags strongly, so a weak one would fail every request.
func (r *HTTPReaderAt) setETag(resp *http.Response) {
	if e := resp.Header.Get("ETag"); e != "" && !strings.HasPrefix(e, "W/") {
		r.etag = e
	}
}

// readBody reads exactly len(p) bytes of a response body. A body that
// ends early is retried: a connection dropped mid-body is transient.
func (r *HTTPReaderAt) readBody(body io.Reader, p []byte) error {
	n, err := io.ReadFull(body, p)
	r.bytes.Add(int64(n))
	if err == nil {
		return nil
	}
	if cerr := r.ctx.Err(); cerr != nil {
		return cerr
	}
	if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF
	}
	return transient{fmt.Errorf("the body ended after %d of %d bytes: %w", n, len(p), err)}
}

// transient marks an error worth retrying.
type transient struct{ err error }

func (t transient) Error() string { return t.err.Error() }
func (t transient) Unwrap() error { return t.err }

// do sends a request for bytes [off, end) (none if end is 0) until
// handle accepts a response, retrying transient failures. handle sees
// every response except 429 and 5xx, which are retried; it returns a
// transient error to have the request retried. do returns an *HTTPError.
func (r *HTTPReaderAt) do(method string, off, end int64, handle func(*http.Response) error) error {
	var err error
	for attempt := 0; ; attempt++ {
		var wait time.Duration
		wait, err = r.attempt(method, off, end, handle)
		var t transient
		if err == nil || !errors.As(err, &t) {
			break
		}
		if attempt == r.maxRetries {
			err = t.err
			break
		}
		if serr := sleep(r.ctx, r.retryWait(attempt, wait)); serr != nil {
			err = serr
			break
		}
	}
	if err == nil {
		return nil
	}
	return &HTTPError{Method: method, URL: r.safeURL, Off: off, End: end, Err: err}
}

// attempt sends one request. It returns the wait a server asked for with
// Retry-After, if any, and a transient error if the request should be
// retried.
func (r *HTTPReaderAt) attempt(method string, off, end int64, handle func(*http.Response) error) (time.Duration, error) {
	req, err := http.NewRequestWithContext(r.ctx, method, r.url, nil)
	if err != nil {
		return 0, err
	}
	for k, v := range r.header {
		req.Header[k] = v
	}
	// Transparent gzip would make the range one of the compressed bytes.
	req.Header.Set("Accept-Encoding", "identity")
	if end > off {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, end-1))
	}
	if r.etag != "" {
		req.Header.Set("If-Match", r.etag)
	}
	r.requests.Add(1)
	resp, err := r.client.Do(req)
	if err != nil {
		if cerr := r.ctx.Err(); cerr != nil {
			return 0, cerr
		}
		// A *url.Error repeats the URL, query string and all.
		var ue *url.Error
		if errors.As(err, &ue) {
			err = fmt.Errorf("%s: %w", ue.Op, ue.Err)
		}
		return 0, transient{err}
	}
	defer func() {
		// Drain a little, so the connection can be reused.
		_, _ = io.CopyN(io.Discard, resp.Body, 4<<10)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		return retryAfter(resp), transient{statusError(resp)}
	}
	return 0, handle(resp)
}

// retryWait is the wait before retry attempt+1: the backoff doubled per
// attempt with jitter, or what the server asked for if that is longer,
// and never more than maxRetryWait.
func (r *HTTPReaderAt) retryWait(attempt int, asked time.Duration) time.Duration {
	d := r.backoff << min(attempt, 20)
	if d <= 0 || d > maxRetryWait {
		d = maxRetryWait
	}
	d = d/2 + rand.N(d/2+1) // many readers backing off together should not return together
	return min(max(d, asked), maxRetryWait)
}

// sleep waits d, or until ctx is done.
func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// retryAfter reads a Retry-After header in seconds, or returns 0.
func retryAfter(resp *http.Response) time.Duration {
	s, err := strconv.Atoi(strings.TrimSpace(resp.Header.Get("Retry-After")))
	if err != nil || s < 0 {
		return 0
	}
	return time.Duration(min(s, 3600)) * time.Second
}

func statusError(resp *http.Response) *HTTPStatusError {
	return &HTTPStatusError{StatusCode: resp.StatusCode, Status: resp.Status}
}

// parseContentRange parses "bytes first-last/total" or "bytes */total".
// first and last are -1 for "*", total is -1 for "*".
func parseContentRange(s string) (first, last, total int64, err error) {
	bad := func() (int64, int64, int64, error) {
		return 0, 0, 0, fmt.Errorf("malformed Content-Range %q", s)
	}
	rest, ok := strings.CutPrefix(s, "bytes ")
	if !ok {
		return bad()
	}
	rng, tot, ok := strings.Cut(rest, "/")
	if !ok {
		return bad()
	}
	total = -1
	if tot != "*" {
		if total, err = strconv.ParseInt(tot, 10, 64); err != nil || total < 0 {
			return bad()
		}
	}
	if rng == "*" {
		return -1, -1, total, nil
	}
	a, b, ok := strings.Cut(rng, "-")
	if !ok {
		return bad()
	}
	if first, err = strconv.ParseInt(a, 10, 64); err != nil || first < 0 {
		return bad()
	}
	if last, err = strconv.ParseInt(b, 10, 64); err != nil || last < first {
		return bad()
	}
	return first, last, total, nil
}

// redact returns u without user information, query string or fragment.
func redact(u *url.URL) string {
	c := *u
	c.User = nil
	if c.RawQuery != "" || c.ForceQuery {
		c.RawQuery, c.ForceQuery = "...", false
	}
	c.Fragment, c.RawFragment = "", ""
	return c.String()
}
