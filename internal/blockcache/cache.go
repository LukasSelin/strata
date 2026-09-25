// Package blockcache holds decoded blocks of a format adapter, least
// recently used first out, within a byte budget, and shares the loads of
// blocks that several readers ask for at once.
//
// It serves the format adapters, which are modules of their own (DESIGN.md
// §34): cog keeps decoded GeoTIFF blocks in one, and zarr decoded chunks.
// Go's internal rule is by import path, so both may import it although
// they are not in this module; nothing else outside strata can.
package blockcache

import (
	"container/list"
	"context"
	"errors"
	"sync"
)

// Cache holds values, least recently used first out, within a byte
// budget. A source keeps one of decoded blocks: the engine reads a tile
// and its halo, so neighbouring tiles, and bands of rows that do not line
// up with the file's blocks, ask for the same block more than once;
// without the cache each asks decompresses it again.
//
// Loads are shared: a value asked for while another reader is loading it
// waits for that load rather than starting its own. A failed load is not
// kept, so the next read tries again.
//
// Values may be counted references, so that a value's memory can be
// reused once nobody holds it: then hold takes a reference and drop
// gives one back. The cache holds one for as long as it keeps a value,
// and Get takes one for its caller, who gives it back when done. Both
// happen under the cache's lock, so a value Get returns cannot be
// evicted and reused before the caller has its reference.
//
// A Cache is safe for concurrent use.
type Cache[K comparable, V any] struct {
	mu    sync.Mutex
	limit int64
	used  int64
	size  func(V) int64
	hold  func(V) // nil for values that are not counted
	drop  func(V)
	m     map[K]*entry[K, V]
	lru   list.List // of *entry, most recently used at the front
	stats Stats
	// onWait, if set, is called as a reader starts to wait for another's
	// load; tests use it to know that one is waiting.
	onWait func()
}

type entry[K comparable, V any] struct {
	key   K
	elem  *list.Element // nil while loading
	ready chan struct{} // closed once v or err is set
	v     V
	err   error
}

// Stats counts what a Cache has done since it was made.
type Stats struct {
	// Hits is the Gets and Peeks answered with a value the cache held.
	Hits int64
	// Shared is the Gets that waited for another reader's load and took
	// its value.
	Shared int64
	// Loads is the Gets that called their load function, and Failed how
	// many of those loads returned an error.
	Loads, Failed int64
	// Evictions is the values dropped to keep within the limit.
	Evictions int64
	// Bytes and Len are what the cache holds now, as size counts it.
	Bytes int64
	Len   int
}

// New returns a cache holding values of at most limit bytes in total, as
// size counts them. It always keeps the value loaded last, however large.
func New[K comparable, V any](limit int64, size func(V) int64) *Cache[K, V] {
	return &Cache[K, V]{limit: limit, size: size, m: map[K]*entry[K, V]{}}
}

// WithHolds makes the cache count references to its values; see Cache.
func (c *Cache[K, V]) WithHolds(hold, drop func(V)) *Cache[K, V] {
	c.hold, c.drop = hold, drop
	return c
}

// Limit returns the byte budget the cache was made with.
func (c *Cache[K, V]) Limit() int64 { return c.limit }

// Stats returns the cache's counters.
func (c *Cache[K, V]) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := c.stats
	s.Bytes, s.Len = c.used, c.lru.Len()
	return s
}

// Get returns the value for key, from the cache or from load. It returns
// ctx.Err() if ctx is done while it waits for another reader's load, and
// that load's error if it fails, unless the error is the other reader's
// context ending: then, with ctx not done, it loads the value itself. With
// holds, the caller owns one reference to the value, and load must return
// a value with one reference, which becomes the caller's.
func (c *Cache[K, V]) Get(ctx context.Context, key K, load func() (V, error)) (V, error) {
	var zero V
	for {
		c.mu.Lock()
		e, ok := c.m[key]
		if !ok {
			break // with the lock held: load it
		}
		if e.elem != nil {
			c.lru.MoveToFront(e.elem)
			if c.hold != nil {
				c.hold(e.v)
			}
			c.stats.Hits++
			c.mu.Unlock()
			return e.v, nil
		}
		c.mu.Unlock()
		if c.onWait != nil {
			c.onWait()
		}
		select {
		case <-e.ready:
		case <-ctx.Done():
			return zero, ctx.Err()
		}
		if e.err != nil {
			if isContextErr(e.err) && ctx.Err() == nil {
				continue // the loader gave up, and this reader has not
			}
			return zero, e.err
		}
		c.mu.Lock()
		if c.hold == nil {
			c.stats.Shared++
			c.mu.Unlock()
			return e.v, nil
		}
		// A reference is only safe to take while the cache still holds
		// one: if the value was evicted since the load, it may be gone.
		if c.m[key] == e && e.elem != nil {
			c.hold(e.v)
			c.stats.Shared++
			c.mu.Unlock()
			return e.v, nil
		}
		c.mu.Unlock()
	}
	e := &entry[K, V]{key: key, ready: make(chan struct{})}
	c.m[key] = e
	c.stats.Loads++
	c.mu.Unlock()

	v, err := load()

	c.mu.Lock()
	e.v, e.err = v, err
	if err != nil {
		delete(c.m, key)
		c.stats.Failed++
	} else {
		if c.hold != nil {
			c.hold(v) // the cache's own reference
		}
		e.elem = c.lru.PushFront(e)
		c.used += c.size(v)
		c.evict()
	}
	c.mu.Unlock()
	close(e.ready)
	return v, err
}

// Peek returns the value for key if the cache holds it, as Get would, and
// false otherwise: while it is loading too. It never waits or loads.
func (c *Cache[K, V]) Peek(key K) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.m[key]
	if !ok || e.elem == nil {
		var zero V
		return zero, false
	}
	c.lru.MoveToFront(e.elem)
	if c.hold != nil {
		c.hold(e.v)
	}
	c.stats.Hits++
	return e.v, true
}

func isContextErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// evict drops the least recently used values until the cache is within
// its limit, keeping at least the one just added.
func (c *Cache[K, V]) evict() {
	for c.used > c.limit && c.lru.Len() > 1 {
		e := c.lru.Remove(c.lru.Back()).(*entry[K, V])
		delete(c.m, e.key)
		c.used -= c.size(e.v)
		c.stats.Evictions++
		if c.drop != nil {
			c.drop(e.v)
		}
	}
}
