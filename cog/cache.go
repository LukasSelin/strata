package cog

import (
	"container/list"
	"context"
	"sync"
)

// cache holds values, least recently used first out, within a byte
// budget. A Source keeps one of decoded blocks: the engine reads a tile
// and its halo, so neighbouring tiles, and bands of rows that do not line
// up with the file's blocks, ask for the same block more than once;
// without the cache each asks decompresses it again. A File whose bands
// are interleaved keeps one of compressed blocks, which its sources share,
// so each block is fetched once for every band rather than once per band.
//
// Loads are shared: a value asked for while another reader is loading it
// waits for that load rather than starting its own. A failed load is not
// kept, so the next read tries again.
//
// Values may be counted references, so that a value's memory can be
// reused once nobody holds it: then hold takes a reference and drop
// gives one back. The cache holds one for as long as it keeps a value,
// and get takes one for its caller, who gives it back when done. Both
// happen under the cache's lock, so a value get returns cannot be
// evicted and reused before the caller has its reference.
type cache[K comparable, V any] struct {
	mu    sync.Mutex
	limit int64
	used  int64
	size  func(V) int64
	hold  func(V) // nil for values that are not counted
	drop  func(V)
	m     map[K]*centry[K, V]
	lru   list.List // of *centry, most recently used at the front
}

type centry[K comparable, V any] struct {
	key   K
	elem  *list.Element // nil while loading
	ready chan struct{} // closed once v or err is set
	v     V
	err   error
}

// newCache returns a cache holding values of at most limit bytes in
// total, as size counts them.
func newCache[K comparable, V any](limit int64, size func(V) int64) *cache[K, V] {
	return &cache[K, V]{limit: limit, size: size, m: map[K]*centry[K, V]{}}
}

// withHolds makes the cache count references to its values; see cache.
func (c *cache[K, V]) withHolds(hold, drop func(V)) *cache[K, V] {
	c.hold, c.drop = hold, drop
	return c
}

// get returns the value for key, from the cache or from load. It returns
// ctx.Err() if ctx is done while it waits for another reader's load. With
// holds, the caller owns one reference to the value, and load must return
// a value with one reference, which becomes the caller's.
func (c *cache[K, V]) get(ctx context.Context, key K, load func() (V, error)) (V, error) {
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
			c.mu.Unlock()
			return e.v, nil
		}
		c.mu.Unlock()
		select {
		case <-e.ready:
		case <-ctx.Done():
			var zero V
			return zero, ctx.Err()
		}
		if e.err != nil {
			var zero V
			return zero, e.err
		}
		if c.hold == nil {
			return e.v, nil
		}
		// A reference is only safe to take while the cache still holds
		// one: if the value was evicted since the load, it may be gone.
		c.mu.Lock()
		if c.m[key] == e && e.elem != nil {
			c.hold(e.v)
			c.mu.Unlock()
			return e.v, nil
		}
		c.mu.Unlock()
	}
	e := &centry[K, V]{key: key, ready: make(chan struct{})}
	c.m[key] = e
	c.mu.Unlock()

	v, err := load()

	c.mu.Lock()
	e.v, e.err = v, err
	if err != nil {
		delete(c.m, key)
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

// evict drops the least recently used values until the cache is within
// its limit, keeping at least the one just added.
func (c *cache[K, V]) evict() {
	for c.used > c.limit && c.lru.Len() > 1 {
		e := c.lru.Remove(c.lru.Back()).(*centry[K, V])
		delete(c.m, e.key)
		c.used -= c.size(e.v)
		if c.drop != nil {
			c.drop(e.v)
		}
	}
}
