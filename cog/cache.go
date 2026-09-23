package cog

import (
	"container/list"
	"context"
	"sync"
)

// cache holds decoded blocks, least recently used first out, within a
// byte budget. The engine reads a tile and its halo, so neighbouring
// tiles, and bands of rows that do not line up with the file's blocks,
// ask for the same block more than once; without the cache each asks
// decompresses it again.
//
// Loads are shared: a block asked for while another reader is decoding
// it waits for that decode rather than starting its own. A failed load
// is not kept, so the next read tries again.
type cache struct {
	mu    sync.Mutex
	limit int64
	used  int64
	m     map[int]*centry
	lru   list.List // of *centry, most recently used at the front
}

type centry struct {
	key   int
	elem  *list.Element // nil while loading
	ready chan struct{} // closed once b or err is set
	b     *block
	err   error
}

func newCache(limit int64) *cache {
	return &cache{limit: limit, m: map[int]*centry{}}
}

// get returns block key, from the cache or from load. It returns
// ctx.Err() if ctx is done while it waits for another reader's load.
func (c *cache) get(ctx context.Context, key int, load func() (*block, error)) (*block, error) {
	c.mu.Lock()
	if e, ok := c.m[key]; ok {
		if e.elem != nil {
			c.lru.MoveToFront(e.elem)
			c.mu.Unlock()
			return e.b, nil
		}
		c.mu.Unlock()
		select {
		case <-e.ready:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return e.b, e.err
	}
	e := &centry{key: key, ready: make(chan struct{})}
	c.m[key] = e
	c.mu.Unlock()

	b, err := load()

	c.mu.Lock()
	e.b, e.err = b, err
	if err != nil {
		delete(c.m, key)
	} else {
		e.elem = c.lru.PushFront(e)
		c.used += b.size()
		c.evict()
	}
	c.mu.Unlock()
	close(e.ready)
	return b, err
}

// evict drops the least recently used blocks until the cache is within
// its limit, keeping at least the one just added.
func (c *cache) evict() {
	for c.used > c.limit && c.lru.Len() > 1 {
		e := c.lru.Remove(c.lru.Back()).(*centry)
		delete(c.m, e.key)
		c.used -= e.b.size()
	}
}
