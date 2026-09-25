package zarr

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/blockcache"
	"github.com/LukasSelin/strata/raster"
	zarrv3 "github.com/LukasSelin/zarr"
)

// The decoded-chunk cache of a source whose SourceOptions.CacheBytes is 0
// holds DefaultCacheRows rows of the array's chunks, but never less than
// DefaultCacheBytes nor more than MaxDefaultCacheBytes, as cog's does
// (benchmarks/cog/RESULTS.md, "The cache"). A row of chunks is what one
// band of tiles needs, so the cache scales with the raster's width. More
// workers, taller tiles or a wider raster than the cap allows need
// CacheBytes set.
const (
	DefaultCacheRows     = 8
	DefaultCacheBytes    = 64 << 20
	MaxDefaultCacheBytes = 1 << 30
)

// DefaultReadConcurrency is how many chunks one ReadWindow loads at once
// when SourceOptions.ReadConcurrency is 0.
const DefaultReadConcurrency = 8

// SourceOptions selects what a Source reads.
type SourceOptions struct {
	// Index fixes the dimensions before the last two: one index for each,
	// so its length is the array's dimensions less two. Nil reads a 2-D
	// array.
	Index []int
	// IgnoreFill reads every cell as valid, rather than those whose
	// element equals the array's fill_value as invalid.
	IgnoreFill bool
	// CacheBytes bounds the memory of decoded chunks the source keeps, so
	// chunks that several windows touch are decoded once: 0 means the
	// default (DefaultCacheRows rows of chunks, within DefaultCacheBytes
	// and MaxDefaultCacheBytes), a negative value no cache. The cache
	// always holds at least the most recent chunk.
	CacheBytes int64
	// ReadConcurrency is how many chunks one ReadWindow loads at once: 0
	// means DefaultReadConcurrency, 1 one at a time.
	ReadConcurrency int
}

// Source is an engine.RasterSource reading a 2-D slice of a Zarr array.
// It is safe for concurrent ReadWindow calls.
type Source struct {
	a      *zarrv3.Array
	w, h   int // the raster: the array's last two dimensions
	cw, ch int // the chunk's
	// lead is the index of the chunk that holds Index along the leading
	// dimensions, and plane the offset of Index's y-x plane within it.
	lead  []int
	plane int
	fill  bool // the fill value is NoData
	conc  int
	load  func(ctx context.Context, idx []int) (*chunk, error)
	cache *blockcache.Cache[[2]int, *chunk]
	geo   georef
}

var _ engine.RasterSource = (*Source)(nil)

// chunk is a decoded chunk's y-x plane: the cells of every row and column
// of the chunk's shape, the part past the array's end included, and
// their validity, nil where every cell is valid.
type chunk struct {
	vals  []float32
	valid []uint64
}

func (c *chunk) size() int64 { return 4*int64(len(c.vals)) + 8*int64(len(c.valid)) + 64 }

// Open opens the array at path in s and returns a source over it; see
// NewSource.
func Open(ctx context.Context, s zarrv3.Store, path string, opts SourceOptions) (*Source, error) {
	a, err := zarrv3.OpenArray(ctx, s, path)
	if err != nil {
		return nil, err
	}
	return NewSource(a, opts)
}

// NewSource returns a source reading the y-x plane of a that opts select.
// It reads nothing from the store. It returns an error for an array of
// fewer than two dimensions, an empty one, a data type it does not read
// (bool), an Index that does not fit the array, and georeferencing
// attributes that are malformed or rotated.
func NewSource(a *zarrv3.Array, opts SourceOptions) (*Source, error) {
	shape, chunks := a.Shape(), a.ChunkShape()
	n := len(shape)
	if n < 2 {
		return nil, fmt.Errorf("zarr: array %q has %d dimensions; a raster needs 2 or more", a.Path(), n)
	}
	if len(opts.Index) != n-2 {
		return nil, fmt.Errorf("zarr: array %q has %d dimensions, so SourceOptions.Index needs %d indices, not %d",
			a.Path(), n, n-2, len(opts.Index))
	}
	s := &Source{
		a: a, h: shape[n-2], w: shape[n-1], ch: chunks[n-2], cw: chunks[n-1],
		lead: make([]int, n-2), fill: !opts.IgnoreFill, conc: opts.ReadConcurrency,
	}
	if s.w == 0 || s.h == 0 {
		return nil, fmt.Errorf("zarr: array %q is empty: shape %v", a.Path(), shape)
	}
	for k, i := range opts.Index {
		if i < 0 || i >= shape[k] {
			return nil, fmt.Errorf("zarr: array %q: index %d of dimension %d, of size %d", a.Path(), i, k, shape[k])
		}
		s.lead[k] = i / chunks[k]
		s.plane = s.plane*chunks[k] + i%chunks[k]
	}
	s.plane *= s.ch * s.cw
	if s.conc <= 0 {
		s.conc = DefaultReadConcurrency
	}
	var err error
	if s.load, err = loader(a, s.plane, s.ch*s.cw, s.fill); err != nil {
		return nil, err
	}
	if s.geo, err = readGeoref(a); err != nil {
		return nil, fmt.Errorf("zarr: array %q: %w", a.Path(), err)
	}
	size := (*chunk).size
	switch {
	case opts.CacheBytes == 0:
		s.cache = blockcache.New[[2]int](defaultCacheBytes(s.w, s.cw, s.ch), size)
	case opts.CacheBytes > 0:
		s.cache = blockcache.New[[2]int](opts.CacheBytes, size)
	}
	return s, nil
}

// defaultCacheBytes is the cache of a source whose CacheBytes is 0:
// DefaultCacheRows rows of decoded chunks, as chunk.size counts them,
// clamped to [DefaultCacheBytes, MaxDefaultCacheBytes].
func defaultCacheBytes(width, cw, ch int) int64 {
	cells := int64(cw) * int64(ch)
	perChunk := 4*cells + 8*int64(raster.MaskWords(int(cells))) + 64
	across := int64((width + cw - 1) / cw)
	if across > MaxDefaultCacheBytes/(DefaultCacheRows*perChunk) {
		return MaxDefaultCacheBytes // and no overflow on the way
	}
	return min(max(DefaultCacheRows*across*perChunk, DefaultCacheBytes), MaxDefaultCacheBytes)
}

// Array returns the array the source reads.
func (s *Source) Array() *zarrv3.Array { return s.a }

// Size returns the width and height of the plane, the array's last two
// dimensions.
func (s *Source) Size() (width, height int) { return s.w, s.h }

// Masked reports whether some cells may be invalid: whether the fill
// value is NoData, which it is unless SourceOptions.IgnoreFill.
func (s *Source) Masked() bool { return s.fill }

// Georeferenced reports whether the array has a spatial:transform.
func (s *Source) Georeferenced() bool { return s.geo.ok }

// Grid returns the plane's grid, from the array's spatial:transform and
// proj:code attributes; see the package documentation. Without a
// transform it is the grid of the array's own cells.
func (s *Source) Grid() raster.Grid {
	g := raster.Grid{Width: s.w, Height: s.h, ResolutionX: 1, ResolutionY: 1, CRS: raster.CRS{Code: s.geo.crs}}
	if s.geo.ok {
		g.OriginX, g.OriginY = s.geo.originX, s.geo.originY
		g.ResolutionX, g.ResolutionY = s.geo.resX, s.geo.resY
	}
	return g
}

// CacheBytes returns the bound on the source's decoded-chunk cache: the
// one SourceOptions gave, or the default it chose, or 0 for no cache.
func (s *Source) CacheBytes() int64 {
	if s.cache == nil {
		return 0
	}
	return s.cache.Limit()
}

// CacheStats counts what the decoded-chunk cache has done.
type CacheStats struct {
	// Hits is the chunks a read took from the cache, and Shared those it
	// took from another read that was loading them at the time.
	Hits, Shared int64
	// Loads is the chunks read and decoded from the store, and Failed how
	// many of those failed.
	Loads, Failed int64
	// Evictions is the chunks dropped to stay within CacheBytes.
	Evictions int64
	// Chunks and Bytes are what the cache holds now.
	Chunks int
	Bytes  int64
}

// CacheStats returns the cache's counters, all 0 for a source without a
// cache.
func (s *Source) CacheStats() CacheStats {
	if s.cache == nil {
		return CacheStats{}
	}
	st := s.cache.Stats()
	return CacheStats{
		Hits: st.Hits, Shared: st.Shared, Loads: st.Loads, Failed: st.Failed,
		Evictions: st.Evictions, Chunks: st.Len, Bytes: st.Bytes,
	}
}

// ReadWindow reads the region at (x, y) into dst; see
// engine.RasterSource. It takes each chunk the region touches from the
// cache, or reads and decodes it, up to SourceOptions.ReadConcurrency at
// once.
// It returns ctx.Err() if ctx is done before every chunk is read, and
// errors from the store or the codecs wrapped with the chunk that failed.
func (s *Source) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	if err := dst.Validate(); err != nil {
		panic(fmt.Sprintf("zarr: Source.ReadWindow: dst: %v", err))
	}
	if x < 0 || y < 0 || x > s.w-dst.Width || y > s.h-dst.Height {
		panic(fmt.Sprintf("zarr: Source.ReadWindow: %d×%d region at (%d, %d) outside %d×%d raster",
			dst.Width, dst.Height, x, y, s.w, s.h))
	}
	if s.fill && dst.Valid == nil {
		panic("zarr: Source.ReadWindow: the source is Masked and dst has no validity mask")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Chunks the cache holds are copied here; the rest are loaded, and
	// then only are goroutines worth starting.
	var misses [][2]int
	for cy := y / s.ch; cy <= (y+dst.Height-1)/s.ch; cy++ {
		for cx := x / s.cw; cx <= (x+dst.Width-1)/s.cw; cx++ {
			c := [2]int{cy, cx}
			if s.cache != nil {
				if b, ok := s.cache.Peek(c); ok {
					s.copyChunk(dst, x, y, c, b, nil)
					continue
				}
			}
			misses = append(misses, c)
		}
	}
	if len(misses) <= 1 || s.conc == 1 {
		for _, c := range misses {
			if err := s.readChunk(ctx, dst, x, y, c, nil); err != nil {
				return err
			}
		}
		return nil
	}

	// Chunks share mask words where they meet, so validity is copied
	// under maskMu; their cells are disjoint and copied without it.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		maskMu sync.Mutex
		mu     sync.Mutex
		next   int
		first  error
		wg     sync.WaitGroup
	)
	for range min(len(misses), s.conc) {
		wg.Go(func() {
			for {
				mu.Lock()
				i := next
				next++
				stop := first != nil
				mu.Unlock()
				if i >= len(misses) || stop {
					return
				}
				if err := s.readChunk(ctx, dst, x, y, misses[i], &maskMu); err != nil {
					mu.Lock()
					if first == nil {
						first = err
					}
					mu.Unlock()
					cancel()
					return
				}
			}
		})
	}
	wg.Wait()
	return first
}

// readChunk loads the chunk at c, (row, column) in the chunk grid, and
// copies it into dst; see copyChunk.
func (s *Source) readChunk(ctx context.Context, dst raster.Float32Raster, x, y int, c [2]int, maskMu *sync.Mutex) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b, err := s.chunk(ctx, c)
	if err != nil {
		if ctx.Err() != nil && errors.Is(err, ctx.Err()) {
			return err
		}
		return fmt.Errorf("zarr: array %q, chunk (%d, %d) of the plane: %w", s.a.Path(), c[0], c[1], err)
	}
	s.copyChunk(dst, x, y, c, b, maskMu)
	return nil
}

// copyChunk copies the part of b, the chunk at c, that lies inside dst,
// whose top-left is (x, y). It takes maskMu, if not nil, around the
// validity it writes.
func (s *Source) copyChunk(dst raster.Float32Raster, x, y int, c [2]int, b *chunk, maskMu *sync.Mutex) {
	bx0, by0 := c[1]*s.cw, c[0]*s.ch
	cx0, cx1 := max(x, bx0), min(x+dst.Width, bx0+s.cw)
	cy0, cy1 := max(y, by0), min(y+dst.Height, by0+s.ch)
	n := cx1 - cx0
	for yy := cy0; yy < cy1; yy++ {
		src := (yy-by0)*s.cw + cx0 - bx0
		d := (yy-y)*dst.Stride + cx0 - x
		copy(dst.Data[d:d+n], b.vals[src:src+n])
	}
	if dst.Valid == nil {
		return
	}
	if maskMu != nil {
		maskMu.Lock()
		defer maskMu.Unlock()
	}
	for yy := cy0; yy < cy1; yy++ {
		src := (yy-by0)*s.cw + cx0 - bx0
		d := dst.ValidOffset + (yy-y)*dst.Stride + cx0 - x
		if b.valid == nil {
			raster.MaskFillRange(dst.Valid, d, n, true)
		} else {
			raster.MaskCopyRange(dst.Valid, d, b.valid, src, n)
		}
	}
}

// chunk returns the decoded chunk at c, through the cache.
func (s *Source) chunk(ctx context.Context, c [2]int) (*chunk, error) {
	load := func() (*chunk, error) {
		idx := append(append(make([]int, 0, len(s.lead)+2), s.lead...), c[0], c[1])
		return s.load(ctx, idx)
	}
	if s.cache == nil {
		return load()
	}
	return s.cache.Get(ctx, c, load)
}
