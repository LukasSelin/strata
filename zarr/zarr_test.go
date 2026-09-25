package zarr

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/rand/v2"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/LukasSelin/strata/raster"
	zarrv3 "github.com/LukasSelin/zarr"
)

// newArray creates an array in a memory store and writes data, the whole
// of it, in C order.
func newArray[T number](t testing.TB, store zarrv3.Store, o zarrv3.ArrayOptions, data []T) *zarrv3.Array {
	t.Helper()
	ctx := context.Background()
	o.DataType = zarrv3.DataTypeOf[T]()
	a, err := zarrv3.CreateArray(ctx, store, "a", o)
	if err != nil {
		t.Fatal(err)
	}
	if err := zarrv3.Write(ctx, a, make([]int, len(o.Shape)), o.Shape, data); err != nil {
		t.Fatal(err)
	}
	return a
}

// want is the raster a source over the y-x plane at index of a should
// read, from the library's own whole-array Read: each element as float32,
// and valid unless it equals fill (any NaN for a NaN fill), if masked.
func want[T number](t testing.TB, a *zarrv3.Array, index []int, masked bool) raster.Float32Raster {
	t.Helper()
	shape := a.Shape()
	n := len(shape)
	start := append(append([]int{}, index...), 0, 0)
	count := append(make([]int, 0, n), shape...)
	for k := range index {
		count[k] = 1
	}
	data, err := zarrv3.Read[T](context.Background(), a, start, count)
	if err != nil {
		t.Fatal(err)
	}
	fill := a.FillValue().(T)
	r := raster.NewFloat32(shape[n-1], shape[n-2], make([]float32, len(data)))
	if masked {
		r.Valid = raster.NewMask(len(data))
	}
	for i, v := range data {
		r.Data[i] = exact(v)
		if masked && (v == fill || math.IsNaN(float64(fill)) && math.IsNaN(float64(v))) {
			raster.MaskSet(r.Valid, i, false)
		}
	}
	return r
}

// exact rounds v to float32 through math/big, once, to nearest even and
// to ±Inf past float32's range; float32s are kept bit for bit.
func exact[T number](v T) float32 {
	var f big.Float
	switch x := any(v).(type) {
	case float32:
		return x
	case float64:
		f.SetFloat64(x)
	case uint64:
		f.SetUint64(x)
	case uint8, uint16, uint32:
		f.SetUint64(reflect.ValueOf(x).Uint())
	default:
		f.SetInt64(reflect.ValueOf(x).Int())
	}
	r, _ := f.Float32()
	return r
}

// window returns a masked w×h window inside a larger raster whose every
// cell and bit is garbage, so that a read writing outside its cells, or
// leaving some unwritten, shows; and the larger raster.
func window(rng *rand.Rand, w, h int) (inner, outer raster.Float32Raster) {
	pad := rng.IntN(3)
	stride := w + 2*pad + rng.IntN(70)
	oh := h + 2*pad
	outer = raster.NewFloat32Stride(stride, oh, stride, make([]float32, stride*oh))
	outer.Valid = make([]uint64, raster.MaskWords(stride*oh))
	for i := range outer.Data {
		outer.Data[i] = -12345
	}
	for i := range outer.Valid {
		outer.Valid[i] = rng.Uint64()
	}
	return outer.Window(pad, pad, w, h), outer
}

// check reads the window at (x, y) of size w×h from src and compares it,
// cells and validity, with the same window of ref.
func check(t *testing.T, name string, src *Source, ref raster.Float32Raster, rng *rand.Rand, x, y, w, h int) {
	t.Helper()
	dst, outer := window(rng, w, h)
	before := append([]uint64{}, outer.Valid...)
	if err := src.ReadWindow(context.Background(), dst, x, y); err != nil {
		t.Fatalf("%s: window %d×%d at (%d, %d): %v", name, w, h, x, y, err)
	}
	for yy := range h {
		for xx := range w {
			i := ref.Index(x+xx, y+yy)
			valid := ref.Valid == nil || raster.MaskGet(ref.Valid, i)
			if got := dst.IsValid(xx, yy); got != valid {
				t.Fatalf("%s: window %d×%d at (%d, %d): cell (%d, %d) valid %v, want %v",
					name, w, h, x, y, x+xx, y+yy, got, valid)
			}
			g, e := dst.Data[dst.Index(xx, yy)], ref.Data[i]
			if valid && math.Float32bits(g) != math.Float32bits(e) {
				t.Fatalf("%s: window %d×%d at (%d, %d): cell (%d, %d) = %v, want %v",
					name, w, h, x, y, x+xx, y+yy, g, e)
			}
		}
	}
	// Nothing outside the window's cells changed.
	pad := dst.ValidOffset / outer.Stride
	for yy := range outer.Height {
		for xx := range outer.Width {
			if xx >= pad && xx < pad+w && yy >= pad && yy < pad+h {
				continue
			}
			i := outer.Index(xx, yy)
			if outer.Data[i] != -12345 || raster.MaskGet(outer.Valid, i) != raster.MaskGet(before, i) {
				t.Fatalf("%s: window %d×%d at (%d, %d): cell (%d, %d) of the raster around it changed",
					name, w, h, x, y, xx, yy)
			}
		}
	}
}

// windows checks corners, edges and random windows of src against ref.
func windows(t *testing.T, name string, src *Source, ref raster.Float32Raster, rng *rand.Rand) {
	t.Helper()
	w, h := src.Size()
	ch, cw := src.ch, src.cw
	fixed := [][4]int{
		{0, 0, w, h},                             // everything
		{0, 0, 1, 1},                             // one cell
		{cw - 1, ch - 1, 3, 3},                   // across a chunk corner
		{cw - 2, ch - 2, cw + 4, ch + 4},         // across four chunks and more
		{w - 1, h - 1, 1, 1},                     // the last cell, in a partial chunk
		{w - cw - 3, h - ch - 3, cw + 3, ch + 3}, // the partial corner chunk and its neighbours
		{0, h - 2, w, 2},                         // a strip along the bottom edge
		{w - 2, 0, 2, h},                         // and the right one
	}
	for _, f := range fixed {
		x, y, ww, hh := max(f[0], 0), max(f[1], 0), f[2], f[3]
		ww, hh = min(ww, w-x), min(hh, h-y)
		check(t, name, src, ref, rng, x, y, ww, hh)
	}
	for range 40 {
		ww, hh := 1+rng.IntN(w), 1+rng.IntN(h)
		check(t, name, src, ref, rng, rng.IntN(w-ww+1), rng.IntN(h-hh+1), ww, hh)
	}
}

// fillSome returns n random elements, about a tenth of them fill.
func fillSome[T number](rng *rand.Rand, n int, fill T, gen func() T) []T {
	d := make([]T, n)
	for i := range d {
		if rng.IntN(10) == 0 {
			d[i] = fill
		} else {
			d[i] = gen()
		}
	}
	return d
}

func testType[T number](t *testing.T, fill T, gen func(*rand.Rand) T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const w, h = 53, 37 // not a multiple of either chunk side
	data := fillSome(rng, w*h, fill, func() T { return gen(rng) })
	codecs := []zarrv3.Codec{zarrv3.BytesCodec{Endian: zarrv3.Little}, zarrv3.GzipCodec{Level: 5}}
	a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
		Shape: []int{h, w}, ChunkShape: []int{8, 16}, FillValue: fill, Codecs: codecs,
	}, data)
	ref := want[T](t, a, nil, true)
	for _, o := range []SourceOptions{
		{},
		{CacheBytes: -1},
		{CacheBytes: 1, ReadConcurrency: 1},
		{CacheBytes: 3000, ReadConcurrency: 3},
	} {
		src, err := NewSource(a, o)
		if err != nil {
			t.Fatal(err)
		}
		if !src.Masked() {
			t.Fatal("a source with a fill value is not Masked")
		}
		windows(t, fmt.Sprintf("%s %+v", a.DataType(), o), src, ref, rng)
	}
	src, err := NewSource(a, SourceOptions{IgnoreFill: true})
	if err != nil {
		t.Fatal(err)
	}
	if src.Masked() {
		t.Fatal("IgnoreFill: the source is Masked")
	}
	windows(t, fmt.Sprintf("%s IgnoreFill", a.DataType()), src, want[T](t, a, nil, false), rng)
}

// TestTypes checks windows of every element type against the library's
// whole-array Read, across chunk corners and at the partial chunks of the
// array's edge.
func TestTypes(t *testing.T) {
	t.Run("int8", func(t *testing.T) {
		testType(t, int8(-128), func(r *rand.Rand) int8 { return int8(r.IntN(256) - 128) })
	})
	t.Run("int16", func(t *testing.T) {
		testType(t, int16(-9999), func(r *rand.Rand) int16 { return int16(r.IntN(65536) - 32768) })
	})
	t.Run("int32", func(t *testing.T) { testType(t, int32(0), func(r *rand.Rand) int32 { return r.Int32() - 1<<30 }) })
	t.Run("int64", func(t *testing.T) {
		testType(t, int64(1<<53+1), func(r *rand.Rand) int64 { return 1<<53 + r.Int64N(4) - 1 })
	})
	t.Run("uint8", func(t *testing.T) { testType(t, uint8(255), func(r *rand.Rand) uint8 { return uint8(r.UintN(256)) }) })
	t.Run("uint16", func(t *testing.T) {
		testType(t, uint16(65535), func(r *rand.Rand) uint16 { return uint16(r.UintN(65536)) })
	})
	t.Run("uint32", func(t *testing.T) { testType(t, uint32(7), func(r *rand.Rand) uint32 { return r.Uint32() }) })
	t.Run("uint64", func(t *testing.T) {
		testType(t, uint64(math.MaxUint64), func(r *rand.Rand) uint64 { return r.Uint64() })
	})
	t.Run("float32", func(t *testing.T) {
		testType(t, float32(-3.4e38), func(r *rand.Rand) float32 { return float32(r.NormFloat64() * 100) })
	})
	t.Run("float64", func(t *testing.T) {
		testType(t, 1e300, func(r *rand.Rand) float64 {
			if r.IntN(20) == 0 {
				return -1e300 // past float32's range: -Inf
			}
			return r.NormFloat64() * 1e6
		})
	})
}

// TestFillValues checks the masks fill values give: in the element type,
// before the conversion, and any NaN for a NaN fill.
func TestFillValues(t *testing.T) {
	ctx := context.Background()
	read := func(t *testing.T, a *zarrv3.Array) raster.Float32Raster {
		t.Helper()
		src, err := NewSource(a, SourceOptions{})
		if err != nil {
			t.Fatal(err)
		}
		w, h := src.Size()
		dst := raster.NewFloat32(w, h, make([]float32, w*h))
		dst.Valid = raster.NewMask(w * h)
		if err := src.ReadWindow(ctx, dst, 0, 0); err != nil {
			t.Fatal(err)
		}
		return dst
	}
	valid := func(r raster.Float32Raster) string {
		var b strings.Builder
		for i := range r.Width {
			if r.IsValid(i, 0) {
				b.WriteByte('1')
			} else {
				b.WriteByte('0')
			}
		}
		return b.String()
	}
	quiet, payload := math.Float32frombits(0x7fc00000), math.Float32frombits(0x7f800001)
	neg := math.Float32frombits(0xffc00123)
	t.Run("float32 NaN", func(t *testing.T) {
		a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
			Shape: []int{1, 6}, ChunkShape: []int{1, 4}, FillValue: math.NaN(),
		}, []float32{1, quiet, payload, neg, float32(math.Inf(1)), 0})
		if got := valid(read(t, a)); got != "100011" {
			t.Errorf("validity %s, want 100011", got)
		}
	})
	t.Run("float64 NaN", func(t *testing.T) {
		a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
			Shape: []int{1, 4}, ChunkShape: []int{1, 3}, FillValue: math.NaN(),
		}, []float64{math.Float64frombits(0x7ff0000000000001), 2, math.NaN(), math.Inf(-1)})
		if got := valid(read(t, a)); got != "0101" {
			t.Errorf("validity %s, want 0101", got)
		}
	})
	t.Run("zero matches minus zero", func(t *testing.T) {
		a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
			Shape: []int{1, 3}, ChunkShape: []int{1, 2},
		}, []float32{float32(math.Copysign(0, -1)), 0, 1})
		if got := valid(read(t, a)); got != "001" {
			t.Errorf("validity %s, want 001", got)
		}
	})
	t.Run("float64 fill inexact in float32", func(t *testing.T) {
		// 0.1 and its float32 neighbour convert to the same float32.
		near := math.Nextafter(0.1, 1)
		a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
			Shape: []int{1, 3}, ChunkShape: []int{1, 3}, FillValue: 0.1,
		}, []float64{0.1, near, 5})
		r := read(t, a)
		if got := valid(r); got != "011" {
			t.Errorf("validity %s, want 011", got)
		}
		if r.Data[1] != float32(0.1) {
			t.Errorf("cell 1 is %v, want float32(0.1)", r.Data[1])
		}
	})
	t.Run("int64 fill inexact in float32", func(t *testing.T) {
		a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
			Shape: []int{1, 3}, ChunkShape: []int{1, 2}, FillValue: int64(1<<40 + 1),
		}, []int64{1 << 40, 1<<40 + 1, 1<<40 + 2})
		if got := valid(read(t, a)); got != "101" {
			t.Errorf("validity %s, want 101", got)
		}
	})
	t.Run("chunks never written", func(t *testing.T) {
		store := zarrv3.NewMemoryStore()
		a, err := zarrv3.CreateArray(ctx, store, "a", zarrv3.ArrayOptions{
			Shape: []int{2, 5}, ChunkShape: []int{2, 2}, DataType: zarrv3.Uint16, FillValue: 9,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := zarrv3.WriteChunk(ctx, a, []int{0, 1}, []uint16{1, 2, 3, 9}); err != nil {
			t.Fatal(err)
		}
		if got := valid(read(t, a)); got != "00110" {
			t.Errorf("validity %s, want 00110", got)
		}
	})
}

// TestIndex checks every y-x plane of a 4-D (time, band, y, x) array,
// whose chunks span several time steps.
func TestIndex(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	shape := []int{3, 2, 21, 19}
	data := fillSome(rng, 3*2*21*19, float32(-1), func() float32 { return rng.Float32() })
	a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{
		Shape: shape, ChunkShape: []int{2, 1, 8, 5}, FillValue: -1,
		DimensionNames: []string{"time", "band", "y", "x"},
	}, data)
	for tm := range shape[0] {
		for b := range shape[1] {
			idx := []int{tm, b}
			src, err := NewSource(a, SourceOptions{Index: idx, CacheBytes: 2000})
			if err != nil {
				t.Fatal(err)
			}
			windows(t, fmt.Sprintf("index %v", idx), src, want[float32](t, a, idx, true), rng)
		}
	}
}

// rangeless hides a store's GetRange, so that a sharded array is read a
// shard at a time.
type rangeless struct{ zarrv3.Store }

// TestSharded checks sharded arrays, read with and without range reads.
func TestSharded(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const w, h = 45, 30
	data := fillSome(rng, w*h, int16(-1), func() int16 { return int16(rng.IntN(2000)) })
	for _, s := range []struct {
		name  string
		store zarrv3.Store
	}{{"ranges", zarrv3.NewMemoryStore()}, {"whole shards", rangeless{zarrv3.NewMemoryStore()}}} {
		a := newArray(t, s.store, zarrv3.ArrayOptions{
			Shape: []int{h, w}, ChunkShape: []int{4, 6}, ShardShape: []int{12, 18}, FillValue: -1,
			Codecs: []zarrv3.Codec{zarrv3.BytesCodec{Endian: zarrv3.Little}, zarrv3.GzipCodec{Level: 1}},
		}, data)
		src, err := NewSource(a, SourceOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if src.cw != 6 || src.ch != 4 {
			t.Fatalf("the source caches %d×%d chunks, want the inner 6×4", src.cw, src.ch)
		}
		windows(t, s.name, src, want[int16](t, a, nil, true), rng)
	}
}

// TestRefused checks the arrays and options NewSource refuses.
func TestRefused(t *testing.T) {
	ctx := context.Background()
	create := func(o zarrv3.ArrayOptions) *zarrv3.Array {
		a, err := zarrv3.CreateArray(ctx, zarrv3.NewMemoryStore(), "a", o)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	f32 := zarrv3.Float32
	for _, tc := range []struct {
		name string
		a    *zarrv3.Array
		o    SourceOptions
		want string
	}{
		{"1-D", create(zarrv3.ArrayOptions{Shape: []int{5}, ChunkShape: []int{5}, DataType: f32}), SourceOptions{}, "2 or more"},
		{"bool", create(zarrv3.ArrayOptions{Shape: []int{2, 2}, ChunkShape: []int{2, 2}, DataType: zarrv3.Bool}), SourceOptions{}, "bool is not supported"},
		{"empty", create(zarrv3.ArrayOptions{Shape: []int{0, 2}, ChunkShape: []int{2, 2}, DataType: f32}), SourceOptions{}, "empty"},
		{"no index", create(zarrv3.ArrayOptions{Shape: []int{2, 2, 2}, ChunkShape: []int{1, 2, 2}, DataType: f32}), SourceOptions{}, "needs 1 indices, not 0"},
		{"index on 2-D", create(zarrv3.ArrayOptions{Shape: []int{2, 2}, ChunkShape: []int{2, 2}, DataType: f32}), SourceOptions{Index: []int{0}}, "needs 0 indices"},
		{"index outside", create(zarrv3.ArrayOptions{Shape: []int{2, 2, 2}, ChunkShape: []int{1, 2, 2}, DataType: f32}), SourceOptions{Index: []int{2}}, "index 2 of dimension 0"},
	} {
		_, err := NewSource(tc.a, tc.o)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want one containing %q", tc.name, err, tc.want)
		}
	}
}

// TestGrid checks the grid from the georeferencing attributes.
func TestGrid(t *testing.T) {
	ctx := context.Background()
	open := func(attrs map[string]any) (*Source, error) {
		a, err := zarrv3.CreateArray(ctx, zarrv3.NewMemoryStore(), "a", zarrv3.ArrayOptions{
			Shape: []int{4, 6}, ChunkShape: []int{2, 2}, DataType: zarrv3.Float32, Attributes: attrs,
		})
		if err != nil {
			t.Fatal(err)
		}
		return NewSource(a, SourceOptions{})
	}
	transform := []float64{10, 0, 500000, 0, -10, 6600000}
	for _, tc := range []struct {
		name  string
		attrs map[string]any
		want  raster.Grid
		geo   bool
	}{
		{"none", nil, raster.Grid{Width: 6, Height: 4, ResolutionX: 1, ResolutionY: 1}, false},
		{"crs only", map[string]any{"proj:code": "EPSG:3006"},
			raster.Grid{Width: 6, Height: 4, ResolutionX: 1, ResolutionY: 1, CRS: raster.CRS{Code: "EPSG:3006"}}, false},
		{"pixel", map[string]any{"spatial:transform": transform, "proj:code": "EPSG:32633"},
			raster.Grid{Width: 6, Height: 4, ResolutionX: 10, ResolutionY: -10, OriginX: 500000, OriginY: 6600000, CRS: raster.CRS{Code: "EPSG:32633"}}, true},
		{"node", map[string]any{"spatial:transform": transform, "spatial:registration": "node"},
			raster.Grid{Width: 6, Height: 4, ResolutionX: 10, ResolutionY: -10, OriginX: 499995, OriginY: 6600005}, true},
	} {
		src, err := open(tc.attrs)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got := src.Grid(); got != tc.want {
			t.Errorf("%s: grid %+v, want %+v", tc.name, got, tc.want)
		}
		if src.Georeferenced() != tc.geo {
			t.Errorf("%s: Georeferenced %v, want %v", tc.name, src.Georeferenced(), tc.geo)
		}
	}
	for _, tc := range []struct {
		name  string
		attrs map[string]any
		want  string
	}{
		{"rotated", map[string]any{"spatial:transform": []float64{10, 1, 0, 0, -10, 0}}, "rotated"},
		{"short", map[string]any{"spatial:transform": []float64{10, 0, 0, 0, -10}}, "want 6"},
		{"not numbers", map[string]any{"spatial:transform": "10 0 0 0 -10 0"}, "spatial:transform"},
		{"zero resolution", map[string]any{"spatial:transform": []float64{0, 0, 0, 0, -10, 0}}, "resolution of 0"},
		{"registration", map[string]any{"spatial:registration": "corner"}, "want \"pixel\" or \"node\""},
		{"crs a number", map[string]any{"proj:code": 3006}, "proj:code"},
	} {
		if _, err := open(tc.attrs); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want one containing %q", tc.name, err, tc.want)
		}
	}
}

// TestPanics checks ReadWindow's programming errors.
func TestPanics(t *testing.T) {
	a := newArray(t, zarrv3.NewMemoryStore(), zarrv3.ArrayOptions{Shape: []int{4, 4}, ChunkShape: []int{2, 2}}, make([]float32, 16))
	src, err := NewSource(a, SourceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	masked := func(w, h int) raster.Float32Raster {
		r := raster.NewFloat32(w, h, make([]float32, w*h))
		r.Valid = raster.NewMask(w * h)
		return r
	}
	for _, tc := range []struct {
		name string
		dst  raster.Float32Raster
		x, y int
		want string
	}{
		{"outside", masked(2, 2), 3, 0, "outside 4×4 raster"},
		{"no mask", raster.NewFloat32(2, 2, make([]float32, 4)), 0, 0, "no validity mask"},
		{"invalid dst", raster.Float32Raster{Width: 2, Height: 2, Stride: 2}, 0, 0, "dst"},
	} {
		func() {
			defer func() {
				r := recover()
				if s, _ := r.(string); !strings.Contains(s, tc.want) {
					t.Errorf("%s: panic %v, want one containing %q", tc.name, r, tc.want)
				}
			}()
			_ = src.ReadWindow(context.Background(), tc.dst, tc.x, tc.y)
		}()
	}
}

// counting is a store that counts Get and GetRange calls, and fails or
// blocks them on request.
type counting struct {
	*zarrv3.MemoryStore
	mu    sync.Mutex
	gets  map[string]int
	fail  map[string]int // key: how many more calls fail
	block chan struct{}  // if not nil, chunk reads wait for it or ctx
	start chan struct{}  // if not nil, receives once per chunk read begun
}

var errInjected = errors.New("injected failure")

func newCounting() *counting {
	return &counting{MemoryStore: zarrv3.NewMemoryStore(), gets: map[string]int{}, fail: map[string]int{}}
}

func (c *counting) enter(ctx context.Context, key string) error {
	if strings.HasSuffix(key, "zarr.json") {
		return nil
	}
	c.mu.Lock()
	c.gets[key]++
	fail := c.fail[key] > 0
	if fail {
		c.fail[key]--
	}
	block, start := c.block, c.start
	c.mu.Unlock()
	if start != nil {
		start <- struct{}{}
	}
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if fail {
		return errInjected
	}
	return nil
}

func (c *counting) Get(ctx context.Context, key string) ([]byte, error) {
	if err := c.enter(ctx, key); err != nil {
		return nil, err
	}
	return c.MemoryStore.Get(ctx, key)
}

func (c *counting) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := c.enter(ctx, key); err != nil {
		return nil, err
	}
	return c.MemoryStore.GetRange(ctx, key, off, n)
}

func (c *counting) total() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, v := range c.gets {
		n += v
	}
	return n
}

// cacheArray is an 8×8 float32 array of 4×4 chunks, each chunk's cells
// its index in the chunk grid, row-major.
func cacheArray(t *testing.T, store zarrv3.Store) *zarrv3.Array {
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32((i/8/4)*2 + i%8/4)
	}
	return newArray(t, store, zarrv3.ArrayOptions{Shape: []int{8, 8}, ChunkShape: []int{4, 4}, FillValue: -1}, data)
}

func read(src *Source, x, y, w, h int) (raster.Float32Raster, error) {
	dst := raster.NewFloat32(w, h, make([]float32, w*h))
	dst.Valid = raster.NewMask(w * h)
	return dst, src.ReadWindow(context.Background(), dst, x, y)
}

// chunkBytes is what the cache counts for one of cacheArray's chunks.
const chunkBytes = 4*16 + 64

// TestCacheCounts checks hits, loads and eviction against the store's
// own count of reads.
func TestCacheCounts(t *testing.T) {
	store := newCounting()
	a := cacheArray(t, store)
	src, err := NewSource(a, SourceOptions{CacheBytes: 2 * chunkBytes, ReadConcurrency: 1})
	if err != nil {
		t.Fatal(err)
	}
	if src.CacheBytes() != 2*chunkBytes {
		t.Fatalf("CacheBytes %d, want %d", src.CacheBytes(), 2*chunkBytes)
	}
	steps := []struct {
		x, y, w, h int
		want       CacheStats
	}{
		{0, 0, 2, 2, CacheStats{Loads: 1, Chunks: 1, Bytes: chunkBytes}},                            // chunk 0
		{1, 1, 3, 3, CacheStats{Hits: 1, Loads: 1, Chunks: 1, Bytes: chunkBytes}},                   // chunk 0 again
		{3, 0, 2, 1, CacheStats{Hits: 2, Loads: 2, Chunks: 2, Bytes: 2 * chunkBytes}},               // 0 and 1
		{0, 4, 1, 1, CacheStats{Hits: 2, Loads: 3, Evictions: 1, Chunks: 2, Bytes: 2 * chunkBytes}}, // 2 evicts 0
		{0, 0, 1, 1, CacheStats{Hits: 2, Loads: 4, Evictions: 2, Chunks: 2, Bytes: 2 * chunkBytes}}, // 0 evicts 1
	}
	for i, s := range steps {
		r, err := read(src, s.x, s.y, s.w, s.h)
		if err != nil {
			t.Fatal(err)
		}
		if got := src.CacheStats(); got != s.want {
			t.Errorf("step %d: stats %+v, want %+v", i, got, s.want)
		}
		if r.Data[0] != float32((s.y/4)*2+s.x/4) {
			t.Errorf("step %d: cell %v", i, r.Data[0])
		}
	}
	if got := store.total(); got != 4 {
		t.Errorf("the store served %d chunk reads, want 4, one per load", got)
	}

	none, _ := NewSource(a, SourceOptions{CacheBytes: -1})
	if none.CacheBytes() != 0 || none.CacheStats() != (CacheStats{}) {
		t.Errorf("no cache: CacheBytes %d, stats %+v", none.CacheBytes(), none.CacheStats())
	}
	def, _ := NewSource(a, SourceOptions{})
	if def.CacheBytes() != DefaultCacheBytes {
		t.Errorf("default CacheBytes %d, want %d", def.CacheBytes(), DefaultCacheBytes)
	}
}

// TestDefaultCacheBytes checks the default cache: DefaultCacheRows rows
// of chunks, within its floor and cap.
func TestDefaultCacheBytes(t *testing.T) {
	row := func(width, cw, ch int) int64 {
		cells := int64(cw * ch)
		return int64((width+cw-1)/cw) * (4*cells + 8*int64(raster.MaskWords(int(cells))) + 64)
	}
	for _, tc := range []struct {
		name          string
		width, cw, ch int
		want          int64
	}{
		{"wide", 16384, 512, 512, DefaultCacheRows * row(16384, 512, 512)},
		{"narrow: the floor", 700, 256, 256, DefaultCacheBytes},
		{"very wide: the cap", 400000, 512, 512, MaxDefaultCacheBytes},
		{"absurdly wide: the cap, no overflow", 1 << 40, 16384, 16384, MaxDefaultCacheBytes},
	} {
		if got := defaultCacheBytes(tc.width, tc.cw, tc.ch); got != tc.want {
			t.Errorf("%s: %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestSharedLoad checks that readers asking for a chunk while another is
// loading it wait for that load; run it with -race.
func TestSharedLoad(t *testing.T) {
	store := newCounting()
	a := cacheArray(t, store)
	src, err := NewSource(a, SourceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	store.block, store.start = make(chan struct{}), make(chan struct{}, 1)
	const readers = 12
	var wg sync.WaitGroup
	errs := make([]error, readers)
	wg.Go(func() { _, errs[0] = read(src, 0, 0, 3, 3) })
	<-store.start // the first reader is loading chunk 0
	for i := 1; i < readers; i++ {
		wg.Go(func() { _, errs[i] = read(src, i%3, i%2, 1, 1) })
	}
	close(store.block)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("reader %d: %v", i, err)
		}
	}
	st := src.CacheStats()
	if st.Loads != 1 || st.Hits+st.Shared != readers-1 || store.total() != 1 {
		t.Errorf("stats %+v and %d store reads; want 1 load, %d served from it", st, store.total(), readers-1)
	}
}

// TestFailedLoadRetried checks that a chunk whose read failed is not
// kept: the error names the chunk, and the next read tries again.
func TestFailedLoadRetried(t *testing.T) {
	store := newCounting()
	a := cacheArray(t, store)
	src, err := NewSource(a, SourceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	store.fail[a.ChunkKey([]int{1, 1})] = 1
	_, err = read(src, 2, 2, 4, 4) // four chunks, one failing
	if !errors.Is(err, errInjected) || !strings.Contains(err.Error(), "chunk (1, 1)") {
		t.Fatalf("error %v, want the injected one, naming chunk (1, 1)", err)
	}
	r, err := read(src, 4, 4, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Data[0] != 3 {
		t.Errorf("cell %v, want 3", r.Data[0])
	}
	// The failure cancels the read's other loads, which may fail too.
	if st := src.CacheStats(); st.Failed < 1 || st.Chunks < 1 {
		t.Errorf("stats %+v, want a failed load and the chunk kept since", st)
	}
}

// TestCancel checks that a read returns ctx.Err() once ctx is done,
// before or during a load, that the load it abandoned is not kept, and
// that another reader waiting for that load loads the chunk itself.
func TestCancel(t *testing.T) {
	store := newCounting()
	a := cacheArray(t, store)
	src, err := NewSource(a, SourceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	dst := raster.NewFloat32(8, 8, make([]float32, 64))
	dst.Valid = raster.NewMask(64)

	done, cancel := context.WithCancel(context.Background())
	cancel()
	if err := src.ReadWindow(done, dst, 0, 0); !errors.Is(err, context.Canceled) {
		t.Errorf("a done context: %v, want context.Canceled", err)
	}
	if store.total() != 0 {
		t.Errorf("a done context read %d chunks", store.total())
	}

	store.block, store.start = make(chan struct{}), make(chan struct{}, 8)
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() { first <- src.ReadWindow(ctx, dst, 0, 0) }()
	for range 4 {
		<-store.start // every chunk is loading
	}
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled while loading: %v, want context.Canceled", err)
	}
	if st := src.CacheStats(); st.Chunks != 0 {
		t.Errorf("abandoned loads were kept: %+v", st)
	}

	// The next read, with a live context, loads them again.
	store.mu.Lock()
	store.block, store.start = nil, nil
	store.mu.Unlock()
	r, err := read(src, 3, 3, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if r.Data[3] != 3 {
		t.Errorf("cell (4, 4) is %v, want 3", r.Data[3])
	}
}
