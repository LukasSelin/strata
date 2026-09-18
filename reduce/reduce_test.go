package reduce_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

var (
	nan      = float32(math.NaN())
	otherNaN = math.Float32frombits(0xffc0_0001)
	posInf   = float32(math.Inf(1))
	negInf   = float32(math.Inf(-1))
	negZero  = float32(math.Copysign(0, -1))
)

// ref is the reduction written the obvious way, cell by cell, with no
// engine and no vector kernel.
func ref(r raster.Float32Raster) (mn, mx float32, count int64) {
	mn, mx = posInf, negInf
	for y := range r.Height {
		for x := range r.Width {
			if !r.IsValid(x, y) {
				continue
			}
			v := r.Data[r.Index(x, y)]
			mn, mx = min(mn, v), max(mx, v)
			count++
		}
	}
	if count == 0 {
		return nan, nan, 0
	}
	return mn, mx, count
}

// same compares two results the way the package promises them: bit for
// bit, so a lost -0 fails, except that any NaN matches any NaN — the
// canonical-NaN rule is checked on its own in TestNaNIsCanonical.
func same(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// TestValues covers the DESIGN.md §39 table over every execution path:
// ordinary values, NaN, infinities, negatives, -0 and +0, NoData, odd
// dimensions, non-contiguous strides, vector tails and tiny arrays.
func TestValues(t *testing.T) {
	cases := []struct {
		name   string
		data   []float32
		valid  []bool // nil: no mask
		mn, mx float32
		count  int64
	}{
		{name: "one cell", data: []float32{7}, mn: 7, mx: 7, count: 1},
		{name: "ordinary", data: []float32{3, 1, 4, 1, 5, 9, 2, 6}, mn: 1, mx: 9, count: 8},
		{name: "negatives", data: []float32{-3, -1, -4}, mn: -4, mx: -1, count: 3},
		{
			name: "infinities", data: []float32{posInf, 1, negInf},
			mn: negInf, mx: posInf, count: 3,
		},
		{
			// A NaN in a valid cell is a value, not NoData, so it
			// absorbs both extremes.
			name: "nan absorbs", data: []float32{1, nan, 3},
			mn: nan, mx: nan, count: 3,
		},
		{
			name: "nan under an invalid cell is never read",
			data: []float32{1, nan, 3}, valid: []bool{true, false, true},
			mn: 1, mx: 3, count: 2,
		},
		{
			// Go's builtins order -0 below +0, so both survive.
			name: "signed zeros", data: []float32{0, negZero},
			mn: negZero, mx: 0, count: 2,
		},
		{
			name: "signed zeros reversed", data: []float32{negZero, 0},
			mn: negZero, mx: 0, count: 2,
		},
		{
			name: "all invalid", data: []float32{1, 2, 3}, valid: []bool{false, false, false},
			mn: nan, mx: nan, count: 0,
		},
		{
			name: "one valid", data: []float32{1, 2, 3}, valid: []bool{false, true, false},
			mn: 2, mx: 2, count: 1,
		},
		{
			name: "extremes", data: []float32{math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32},
			mn: -math.MaxFloat32, mx: math.MaxFloat32, count: 3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Every length around the 8-lane vector boundary, so the
			// tail of a SIMD kernel is exercised by every case.
			for _, w := range []int{len(tc.data), 1, 7, 8, 9, 16, 17, 64, 65} {
				if w < len(tc.data) {
					continue
				}
				r := layOut(t, tc.data, tc.valid, w)
				checkAll(t, fmt.Sprintf("%s w=%d", tc.name, w), r, tc.mn, tc.mx, tc.count)
			}
		})
	}
}

// layOut lays data out as a w-wide raster, repeating it until the rows
// are full, in a window of a larger root whose Stride exceeds Width and
// is not a multiple of 64, with an odd mask offset. So every case is also
// a non-contiguous-stride case.
func layOut(t *testing.T, data []float32, valid []bool, w int) raster.Float32Raster {
	t.Helper()
	h := (len(data) + w - 1) / w
	rootW, rootH := w+3, h+2
	stride := rootW + 37
	if stride%64 == 0 {
		stride++
	}
	n := (rootH-1)*stride + rootW
	root := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
	for i := range root.Data {
		root.Data[i] = 12345 // stale values outside the window
	}
	masked := valid != nil
	if masked {
		root.ValidOffset = 3
		root.Valid = make([]uint64, raster.MaskWords(root.ValidOffset+n)+1)
		for k := range root.Valid {
			root.Valid[k] = ^uint64(0) // stale bits outside the window
		}
	}
	win := root.Window(2, 1, w, h)
	for i := range w * h {
		x, y := i%w, i/w
		// Repeat the case's cells to fill the rectangle; a repeat cannot
		// change a min, a max, or which cells are valid.
		j := i % len(data)
		win.Data[win.Index(x, y)] = data[j]
		if masked {
			win.SetValid(x, y, valid[j])
		}
	}
	return win
}

// checkAll runs every execution path and checks they agree with each
// other, with the naive reference, and with the case's expectation.
func checkAll(t *testing.T, id string, r raster.Float32Raster, mn, mx float32, count int64) {
	t.Helper()
	// The rectangle repeats the case's cells to fill its rows, which
	// cannot change a minimum or a maximum but does multiply the count.
	// So the case pins the extremes and the naive reference pins the
	// count; they must agree before anything else is compared.
	rmn, rmx, want := ref(r)
	if !same(rmn, mn) || !same(rmx, mx) {
		t.Fatalf("%s: the reference gives %v %v, want %v %v", id, rmn, rmx, mn, mx)
	}
	if (want == 0) != (count == 0) {
		t.Fatalf("%s: the reference counts %d valid cells, want a multiple of %d", id, want, count)
	}

	gotMn, gotMx, gotN := reduce.MinMax(r)
	if !same(gotMn, mn) || !same(gotMx, mx) || gotN != want {
		t.Fatalf("%s: MinMax = %v %v %d, want %v %v %d", id, gotMn, gotMx, gotN, mn, mx, want)
	}
	if c := reduce.Count(r); c != want {
		t.Fatalf("%s: Count = %d, want %d", id, c, want)
	}

	ctx := context.Background()
	src := engine.NewMemorySource(r)
	for _, opts := range tilings() {
		mn2, mx2, n2, err := reduce.MinMaxTiled(ctx, r, opts)
		if err != nil {
			t.Fatalf("%s %+v: %v", id, opts, err)
		}
		if !same(mn2, mn) || !same(mx2, mx) || n2 != want {
			t.Fatalf("%s tiled %+v: %v %v %d, want %v %v %d", id, opts, mn2, mx2, n2, mn, mx, want)
		}
		mn3, mx3, n3, err := reduce.MinMaxChunked(ctx, src, opts)
		if err != nil {
			t.Fatalf("%s %+v: %v", id, opts, err)
		}
		if !same(mn3, mn) || !same(mx3, mx) || n3 != want {
			t.Fatalf("%s chunked %+v: %v %v %d, want %v %v %d", id, opts, mn3, mx3, n3, mn, mx, want)
		}
		if n, err := reduce.CountTiled(ctx, r, opts); err != nil || n != want {
			t.Fatalf("%s CountTiled %+v: %d, %v, want %d", id, opts, n, err, want)
		}
		if n, err := reduce.CountChunked(ctx, src, opts); err != nil || n != want {
			t.Fatalf("%s CountChunked %+v: %d, %v, want %d", id, opts, n, err, want)
		}
	}
}

func tilings() []engine.Options {
	var out []engine.Options
	for _, tw := range []int{0, 1, 7, 64} {
		for _, th := range []int{0, 1, 5} {
			for _, wk := range []int{1, 3, runtime.GOMAXPROCS(0)} {
				out = append(out, engine.Options{TileWidth: tw, TileHeight: th, Workers: wk})
			}
		}
	}
	return out
}

// TestNaNIsCanonical is the rule the package documents and DESIGN.md §49
// does not state: Go's min and max do not say which NaN's payload
// survives, and a vector fold and a scalar one carry different ones out
// of a raster holding several, so the result would otherwise depend on
// the tiling, the worker count and the backend. A NaN result is therefore
// always the canonical quiet NaN.
func TestNaNIsCanonical(t *testing.T) {
	want := math.Float32bits(nan)
	data := []float32{1, otherNaN, 2, math.Float32frombits(0x7fc0_1234), 3, nan, 4}
	for _, w := range []int{1, 3, 7, 8, 9, 16} {
		if w > len(data) {
			continue
		}
		r := layOut(t, data, nil, w)
		mn, mx, _ := reduce.MinMax(r)
		if math.Float32bits(mn) != want || math.Float32bits(mx) != want {
			t.Fatalf("w=%d: MinMax = %#x %#x, want the canonical NaN %#x",
				w, math.Float32bits(mn), math.Float32bits(mx), want)
		}
		for _, opts := range tilings() {
			mn, mx, _, err := reduce.MinMaxTiled(context.Background(), r, opts)
			if err != nil {
				t.Fatal(err)
			}
			if math.Float32bits(mn) != want || math.Float32bits(mx) != want {
				t.Fatalf("w=%d %+v: MinMax = %#x %#x, want %#x",
					w, opts, math.Float32bits(mn), math.Float32bits(mx), want)
			}
		}
	}
}

// TestInvalidDataNeverRead is the §31 rule for a fold: Data under an
// invalid cell is unspecified, so overwriting every invalid cell with
// anything at all must not move the answer.
func TestInvalidDataNeverRead(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	w, h := 61, 37
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	for y := range h {
		for x := range w {
			r.Data[r.Index(x, y)] = float32(rng.NormFloat64() * 1000)
			r.SetValid(x, y, rng.IntN(3) != 0)
		}
	}
	mn, mx, n := reduce.MinMax(r)

	hazards := []float32{nan, posInf, negInf, math.MaxFloat32, -math.MaxFloat32, 0, negZero}
	for y := range h {
		for x := range w {
			if !r.IsValid(x, y) {
				r.Data[r.Index(x, y)] = hazards[rng.IntN(len(hazards))]
			}
		}
	}
	mn2, mx2, n2 := reduce.MinMax(r)
	if !same(mn, mn2) || !same(mx, mx2) || n != n2 {
		t.Fatalf("scrambling invalid cells changed %v %v %d to %v %v %d", mn, mx, n, mn2, mx2, n2)
	}
}

// TestCancelledReturnsNoValue is DESIGN.md §49's rule that a fold differs
// from a map here: no partial answer comes back with the error.
func TestCancelledReturnsNoValue(t *testing.T) {
	w, h := 64, 64
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = float32(i) + 1
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	opts := engine.Options{TileWidth: 8, TileHeight: 8, Workers: 2}

	mn, mx, n, err := reduce.MinMaxTiled(ctx, r, opts)
	if err == nil {
		t.Fatal("MinMaxTiled: no error from a cancelled context")
	}
	if mn != 0 || mx != 0 || n != 0 {
		t.Fatalf("MinMaxTiled: got %v %v %d with an error, want zeroes", mn, mx, n)
	}
	if c, err := reduce.CountTiled(ctx, r, opts); err == nil || c != 0 {
		t.Fatalf("CountTiled: got %d, %v, want 0 and an error", c, err)
	}
	src := engine.NewMemorySource(r)
	if _, _, n, err := reduce.MinMaxChunked(ctx, src, opts); err == nil || n != 0 {
		t.Fatalf("MinMaxChunked: got %d, %v, want 0 and an error", n, err)
	}
	if c, err := reduce.CountChunked(ctx, src, opts); err == nil || c != 0 {
		t.Fatalf("CountChunked: got %d, %v, want 0 and an error", c, err)
	}
}

// TestPanics checks that operand errors panic rather than returning.
func TestPanics(t *testing.T) {
	mustPanic(t, "raster: dimensions must be positive", func() {
		reduce.MinMax(raster.Float32Raster{Data: make([]float32, 4), Width: 0, Height: 2, Stride: 2})
	})
	mustPanic(t, "negative Options", func() {
		_, _, _, _ = reduce.MinMaxTiled(context.Background(), newRaster(4, 4), engine.Options{Workers: -1})
	})
}

func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		v := recover()
		if v == nil {
			t.Fatalf("no panic, want one containing %q", want)
		}
		if msg := fmt.Sprint(v); !contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func newRaster(w, h int) raster.Float32Raster {
	return raster.NewFloat32(w, h, make([]float32, w*h))
}
