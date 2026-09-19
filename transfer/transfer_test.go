package transfer_test

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/transfer"
)

// The tests compare every operation against a naive per-cell reference
// built from Index and IsValid, over every combination of operand
// layout, mask presence and aliasing, at sizes around the 8-lane and
// 64-bit word boundaries, and then require the plain, Tiled and Chunked
// forms to agree bit for bit for every tiling and worker count. They
// also check that nothing outside dst's cells changes.

type layout int

const (
	compactLayout layout = iota // Stride == Width
	stridedLayout               // Stride > Width, arbitrary
	alignedLayout               // Stride a multiple of 64
	nestedLayout                // window of a window of a strided raster with a mask offset
)

func (l layout) String() string {
	return [...]string{"compact", "strided", "aligned", "nested"}[l]
}

var layouts = []layout{compactLayout, stridedLayout, alignedLayout, nestedLayout}

var sizes = [][2]int{
	{1, 1}, {7, 3}, {8, 2}, {9, 4}, {15, 1}, {16, 3}, {17, 2},
	{63, 2}, {64, 2}, {65, 3}, {129, 2}, {200, 3},
}

var (
	nan     = float32(math.NaN())
	inf     = float32(math.Inf(1))
	ninf    = float32(math.Inf(-1))
	negZero = float32(math.Copysign(0, -1))
)

var specials = []float32{
	nan,
	math.Float32frombits(0x7fc0_0001), // quiet NaN with payload
	math.Float32frombits(0xffc0_0000), // negative NaN
	inf, ninf,
	0, negZero,
	1, -1, 10, 20, 30, // values that land exactly on the test tables' breaks and knots
	math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
}

func randomValue(rng *rand.Rand) float32 {
	if rng.IntN(10) < 4 {
		return specials[rng.IntN(len(specials))]
	}
	return float32(rng.NormFloat64() * 15)
}

// ops are the operations under test, each with a per-cell reference
// written from the documentation rather than from the implementation,
// and each in all three forms.
var ops = []struct {
	name    string
	ref     func(v float32) float32
	plain   func(dst, src raster.Float32Raster)
	tiled   func(ctx context.Context, dst, src raster.Float32Raster, opts engine.Options) error
	chunked func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, opts engine.Options) error
}{
	{
		name: "Reclass",
		ref:  func(v float32) float32 { return refReclass(v, testBreaks, testValues) },
		plain: func(dst, src raster.Float32Raster) {
			transfer.Reclass(dst, src, testBreaks, testValues)
		},
		tiled: func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.ReclassTiled(ctx, dst, src, testBreaks, testValues, o)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.ReclassChunked(ctx, dst, src, testBreaks, testValues, o)
		},
	},
	{
		name: "Lookup",
		ref:  func(v float32) float32 { return refLookup(v, testXs, testYs) },
		plain: func(dst, src raster.Float32Raster) {
			transfer.Lookup(dst, src, testXs, testYs)
		},
		tiled: func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.LookupTiled(ctx, dst, src, testXs, testYs, o)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.LookupChunked(ctx, dst, src, testXs, testYs, o)
		},
	},
	{
		name: "Rescale",
		ref:  func(v float32) float32 { return float32(v*-2.5) + 7 },
		plain: func(dst, src raster.Float32Raster) {
			transfer.Rescale(dst, src, -2.5, 7)
		},
		tiled: func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.RescaleTiled(ctx, dst, src, -2.5, 7, o)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.RescaleChunked(ctx, dst, src, -2.5, 7, o)
		},
	},
	{
		name: "RescaleRange",
		ref:  func(v float32) float32 { a, b := refCoeffs(0, 90, 1, 5); return float32(v*a) + b },
		plain: func(dst, src raster.Float32Raster) {
			transfer.RescaleRange(dst, src, 0, 90, 1, 5)
		},
		tiled: func(ctx context.Context, dst, src raster.Float32Raster, o engine.Options) error {
			return transfer.RescaleRangeTiled(ctx, dst, src, 0, 90, 1, 5, o)
		},
		chunked: func(ctx context.Context, dst engine.RasterSink, src engine.RasterSource, o engine.Options) error {
			return transfer.RescaleRangeChunked(ctx, dst, src, 0, 90, 1, 5, o)
		},
	},
}

var (
	testBreaks = []float32{-1, 10, 20, 30}
	testValues = []float32{1, 2, 3, 4, 5}
	testXs     = []float32{-1, 0, 10, 20, 30}
	testYs     = []float32{0.5, 1, 2.5, 1.5, negZero}
)

// The references below are written from the package documentation.

func refReclass(v float32, breaks, values []float32) float32 {
	if v != v {
		return v
	}
	k := 0
	for _, b := range breaks {
		if v >= b {
			k++
		}
	}
	return values[k]
}

func refLookup(v float32, xs, ys []float32) float32 {
	if v != v {
		return v
	}
	if v <= xs[0] {
		return ys[0]
	}
	last := len(xs) - 1
	if v >= xs[last] {
		return ys[last]
	}
	for k := 1; k <= last; k++ {
		if v < xs[k] {
			if v == xs[k-1] {
				return ys[k-1] // the curve passes through its knots
			}
			t := (v - xs[k-1]) / (xs[k] - xs[k-1])
			return ys[k-1] + float32(t*(ys[k]-ys[k-1]))
		}
	}
	panic("unreachable")
}

// refCoeffs is RescaleRange's documented resolution of two intervals,
// with the conversions that stop a fused multiply-subtract.
func refCoeffs(inLo, inHi, outLo, outHi float32) (a, b float32) {
	a = float32((float64(outHi) - float64(outLo)) / (float64(inHi) - float64(inLo)))
	b = float32(float64(outLo) - float64(float64(a)*float64(inLo)))
	return a, b
}

// operand is a raster under test plus the root raster that owns its
// memory and the index of its first cell in the root's Data.
type operand struct {
	r     raster.Float32Raster
	root  raster.Float32Raster
	start int
}

func newOperand(rng *rand.Rand, w, h int, lay layout, masked bool) operand {
	rootW, rootH, stride, validOffset := w, h, w, 0
	switch lay {
	case stridedLayout:
		stride = w + 1 + rng.IntN(70)
	case alignedLayout:
		stride = (w/64 + 1) * 64
	case nestedLayout:
		rootW, rootH = w+13, h+7
		stride = rootW + rng.IntN(9)
		validOffset = 1 + rng.IntN(100)
	}
	n := (rootH-1)*stride + rootW
	data := make([]float32, n)
	for i := range data {
		data[i] = randomValue(rng)
	}
	root := raster.NewFloat32Stride(rootW, rootH, stride, data)
	if masked {
		root.Valid = raster.NewMask(validOffset + n + rng.IntN(130))
		root.ValidOffset = validOffset
		for i := range len(root.Valid) * 64 {
			if rng.IntN(4) == 0 {
				raster.MaskSet(root.Valid, i, false)
			}
		}
	}
	op := operand{r: root, root: root}
	if lay == nestedLayout {
		mid := root.Window(3, 2, w+9, h+4)
		x, y := rng.IntN(10), rng.IntN(5)
		op.r = mid.Window(x, y, w, h)
		op.start = (2+y)*stride + 3 + x
	}
	return op
}

type cell struct {
	v     float32
	valid bool
}

func snapshot(r raster.Float32Raster) []cell {
	cs := make([]cell, 0, r.Width*r.Height)
	for y := range r.Height {
		for x := range r.Width {
			cs = append(cs, cell{r.Data[r.Index(x, y)], r.IsValid(x, y)})
		}
	}
	return cs
}

// sameFloat compares bitwise, except that any NaN equals any NaN.
func sameFloat(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

type frozen struct {
	dst   operand
	data  []float32
	valid []uint64
}

func freeze(dst operand) frozen {
	return frozen{
		dst:   dst,
		data:  append([]float32(nil), dst.root.Data...),
		valid: append([]uint64(nil), dst.root.Valid...),
	}
}

func (f frozen) checkUntouched(t *testing.T) {
	t.Helper()
	root, r := f.dst.root, f.dst.r
	inDst := func(i int) bool {
		i -= f.dst.start
		if i < 0 {
			return false
		}
		return i/r.Stride < r.Height && i%r.Stride < r.Width
	}
	for i, v := range root.Data {
		if !inDst(i) && math.Float32bits(v) != math.Float32bits(f.data[i]) {
			t.Fatalf("root cell %d outside dst changed: %v -> %v", i, f.data[i], v)
		}
	}
	for bit := range len(f.valid) * 64 {
		i := bit - root.ValidOffset
		if (i < 0 || i >= len(root.Data) || !inDst(i)) &&
			raster.MaskGet(root.Valid, bit) != raster.MaskGet(f.valid, bit) {
			t.Fatalf("mask bit %d outside dst changed", bit)
		}
	}
}

func checkResult(t *testing.T, id string, dst raster.Float32Raster, src []cell, ref func(float32) float32) {
	t.Helper()
	for y := range dst.Height {
		for x := range dst.Width {
			in := src[y*dst.Width+x]
			if got := dst.IsValid(x, y); got != in.valid {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", id, x, y, got, in.valid)
			}
			want := ref(in.v)
			if got := dst.Data[dst.Index(x, y)]; !sameFloat(got, want) {
				t.Fatalf("%s: cell (%d, %d) from %v (%#08x) = %v (%#08x), want %v (%#08x)",
					id, x, y, in.v, math.Float32bits(in.v),
					got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}
	}
}

// TestValues runs every operation over every layout, size and mask
// combination against the per-cell reference, and checks that nothing
// outside dst changed.
func TestValues(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			for _, size := range sizes {
				for _, dl := range layouts {
					for _, sl := range layouts {
						for _, masked := range []bool{false, true} {
							w, h := size[0], size[1]
							src := newOperand(rng, w, h, sl, masked)
							dst := newOperand(rng, w, h, dl, masked)
							before := snapshot(src.r)
							f := freeze(dst)
							op.plain(dst.r, src.r)
							id := op.name + "/" + dl.String() + "/" + sl.String()
							checkResult(t, id, dst.r, before, op.ref)
							f.checkUntouched(t)
						}
					}
				}
			}
		})
	}
}

// TestInPlace checks that dst may be src exactly, which is the case a
// caller hits when overwriting a factor raster with its class.
func TestInPlace(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, op := range ops {
		for _, lay := range layouts {
			for _, masked := range []bool{false, true} {
				r := newOperand(rng, 37, 5, lay, masked)
				before := snapshot(r.r)
				op.plain(r.r, r.r)
				checkResult(t, op.name+"/inPlace/"+lay.String(), r.r, before, op.ref)
			}
		}
	}
}

// tilings is the grid of engine options every path is compared over: no
// tiling, degenerate one-cell tiles, tiles that do not divide the
// raster, and one, several and GOMAXPROCS workers.
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

// TestPathsAgree is the package's central guarantee: the plain, Tiled
// and Chunked forms write the same bits for every tile size and worker
// count.
func TestPathsAgree(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewPCG(5, 6))
	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			for _, size := range [][2]int{{1, 1}, {9, 4}, {65, 3}, {200, 3}} {
				for _, masked := range []bool{false, true} {
					w, h := size[0], size[1]
					src := newOperand(rng, w, h, nestedLayout, masked)
					want := newOperand(rng, w, h, compactLayout, masked)
					op.plain(want.r, src.r)
					wantCells := snapshot(want.r)

					for _, opts := range tilings() {
						got := newOperand(rng, w, h, stridedLayout, masked)
						if err := op.tiled(ctx, got.r, src.r, opts); err != nil {
							t.Fatalf("Tiled %+v: %v", opts, err)
						}
						compare(t, "Tiled", opts, got.r, wantCells)

						sink := engine.NewMemorySink(newOperand(rng, w, h, compactLayout, masked).r)
						if err := op.chunked(ctx, sink, engine.NewMemorySource(src.r), opts); err != nil {
							t.Fatalf("Chunked %+v: %v", opts, err)
						}
						compare(t, "Chunked", opts, sink.Raster(), wantCells)
					}
				}
			}
		})
	}
}

func compare(t *testing.T, path string, opts engine.Options, got raster.Float32Raster, want []cell) {
	t.Helper()
	for y := range got.Height {
		for x := range got.Width {
			w := want[y*got.Width+x]
			if v := got.IsValid(x, y); v != w.valid {
				t.Fatalf("%s %+v: cell (%d, %d) valid = %v, want %v", path, opts, x, y, v, w.valid)
			}
			if v := got.Data[got.Index(x, y)]; !sameFloat(v, w.v) {
				t.Fatalf("%s %+v: cell (%d, %d) = %v (%#08x), want %v (%#08x)",
					path, opts, x, y, v, math.Float32bits(v), w.v, math.Float32bits(w.v))
			}
		}
	}
}

// TestKnotsAreExact checks the documented promise that Lookup returns
// each knot's y bit for bit, including next to a NaN, an infinity and a
// negative zero, which interpolating from the knot would not.
func TestKnotsAreExact(t *testing.T) {
	xs := []float32{-3, 0, 7.25, 1000, 2000}
	ys := []float32{5, nan, inf, negZero, 20}
	src := raster.NewFloat32(len(xs), 1, append([]float32(nil), xs...))
	dst := raster.NewFloat32Like(src)
	transfer.Lookup(dst, src, xs, ys)
	for i, want := range ys {
		if got := dst.Data[i]; !sameFloat(got, want) {
			t.Errorf("knot %d (x=%v): got %v (%#08x), want %v (%#08x)",
				i, xs[i], got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
	if !sameFloat(dst.Data[3], negZero) {
		t.Errorf("a -0 knot came back %#08x, want %#08x", math.Float32bits(dst.Data[3]), math.Float32bits(negZero))
	}
}

// TestUnitCurveIsClamp pins the one exact bridge to package algebra: the
// curve through (0,0) and (1,1) is algebra.Clamp to [0, 1], bit for bit.
// It does not generalize — see the package documentation.
func TestUnitCurveIsClamp(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	src := newOperand(rng, 129, 3, nestedLayout, true)
	viaLookup := newOperand(rng, 129, 3, stridedLayout, true)
	viaClamp := newOperand(rng, 129, 3, compactLayout, true)

	unit := []float32{0, 1}
	transfer.Lookup(viaLookup.r, src.r, unit, unit)
	algebra.Clamp(viaClamp.r, src.r, 0, 1)
	compare(t, "unit curve", engine.Options{}, viaLookup.r, snapshot(viaClamp.r))
}

// TestRescaleIdentity pins the documented difference between a zero and
// a negative zero offset: only the latter is the bit identity, because
// v + 0 turns -0 into +0.
func TestRescaleIdentity(t *testing.T) {
	src := raster.NewFloat32(4, 1, []float32{negZero, 0, -1, nan})
	dst := raster.NewFloat32Like(src)

	transfer.Rescale(dst, src, 1, negZero)
	for i, want := range src.Data {
		if !sameFloat(dst.Data[i], want) {
			t.Errorf("b = -0: cell %d = %#08x, want %#08x", i, math.Float32bits(dst.Data[i]), math.Float32bits(want))
		}
	}

	transfer.Rescale(dst, src, 1, 0)
	if math.Float32bits(dst.Data[0]) != math.Float32bits(0) {
		t.Errorf("b = +0 left -0 as %#08x, want +0", math.Float32bits(dst.Data[0]))
	}
}

// rangeCases are intervals RescaleRange is checked over, from the
// ordinary to the ones whose coefficients are extreme.
var rangeCases = [][4]float32{
	{0, 90, 0, 1}, {0, 90, 1, 5}, {0, 1, 0, 1}, {-1, 1, 0, 100},
	{11.2, 21.3, 4, 5}, {2.5, 7.5, 1, 9}, {-40, 40, 0.1, 0.9},
	{1e-20, 3e-20, -7, 7}, {1e20, 3e20, -1, 1}, {-1000, 1000, 2, 2},
}

// TestRescaleRangeIsRescale checks the one exactness RescaleRange
// actually promises: it is Rescale with the resolved coefficients, bit
// for bit, on every cell.
func TestRescaleRangeIsRescale(t *testing.T) {
	for _, c := range rangeCases {
		inLo, inHi, outLo, outHi := c[0], c[1], c[2], c[3]
		src := raster.NewFloat32(5, 1, []float32{inLo, inHi, (inLo + inHi) / 2, inLo - 1, inHi * 3})
		dst := raster.NewFloat32Like(src)
		transfer.RescaleRange(dst, src, inLo, inHi, outLo, outHi)

		a, b := refCoeffs(inLo, inHi, outLo, outHi)
		viaRescale := raster.NewFloat32Like(src)
		transfer.Rescale(viaRescale, src, a, b)
		for i := range src.Data {
			if !sameFloat(dst.Data[i], viaRescale.Data[i]) {
				t.Errorf("%v: cell %d: RescaleRange gave %#08x, Rescale with the resolved coefficients %#08x",
					c, i, math.Float32bits(dst.Data[i]), math.Float32bits(viaRescale.Data[i]))
			}
		}
	}
}

// TestRescaleRangeEndpoints pins how close the endpoints land. The error
// is float32 precision relative to the output span, not to the endpoint:
// the offset absorbs the scaled inLo, so an endpoint near zero can be
// several of its own ulps out once that cancels. Mapping [-40, 40] onto
// [0.1, 0.9] gives an offset of 0.5 and returns 0.100000024 at -40,
// three ulps of 0.1 but a quarter of an ulp of the span.
//
// A zero inLo is exact, and provably so rather than by observation: the
// offset is then outLo itself and the scaled cell is zero.
func TestRescaleRangeEndpoints(t *testing.T) {
	for _, c := range rangeCases {
		inLo, inHi, outLo, outHi := c[0], c[1], c[2], c[3]
		src := raster.NewFloat32(2, 1, []float32{inLo, inHi})
		dst := raster.NewFloat32Like(src)
		transfer.RescaleRange(dst, src, inLo, inHi, outLo, outHi)

		// One part in 2^-20 of the span, comfortably inside float32's
		// 2^-24 relative precision after a handful of roundings.
		span := math.Abs(float64(outHi) - float64(outLo))
		tol := math.Max(span, math.Abs(float64(outLo))) / (1 << 20)
		for i, want := range []float32{outLo, outHi} {
			if got := math.Abs(float64(dst.Data[i]) - float64(want)); got > tol {
				t.Errorf("%v: endpoint %d mapped to %v, %g from %v (tolerance %g)",
					c, i, dst.Data[i], got, want, tol)
			}
		}
		if inLo == 0 && !sameFloat(dst.Data[0], outLo) {
			t.Errorf("%v: a zero inLo mapped to %v (%#08x), want outLo %v (%#08x) exactly",
				c, dst.Data[0], math.Float32bits(dst.Data[0]), outLo, math.Float32bits(outLo))
		}
	}
}

// TestTwoKnotLookupIsExact is the counterpart: the curve through the two
// corners returns both of them bit for bit for every interval, which is
// what the documentation points a caller needing exact endpoints at.
func TestTwoKnotLookupIsExact(t *testing.T) {
	for _, c := range rangeCases {
		inLo, inHi, outLo, outHi := c[0], c[1], c[2], c[3]
		src := raster.NewFloat32(4, 1, []float32{inLo, inHi, inLo - 1, inHi * 3})
		dst := raster.NewFloat32Like(src)
		transfer.Lookup(dst, src, []float32{inLo, inHi}, []float32{outLo, outHi})
		for i, want := range []float32{outLo, outHi, outLo, outHi} {
			if !sameFloat(dst.Data[i], want) {
				t.Errorf("%v: cell %d (%v) = %v (%#08x), want %v (%#08x)",
					c, i, src.Data[i], dst.Data[i], math.Float32bits(dst.Data[i]), want, math.Float32bits(want))
			}
		}
	}
}

// TestRescaleIsNotFused pins the separate rounding of the multiply and
// the add, the property that makes the result the same on amd64 and on
// arm64. See vec.TestAffineIsNotFused for the arithmetic.
func TestRescaleIsNotFused(t *testing.T) {
	const eps = 1.0 / (1 << 23)
	a := float32(1 + eps)
	b := -float32(1 + 2*eps)
	src := raster.NewFloat32(1, 1, []float32{a})
	dst := raster.NewFloat32Like(src)
	transfer.Rescale(dst, src, a, b)
	if got := dst.Data[0]; got != 0 || math.Signbit(float64(got)) {
		t.Errorf("Rescale = %v (%#08x), want +0; the multiply-add was fused", got, math.Float32bits(got))
	}
}

func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", want)
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic %v (%T), want a string containing %q", r, r, want)
		}
		if !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

func TestPanics(t *testing.T) {
	ctx := context.Background()
	r := func(w, h int) raster.Float32Raster { return raster.NewFloat32(w, h, make([]float32, w*h)) }
	a := r(4, 3)
	masked := r(4, 3)
	masked.Valid = raster.NewMask(12)

	// Tables.
	mustPanic(t, "transfer.Reclass: values must hold one more element than breaks, got 2 values and 2 breaks",
		func() { transfer.Reclass(a, a, []float32{0, 1}, []float32{1, 2}) })
	mustPanic(t, "transfer.Reclass: breaks must be strictly increasing, but breaks[1] = 0 is not above breaks[0] = 0",
		func() { transfer.Reclass(a, a, []float32{0, 0}, []float32{1, 2, 3}) })
	mustPanic(t, "transfer.Reclass: breaks must not hold NaN, but breaks[0] is NaN",
		func() { transfer.Reclass(a, a, []float32{nan}, []float32{1, 2}) })
	mustPanic(t, "transfer.Lookup: xs and ys must have equal length, got 2 and 1",
		func() { transfer.Lookup(a, a, []float32{0, 1}, []float32{1}) })
	mustPanic(t, "transfer.Lookup: the table needs at least one knot",
		func() { transfer.Lookup(a, a, nil, nil) })
	mustPanic(t, "transfer.Lookup: xs must be strictly increasing",
		func() { transfer.Lookup(a, a, []float32{1, 0}, []float32{1, 2}) })
	mustPanic(t, "transfer.Lookup: xs must be finite, but xs[1] = +Inf",
		func() { transfer.Lookup(a, a, []float32{0, inf}, []float32{1, 2}) })
	mustPanic(t, "transfer.RescaleRange: inLo and inHi must be finite and different, got 3 and 3",
		func() { transfer.RescaleRange(a, a, 3, 3, 0, 1) })
	mustPanic(t, "transfer.RescaleRange: inLo and inHi must be finite and different, got 0 and +Inf",
		func() { transfer.RescaleRange(a, a, 0, inf, 0, 1) })
	mustPanic(t, "transfer.RescaleRange: outLo and outHi must be finite, got 0 and NaN",
		func() { transfer.RescaleRange(a, a, 0, 1, 0, nan) })
	mustPanic(t, "gives a scale of +Inf",
		func() { transfer.RescaleRange(a, a, 0, math.SmallestNonzeroFloat32, 0, math.MaxFloat32) })

	// Operands.
	mustPanic(t, "transfer.Reclass: src dimensions differ from dst",
		func() { transfer.Reclass(a, r(3, 4), testBreaks, testValues) })
	mustPanic(t, "transfer.Rescale: src dimensions differ from dst",
		func() { transfer.Rescale(a, r(5, 3), 1, 0) })
	mustPanic(t, "transfer.Lookup: an input has a validity mask but dst.Valid is nil",
		func() { transfer.Lookup(a, masked, testXs, testYs) })
	mustPanic(t, "transfer.Rescale: dst overlaps src at a different offset or stride", func() {
		root := raster.NewFloat32(10, 4, make([]float32, 40))
		transfer.Rescale(root.Window(0, 0, 5, 4), root.Window(1, 0, 5, 4), 1, 0)
	})

	// The tables are checked on every path, with the entry point's name.
	mustPanic(t, "transfer.ReclassTiled: breaks must not hold NaN",
		func() { _ = transfer.ReclassTiled(ctx, a, a, []float32{nan}, []float32{1, 2}, engine.Options{}) })
	mustPanic(t, "transfer.LookupChunked: the table needs at least one knot", func() {
		_ = transfer.LookupChunked(ctx, engine.NewMemorySink(a), engine.NewMemorySource(a), nil, nil, engine.Options{})
	})
	mustPanic(t, "transfer.RescaleRangeTiled: inLo and inHi must be finite and different",
		func() { _ = transfer.RescaleRangeTiled(ctx, a, a, 1, 1, 0, 1, engine.Options{}) })
}

// TestTableRejectedBeforeAnyWrite checks that a bad table leaves dst
// exactly as it was: the checks run before the operand checks touch
// anything and before a single cell is written.
func TestTableRejectedBeforeAnyWrite(t *testing.T) {
	src := raster.NewFloat32(8, 2, make([]float32, 16))
	dst := raster.NewFloat32(8, 2, make([]float32, 16))
	for i := range dst.Data {
		dst.Data[i] = 12345
	}
	before := append([]float32(nil), dst.Data...)
	mustPanic(t, "transfer.Reclass", func() { transfer.Reclass(dst, src, []float32{1, 0}, []float32{1, 2, 3}) })
	for i, v := range dst.Data {
		if math.Float32bits(v) != math.Float32bits(before[i]) {
			t.Fatalf("cell %d was written before the table was rejected", i)
		}
	}
}

func TestNoAllocs(t *testing.T) {
	const w, h = 100, 70
	a := raster.NewFloat32(w, h, make([]float32, w*h))
	strided := raster.NewFloat32Stride(w+9, h+3, w+20, make([]float32, (h+2)*(w+20)+w+9)).Window(4, 2, w, h)
	masked := raster.NewFloat32Like(a)
	masked.Valid = raster.NewMask(w * h)
	dst := raster.NewFloat32Like(masked)
	cases := map[string]func(){
		"Reclass/compact":      func() { transfer.Reclass(a, a, testBreaks, testValues) },
		"Reclass/strided":      func() { transfer.Reclass(strided, a, testBreaks, testValues) },
		"Reclass/masked":       func() { transfer.Reclass(dst, masked, testBreaks, testValues) },
		"Lookup/compact":       func() { transfer.Lookup(a, a, testXs, testYs) },
		"Lookup/strided":       func() { transfer.Lookup(strided, a, testXs, testYs) },
		"Lookup/masked":        func() { transfer.Lookup(dst, masked, testXs, testYs) },
		"Rescale/compact":      func() { transfer.Rescale(a, a, 2, 1) },
		"Rescale/strided":      func() { transfer.Rescale(strided, a, 2, 1) },
		"Rescale/masked":       func() { transfer.Rescale(dst, masked, 2, 1) },
		"RescaleRange/compact": func() { transfer.RescaleRange(a, a, 0, 90, 0, 1) },
		"RescaleRange/masked":  func() { transfer.RescaleRange(dst, masked, 0, 90, 0, 1) },
	}
	for name, f := range cases {
		if n := testing.AllocsPerRun(10, f); n != 0 {
			t.Errorf("%s: %v allocs per run", name, n)
		}
	}
}

// TestCancellation checks that a cancelled context stops a tiled run and
// is reported, as package engine promises.
func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	src := raster.NewFloat32(64, 64, make([]float32, 64*64))
	dst := raster.NewFloat32Like(src)
	for _, op := range ops {
		if err := op.tiled(ctx, dst, src, engine.Options{TileHeight: 1}); !errors.Is(err, context.Canceled) {
			t.Errorf("%s: err = %v, want context.Canceled", op.name, err)
		}
	}
}
