package algebra_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/raster"
)

// The tests compare every operation against a naive per-cell reference
// built from Index and IsValid, over every combination of operand layout,
// mask presence and aliasing, at sizes around the 8-lane and 64-bit word
// boundaries. They also check that nothing outside dst's cells changes.

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
	{63, 2}, {64, 2}, {65, 3}, {127, 1}, {129, 2}, {200, 3},
}

var specials = []float32{
	float32(math.NaN()),
	math.Float32frombits(0x7fc0_0001), // quiet NaN with payload
	math.Float32frombits(0xffc0_0000), // negative NaN
	float32(math.Inf(1)), float32(math.Inf(-1)),
	0, float32(math.Copysign(0, -1)),
	1, -1, math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
}

func randomValue(rng *rand.Rand) float32 {
	if rng.IntN(10) < 3 {
		return specials[rng.IntN(len(specials))]
	}
	return float32(rng.NormFloat64() * 100)
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

// cell is the reference view of one input cell.
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

// frozen records dst's root so checkUntouched can verify that only dst's
// cells and their validity bits changed.
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

func checkResult(t *testing.T, dst raster.Float32Raster, want []cell) {
	t.Helper()
	for y := range dst.Height {
		for x := range dst.Width {
			w := want[y*dst.Width+x]
			if got := dst.IsValid(x, y); got != w.valid {
				t.Fatalf("cell (%d, %d) valid = %v, want %v", x, y, got, w.valid)
			}
			if !w.valid {
				continue
			}
			if got := dst.Data[dst.Index(x, y)]; !sameFloat(got, w.v) {
				t.Fatalf("cell (%d, %d) = %v (%#08x), want %v (%#08x)",
					x, y, got, math.Float32bits(got), w.v, math.Float32bits(w.v))
			}
		}
	}
}

type aliasing int

const (
	noAlias   aliasing = iota
	dstIsA             // algebra.Op(a, a, b)
	dstIsB             // algebra.Op(b, a, b)
	aIsB               // algebra.Op(dst, a, a)
	dstIsAIsB          // algebra.Op(a, a, a)
)

var binaryOps = []struct {
	name string
	fn   func(dst, a, b raster.Float32Raster)
	ref  func(a, b float32) float32
}{
	{"Add", algebra.Add, func(a, b float32) float32 { return a + b }},
	{"Sub", algebra.Sub, func(a, b float32) float32 { return a - b }},
	{"Mul", algebra.Mul, func(a, b float32) float32 { return a * b }},
	{"Min", algebra.Min, func(a, b float32) float32 { return min(a, b) }},
	{"Max", algebra.Max, func(a, b float32) float32 { return max(a, b) }},
	// Mask(dst, src, mask) keeps src's value and the AND of the two
	// validities, so it is a binary operation whose reference is a.
	{"Mask", algebra.Mask, func(a, _ float32) float32 { return a }},
}

func TestBinaryMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, op := range binaryOps {
		t.Run(op.name, func(t *testing.T) {
			cases := 0
			for _, sz := range sizes {
				w, h := sz[0], sz[1]
				for _, la := range layouts {
					for _, lb := range layouts {
						for _, ld := range layouts {
							for masks := range 4 {
								aMasked, bMasked := masks&1 != 0, masks&2 != 0
								for alias := noAlias; alias <= dstIsAIsB; alias++ {
									runBinary(t, rng, op.fn, op.ref, w, h, la, lb, ld, aMasked, bMasked, alias)
									cases++
								}
							}
						}
					}
				}
			}
			t.Logf("%d cases", cases)
		})
	}
}

func runBinary(t *testing.T, rng *rand.Rand, fn func(dst, a, b raster.Float32Raster),
	ref func(a, b float32) float32, w, h int, la, lb, ld layout, aMasked, bMasked bool, alias aliasing) {
	t.Helper()
	a := newOperand(rng, w, h, la, aMasked)
	b := newOperand(rng, w, h, lb, bMasked)
	switch alias {
	case aIsB, dstIsAIsB:
		b = a
		bMasked = aMasked
	}
	var dst operand
	switch alias {
	case dstIsA, dstIsAIsB:
		dst = a
	case dstIsB:
		dst = b
	default:
		// dst needs a mask when an input has one; otherwise try both.
		dst = newOperand(rng, w, h, ld, aMasked || bMasked || rng.IntN(2) == 0)
	}
	if (aMasked || bMasked) && dst.r.Valid == nil {
		return // would panic by design; covered by TestPanics
	}

	as, bs := snapshot(a.r), snapshot(b.r)
	want := make([]cell, len(as))
	for i := range want {
		want[i] = cell{ref(as[i].v, bs[i].v), as[i].valid && bs[i].valid}
	}
	f := freeze(dst)
	fn(dst.r, a.r, b.r)

	defer func() {
		if t.Failed() {
			t.Logf("case %dx%d a=%v masked=%v b=%v masked=%v dst=%v alias=%d",
				w, h, la, aMasked, lb, bMasked, ld, alias)
		}
	}()
	checkResult(t, dst.r, want)
	f.checkUntouched(t)
}

func TestClampMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	inf := float32(math.Inf(1))
	bounds := [][2]float32{{0, 100}, {-1, 1}, {float32(math.Copysign(0, -1)), 0}, {5, -5}, {-inf, inf}}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for _, ls := range layouts {
			for _, ld := range layouts {
				for _, masked := range []bool{false, true} {
					for _, inPlace := range []bool{false, true} {
						for _, bd := range bounds {
							src := newOperand(rng, w, h, ls, masked)
							dst := src
							if !inPlace {
								dst = newOperand(rng, w, h, ld, masked || rng.IntN(2) == 0)
							}
							ss := snapshot(src.r)
							want := make([]cell, len(ss))
							for i, c := range ss {
								want[i] = cell{min(max(c.v, bd[0]), bd[1]), c.valid}
							}
							f := freeze(dst)
							algebra.Clamp(dst.r, src.r, bd[0], bd[1])
							func() {
								defer func() {
									if t.Failed() {
										t.Logf("case %dx%d src=%v masked=%v dst=%v inPlace=%v bounds=%v",
											w, h, ls, masked, ld, inPlace, bd)
									}
								}()
								checkResult(t, dst.r, want)
								f.checkUntouched(t)
							}()
						}
					}
				}
			}
		}
	}
}

// TestSiblingWindows writes into a window of the same raster, and so the
// same mask, as an input, at positions that share no cells with it.
func TestSiblingWindows(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for _, extra := range []int{0, 1, 5, 64} {
			rootW, rootH := 2*w+3, 2*h+1
			stride := rootW + extra
			n := (rootH-1)*stride + rootW
			data := make([]float32, n)
			for i := range data {
				data[i] = randomValue(rng)
			}
			root := raster.NewFloat32Stride(rootW, rootH, stride, data)
			root.Valid = raster.NewMask(n + 70)
			root.ValidOffset = 70
			for i := range len(root.Valid) * 64 {
				if rng.IntN(4) == 0 {
					raster.MaskSet(root.Valid, i, false)
				}
			}
			at := func(x, y int) operand {
				return operand{r: root.Window(x, y, w, h), root: root, start: y*stride + x}
			}
			a := at(0, 1)
			for _, d := range []operand{at(w+2, 0), at(w+1, h+1), at(0, h+1), at(w+3, 1)} {
				b := newOperand(rng, w, h, layouts[rng.IntN(len(layouts))], rng.IntN(2) == 0)
				as, bs := snapshot(a.r), snapshot(b.r)
				want := make([]cell, len(as))
				for i := range want {
					want[i] = cell{as[i].v - bs[i].v, as[i].valid && bs[i].valid}
				}
				f := freeze(d)
				algebra.Sub(d.r, a.r, b.r)
				checkResult(t, d.r, want)
				f.checkUntouched(t)

				as = snapshot(a.r)
				for i, c := range as {
					want[i] = cell{min(max(c.v, -10), 10), c.valid}
				}
				f = freeze(d)
				algebra.Clamp(d.r, a.r, -10, 10)
				checkResult(t, d.r, want)
				f.checkUntouched(t)

				// Mask copies between sibling windows, so dst's bits and
				// src's are disjoint ranges of one mask array.
				as, bs = snapshot(a.r), snapshot(b.r)
				for i := range want {
					want[i] = cell{as[i].v, as[i].valid && bs[i].valid}
				}
				f = freeze(d)
				algebra.Mask(d.r, a.r, b.r)
				checkResult(t, d.r, want)
				f.checkUntouched(t)
			}
		}
	}
}

// TestOverlapDetection places two windows of one raster everywhere and
// checks that an operation panics exactly when they share some cells
// without being the same window.
func TestOverlapDetection(t *testing.T) {
	for _, stride := range []int{7, 8, 11} {
		root := raster.NewFloat32Stride(7, 6, stride, make([]float32, 5*stride+7))
		for _, sz := range [][2]int{{1, 1}, {3, 2}, {7, 1}, {4, 3}, {6, 6}} {
			w, h := sz[0], sz[1]
			type pos struct{ x, y int }
			var ps []pos
			for y := 0; y+h <= 6; y++ {
				for x := 0; x+w <= 7; x++ {
					ps = append(ps, pos{x, y})
				}
			}
			for _, p := range ps {
				for _, q := range ps {
					dst, src := root.Window(p.x, p.y, w, h), root.Window(q.x, q.y, w, h)
					share := p != q && p.x < q.x+w && q.x < p.x+w && p.y < q.y+h && q.y < p.y+h
					panicked := func() (panicked bool) {
						defer func() { panicked = recover() != nil }()
						algebra.Clamp(dst, src, 0, 1)
						return false
					}()
					if panicked != share {
						t.Fatalf("stride %d, %dx%d windows at %v and %v: panicked = %v, want %v",
							stride, w, h, p, q, panicked, share)
					}
				}
			}
		}
	}
}

func TestAllValidLeavesNilMask(t *testing.T) {
	a := raster.NewFloat32(3, 2, []float32{1, 2, 3, 4, 5, 6})
	dst := raster.NewFloat32Like(a)
	algebra.Add(dst, a, a)
	algebra.Clamp(dst, dst, 3, 10)
	if dst.Valid != nil {
		t.Fatal("dst gained a mask")
	}
	for i, want := range []float32{3, 4, 6, 8, 10, 10} {
		if dst.Data[i] != want {
			t.Fatalf("Data = %v", dst.Data)
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
		"Add/compact":        func() { algebra.Add(a, a, a) },
		"Add/strided":        func() { algebra.Add(strided, a, strided) },
		"Clamp/compact":      func() { algebra.Clamp(a, a, 0, 1) },
		"Clamp/strided":      func() { algebra.Clamp(strided, a, 0, 1) },
		"Max/masked":         func() { algebra.Max(dst, masked, a) },
		"Mul/masked/strided": func() { algebra.Mul(dst, masked, strided) },
		"Mask/masked":        func() { algebra.Mask(dst, a, masked) },
		"Mask/inPlace":       func() { algebra.Mask(dst, dst, masked) },
	}
	for name, f := range cases {
		if n := testing.AllocsPerRun(10, f); n != 0 {
			t.Errorf("%s: %v allocs per run", name, n)
		}
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
		msg, _ := r.(string)
		if !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

func TestPanics(t *testing.T) {
	r := func(w, h int) raster.Float32Raster { return raster.NewFloat32(w, h, make([]float32, w*h)) }
	a, b := r(4, 3), r(4, 3)
	masked := r(4, 3)
	masked.Valid = raster.NewMask(12)

	mustPanic(t, "algebra.Add: b dimensions differ from dst", func() { algebra.Add(a, a, r(3, 4)) })
	mustPanic(t, "algebra.Sub: a dimensions differ from dst", func() { algebra.Sub(a, r(4, 2), b) })
	mustPanic(t, "algebra.Clamp: src dimensions differ from dst", func() { algebra.Clamp(a, r(5, 3), 0, 1) })
	mustPanic(t, "algebra.Mul: a: raster: data has", func() {
		short := a
		short.Data = short.Data[:5]
		algebra.Mul(b, short, b)
	})
	mustPanic(t, "algebra.Min: b: raster: mask has", func() {
		bad := masked
		bad.ValidOffset = 60
		algebra.Min(masked, a, bad)
	})

	mustPanic(t, "algebra.Mask: mask dimensions differ from dst", func() { algebra.Mask(a, b, r(3, 4)) })

	mustPanic(t, "dst.Valid is nil", func() { algebra.Max(a, b, masked) })
	mustPanic(t, "dst.Valid is nil", func() { algebra.Mask(a, b, masked) })
	mustPanic(t, "dst.Valid is nil", func() { algebra.Clamp(a, masked, 0, 1) })

	big := raster.NewFloat32(10, 10, make([]float32, 100))
	w1, w2 := big.Window(0, 0, 4, 3), big.Window(1, 0, 4, 3)
	mustPanic(t, "dst overlaps a at a different offset", func() { algebra.Add(w2, w1, b) })
	mustPanic(t, "dst overlaps src at a different offset", func() { algebra.Clamp(w1, w2, 0, 1) })
	mustPanic(t, "dst overlaps mask at a different offset", func() { algebra.Mask(w2, b, w1) })
	restrided := raster.NewFloat32Stride(4, 3, 5, big.Data)
	mustPanic(t, "dst overlaps b at a different offset or stride", func() { algebra.Add(w1, a, restrided) })
	// Overlapping inputs are fine; only dst matters.
	algebra.Add(a, w1, w2)
}

func ExampleAdd() {
	a := raster.NewFloat32(3, 1, []float32{1, 2, 3})
	b := raster.NewFloat32(3, 1, []float32{10, 20, 30})
	b.Valid = raster.NewMask(3)
	b.SetValid(1, 0, false)

	dst := raster.NewFloat32Like(b) // has a mask because b does
	algebra.Add(dst, a, b)
	for x := range dst.Width {
		if dst.IsValid(x, 0) {
			fmt.Println(dst.Data[x])
		} else {
			fmt.Println("nodata")
		}
	}
	// Output:
	// 11
	// nodata
	// 33
}

func ExampleMask() {
	src := raster.NewFloat32(3, 1, []float32{1, 2, 3})
	// Only cloud's validity is read; its values never are.
	cloud := raster.NewFloat32(3, 1, []float32{-999, -999, -999})
	cloud.Valid = raster.NewMask(3)
	cloud.SetValid(1, 0, false)

	dst := raster.NewFloat32Like(cloud) // has a mask because cloud does
	algebra.Mask(dst, src, cloud)
	for x := range dst.Width {
		if dst.IsValid(x, 0) {
			fmt.Println(dst.Data[x])
		} else {
			fmt.Println("nodata")
		}
	}
	// Output:
	// 1
	// nodata
	// 3
}
