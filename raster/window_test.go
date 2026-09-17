package raster

import (
	"math"
	"testing"
)

// shapes covers compact and padded strides, odd sizes, and masks whose
// rows cross 64-bit word boundaries at non-zero bit offsets.
var windowShapes = []struct{ w, h, stride int }{
	{1, 1, 1},
	{5, 3, 5},
	{7, 5, 9},
	{11, 7, 64},
	{67, 3, 70},
}

// newMasked builds a raster of the given shape with a mask in which
// roughly a third of the cells, in an irregular pattern, are invalid.
func newMasked(w, h, stride int) Float32Raster {
	r := NewFloat32Stride(w, h, stride, seq(dataLen(w, h, stride)))
	r.Valid = NewMask(len(r.Data))
	for y := range h {
		for x := range w {
			if (x*7+y*13)%3 == 0 {
				r.SetValid(x, y, false)
			}
		}
	}
	return r
}

// checkView verifies every cell of view against parent at offset (ox, oy):
// the same value, the same validity, and the same memory.
func checkView(t *testing.T, parent, view Float32Raster, ox, oy int) {
	t.Helper()
	if err := view.Validate(); err != nil {
		t.Fatalf("window at (%d, %d): %v", ox, oy, err)
	}
	for y := range view.Height {
		row := view.Row(y)
		for x := range view.Width {
			p := &parent.Data[parent.Index(ox+x, oy+y)]
			if &row[x] != p {
				t.Fatalf("window at (%d, %d) cell (%d, %d) does not alias parent", ox, oy, x, y)
			}
			if view.IsValid(x, y) != parent.IsValid(ox+x, oy+y) {
				t.Fatalf("window at (%d, %d) cell (%d, %d): validity %v, parent %v",
					ox, oy, x, y, view.IsValid(x, y), parent.IsValid(ox+x, oy+y))
			}
		}
	}
}

// TestWindowExhaustive checks every possible window of each shape, and
// every window of a representative inner window, against the parent.
func TestWindowExhaustive(t *testing.T) {
	for _, s := range windowShapes {
		for _, masked := range []bool{false, true} {
			r := NewFloat32Stride(s.w, s.h, s.stride, seq(dataLen(s.w, s.h, s.stride)))
			if masked {
				r = newMasked(s.w, s.h, s.stride)
			}
			// Keep the 67-wide case tractable: step through its windows.
			step := 1
			if s.w > 16 {
				step = 5
			}
			for y := 0; y < s.h; y++ {
				for x := 0; x < s.w; x += step {
					for h := 1; y+h <= s.h; h++ {
						for w := 1; x+w <= s.w; w += step {
							win := r.Window(x, y, w, h)
							if win.Stride != r.Stride || (masked && &win.Valid[0] != &r.Valid[0]) {
								t.Fatalf("%+v: window header %+v", s, win)
							}
							checkView(t, r, win, x, y)
						}
					}
				}
			}

			// Windows of a window: compare against the root at the summed offset.
			ix, iy := s.w/3, s.h/3
			inner := r.Window(ix, iy, s.w-ix, s.h-iy)
			for y := range inner.Height {
				for x := 0; x < inner.Width; x += step {
					nested := inner.Window(x, y, inner.Width-x, inner.Height-y)
					checkView(t, r, nested, ix+x, iy+y)
					checkView(t, inner, nested, x, y)
				}
			}
		}
	}
}

func TestWindowSharesMemoryBothWays(t *testing.T) {
	r := NewFloat32Stride(9, 7, 12, make([]float32, dataLen(9, 7, 12)))
	a := r.Window(2, 1, 6, 5)
	b := a.Window(1, 2, 4, 3) // root cell (3, 3) is b's (0, 0)

	b.Row(0)[0] = 5
	if got := cell(r, 3, 3); got != 5 {
		t.Fatalf("write via nested window: root sees %v, want 5", got)
	}
	if got := cell(a, 1, 2); got != 5 {
		t.Fatalf("write via nested window: middle window sees %v, want 5", got)
	}

	r.Row(5)[6] = 8 // b's (3, 2)
	if got := b.Row(2)[3]; got != 8 {
		t.Fatalf("write via root: nested window sees %v, want 8", got)
	}

	// Sibling windows that overlap see each other's writes.
	c := r.Window(5, 4, 4, 3)
	c.Row(0)[0] = float32(math.Inf(1)) // root (5, 4) = b's (2, 1)
	if got := b.Row(1)[2]; !math.IsInf(float64(got), 1) {
		t.Fatalf("write via sibling window: nested window sees %v", got)
	}
}

func TestWindowRejectsOutOfBounds(t *testing.T) {
	r := NewFloat32Stride(8, 6, 10, make([]float32, dataLen(8, 6, 10)))
	bad := []struct {
		name       string
		x, y, w, h int
		want       string
	}{
		{"zero width", 0, 0, 0, 1, "must be positive"},
		{"zero height", 0, 0, 1, 0, "must be positive"},
		{"negative width", 2, 2, -1, 1, "must be positive"},
		{"negative x", -1, 0, 2, 2, "outside"},
		{"negative y", 0, -1, 2, 2, "outside"},
		{"too wide", 1, 0, 8, 1, "outside"},
		{"too tall", 0, 1, 1, 6, "outside"},
		{"x past edge", 8, 0, 1, 1, "outside"},
		{"y past edge", 0, 6, 1, 1, "outside"},
		// Would reach the stride padding of row 0 rather than past Data.
		{"into padding", 7, 0, 2, 1, "outside"},
		{"overflow", 1, 0, math.MaxInt, 1, "outside"},
	}
	for _, c := range bad {
		t.Run(c.name, func(t *testing.T) {
			mustPanic(t, c.want, func() { r.Window(c.x, c.y, c.w, c.h) })
		})
	}

	// Bounds are the window's own, not the parent's.
	win := r.Window(2, 2, 4, 3)
	win.Window(0, 0, 4, 3) // whole window is fine
	mustPanic(t, "outside", func() { win.Window(1, 0, 4, 1) })
	mustPanic(t, "outside", func() { win.Window(0, 3, 1, 1) })
	mustPanic(t, "outside", func() { win.Window(-1, 0, 1, 1) })
}

func TestWindowValidityPropagation(t *testing.T) {
	// 67 wide with stride 70 puts window offsets at arbitrary bit positions
	// and makes rows straddle words.
	r := newMasked(67, 5, 70)
	a := r.Window(40, 1, 27, 4) // ValidOffset 110
	b := a.Window(20, 2, 7, 2)  // root (60, 3), ValidOffset 270
	if a.ValidOffset != 110 || b.ValidOffset != 270 {
		t.Fatalf("offsets = %d, %d, want 110, 270", a.ValidOffset, b.ValidOffset)
	}

	// Parent → nested window.
	r.SetValid(61, 3, true)
	r.SetValid(62, 3, false)
	if !b.IsValid(1, 0) || b.IsValid(2, 0) {
		t.Fatal("validity set on root not visible in nested window")
	}

	// Nested window → parent and middle window, flipping each state.
	for _, v := range []bool{false, true, false} {
		b.SetValid(6, 1, v) // root (66, 4): the last cell, last bit of the mask
		if r.IsValid(66, 4) != v || a.IsValid(26, 3) != v {
			t.Fatalf("SetValid(%v) via nested window not visible upstream", v)
		}
		if bit := r.ValidOffset + r.Index(66, 4); MaskGet(r.Valid, bit) != v {
			t.Fatalf("mask bit %d = %v, want %v", bit, !v, v)
		}
	}

	// Writing validity through a window touches exactly one bit.
	before := append([]uint64(nil), r.Valid...)
	a.SetValid(3, 0, !a.IsValid(3, 0))
	diff := 0
	for k := range before {
		diff += popcount(before[k] ^ r.Valid[k])
	}
	if diff != 1 {
		t.Fatalf("SetValid through window changed %d bits, want 1", diff)
	}

	// Padding and trailing bits are never touched.
	if tail := uint(len(r.Data) & 63); tail != 0 && r.Valid[len(r.Valid)-1]>>tail != 0 {
		t.Fatal("bits past the last cell were set")
	}

	// Windows of a maskless raster stay maskless.
	plain := NewFloat32(5, 5, make([]float32, 25)).Window(1, 1, 3, 3)
	if plain.Valid != nil || plain.ValidOffset != 0 || !plain.IsValid(2, 2) {
		t.Fatalf("maskless window got validity state %+v", plain)
	}
}

func popcount(v uint64) int {
	n := 0
	for ; v != 0; v &= v - 1 {
		n++
	}
	return n
}
