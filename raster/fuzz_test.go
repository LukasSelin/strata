package raster

import (
	"math/big"
	"strings"
	"testing"

	"strata/internal/fuzzdata"
)

// The fuzz tests check the package's arithmetic on shapes and bit ranges
// against exact references (math/big for shapes, one bit at a time for
// masks), and that every rejection is a panic with a "raster:" message
// rather than a runtime error from a later index.

// requireRasterPanic runs f and returns whether it panicked, failing the
// test if the panic is not one of the package's own.
func requireRasterPanic(t *testing.T, name string, f func()) (panicked bool) {
	t.Helper()
	defer func() {
		v := recover()
		if v == nil {
			return
		}
		panicked = true
		if s, ok := v.(string); !ok || !strings.HasPrefix(s, "raster: ") {
			t.Fatalf("%s: panic %v (%T), want a \"raster: \" message", name, v, v)
		}
	}()
	f()
	return false
}

// exactShape reports whether a width×height raster with the given stride
// fits in dataLen cells, and if Valid is set, in maskBits bits at
// validOffset, computed without overflow.
func exactShape(width, height, stride, dataLen int, masked bool, maskBits, validOffset int) bool {
	if width <= 0 || height <= 0 || stride < width {
		return false
	}
	need := big.NewInt(int64(height - 1))
	need.Mul(need, big.NewInt(int64(stride)))
	need.Add(need, big.NewInt(int64(width)))
	if need.Cmp(big.NewInt(int64(dataLen))) > 0 {
		return false
	}
	if !masked {
		return true
	}
	if validOffset < 0 {
		return false
	}
	need.Add(need, big.NewInt(int64(validOffset)))
	return need.Cmp(big.NewInt(int64(maskBits))) <= 0
}

// FuzzValidate checks that Validate accepts exactly the consistent
// rasters, including shapes whose cell count overflows int, and that
// every accessor works on the rasters it accepts.
func FuzzValidate(f *testing.F) {
	f.Add(3, 2, 4, uint16(7), false, uint16(0), 0)
	f.Add(1, 1<<32+1, 1<<32, uint16(1), false, uint16(0), 0)
	f.Add(2, 2, 2, uint16(4), true, uint16(1), 60)
	f.Add(2, 2, 2, uint16(4), true, uint16(1), 1<<63-2)
	f.Add(1<<62, 5, 1<<62, uint16(100), false, uint16(0), 0)
	f.Add(-1, 3, 3, uint16(9), false, uint16(0), 0)
	f.Fuzz(func(t *testing.T, width, height, stride int, dataLen uint16, masked bool, maskWords uint16, validOffset int) {
		r := Float32Raster{Data: make([]float32, dataLen), Width: width, Height: height, Stride: stride}
		if masked {
			r.Valid = make([]uint64, maskWords)
			r.ValidOffset = validOffset
		}
		want := exactShape(width, height, stride, int(dataLen), masked, 64*int(maskWords), validOffset)
		err := r.Validate()
		if (err == nil) != want {
			t.Fatalf("Validate(%d×%d stride %d, %d cells, mask %v %d words at %d) = %v, want valid %v",
				width, height, stride, dataLen, masked, maskWords, validOffset, err, want)
		}

		nopanic := !requireRasterPanic(t, "NewFloat32Stride", func() {
			n := NewFloat32Stride(width, height, stride, r.Data)
			if err := n.Validate(); err != nil {
				t.Fatalf("NewFloat32Stride returned an invalid raster: %v", err)
			}
		})
		if nopanic != exactShape(width, height, stride, int(dataLen), false, 0, 0) {
			t.Fatalf("NewFloat32Stride(%d, %d, %d, %d cells) panicked = %v", width, height, stride, dataLen, !nopanic)
		}
		if err != nil {
			return
		}

		// Every cell of an accepted raster is addressable.
		for _, y := range []int{0, height - 1} {
			if row := r.Row(y); len(row) != width {
				t.Fatalf("Row(%d) has %d cells, want %d", y, len(row), width)
			}
			for _, x := range []int{0, width - 1} {
				_ = r.Data[r.Index(x, y)]
				_ = r.IsValid(x, y)
				if masked {
					r.SetValid(x, y, true)
				}
				win := r.Window(x, y, width-x, height-y)
				if err := win.Validate(); err != nil {
					t.Fatalf("Window(%d, %d) of a valid raster: %v", x, y, err)
				}
			}
		}
		like := NewFloat32Like(r)
		if err := like.Validate(); err != nil {
			t.Fatalf("NewFloat32Like: %v", err)
		}
	})
}

// FuzzWindow checks Window and Grid.Window against their bounds for any
// arguments, and that windows of windows address their parent's cells and
// validity bits.
func FuzzWindow(f *testing.F) {
	f.Add([]byte{5, 4, 3, 7, 1, 1, 3, 2, 1, 0, 2, 1})
	f.Add([]byte{70, 3, 64, 100, 0, 0, 70, 3, 5, 1, 60, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		w, h := d.Range(1, 80), d.Range(1, 8)
		stride := w + d.IntN(80)
		off := d.IntN(130)
		n := (h-1)*stride + w
		root := NewFloat32Stride(w, h, stride, make([]float32, n))
		for i := range root.Data {
			root.Data[i] = float32(i)
		}
		root.Valid = make([]uint64, MaskWords(off+n))
		root.ValidOffset = off
		for k := range root.Valid {
			root.Valid[k] = d.Uint64()
		}
		grid := Grid{Width: w, Height: h, ResolutionX: 2, ResolutionY: -3, OriginX: 10, OriginY: 20}

		// Arguments near and past the bounds, including extremes.
		arg := func(limit int) int {
			switch d.IntN(8) {
			case 0:
				return -1
			case 1:
				return 1<<63 - 1
			case 2:
				return -1 << 63
			default:
				return d.Range(0, limit+1)
			}
		}
		cur, ox, oy := root, 0, 0
		for range 3 {
			x, y, ww, hh := arg(cur.Width), arg(cur.Height), arg(cur.Width), arg(cur.Height)
			inside := ww > 0 && hh > 0 && x >= 0 && y >= 0 && x <= cur.Width-ww && y <= cur.Height-hh
			var win Float32Raster
			if requireRasterPanic(t, "Window", func() { win = cur.Window(x, y, ww, hh) }) == inside {
				t.Fatalf("Window(%d, %d, %d, %d) of %d×%d: panicked = %v", x, y, ww, hh, cur.Width, cur.Height, inside)
			}
			g := Grid{Width: cur.Width, Height: cur.Height}
			if requireRasterPanic(t, "Grid.Window", func() { g.Window(x, y, ww, hh) }) == inside {
				t.Fatalf("Grid.Window(%d, %d, %d, %d) of %d×%d: panicked = %v", x, y, ww, hh, cur.Width, cur.Height, inside)
			}
			if !inside {
				return
			}
			ox, oy = ox+x, oy+y
			if err := win.Validate(); err != nil {
				t.Fatalf("window: %v", err)
			}
			for _, c := range [][2]int{{0, 0}, {ww - 1, hh - 1}, {d.IntN(ww), d.IntN(hh)}} {
				if got, want := win.Data[win.Index(c[0], c[1])], root.Data[root.Index(ox+c[0], oy+c[1])]; got != want {
					t.Fatalf("window cell %v = %v, root cell = %v", c, got, want)
				}
				if got, want := win.IsValid(c[0], c[1]), root.IsValid(ox+c[0], oy+c[1]); got != want {
					t.Fatalf("window validity %v = %v, root = %v", c, got, want)
				}
				win.SetValid(c[0], c[1], !win.IsValid(c[0], c[1]))
				if win.IsValid(c[0], c[1]) != root.IsValid(ox+c[0], oy+c[1]) {
					t.Fatalf("SetValid through a window not visible in the root")
				}
			}
			gw := grid.Window(ox, oy, ww, hh)
			if gw.OriginX != 10+2*float64(ox) || gw.OriginY != 20-3*float64(oy) || gw.Width != ww || gw.Height != hh {
				t.Fatalf("Grid.Window(%d, %d, %d, %d) = %+v", ox, oy, ww, hh, gw)
			}
			cur = win
		}
	})
}

// FuzzMaskRanges checks MaskFillRange, MaskCopyRange and MaskAndRange
// against bit-at-a-time references, for arbitrary offsets and lengths,
// in place at the same offset, and that they panic exactly when a range
// leaves its mask.
func FuzzMaskRanges(f *testing.F) {
	f.Add([]byte{2, 0, 3, 61, 70, 5, 0})
	f.Add([]byte{4, 1, 0, 64, 128, 64, 1})
	f.Add([]byte{3, 2, 1, 63, 1, 129, 2})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		words := func() []uint64 {
			m := make([]uint64, d.Range(0, 5))
			for k := range m {
				m[k] = d.Uint64()
			}
			return m
		}
		dst, a, b := words(), words(), words()
		op := d.IntN(3)
		inPlace := d.IntN(3) // 0: separate masks, 1: a is dst, 2: a and b are dst
		off := func(m []uint64) int { return d.Range(-2, len(m)*64+2) }
		dstOff, aOff, bOff := off(dst), off(a), off(b)
		n := d.Range(-2, 5*64+2)
		switch inPlace {
		case 1:
			a, aOff = dst, dstOff
		case 2:
			a, aOff, b, bOff = dst, dstOff, dst, dstOff
		}
		valid := d.Bool()

		inside := func(m []uint64, off int) bool { return off >= 0 && n >= 0 && off+n <= len(m)*64 }
		get := func(m []uint64, i int) bool { return m[i>>6]>>(i&63)&1 != 0 }
		want := append([]uint64(nil), dst...)
		ok := inside(dst, dstOff)
		switch op {
		case 0:
			if ok {
				for i := range n {
					MaskSet(want, dstOff+i, valid)
				}
			}
		case 1:
			ok = ok && inside(a, aOff)
			if ok {
				for i := range n {
					MaskSet(want, dstOff+i, get(a, aOff+i))
				}
			}
		case 2:
			ok = ok && inside(a, aOff) && inside(b, bOff)
			if ok {
				for i := range n {
					MaskSet(want, dstOff+i, get(a, aOff+i) && get(b, bOff+i))
				}
			}
		}

		panicked := requireRasterPanic(t, "mask range", func() {
			switch op {
			case 0:
				MaskFillRange(dst, dstOff, n, valid)
			case 1:
				MaskCopyRange(dst, dstOff, a, aOff, n)
			case 2:
				MaskAndRange(dst, dstOff, a, aOff, b, bOff, n)
			}
		})
		if panicked == ok {
			t.Fatalf("op %d: dst %d words at %d, a %d at %d, b %d at %d, n %d: panicked = %v",
				op, len(dst), dstOff, len(a), aOff, len(b), bOff, n, panicked)
		}
		for k := range want {
			if dst[k] != want[k] {
				t.Fatalf("op %d in place %d: dst %d words at %d, a at %d, b at %d, n %d: word %d = %#016x, want %#016x",
					op, inPlace, len(dst), dstOff, aOff, bOff, n, k, dst[k], want[k])
			}
		}
	})
}
