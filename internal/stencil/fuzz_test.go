package stencil

import (
	"math"
	"strings"
	"testing"

	"strata/internal/fuzzdata"
)

// FuzzErodeBox checks ErodeBox against a per-cell reference for any
// radius up to 3, one to three sources at arbitrary offsets and strides,
// radius 0 in place, and destinations whose own strides and offsets
// differ; that bits outside the destination's cells never change; and
// that a mask too short for its region panics with a "stencil:" message
// before anything is written.
func FuzzErodeBox(f *testing.F) {
	f.Add([]byte{1, 5, 3, 1, 0, 0, 0, 0})
	f.Add([]byte{63, 2, 1, 2, 1, 64, 3, 1})
	f.Add([]byte{0, 0, 0, 0, 1, 1, 1, 1, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		w, h, r := d.Range(1, 140), d.Range(1, 6), d.Range(0, 3)
		sw, sh := w+2*r, h+2*r
		region := func(width, height int) MaskRegion {
			off, stride := d.IntN(200), width+d.IntN(80)
			words := (off+(height-1)*stride+width+63)>>6 + d.IntN(3)
			if d.IntN(10) == 0 {
				words -= 1 + d.IntN(2) // may be too short
			}
			bits := make([]uint64, max(words, 0))
			for k := range bits {
				switch d.IntN(3) {
				case 0:
					bits[k] = d.Uint64()
				case 1:
					bits[k] = ^uint64(0) &^ (1 << d.IntN(64))
				default:
					bits[k] = ^uint64(0)
				}
			}
			return MaskRegion{bits, off, stride}
		}
		fits := func(m MaskRegion, width, height int) bool {
			return len(m.Bits)*64 >= m.Off+(height-1)*m.Stride+width
		}
		srcs := make([]MaskRegion, d.Range(1, 3))
		for i := range srcs {
			srcs[i] = region(sw, sh)
		}
		dst := region(w, h)
		if r == 0 && d.Bool() {
			dst = srcs[d.IntN(len(srcs))] // in place
		}
		ok := fits(dst, w, h)
		for _, s := range srcs {
			ok = ok && fits(s, sw, sh)
		}

		get := func(m MaskRegion, x, y int) bool {
			i := m.Off + y*m.Stride + x
			return m.Bits[i>>6]>>uint(i&63)&1 != 0
		}
		var want []bool
		if ok {
			want = make([]bool, w*h)
			for y := range h {
				for x := range w {
					v := true
					for _, s := range srcs {
						for j := 0; v && j <= 2*r; j++ {
							for i := 0; v && i <= 2*r; i++ {
								v = get(s, x+i, y+j)
							}
						}
					}
					want[y*w+x] = v
				}
			}
		}
		before := append([]uint64(nil), dst.Bits...)
		panicked := func() (p bool) {
			defer func() {
				if v := recover(); v != nil {
					p = true
					if msg, isStr := v.(string); !isStr || !strings.HasPrefix(msg, "stencil: ") {
						t.Fatalf("panic %v, want a \"stencil: \" message", v)
					}
				}
			}()
			ErodeBox(dst, srcs, w, h, r, make([]uint64, ErodeScratch(w, r)))
			return false
		}()
		if panicked == ok {
			t.Fatalf("w %d h %d r %d: regions fit = %v but panicked = %v", w, h, r, ok, panicked)
		}
		if panicked {
			for k := range before {
				if dst.Bits[k] != before[k] {
					t.Fatalf("a panicking ErodeBox changed word %d", k)
				}
			}
			return
		}
		for y := range h {
			for x := range w {
				if got := get(dst, x, y); got != want[y*w+x] {
					t.Fatalf("w %d h %d r %d srcs %d: cell (%d, %d) = %v, want %v", w, h, r, len(srcs), x, y, got, want[y*w+x])
				}
			}
		}
		for i := range len(before) * 64 {
			j := i - dst.Off
			inDst := j >= 0 && j/dst.Stride < h && j%dst.Stride < w
			if !inDst && (dst.Bits[i>>6]^before[i>>6])>>uint(i&63)&1 != 0 {
				t.Fatalf("w %d h %d r %d: bit %d outside the destination changed", w, h, r, i)
			}
		}
	})
}

// FuzzHornRows runs every Horn row kernel on the current backend and on
// the scalar one over arbitrary elevations and parameters, and requires
// identical results (any NaN matching any NaN), no reads past n+2 input
// cells (the extra cells are poisoned) and no writes past n.
func FuzzHornRows(f *testing.F) {
	f.Add([]byte{0, 9, 1})
	f.Add([]byte{1, 16, 0, 1})
	f.Add([]byte{2, 33, 1, 0})
	f.Add([]byte{3, 8, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		defer UseScalar(false)
		d := fuzzdata.New(data)
		kind, n := d.IntN(4), d.Range(0, 40)
		const extra = 3
		rows := make([][]float32, 3)
		for k := range rows {
			rows[k] = make([]float32, n+2+extra)
			for i := range n + 2 {
				rows[k][i] = d.Float32()
			}
		}
		param := func() float32 {
			if d.IntN(3) == 0 {
				return d.Float32()
			}
			return float32(int8(d.Byte())) / 16
		}
		kx, ky, p1, p2, p3 := param(), param(), param(), param(), param()
		flag := d.Bool()

		// run poisons the cells past n+2 with poison, so that a backend
		// reading them disagrees with a run poisoned differently.
		run := func(poison float32) (a, b []float32) {
			for _, r := range rows {
				for i := n + 2; i < len(r); i++ {
					r[i] = poison
				}
			}
			a, b = make([]float32, n+extra), make([]float32, n+extra)
			for i := n; i < len(a); i++ {
				a[i], b[i] = 7, 7
			}
			dst := a[:n]
			switch kind {
			case 0:
				HornGradientRow(dst, b[:n], rows[0], rows[1], rows[2], kx, ky)
			case 1:
				HornSlopeRow(dst, rows[0], rows[1], rows[2], kx, ky, p1, flag)
			case 2:
				HornAspectRow(dst, rows[0], rows[1], rows[2], kx, ky, p1, flag)
			case 3:
				HornHillshadeRow(dst, rows[0], rows[1], rows[2], kx, ky, p1, p2, p3)
			}
			return a, b
		}
		UseScalar(true)
		wa, wb := run(0)
		UseScalar(false)
		ga, gb := run(1e30)
		for i := range wa {
			if !sameResult(ga[i], wa[i]) || !sameResult(gb[i], wb[i]) {
				t.Fatalf("kind %d n %d kx %v ky %v params %v %v %v %v: cell %d = %v, %v on %s, want %v, %v (scalar)",
					kind, n, kx, ky, p1, p2, p3, flag, i, ga[i], gb[i], Backend(), wa[i], wb[i])
			}
		}
		for i := n; i < len(wa); i++ {
			if wa[i] != 7 || ga[i] != 7 || wb[i] != 7 || gb[i] != 7 {
				t.Fatalf("kind %d n %d: wrote past the output at %d", kind, n, i)
			}
		}
	})
}

// sameResult compares bitwise, except that any NaN matches any NaN.
func sameResult(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// FuzzAtan checks the documented accuracy and special cases of Atan32
// (for x >= 0) and Atan2F32 against package math.
func FuzzAtan(f *testing.F) {
	f.Add(float32(1), float32(1))
	f.Add(float32(0), float32(math.Copysign(0, -1)))
	f.Add(float32(math.Inf(-1)), float32(math.Inf(1)))
	f.Add(float32(-3e-39), float32(-1e38))
	f.Add(float32(2.547), float32(-0.4142135))
	f.Fuzz(func(t *testing.T, y, x float32) {
		if ax := float32(math.Abs(float64(x))); !math.IsNaN(float64(ax)) {
			got, want := Atan32(ax), math.Atan(float64(ax))
			if math.Abs(float64(got)-want) > 1.5e-7 {
				t.Fatalf("Atan32(%v) = %v, math.Atan = %v, error %.3g", ax, got, want, math.Abs(float64(got)-want))
			}
		}
		got, want := Atan2F32(y, x), math.Atan2(float64(y), float64(x))
		if math.IsNaN(want) != (got != got) {
			t.Fatalf("Atan2F32(%v, %v) = %v, math.Atan2 = %v", y, x, got, want)
		}
		if math.IsNaN(want) {
			return
		}
		if math.Signbit(float64(got)) != math.Signbit(want) {
			t.Fatalf("Atan2F32(%v, %v) = %v, math.Atan2 = %v: sign differs", y, x, got, want)
		}
		if e := math.Abs(float64(got) - want); e > 3e-7 {
			t.Fatalf("Atan2F32(%v, %v) = %v, math.Atan2 = %v, error %.3g", y, x, got, want, e)
		}
	})
}
