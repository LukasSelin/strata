package stencil

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestAtan32Accuracy(t *testing.T) {
	// A stride through every non-negative finite float32 bit pattern, plus
	// the neighbourhoods of the reduction thresholds. The exhaustive run
	// quoted in Atan32's documentation found a maximum of 1.41e-7.
	const bound = 1.5e-7
	check := func(x float32) {
		got := float64(Atan32(x))
		if e := math.Abs(got - math.Atan(float64(x))); e > bound {
			t.Fatalf("Atan32(%g) = %g, math.Atan = %g, error %.3g > %.3g",
				x, got, math.Atan(float64(x)), e, bound)
		}
	}
	for b := uint32(0); b < 0x7F800000; b += 9973 {
		check(math.Float32frombits(b))
	}
	for _, th := range []float32{atanTanPi8, atanTan3Pi8, 1} {
		x := th
		for range 1000 {
			x = math.Nextafter32(x, 0)
		}
		for range 2000 {
			check(x)
			x = math.Nextafter32(x, 10)
		}
	}
	if got := Atan32(0); math.Float32bits(got) != 0 {
		t.Errorf("Atan32(+0) = %g (%#x), want +0", got, math.Float32bits(got))
	}
	if got := Atan32(float32(math.Inf(1))); got != float32(math.Pi/2) {
		t.Errorf("Atan32(+Inf) = %g, want π/2", got)
	}
	if got := Atan32(float32(math.NaN())); got == got {
		t.Errorf("Atan32(NaN) = %g, want NaN", got)
	}
}

func TestRowLengthPanics(t *testing.T) {
	mustPanic(t, "short row", func() {
		HornSlopeRow(make([]float32, 4), make([]float32, 6), make([]float32, 5), make([]float32, 6), 1, 1, 1, false)
	})
	mustPanic(t, "dx/dy length", func() {
		HornGradientRow(make([]float32, 4), make([]float32, 3), make([]float32, 6), make([]float32, 6), make([]float32, 6), 1, 1)
	})
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: did not panic", name)
		}
	}()
	f()
}

// naiveErode is the per-cell reference for Erode3x3.
func naiveErode(dst []uint64, dstOff, dstStride int, src []uint64, srcOff, srcStride, w, h int) {
	get := func(x, y int) bool {
		i := srcOff + y*srcStride + x
		return src[i>>6]>>uint(i&63)&1 != 0
	}
	for y := range h {
		for x := range w {
			ok := x > 0 && y > 0 && x < w-1 && y < h-1
			for dy := -1; ok && dy <= 1; dy++ {
				for dx := -1; ok && dx <= 1; dx++ {
					ok = get(x+dx, y+dy)
				}
			}
			i := dstOff + y*dstStride + x
			if ok {
				dst[i>>6] |= 1 << uint(i&63)
			} else {
				dst[i>>6] &^= 1 << uint(i&63)
			}
		}
	}
}

func TestErode3x3MatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, w := range []int{1, 2, 3, 4, 7, 62, 63, 64, 65, 66, 127, 128, 129, 200} {
		for _, h := range []int{1, 2, 3, 4, 7} {
			for _, extra := range []int{0, 1, 13, (64 - w%64) % 64} {
				stride := w + extra
				for _, offs := range [][2]int{{0, 0}, {1, 63}, {63, 1}, {64, 100}, {37, 64}} {
					n := offs[0] + offs[1] + (h-1)*stride + w + 64
					words := (n + 63) >> 6
					src := make([]uint64, words)
					for k := range src {
						// Mostly-valid masks so interior cells survive the
						// erosion often enough to be checked.
						src[k] = rng.Uint64() | rng.Uint64() | rng.Uint64()
					}
					dst := make([]uint64, words)
					for k := range dst {
						dst[k] = rng.Uint64()
					}
					want := append([]uint64(nil), dst...)
					naiveErode(want, offs[1], stride, src, offs[0], stride, w, h)
					Erode3x3(dst, offs[1], stride, src, offs[0], stride, w, h)
					for k := range dst {
						if dst[k] != want[k] {
							t.Fatalf("w=%d h=%d stride=%d offs=%v: word %d = %#016x, want %#016x",
								w, h, stride, offs, k, dst[k], want[k])
						}
					}
				}
			}
		}
	}
}

func TestErode3x3DifferentStrides(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, w := range []int{63, 64, 65} {
		h := 6
		srcStride, dstStride := w+5, 64*((w+63)/64)
		src := make([]uint64, 16)
		dst := make([]uint64, 16)
		for k := range src {
			src[k] = rng.Uint64() | rng.Uint64()
			dst[k] = rng.Uint64()
		}
		want := append([]uint64(nil), dst...)
		naiveErode(want, 3, dstStride, src, 70, srcStride, w, h)
		Erode3x3(dst, 3, dstStride, src, 70, srcStride, w, h)
		for k := range dst {
			if dst[k] != want[k] {
				t.Fatalf("w=%d: word %d = %#016x, want %#016x", w, k, dst[k], want[k])
			}
		}
	}
}

func TestClearBorder(t *testing.T) {
	const w, h, stride, off = 65, 4, 70, 5
	m := make([]uint64, 8)
	for k := range m {
		m[k] = ^uint64(0)
	}
	ClearBorder(m, off, stride, w, h)
	for i := range len(m) * 64 {
		set := m[i>>6]>>uint(i&63)&1 != 0
		want := true
		if j := i - off; j >= 0 && j < (h-1)*stride+w && j%stride < w {
			x, y := j%stride, j/stride
			want = !(x == 0 || y == 0 || x == w-1 || y == h-1)
		}
		if set != want {
			t.Fatalf("bit %d = %v, want %v", i, set, want)
		}
	}
}
