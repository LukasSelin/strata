package vec

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
)

// validRef is ValidBits's documentation for one cell, written from it
// rather than from scalarValidBits: invalid where the cell equals the
// fill as a float, or is any NaN when the fill is one.
func validRef(c, fill float32) bool {
	if math.IsNaN(float64(fill)) {
		return !math.IsNaN(float64(c))
	}
	return float64(c) != float64(fill)
}

// checkValidBits runs ValidBits on src over a dst full of garbage and
// requires every bit to match validRef, the bits past len(src) cleared.
func checkValidBits(t *testing.T, src []float32, fill float32) {
	t.Helper()
	dst := make([]uint64, (len(src)+63)/64)
	for i := range dst {
		dst[i] = 0xdead_beef_f00d_cafe
	}
	ValidBits(dst, src, fill)
	for i, c := range src {
		got := dst[i>>6]>>(i&63)&1 != 0
		if want := validRef(c, fill); got != want {
			t.Fatalf("%s, fill %v (%#08x), n %d: cell %d = %v (%#08x): valid %v, want %v",
				Backend(), fill, math.Float32bits(fill), len(src), i, c, math.Float32bits(c), got, want)
		}
	}
	if r := len(src) & 63; r != 0 && dst[len(dst)-1]>>r != 0 {
		t.Fatalf("%s, n %d: bits past the last cell set: %#x", Backend(), len(src), dst[len(dst)-1])
	}
}

// validFills are the fill values the tests use: the zeros, NaNs with
// and without payloads, infinities, a subnormal, and typical NoData.
var validFills = []float32{
	0, negZero, nan,
	math.Float32frombits(0xffc0_0001), math.Float32frombits(0x7f80_0001),
	inf, ninf, math.SmallestNonzeroFloat32,
	-9999, 65535, -3.4028235e38, 1000.25,
}

// validCells returns n cells of which about one in four is a hard case
// for fill: the fill itself, its other zero, a NaN of some payload, or
// a neighbour one ulp away.
func validCells(rng *rand.Rand, n int, fill float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		switch rng.IntN(12) {
		case 0:
			s[i] = fill
		case 1:
			s[i] = -fill // -0 for 0, and a NaN of the other sign
		case 2:
			s[i] = math.Float32frombits(0x7f80_0000 | uint32(rng.IntN(1<<23)) | 1) // a NaN
		case 3:
			s[i] = math.Nextafter32(fill, inf)
		case 4:
			s[i] = math.Float32frombits(rng.Uint32())
		default:
			s[i] = float32(rng.NormFloat64() * 1000)
		}
	}
	return s
}

// TestValidBits holds both backends to validRef over random rows with
// fills, NaNs, both zeros and ragged tails, at every offset of a backing
// array, so lanes start at every alignment.
func TestValidBits(t *testing.T) {
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(3, 1))
	for _, scalar := range []bool{true, false} {
		UseScalar(scalar)
		for _, fill := range validFills {
			for _, n := range []int{0, 1, 7, 8, 9, 31, 63, 64, 65, 127, 128, 129, 200, 511, 512, 1000} {
				back := validCells(rng, n+7, fill)
				for off := range 8 {
					checkValidBits(t, back[off:off+n], fill)
				}
			}
		}
	}
}

// TestValidBitsSIMDMatchesScalar compares the backends' words directly
// on random rows, the property the SIMD contract states.
func TestValidBitsSIMDMatchesScalar(t *testing.T) {
	defer UseScalar(false)
	if simdKernels == nil {
		t.Skip("no SIMD kernels in this build or on this CPU")
	}
	rng := rand.New(rand.NewPCG(5, 9))
	for range 200 {
		fill := validFills[rng.IntN(len(validFills))]
		src := validCells(rng, rng.IntN(700), fill)
		words := (len(src) + 63) / 64
		want, got := make([]uint64, words), make([]uint64, words)
		scalarValidBits(want, src, fill)
		simdKernels.validBits(got, src, fill)
		for w := range want {
			if got[w] != want[w] {
				t.Fatalf("fill %v, n %d, word %d: %s %#016x, scalar %#016x", fill, len(src), w, simdName, got[w], want[w])
			}
		}
	}
}

func TestValidBitsPanicsOnLength(t *testing.T) {
	for _, c := range []struct{ words, cells int }{{0, 1}, {1, 0}, {1, 65}, {2, 64}} {
		func() {
			defer func() {
				v := recover()
				if msg, ok := v.(string); !ok || !strings.HasPrefix(msg, "vec: ") {
					t.Errorf("%d words for %d cells: panic %v, want a \"vec: \" message", c.words, c.cells, v)
				}
			}()
			ValidBits(make([]uint64, c.words), make([]float32, c.cells), 0)
		}()
	}
}

// FuzzValidBits runs ValidBits on both backends over arbitrary bit
// patterns, lengths around the word and lane boundaries, and slices at
// any offset, and requires validRef's bits and nothing written past dst.
func FuzzValidBits(f *testing.F) {
	f.Add([]byte{0, 70, 3, 5, 0})
	f.Add([]byte{9, 64, 0, 0, 4, 200, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		defer UseScalar(false)
		d := fuzzdata.New(data)
		fill := d.Float32()
		n := d.Range(0, 200)
		off := d.IntN(8)
		back := make([]float32, n+off)
		for i := range back {
			if d.IntN(4) == 0 {
				back[i] = fill
			} else {
				back[i] = d.Float32()
			}
		}
		src := back[off:]
		for _, scalar := range []bool{true, false} {
			UseScalar(scalar)
			words := (n + 63) / 64
			guard := make([]uint64, words+2)
			guard[0], guard[words+1] = 1, 2
			dst := guard[1 : words+1 : words+1]
			checkValidBitsInto(t, dst, src, fill)
			if guard[0] != 1 || guard[words+1] != 2 {
				t.Fatalf("%s, n %d: wrote outside dst", Backend(), n)
			}
		}
	})
}

func checkValidBitsInto(t *testing.T, dst []uint64, src []float32, fill float32) {
	t.Helper()
	ValidBits(dst, src, fill)
	for i, c := range src {
		if got, want := dst[i>>6]>>(i&63)&1 != 0, validRef(c, fill); got != want {
			t.Fatalf("%s, fill %#08x, n %d: cell %d (%#08x): valid %v, want %v",
				Backend(), math.Float32bits(fill), len(src), i, math.Float32bits(c), got, want)
		}
	}
	if r := len(src) & 63; r != 0 && dst[len(dst)-1]>>r != 0 {
		t.Fatalf("%s, n %d: bits past the last cell set", Backend(), len(src))
	}
}

// BenchmarkValidBits times one row of float32 cells, most valid, against
// an ordinary fill and a NaN one, on both backends.
//
//	GOEXPERIMENT=simd go test -run - -bench ValidBits ./internal/vec
func BenchmarkValidBits(b *testing.B) {
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(1, 2))
	const n = 11264 // a benchmark raster's row
	src := validCells(rng, n, -9999)
	dst := make([]uint64, (n+63)/64)
	for _, scalar := range []bool{true, false} {
		UseScalar(scalar)
		for _, fill := range []float32{-9999, nan} {
			b.Run(fmt.Sprintf("backend=%s/fill=%v", Backend(), fill), func(b *testing.B) {
				b.SetBytes(4 * n)
				for b.Loop() {
					ValidBits(dst, src, fill)
				}
			})
		}
	}
}

// TestValidWord holds ValidWord to validRef on every length up to a word,
// and to a panic past it.
func TestValidWord(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 6))
	for _, fill := range validFills {
		for n := range 65 {
			src := validCells(rng, n, fill)
			w := ValidWord(src, fill)
			for i, c := range src {
				if got := w>>i&1 != 0; got != validRef(c, fill) {
					t.Fatalf("fill %#08x, n %d: cell %d (%#08x): valid %v", math.Float32bits(fill), n, i, math.Float32bits(c), got)
				}
			}
			if n < 64 && w>>n != 0 {
				t.Fatalf("fill %#08x, n %d: bits past the last cell set: %#x", math.Float32bits(fill), n, w)
			}
		}
	}
	defer func() {
		if msg, ok := recover().(string); !ok || !strings.HasPrefix(msg, "vec: ") {
			t.Errorf("65 cells: panic %q, want a \"vec: \" message", msg)
		}
	}()
	ValidWord(make([]float32, 65), 0)
}
