//go:build goexperiment.simd && amd64

package curve

import (
	"fmt"
	"math"
	"math/rand/v2"
	"simd/archsimd"
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
)

func requireAVX2(t testing.TB) {
	t.Helper()
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not available on this CPU")
	}
}

// hazardCells fills a row with the cells the vector kernels could get
// wrong: NaN (two payloads), the infinities, both zeros, every table
// entry exactly, one ulp either side of it, midpoints, and cells below
// and above the whole table, shuffled so each lands in every lane
// position across the lengths the tests run.
func hazardCells(rng *rand.Rand, n int, table []float32) []float32 {
	pool := []float32{
		nan, math.Float32frombits(0xffc0_0001), inf, ninf, 0, negZero,
		math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
	}
	for i, x := range table {
		pool = append(pool, x, math.Nextafter32(x, ninf), math.Nextafter32(x, inf))
		if i+1 < len(table) {
			pool = append(pool, x+(table[i+1]-x)/2)
		}
	}
	if len(table) > 0 {
		pool = append(pool, table[0]-1, table[len(table)-1]+1)
	}
	out := make([]float32, n)
	for i := range out {
		if rng.IntN(4) == 0 {
			out[i] = float32(rng.NormFloat64() * 100)
		} else {
			out[i] = pool[rng.IntN(len(pool))]
		}
	}
	return out
}

// hazardTable builds n strictly increasing knots or breaks, about a
// quarter of them one ulp from the next, and a y side that includes NaN,
// the infinities and both zeros when special is set.
func hazardTable(rng *rand.Rand, n int, special bool) (xs, ys []float32) {
	xs = make([]float32, n)
	ys = make([]float32, n)
	x := float32(rng.NormFloat64() * 50)
	for i := range xs {
		xs[i] = x
		switch rng.IntN(4) {
		case 0:
			x = math.Nextafter32(x, inf) // a one-ulp segment
		default:
			x += float32(rng.Float64()*20) + 0.001
		}
		ys[i] = float32(rng.NormFloat64() * 10)
		if special && rng.IntN(5) == 0 {
			ys[i] = []float32{nan, inf, ninf, negZero, 0}[rng.IntN(5)]
		}
	}
	return xs, ys
}

// TestAVX2MatchesScalar compares each AVX2 kernel against its scalar
// counterpart bit for bit, any NaN matching any NaN, over lengths that
// straddle the lane width and table sizes on both sides of the dispatch
// limits, so both the lane loop and the scalar hand-off are covered.
func TestAVX2MatchesScalar(t *testing.T) {
	requireAVX2(t)
	rng := rand.New(rand.NewPCG(41, 42))
	lengths := []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100}
	sizes := []int{1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, reclassVecMax, reclassVecMax + 1, lookupVecMax, lookupVecMax + 1, 70}
	for _, size := range sizes {
		for _, special := range []bool{false, true} {
			xs, ys := hazardTable(rng, size, special)
			for _, n := range lengths {
				src := hazardCells(rng, n, xs)
				want, got := make([]float32, n), make([]float32, n)

				name := fmt.Sprintf("Reclass/breaks=%d/special=%v/n=%d", size-1, special, n)
				scalarReclassFloat32(want, src, xs[:size-1], ys)
				reclassFloat32AVX2(got, src, xs[:size-1], ys)
				assertSlicesEqual(t, name, got, want)

				name = fmt.Sprintf("Lookup/knots=%d/special=%v/n=%d", size, special, n)
				scalarLookupFloat32(want, src, xs, ys)
				lookupFloat32AVX2(got, src, xs, ys)
				assertSlicesEqual(t, name, got, want)
			}
		}
	}
}

// TestAVX2NaNPayload checks the vector path, not only the scalar tail,
// returns a NaN cell's own bits.
func TestAVX2NaNPayload(t *testing.T) {
	requireAVX2(t)
	src := make([]float32, 3*lane)
	for i := range src {
		src[i] = math.Float32frombits(0x7fc0_0000 | uint32(i+1))
		if i%2 == 1 {
			src[i] = math.Float32frombits(0xffc0_0000 | uint32(i+1))
		}
	}
	dst := make([]float32, len(src))
	reclassFloat32AVX2(dst, src, []float32{0}, []float32{1, 2})
	for i := range dst {
		if math.Float32bits(dst[i]) != math.Float32bits(src[i]) {
			t.Fatalf("Reclass cell %d: got %#08x, want %#08x", i, math.Float32bits(dst[i]), math.Float32bits(src[i]))
		}
	}
	lookupFloat32AVX2(dst, src, []float32{0, 1}, []float32{1, 2})
	for i := range dst {
		if math.Float32bits(dst[i]) != math.Float32bits(src[i]) {
			t.Fatalf("Lookup cell %d: got %#08x, want %#08x", i, math.Float32bits(dst[i]), math.Float32bits(src[i]))
		}
	}
}

// FuzzAVX2MatchesScalar runs both AVX2 kernels against the scalar ones
// on arbitrary cells and arbitrary valid tables of up to 70 entries, past
// both dispatch limits, and requires nothing outside dst to change.
func FuzzAVX2MatchesScalar(f *testing.F) {
	f.Add([]byte{0, 17, 5, 1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{1, 40, 9, 9, 1, 1, 2, 2, 3, 3, 4, 4})
	f.Add([]byte{1, 8, 70, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		requireAVX2(t)
		d := fuzzdata.New(data)
		lookup := d.Bool()
		n := d.Range(0, 80)
		size := d.Range(1, 70)

		const guard = 9
		backing := make([]float32, n+2*guard)
		for i := range backing {
			backing[i] = d.Float32()
		}
		before := append([]float32(nil), backing...)
		xs := increasing(d, size)
		ys := arbitrary(d, size)
		src := arbitrary(d, n)
		// Put some cells on the table itself, where the answers switch.
		for i := range src {
			if d.IntN(3) == 0 {
				src[i] = xs[d.IntN(size)]
			}
		}
		dst := backing[guard : guard+n]
		want := make([]float32, n)
		if lookup {
			scalarLookupFloat32(want, src, xs, ys)
			lookupFloat32AVX2(dst, src, xs, ys)
		} else {
			scalarReclassFloat32(want, src, xs[:size-1], ys)
			reclassFloat32AVX2(dst, src, xs[:size-1], ys)
		}
		for i := range want {
			if !same(dst[i], want[i]) {
				t.Fatalf("lookup=%v n=%d size=%d: cell %d (%v, %#08x) = %v (%#08x), scalar %v (%#08x)\nxs %v\nys %v",
					lookup, n, size, i, src[i], math.Float32bits(src[i]),
					dst[i], math.Float32bits(dst[i]), want[i], math.Float32bits(want[i]), xs, ys)
			}
		}
		for i := range backing {
			if (i < guard || i >= guard+n) && math.Float32bits(backing[i]) != math.Float32bits(before[i]) {
				t.Fatalf("lookup=%v n=%d: wrote backing element %d outside dst", lookup, n, i)
			}
		}
	})
}

// TestBackendSelection checks that the SIMD set is installed at init,
// that UseScalar(true) replaces all of it, and that UseScalar(false)
// puts every kernel back.
func TestBackendSelection(t *testing.T) {
	requireAVX2(t)
	defer UseScalar(false)
	if Backend() != "avx2" {
		t.Fatalf("Backend() = %q, want avx2", Backend())
	}
	assertKernels(t, "init", kernelsInUse(), *simdKernels)
	assertKernels(t, "SIMD set", *simdKernels, kernelSet{reclass: reclassFloat32AVX2, lookup: lookupFloat32AVX2})
	UseScalar(true)
	if Backend() != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", Backend())
	}
	assertKernels(t, "UseScalar(true)", kernelsInUse(), scalarKernels)
	UseScalar(false)
	if Backend() != "avx2" {
		t.Fatalf("after UseScalar(false), Backend() = %q", Backend())
	}
	assertKernels(t, "UseScalar(false)", kernelsInUse(), *simdKernels)
}
