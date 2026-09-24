package kern

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// The installed kernels against the scalar ones, bit for bit, on
// whichever SIMD set this build and CPU have: AVX2, NEON, or none, when
// both sides are scalar and the tests only pin the scalar kernels'
// own contracts. Rows are random bytes, of every width up to a few
// vectors past the widest step and some wider ragged ones, so every tail
// length follows every number of whole vectors.

// widths are every row width to 300 samples, then some wider ones, whole
// and ragged.
func widths() []int {
	var ws []int
	for n := range 301 {
		ws = append(ws, n)
	}
	return append(ws, 512, 1000, 1024, 4093, 4096+31)
}

func randomBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	return b
}

// both runs k on scalar kernels, then on the installed ones, on copies
// of row, and returns each side's vals and row afterwards.
func both(n int, row []byte, k func(vals []float32, row []byte)) (sv, gv []float32, sr, gr []byte) {
	sv, gv = make([]float32, n), make([]float32, n)
	sr, gr = slices.Clone(row), slices.Clone(row)
	UseScalar(true)
	k(sv, sr)
	UseScalar(false)
	k(gv, gr)
	return sv, gv, sr, gr
}

func sameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s (%s): sample %d of %d has bits %08x, want %08x",
				what, Backend(), i, len(want), math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestPlanesRowBackends(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	for _, n := range widths() {
		row := randomBytes(rng, 4*n)
		want, got, _, _ := both(n, row, PlanesRow)
		sameBits(t, "PlanesRow", got, want)
	}
}

// TestPlanesRowReference holds the scalar kernel to libtiff's fpAcc,
// written out a byte at a time, so the backends agree with the format
// and not only with each other.
func TestPlanesRowReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(33, 34))
	for _, n := range []int{0, 1, 5, 32, 33, 512} {
		row := randomBytes(rng, 4*n)
		ref := slices.Clone(row)
		for i := 1; i < len(ref); i++ {
			ref[i] += ref[i-1]
		}
		want := make([]float32, n)
		for i := range want {
			want[i] = math.Float32frombits(uint32(ref[3*n+i]) | uint32(ref[2*n+i])<<8 | uint32(ref[n+i])<<16 | uint32(ref[i])<<24)
		}
		for _, scalar := range []bool{true, false} {
			UseScalar(scalar)
			got := make([]float32, n)
			PlanesRow(got, slices.Clone(row))
			sameBits(t, "PlanesRow against fpAcc", got, want)
		}
	}
	UseScalar(false)
}

func TestSumBytesBackends(t *testing.T) {
	rng := rand.New(rand.NewPCG(35, 36))
	for _, n := range widths() {
		row := randomBytes(rng, n)
		_, _, want, got := both(0, row, func(_ []float32, r []byte) { SumBytes(r) })
		if !slices.Equal(got, want) {
			t.Fatalf("SumBytes (%s), %d bytes: differs from the scalar sum", Backend(), n)
		}
	}
}

func TestUint8RowBackends(t *testing.T) {
	rng := rand.New(rand.NewPCG(37, 38))
	for _, n := range widths() {
		row := randomBytes(rng, n)
		for _, signed := range []bool{false, true} {
			for _, pred := range []bool{false, true} {
				want, got, _, _ := both(n, row, func(v []float32, r []byte) { Uint8Row(v, r, signed, pred) })
				sameBits(t, "Uint8Row", got, want)
			}
		}
	}
}

func TestUint16RowBackends(t *testing.T) {
	rng := rand.New(rand.NewPCG(39, 40))
	for _, n := range widths() {
		row := randomBytes(rng, 2*n)
		for _, signed := range []bool{false, true} {
			for _, pred := range []bool{false, true} {
				want, got, _, _ := both(n, row, func(v []float32, r []byte) { Uint16Row(v, r, signed, pred) })
				sameBits(t, "Uint16Row", got, want)
			}
		}
	}
}

func TestCopyRowBackends(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	for _, n := range widths() {
		row := randomBytes(rng, 4*n)
		want, got, _, _ := both(n, row, CopyRow)
		sameBits(t, "CopyRow", got, want)
	}
}

// FuzzKernels gives every row kernel the same bytes on both backends.
// The first byte picks the flags; the rest is the row, cut to whole
// samples for each kernel.
func FuzzKernels(f *testing.F) {
	rng := rand.New(rand.NewPCG(43, 44))
	for _, n := range []int{0, 1, 17, 64, 129, 2048 + 7} {
		f.Add(randomBytes(rng, n))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		signed, pred := data[0]&1 != 0, data[0]&2 != 0
		row := data[1:]
		defer UseScalar(false)

		n := len(row) / 4
		want, got, _, _ := both(n, row[:4*n], PlanesRow)
		sameBits(t, "PlanesRow", got, want)
		want, got, _, _ = both(n, row[:4*n], CopyRow)
		sameBits(t, "CopyRow", got, want)

		_, _, ws, gs := both(0, row, func(_ []float32, r []byte) { SumBytes(r) })
		if !slices.Equal(gs, ws) {
			t.Fatalf("SumBytes (%s), %d bytes: differs from the scalar sum", Backend(), len(row))
		}

		want, got, _, _ = both(len(row), row, func(v []float32, r []byte) { Uint8Row(v, r, signed, pred) })
		sameBits(t, "Uint8Row", got, want)

		n = len(row) / 2
		want, got, _, _ = both(n, row[:2*n], func(v []float32, r []byte) { Uint16Row(v, r, signed, pred) })
		sameBits(t, "Uint16Row", got, want)
	})
}
