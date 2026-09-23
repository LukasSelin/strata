//go:build goexperiment.simd && amd64

package kern

import (
	"math"
	"math/rand/v2"
	"simd/archsimd"
	"slices"
	"testing"
)

// The AVX2 kernels against the scalar ones they replace, bit for bit.

func needAVX2(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("no AVX2")
	}
}

func TestPlanesRowAVX2(t *testing.T) {
	needAVX2(t)
	rng := rand.New(rand.NewPCG(17, 18))
	for _, n := range []int{1, 7, 15, 16, 17, 31, 64, 100, 512, 701} {
		row := make([]byte, 4*n)
		for i := range row {
			row[i] = byte(rng.Uint32())
		}
		want, got := make([]float32, n), make([]float32, n)
		scalarPlanesRow(want, slices.Clone(row))
		planesRowAVX2(got, slices.Clone(row))
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%d samples: sample %d has bits %08x, want %08x", n, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

func TestUint16RowAVX2(t *testing.T) {
	needAVX2(t)
	rng := rand.New(rand.NewPCG(19, 20))
	for _, n := range []int{1, 7, 8, 9, 63, 64, 100, 512, 701} {
		row := make([]byte, 2*n)
		for i := range row {
			row[i] = byte(rng.Uint32())
		}
		for _, signed := range []bool{false, true} {
			for _, pred := range []bool{false, true} {
				want, got := make([]float32, n), make([]float32, n)
				scalarUint16Row(want, slices.Clone(row), signed, pred)
				uint16RowAVX2(got, slices.Clone(row), signed, pred)
				if !slices.Equal(got, want) {
					t.Fatalf("%d samples, signed %v, pred %v: %v, want %v", n, signed, pred, got[:min(n, 8)], want[:min(n, 8)])
				}
			}
		}
	}
}

// TestWordAVX2 tries each mode on values at, next to and far from the
// NoData values, and on the special ones: ±0, NaNs, ±Inf.
func TestWordAVX2(t *testing.T) {
	needAVX2(t)
	rng := rand.New(rand.NewPCG(21, 22))
	nds := []float32{-9999, 0.1, 12, 1e-40, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1))}
	type test struct {
		mode           int
		want, care, lo uint32
		span           uint64
	}
	var tests []test
	for _, v := range nds {
		b := math.Float32bits(v)
		tests = append(tests, test{mode: ModeExact, want: b, care: ^uint32(0)},
			test{mode: ModeRange, lo: b - 2, span: 4})
	}
	tests = append(tests, test{mode: ModeExact, want: 0, care: 0x7fffffff}, test{mode: ModeNaN},
		test{mode: ModeRange, lo: 0x7f7ffffd, span: 2}) // up to MaxFloat32
	for range 300 {
		chunk := make([]float32, 64)
		for i := range chunk {
			switch rng.IntN(6) {
			case 0:
				chunk[i] = nds[rng.IntN(len(nds))]
			case 1: // a neighbour of a NoData value, inside or outside its run
				v := nds[rng.IntN(len(nds))]
				chunk[i] = math.Float32frombits(math.Float32bits(v) + uint32(rng.IntN(7)) - 3)
			case 2:
				chunk[i] = []float32{0, float32(math.Copysign(0, -1)), float32(math.NaN()),
					math.Float32frombits(0xffc00001)}[rng.IntN(4)]
			default:
				chunk[i] = float32(rng.NormFloat64() * 1000)
			}
		}
		for _, tt := range tests {
			got := wordAVX2(chunk, tt.mode, tt.want, tt.care, tt.lo, tt.span)
			if want := scalarWord(chunk, tt.mode, tt.want, tt.care, tt.lo, tt.span); got != want {
				t.Fatalf("test %+v: %064b, want %064b", tt, got, want)
			}
		}
	}
}
