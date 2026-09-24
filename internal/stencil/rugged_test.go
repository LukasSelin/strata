package stencil

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// windowRows returns 2r+1 rows of n+2r cells or a few more, filled as
// TestSIMDRuggednessRowsMatchScalar fills its rows: terrain with
// hazards, and in part of each row, per mode, runs of equal values,
// mixed signed zeros, or elevations near the float32 limit.
func windowRows(rng *rand.Rand, n, r, mode int, special float64) [][]float32 {
	hazards := []float32{
		float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		0, float32(math.Copysign(0, -1)), math.MaxFloat32, -math.MaxFloat32,
		1e30, -1e30, math.SmallestNonzeroFloat32, 1e-30,
	}
	negZero := float32(math.Copysign(0, -1))
	rows := make([][]float32, 2*r+1)
	for j := range rows {
		rows[j] = make([]float32, n+2*r+rng.IntN(3))
		for i := range rows[j] {
			v := float32(rng.NormFloat64()*50 + 500)
			if rng.Float64() < special {
				v = hazards[rng.IntN(len(hazards))]
			}
			if i%16 < 11 {
				switch mode {
				case 1:
					v = 500
				case 2:
					v = [2]float32{0, negZero}[rng.IntN(2)]
				case 3:
					v = float32(rng.Float64()*2-1) * math.MaxFloat32
				}
			}
			rows[j][i] = v
		}
	}
	return rows
}

func sameRugBits(a, b float32) bool {
	return a != a && b != b || math.Float32bits(a) == math.Float32bits(b)
}

// TestRuggednessWindowRowR1IsRuggednessRow holds the any-radius kernel
// at r = 1 to the canonical 3×3 one bit for bit, over hazards, runs of
// equal values, signed zeros and overflowing differences, at lengths
// either side of a block.
func TestRuggednessWindowRowR1IsRuggednessRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(61, 8))
	for _, n := range []int{0, 1, 2, 7, 8, 9, 31, 100, rugBlock - 1, rugBlock, rugBlock + 1, 2*rugBlock + 5} {
		for _, special := range []float64{0, 0.1, 0.5} {
			for mode := range 4 {
				rows := windowRows(rng, n, 1, mode, special)
				for kind := RugTRIRiley; kind <= RugRoughness; kind++ {
					want, got := make([]float32, n), make([]float32, n)
					scalarRuggednessRow(want, rows[0], rows[1], rows[2], kind)
					RuggednessWindowRow(got, rows, kind)
					for i := range want {
						if !sameRugBits(got[i], want[i]) {
							t.Fatalf("kind=%d n=%d mode=%d special=%v cell %d: got %g (%#x), want %g (%#x)",
								kind, n, mode, special, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
						}
					}
				}
			}
		}
	}
}

// naiveRuggedness is RuggednessWindowRow's documented formula for one
// cell, written a cell at a time: the window's cells in row-major order,
// the centre skipped, each sum started from its first term.
func naiveRuggedness(rows [][]float32, i int, kind RuggednessKind) float32 {
	d := len(rows)
	r := d / 2
	c := rows[r][i+r]
	n := float32(d*d - 1)
	var s32 float32
	var s64 float64
	first := true
	hi, lo := rows[0][i], rows[0][i]
	for j := range d {
		for k := range d {
			v := rows[j][i+k]
			hi, lo = max(hi, v), min(lo, v)
			if j == r && k == r {
				continue
			}
			diff := v - c
			switch kind {
			case RugTRIRiley:
				sq := float64(diff) * float64(diff)
				if first {
					s64 = sq
				} else {
					s64 += sq
				}
			case RugTRIWilson:
				a := float32(math.Abs(float64(diff)))
				if first {
					s32 = a
				} else {
					s32 += a
				}
			case RugTPI:
				if first {
					s32 = v
				} else {
					s32 += v
				}
			}
			first = false
		}
	}
	switch kind {
	case RugTRIRiley:
		return float32(math.Sqrt(s64))
	case RugTRIWilson:
		return s32 / n
	case RugTPI:
		return c - s32/n
	}
	return hi - lo
}

// TestRuggednessWindowRowMatchesNaive checks every kind at radii up to
// the terrain package's largest against the per-cell formula, bit for
// bit, at lengths that end inside, at and past a block.
func TestRuggednessWindowRowMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(62, 9))
	for _, r := range []int{1, 2, 3, 5, 8} {
		for _, n := range []int{0, 1, 5, 64, rugBlock - 1, rugBlock, rugBlock + 3, 2*rugBlock + 1} {
			for _, special := range []float64{0, 0.05} {
				for mode := range 4 {
					rows := windowRows(rng, n, r, mode, special)
					for kind := RugTRIRiley; kind <= RugRoughness; kind++ {
						got := make([]float32, n)
						RuggednessWindowRow(got, rows, kind)
						for i := range got {
							if want := naiveRuggedness(rows, i, kind); !sameRugBits(got[i], want) {
								t.Fatalf("r=%d kind=%d n=%d mode=%d cell %d: got %g (%#x), want %g (%#x)",
									r, kind, n, mode, i, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
							}
						}
					}
				}
			}
		}
	}
}

// TestRuggednessWindowRowReadsOnlyItsWindow poisons every cell beyond
// the n+2r a row must provide, and requires no output to change.
func TestRuggednessWindowRowReadsOnlyItsWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(63, 10))
	for _, r := range []int{2, 4} {
		n := rugBlock + 9
		rows := windowRows(rng, n, r, 0, 0)
		for kind := RugTRIRiley; kind <= RugRoughness; kind++ {
			want := make([]float32, n)
			RuggednessWindowRow(want, rows, kind)
			poisoned := make([][]float32, len(rows))
			for j, row := range rows {
				p := append(row[:n+2*r:n+2*r], float32(math.NaN()), 1e30)
				poisoned[j] = p
			}
			got := make([]float32, n)
			RuggednessWindowRow(got, poisoned, kind)
			for i := range got {
				if !sameRugBits(got[i], want[i]) {
					t.Fatalf("r=%d kind=%d: cell %d changed when cells past the window did", r, kind, i)
				}
			}
		}
	}
}

func TestRuggednessWindowRowPanics(t *testing.T) {
	rows := func(d, cells int) [][]float32 {
		out := make([][]float32, d)
		for j := range out {
			out[j] = make([]float32, cells)
		}
		return out
	}
	for _, c := range []struct {
		name string
		rows [][]float32
		kind RuggednessKind
	}{
		{"one row", rows(1, 10), RugTPI},
		{"even rows", rows(4, 10), RugTPI},
		{"short row", append(rows(4, 8), make([]float32, 7)), RugTPI},
		{"unknown kind", rows(5, 8), RugRoughness + 1},
		{"negative kind", rows(5, 8), -1},
	} {
		mustPanic(t, fmt.Sprint(c.name), func() { RuggednessWindowRow(make([]float32, 4), c.rows, c.kind) })
	}
}
