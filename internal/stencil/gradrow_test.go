package stencil

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestFromGradientMatchesFused checks the promise gradrow.go makes: a
// gradient row followed by a from-gradient kernel is bit for bit the
// fused row kernel (any NaN matching any NaN), for slope in every unit,
// aspect both ways round and hillshade, on smooth rows, rows full of
// hazards and flat runs, at every length around the lane widths, on
// the scalar backend and on this build's SIMD one if it has one.
func TestFromGradientMatchesFused(t *testing.T) {
	same := func(a, b float32) bool {
		return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
	}
	hazards := []float32{
		float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		0, float32(math.Copysign(0, -1)), math.MaxFloat32, -math.MaxFloat32,
		1e30, -1e30, math.SmallestNonzeroFloat32, 1e-30,
	}
	type params struct{ kx, ky, scale, flat, c, bx, by float32 }
	ps := []params{
		{0.0125, 0.0125, float32(180 / math.Pi), -1, 180.3122, 127.5, -127.5},
		{0.05, 0.025, 1, 0, 255, 0, 0},
		{1, 1, 100, float32(math.NaN()), 44.28, -251.1, 1e-3},
		{-3, 7e-5, 1, 360, 0, 255, 0},
	}
	backends := []bool{true}
	if simdSlope != nil {
		backends = append(backends, false)
	}
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(47, 3))
	for _, scalar := range backends {
		UseScalar(scalar)
		for n := 0; n <= 70; n++ {
			for _, special := range []float64{0, 0.1, 0.5} {
				r := make([][]float32, 3)
				for k := range r {
					r[k] = make([]float32, n+2)
					for i := range r[k] {
						switch {
						case rng.Float64() < special:
							r[k][i] = hazards[rng.IntN(len(hazards))]
						case i%16 < 5:
							r[k][i] = 500 // flat runs: zero gradients
						default:
							r[k][i] = float32(rng.NormFloat64()*50 + 500)
						}
					}
				}
				for _, p := range ps {
					gx, gy := make([]float32, n), make([]float32, n)
					HornGradientRow(gx, gy, r[0], r[1], r[2], p.kx, p.ky)
					check := func(what string, fused, split []float32) {
						t.Helper()
						for i := range fused {
							if !same(fused[i], split[i]) {
								t.Fatalf("scalar=%v %s n=%d %+v cell %d: from gradient %g (%#x), fused %g (%#x)",
									scalar, what, n, p, i, split[i], math.Float32bits(split[i]),
									fused[i], math.Float32bits(fused[i]))
							}
						}
					}
					for _, atan := range []bool{false, true} {
						fused, split := make([]float32, n), make([]float32, n)
						HornSlopeRow(fused, r[0], r[1], r[2], p.kx, p.ky, p.scale, atan)
						SlopeFromGradientRow(split, gx, gy, p.scale, atan)
						check("slope", fused, split)
					}
					for _, trig := range []bool{false, true} {
						fused, split := make([]float32, n), make([]float32, n)
						HornAspectRow(fused, r[0], r[1], r[2], p.kx, p.ky, p.flat, trig)
						AspectFromGradientRow(split, gx, gy, p.flat, trig)
						check("aspect", fused, split)
					}
					fused, split := make([]float32, n), make([]float32, n)
					HornHillshadeRow(fused, r[0], r[1], r[2], p.kx, p.ky, p.c, p.bx, p.by)
					HillshadeFromGradientRow(split, gx, gy, p.c, p.bx, p.by)
					check("hillshade", fused, split)
				}
			}
		}
	}
}

func TestFromGradientPanics(t *testing.T) {
	s := make([]float32, 4)
	for name, f := range map[string]func(){
		"slope":     func() { SlopeFromGradientRow(s, s[:3], s, 1, false) },
		"aspect":    func() { AspectFromGradientRow(s, s, s[:3], -1, false) },
		"hillshade": func() { HillshadeFromGradientRow(s[:3], s, s, 1, 0, 0) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: mismatched lengths did not panic", name)
				}
			}()
			f()
		}()
	}
}
