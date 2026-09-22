package resamprow

import (
	"math"
	"math/rand/v2"
	"testing"
)

var methods = []Method{Nearest, Bilinear, Cubic, Lanczos, Average}

// ramp applies a 1-D table to the source values 0, 1, …, n-1 in float64.
func apply(a *Axis, vals []float64) []float64 {
	out := make([]float64, len(a.First))
	for c := a.Lo; c < a.Hi; c++ {
		for k := range int(a.Taps[c]) {
			out[c] += float64(a.W[int(a.Off[c])+k]) * vals[int(a.First[c])+k]
		}
	}
	return out
}

func rampVals(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i)
	}
	return v
}

// TestIdentityIsOneTap: an output grid equal to the source gives every
// method a single tap of weight exactly 1 on the same cell, which is what
// makes the identity resampling a bit-exact copy.
func TestIdentityIsOneTap(t *testing.T) {
	for _, m := range methods {
		for _, res := range []float64{1, -1, 0.3, -30, 1e-3} {
			a := NewAxis(m, Spec{N: 17, Origin: 5, Res: res, SrcN: 17, SrcOrigin: 5, SrcRes: res})
			if a.Lo != 0 || a.Hi != 17 {
				t.Fatalf("%v res %v: covered [%d, %d)", m, res, a.Lo, a.Hi)
			}
			for c := range 17 {
				if a.Taps[c] != 1 || a.First[c] != int32(c) || a.W[a.Off[c]] != 1 {
					t.Fatalf("%v res %v: column %d has first %d, taps %d, weight %v",
						m, res, c, a.First[c], a.Taps[c], a.W[a.Off[c]:a.Off[c]+a.Taps[c]])
				}
			}
		}
	}
}

// TestGDALGoldens holds the tables to gdalwarp 3.12.1's output for the
// ramp 0…7 (DESIGN.md §54, probes p2), with the working type Float64.
func TestGDALGoldens(t *testing.T) {
	type golden struct {
		m          Method
		n          int
		origin, rs float64
		want       []float64
	}
	nd := math.NaN() // gdalwarp's nodata: not covered
	for _, g := range []golden{
		{Nearest, 4, 0, 2, []float64{1, 3, 5, 7}},
		{Nearest, 7, 0.5, 1, []float64{1, 2, 3, 4, 5, 6, 7}},
		{Nearest, 10, -1.5, 1, []float64{nd, 0, 1, 2, 3, 4, 5, 6, 7, nd}},
		{Bilinear, 7, 0.5, 1, []float64{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5}},
		{Bilinear, 10, -1.5, 1, []float64{nd, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, nd}},
		{Bilinear, 4, 0, 2, []float64{0.7142857313156128, 2.5, 4.5, 6.285714149475098}},
		{Cubic, 4, 0, 2, []float64{0.5439330339431763, 2.4594595432281494, 4.54054069519043, 6.456067085266113}},
		{Lanczos, 7, 0.5, 1, []float64{0.37163814902305603, 1.5626740455627441, 2.5, 3.5, 4.5, 5.437325954437256, 6.628361701965332}},
		{Lanczos, 4, 0, 2, []float64{0.5060359239578247, 2.449343204498291, 4.550656795501709, 6.493964195251465}},
		{Lanczos, 16, 0, 0.5, []float64{-0.0925142914056778, 0.14245487749576569, 0.6756039261817932, 1.2994424104690552,
			1.7903872728347778, 2.230201005935669, 2.769798994064331, 3.230201005935669, 3.769798994064331, 4.23020076751709,
			4.76979923248291, 5.209612846374512, 5.700557708740234, 6.324396133422852, 6.857544898986816, 7.092514514923096}},
		{Average, 4, 0, 2, []float64{0.5, 2.5, 4.5, 6.5}},
		{Average, 10, -1.5, 1, []float64{nd, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7}},
	} {
		a := NewAxis(g.m, Spec{N: g.n, Origin: g.origin, Res: g.rs, SrcN: 8, SrcOrigin: 0, SrcRes: 1})
		got := apply(&a, rampVals(8))
		for c, w := range g.want {
			covered := c >= a.Lo && c < a.Hi
			if math.IsNaN(w) {
				if covered {
					t.Errorf("%v %d@%v×%v: column %d covered, gdalwarp writes nodata", g.m, g.n, g.origin, g.rs, c)
				}
				continue
			}
			if !covered || math.Abs(got[c]-w) > 2e-6*math.Max(1, math.Abs(w)) {
				t.Errorf("%v %d@%v×%v: column %d = %v (covered %v), gdalwarp %v", g.m, g.n, g.origin, g.rs, c, got[c], covered, w)
			}
		}
	}
}

// TestWidening pins gdalwarp's thresholds: bilinear and cubic widen once
// s > 1/0.95, Lanczos for any s > 1.
func TestWidening(t *testing.T) {
	for _, c := range []struct {
		m     Method
		scale float64
		want  bool
	}{
		{Bilinear, 1.052, false}, {Bilinear, 1.053, true}, {Cubic, 1.052, false}, {Cubic, 1.053, true},
		{Lanczos, 1.0, false}, {Lanczos, 1.001, true}, {Bilinear, 0.5, false}, {Lanczos, 0.5, false},
	} {
		a := NewAxis(c.m, Spec{N: 5, Res: c.scale, SrcN: 10, SrcRes: 1})
		if a.Widened != c.want {
			t.Errorf("%v at %v: widened %v, want %v", c.m, c.scale, a.Widened, c.want)
		}
	}
}

// TestTableInvariants checks every table over random axes: taps lie in
// the source and increase, weights sum to 1 to within rounding, covered
// cells have at least one tap, and a flipped source gives the same taps
// mirrored.
func TestTableInvariants(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range 400 {
		srcN := 1 + rng.IntN(60)
		n := 1 + rng.IntN(60)
		srcRes := math.Pow(2, rng.Float64()*6-3)
		if rng.IntN(2) == 0 {
			srcRes = -srcRes
		}
		res := srcRes * math.Pow(2, rng.Float64()*6-3)
		origin := rng.Float64()*20 - 10
		for _, m := range methods {
			sp := Spec{N: n, Origin: origin, Res: res, SrcN: srcN, SrcRes: srcRes}
			a := NewAxis(m, sp)
			for c := range n {
				covered := c >= a.Lo && c < a.Hi
				if !covered {
					if a.Taps[c] != 0 {
						t.Fatalf("%v %+v: uncovered column %d has taps", m, sp, c)
					}
					continue
				}
				f, k := int(a.First[c]), int(a.Taps[c])
				if k < 1 || f < 0 || f+k > srcN || k > a.MaxTaps {
					t.Fatalf("%v %+v: column %d taps [%d, %d) outside [0, %d)", m, sp, c, f, f+k, srcN)
				}
				sum := 0.0
				for _, w := range a.W[a.Off[c] : int(a.Off[c])+k] {
					sum += float64(w)
				}
				if math.Abs(sum-1) > float64(k)*6e-8 {
					t.Fatalf("%v %+v: column %d weights sum to %v", m, sp, c, sum)
				}
				if m != Average && (a.Centre[c] < a.First[c] || a.Centre[c] >= a.First[c]+a.Taps[c]) {
					t.Fatalf("%v %+v: column %d centre %d not a tap", m, sp, c, a.Centre[c])
				}
			}
			// Mirror the source: the same world, indices reversed.
			flip := Spec{N: n, Origin: origin, Res: res, SrcN: srcN, SrcOrigin: float64(srcN) * srcRes, SrcRes: -srcRes}
			b := NewAxis(m, flip)
			for c := a.Lo; c < a.Hi && m != Nearest; c++ {
				if c < b.Lo || c >= b.Hi || a.Taps[c] != b.Taps[c] || int(a.First[c]) != srcN-int(b.First[c]+b.Taps[c]) {
					// Nearest and cells on exact ties may round the other way; the
					// rest must mirror.
					t.Fatalf("%v %+v: column %d does not mirror", m, sp, c)
				}
				for k := range int(a.Taps[c]) {
					wa, wb := a.W[int(a.Off[c])+k], b.W[int(b.Off[c])+int(b.Taps[c])-1-k]
					if math.Abs(float64(wa-wb)) > 1e-6 {
						t.Fatalf("%v %+v: column %d tap %d weight %v mirrors to %v", m, sp, c, k, wa, wb)
					}
				}
			}
		}
	}
}

// TestCoverageIsContiguous: every output index in [Lo, Hi) has a tap,
// even for cells too small for float64 to tell their edges apart, as
// fuzzing found (a resolution of 5e-16 at an origin of 24), and for
// scales so large that the stretched kernel would span 10¹² cells.
func TestCoverageIsContiguous(t *testing.T) {
	for _, sp := range []Spec{
		{N: 49, Origin: 24, Res: 4.851621023508851e-16, SrcN: 37, SrcOrigin: 26, SrcRes: -4},
		{N: 7, Origin: 0, Res: 1e12, SrcN: 30, SrcRes: 1},
		{N: 50, Origin: 3, Res: 1e-300, SrcN: 5, SrcRes: 1},
	} {
		for _, m := range methods {
			a := NewAxis(m, sp)
			for c := a.Lo; c < a.Hi; c++ {
				if a.Taps[c] < 1 {
					t.Fatalf("%v %+v: covered index %d has no tap", m, sp, c)
				}
			}
		}
	}
}
