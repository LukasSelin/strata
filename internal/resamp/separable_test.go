package resamp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// randomPlan returns a plan over a random source of up to 48×48 cells at
// a random scale and offset, and its source size.
func randomPlan(rng *rand.Rand, m Method) (p *Plan, sw, sh int) {
	sw, sh = 1+rng.IntN(48), 1+rng.IntN(48)
	scales := []float64{0.25, 0.5, 0.73, 1, 1.02, 1.37, 2, 3, 4.5}
	sx, sy := scales[rng.IntN(len(scales))], scales[rng.IntN(len(scales))]
	x := Spec{N: max(1, int(float64(sw)/sx)), Origin: rng.Float64() - 0.5, Res: sx, SrcN: sw, SrcRes: 1}
	y := Spec{N: max(1, int(float64(sh)/sy)), Origin: float64(sh) + rng.Float64() - 0.5, Res: -sy, SrcN: sh, SrcOrigin: float64(sh), SrcRes: -1}
	return NewPlan(m, x, y), sw, sh
}

// edgeValues are the special floats the kernels must carry through
// exactly: signed zeros, infinities, NaN, extremes and subnormals.
var edgeValues = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32, 1e-38, 3.5, -1e20, 7e30,
}

func randomData(rng *rand.Rand, n int, edges bool) []float32 {
	d := make([]float32, n)
	for i := range d {
		if edges && rng.IntN(8) == 0 {
			d[i] = edgeValues[rng.IntN(len(edgeValues))]
		} else {
			d[i] = float32(rng.NormFloat64() * 100)
		}
	}
	return d
}

// separable runs the two passes over the whole covered output.
func separable(p *Plan, src []float32, sw, sh int) []float32 {
	w, h := len(p.X.First), len(p.Y.First)
	out := make([]float32, w*h)
	if p.X.Lo >= p.X.Hi || p.Y.Lo >= p.Y.Hi {
		return out
	}
	fx0, fy0, fx1, fy1 := p.Footprint(p.X.Lo, p.Y.Lo, p.X.Hi, p.Y.Hi)
	cw := p.X.Hi - p.X.Lo
	t := make([]float32, (fy1-fy0)*cw)
	HRows(t, cw, src[fy0*sw+fx0:], sw, fy1-fy0, &p.X, p.X.Lo, p.X.Hi, fx0, make([]float32, HScratch(fx1-fx0)))
	for r := p.Y.Lo; r < p.Y.Hi; r++ {
		wy := p.Y.W[p.Y.Off[r] : p.Y.Off[r]+p.Y.Taps[r]]
		VRow(out[r*w+p.X.Lo:r*w+p.X.Hi], t[(int(p.Y.First[r])-fy0)*cw:], cw, wy)
	}
	return out
}

func sameFloat(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// TestSeparableIsDirect: the two passes give Direct2D's bits for every
// method, scale and edge value, which is the argument that separability
// changes the speed and not the answer (DESIGN.md §54).
func TestSeparableIsDirect(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	for iter := range 500 {
		for _, m := range []Method{Bilinear, Cubic, Lanczos, Average} {
			p, sw, sh := randomPlan(rng, m)
			src := randomData(rng, sw*sh, iter%3 == 0)
			got := separable(p, src, sw, sh)
			w, h := len(p.X.First), len(p.Y.First)
			want := make([]float32, w*h)
			Direct2D(want, w, src, sw, 0, 0, p, 0, w, 0, h)
			for i := range want {
				if !sameFloat(got[i], want[i]) {
					t.Fatalf("%v iter %d: cell %d separable %v, direct %v", m, iter, i, got[i], want[i])
				}
			}
		}
	}
}
