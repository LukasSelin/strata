package accum

import (
	"encoding/binary"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// The reference is the definition: every value converted to a big.Rat,
// which holds a float32 exactly, summed exactly, and rounded once.

func refSum(xs []float32) *big.Rat {
	s := new(big.Rat)
	for _, v := range xs {
		s.Add(s, new(big.Rat).SetFloat64(float64(v)))
	}
	return s
}

func refVariance(xs []float32) *big.Rat {
	n := big.NewRat(int64(len(xs)), 1)
	mean := new(big.Rat).Quo(refSum(xs), n)
	v := new(big.Rat)
	for _, x := range xs {
		d := new(big.Rat).Sub(new(big.Rat).SetFloat64(float64(x)), mean)
		v.Add(v, d.Mul(d, d))
	}
	return v.Quo(v, n)
}

func f64(r *big.Rat) float64 { f, _ := r.Float64(); return f }

// ieeeZero gives an exactly zero reference sum the sign IEEE addition
// does, which big.Rat has no way to hold: -0 only when every value is -0.
func ieeeZero(v float64, xs []float32) float64 {
	if v != 0 {
		return v
	}
	for _, x := range xs {
		if math.Float32bits(x) != 1<<31 {
			return 0
		}
	}
	return math.Copysign(0, -1)
}

func finite(xs []float32) bool {
	for _, v := range xs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

// checkAgainstRef checks every result against the exact reference. xs
// must be finite and non-empty.
func checkAgainstRef(t *testing.T, xs []float32) {
	t.Helper()
	var s Sum
	var m Moments
	s.Add(xs)
	m.Add(xs)
	want := ieeeZero(f64(refSum(xs)), xs)
	if got := s.Value(); !same(got, want) {
		t.Fatalf("Sum = %v, want %v", got, want)
	}
	if got := m.Sum(); !same(got, want) {
		t.Fatalf("Moments.Sum = %v, want %v", got, want)
	}
	wantMean := ieeeZero(f64(new(big.Rat).Quo(refSum(xs), big.NewRat(int64(len(xs)), 1))), xs)
	if got := m.Mean(); !same(got, wantMean) {
		t.Fatalf("Mean = %v, want %v", got, wantMean)
	}
	v := refVariance(xs)
	if got, want := m.Variance(), f64(v); !same(got, want) {
		t.Fatalf("Variance = %v, want %v", got, want)
	}
	// StdDev promises one ulp of the true root. The reference root is
	// taken at twice the precision StdDev uses.
	f := new(big.Float).SetPrec(512).SetRat(v)
	wantSD, _ := f.Sqrt(f).Float64()
	if got := m.StdDev(); math.Abs(got-wantSD) > ulp(wantSD) {
		t.Fatalf("StdDev = %v, want %v within one ulp", got, wantSD)
	}
	if s.Count() != int64(len(xs)) || m.Count() != int64(len(xs)) {
		t.Fatalf("Count = %d, %d, want %d", s.Count(), m.Count(), len(xs))
	}
}

func ulp(x float64) float64 {
	return math.Nextafter(math.Abs(x), math.Inf(1)) - math.Abs(x)
}

// same is bit equality, so that -0 differs from +0 and every NaN is the
// canonical one.
func same(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }

// generators produce values that stress different parts of the bins.
var generators = map[string]func(r *rand.Rand) float32{
	"normal": func(r *rand.Rand) float32 { return float32(r.NormFloat64() * 100) },
	"dem":    func(r *rand.Rand) float32 { return float32(300 + r.Float64()*2000) },
	"allExponents": func(r *rand.Rand) float32 {
		// Any finite bit pattern, subnormals included.
		for {
			v := math.Float32frombits(r.Uint32())
			if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
				return v
			}
		}
	},
	"subnormal": func(r *rand.Rand) float32 {
		return math.Float32frombits(r.Uint32() & 0x807fffff)
	},
	"cancelling": func(r *rand.Rand) float32 {
		// Huge values of both signs around small ones: a float64
		// accumulator loses the small ones entirely.
		switch r.IntN(3) {
		case 0:
			return math.MaxFloat32
		case 1:
			return -math.MaxFloat32
		}
		return float32(r.Float64())
	},
}

func TestAgainstReference(t *testing.T) {
	for name, gen := range generators {
		t.Run(name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(1, 2))
			for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 1000} {
				xs := make([]float32, n)
				for i := range xs {
					xs[i] = gen(r)
				}
				checkAgainstRef(t, xs)
			}
		})
	}
}

// TestCancellation is the case a float64 accumulator gets wrong: the
// small value survives exactly.
func TestCancellation(t *testing.T) {
	xs := []float32{math.MaxFloat32, 1e-30, -math.MaxFloat32, math.MaxFloat32, -math.MaxFloat32}
	var s Sum
	s.Add(xs)
	if got := s.Value(); got != float64(float32(1e-30)) {
		t.Fatalf("Sum = %v, want %v", got, float32(1e-30))
	}
	checkAgainstRef(t, xs)
}

// TestOrderIndependence splits the same values into partials at random
// places, adds them with Add and Add1, combines the partials in a random
// order, and requires the same bits every time.
func TestOrderIndependence(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for name, gen := range generators {
		xs := make([]float32, 777)
		for i := range xs {
			xs[i] = gen(r)
		}
		var ws Sum
		var wm Moments
		ws.Add(xs)
		wm.Add(xs)
		for trial := range 50 {
			perm := r.Perm(len(xs))
			ys := make([]float32, len(xs))
			for i, p := range perm {
				ys[i] = xs[p]
			}
			var parts []*Moments
			var sums []*Sum
			for len(ys) > 0 {
				k := 1 + r.IntN(len(ys))
				p, q := new(Moments), new(Sum)
				if trial%2 == 0 {
					p.Add(ys[:k])
					q.Add(ys[:k])
				} else {
					for _, v := range ys[:k] {
						p.Add1(v)
						q.Add1(v)
					}
				}
				parts, sums = append(parts, p), append(sums, q)
				ys = ys[k:]
			}
			var gm Moments
			var gs Sum
			for _, i := range r.Perm(len(parts)) {
				gm.Combine(parts[i])
				gs.Combine(sums[i])
			}
			for _, c := range []struct {
				what      string
				got, want float64
			}{
				{"Sum", gs.Value(), ws.Value()},
				{"Moments.Sum", gm.Sum(), wm.Sum()},
				{"Mean", gm.Mean(), wm.Mean()},
				{"Variance", gm.Variance(), wm.Variance()},
				{"StdDev", gm.StdDev(), wm.StdDev()},
			} {
				if !same(c.got, c.want) {
					t.Fatalf("%s trial %d: %s = %v, want %v", name, trial, c.what, c.got, c.want)
				}
			}
		}
	}
}

func TestSpecials(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	negZero := float32(math.Copysign(0, -1))
	cases := []struct {
		name            string
		xs              []float32
		sum, mean, vari float64
	}{
		{"empty", nil, 0, math.NaN(), math.NaN()},
		{"nan", []float32{1, nan, 2}, math.NaN(), math.NaN(), math.NaN()},
		{"+inf", []float32{1, inf, 2}, math.Inf(1), math.Inf(1), math.NaN()},
		{"-inf", []float32{1, -inf}, math.Inf(-1), math.Inf(-1), math.NaN()},
		{"both inf", []float32{inf, -inf}, math.NaN(), math.NaN(), math.NaN()},
		{"-0", []float32{negZero, negZero}, math.Copysign(0, -1), math.Copysign(0, -1), 0},
		{"-0 and +0", []float32{negZero, 0}, 0, 0, 0},
		{"cancel to 0", []float32{-3, 3}, 0, 0, 9},
		{"one", []float32{5}, 5, 5, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var s Sum
			var m Moments
			s.Add(c.xs)
			for _, v := range c.xs {
				m.Add1(v)
			}
			if got := s.Value(); !same(got, c.sum) {
				t.Errorf("Sum = %v, want %v", got, c.sum)
			}
			if got := m.Mean(); !same(got, c.mean) {
				t.Errorf("Mean = %v, want %v", got, c.mean)
			}
			if got := m.Variance(); !same(got, c.vari) {
				t.Errorf("Variance = %v, want %v", got, c.vari)
			}
		})
	}
}

// TestNaNIsCanonical feeds NaNs with payloads and requires the canonical
// NaN back, as every reduction returns (DESIGN.md §49).
func TestNaNIsCanonical(t *testing.T) {
	var s Sum
	s.Add([]float32{math.Float32frombits(0x7fc00123), math.Float32frombits(0xffa00001)})
	if got := s.Value(); !same(got, math.NaN()) {
		t.Fatalf("Sum = %#x, want the canonical NaN", math.Float64bits(got))
	}
}

// TestResultsDoNotChangeState reads the results twice, between Adds, and
// checks nothing moved.
func TestResultsDoNotChangeState(t *testing.T) {
	var m Moments
	m.Add([]float32{1, 2, 3, 4, 5})
	before := m
	_, _, _, _ = m.Sum(), m.Mean(), m.Variance(), m.StdDev()
	if m != before {
		t.Fatal("reading a result changed the accumulator")
	}
}

// TestManyLargeValues adds enough of the largest float32 that a bin would
// overflow without normalisation, through Combine, which normalises.
func TestManyLargeValues(t *testing.T) {
	xs := make([]float32, 1<<16)
	for i := range xs {
		xs[i] = math.MaxFloat32
	}
	var part, total Sum
	part.Add(xs)
	for range 1 << 10 { // 2^26 values in all
		total.Combine(&part)
	}
	want := float64(math.MaxFloat32) * (1 << 26)
	if got := total.Value(); got != want {
		t.Fatalf("Sum = %v, want %v", got, want)
	}
}

func TestMaxCount(t *testing.T) {
	var a Sum
	a.n = MaxCount
	defer func() {
		if recover() == nil {
			t.Fatal("no panic past MaxCount")
		}
	}()
	a.Add1(1)
}

// FuzzAccum reads float32s from the fuzzer's bytes, compares with the
// reference when they are finite, and checks a split at an arbitrary
// place gives the same bits either way.
func FuzzAccum(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, uint16(1))
	f.Add([]byte{255, 255, 127, 127, 255, 255, 127, 255, 1, 0, 0, 0}, uint16(2))
	f.Fuzz(func(t *testing.T, data []byte, cut uint16) {
		xs := make([]float32, len(data)/4)
		for i := range xs {
			xs[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		if len(xs) > 0 && finite(xs) {
			checkAgainstRef(t, xs)
		}
		k := int(cut) % (len(xs) + 1)
		var whole, a, b Moments
		whole.Add(xs)
		a.Add(xs[:k])
		b.Add(xs[k:])
		b.Combine(&a)
		for _, p := range [][2]float64{
			{whole.Sum(), b.Sum()}, {whole.Mean(), b.Mean()},
			{whole.Variance(), b.Variance()}, {whole.StdDev(), b.StdDev()},
		} {
			if !same(p[0], p[1]) {
				t.Fatalf("split at %d: %v, whole %v", k, p[1], p[0])
			}
		}
	})
}

// TestBackendsAgree adds the same values with the scalar loop and with
// the vector backend, where the build has one, and requires the same
// exact integer in the bins, not only the same rounded result.
func TestBackendsAgree(t *testing.T) {
	defer useScalar(false)
	r := rand.New(rand.NewPCG(11, 12))
	gens := map[string]func(r *rand.Rand) float32{
		"zeros":    func(r *rand.Rand) float32 { return 0 },
		"negZeros": func(r *rand.Rand) float32 { return float32(math.Copysign(0, -1)) },
		"window": func(r *rand.Rand) float32 {
			// Exactly 32 fields apart: the widest block the vector path takes.
			if r.IntN(2) == 0 {
				return -1
			}
			return 0x1p-32
		},
		"squareWindow": func(r *rand.Rand) float32 {
			// 16 fields apart: the widest block Moments' vector path takes.
			if r.IntN(2) == 0 {
				return -1.5 // field 127; 0x1p-16 is field 111
			}
			return 0x1p-16 * float32(1+r.IntN(1000))
		},
		"pastWindow": func(r *rand.Rand) float32 {
			if r.IntN(2) == 0 {
				return 1
			}
			return 0x1p-33
		},
		"specialsSprinkled": func(r *rand.Rand) float32 {
			if r.IntN(500) == 0 {
				return float32(math.Inf(1 - 2*r.IntN(2)))
			}
			return float32(r.NormFloat64())
		},
	}
	for k, g := range generators {
		gens[k] = g
	}
	for name, gen := range gens {
		for _, n := range []int{63, 64, 65, 128, 1000, 4097} {
			xs := make([]float32, n)
			for i := range xs {
				xs[i] = gen(r)
			}
			var sc, vc Sum
			var sm, vm Moments
			useScalar(true)
			sc.Add(xs)
			sm.Add(xs)
			useScalar(false)
			vc.Add(xs)
			vm.Add(xs)
			if vm.sum.specials != sm.sum.specials {
				t.Fatalf("%s n=%d: Moments specials differ between backends", name, n)
			}
			for _, p := range [][2]*big.Int{
				{exact(vm.sum.bins[0][:], vm.sum.bins[1][:], vm.sum.bins[2][:], vm.sum.bins[3][:]),
					exact(sm.sum.bins[0][:], sm.sum.bins[1][:], sm.sum.bins[2][:], sm.sum.bins[3][:])},
				{exact(vm.sq[0][:], vm.sq[1][:], vm.sq[2][:], vm.sq[3][:]),
					exact(sm.sq[0][:], sm.sq[1][:], sm.sq[2][:], sm.sq[3][:])},
			} {
				if p[0].Cmp(p[1]) != 0 {
					t.Fatalf("%s n=%d: Moments vector %v, scalar %v", name, n, p[0], p[1])
				}
			}
			if sc.specials != vc.specials {
				t.Fatalf("%s n=%d: specials %+v, scalar %+v", name, n, vc.specials, sc.specials)
			}
			got := exact(vc.bins[0][:], vc.bins[1][:], vc.bins[2][:], vc.bins[3][:])
			want := exact(sc.bins[0][:], sc.bins[1][:], sc.bins[2][:], sc.bins[3][:])
			if got.Cmp(want) != 0 {
				t.Fatalf("%s n=%d: vector %v, scalar %v", name, n, got, want)
			}
		}
	}
}
