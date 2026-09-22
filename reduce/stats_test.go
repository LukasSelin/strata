package reduce_test

import (
	"context"
	"fmt"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/accum"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// refStats is Stats written from its definition: every valid cell
// converted to a big.Rat, which holds a float32 exactly, the sum and
// variance computed exactly, and each rounded once. IEEE's rules for NaN,
// the infinities and the sign of an exactly zero sum are applied by hand,
// since big.Rat has none of them.
func refStats(r raster.Float32Raster) reduce.Summary {
	var vals []float32
	for y := range r.Height {
		for x := range r.Width {
			if r.IsValid(x, y) {
				vals = append(vals, r.Data[r.Index(x, y)])
			}
		}
	}
	mn, mx, n := ref(r)
	s := reduce.Summary{Count: n, Min: mn, Max: mx, Mean: math.NaN(), StdDev: math.NaN()}
	if n == 0 {
		return s
	}
	var nans, pos, neg int
	allNegZero := true
	for _, v := range vals {
		switch {
		case v != v:
			nans++
		case math.IsInf(float64(v), 1):
			pos++
		case math.IsInf(float64(v), -1):
			neg++
		}
		allNegZero = allNegZero && math.Float32bits(v) == 1<<31
	}
	switch {
	case nans > 0 || pos > 0 && neg > 0:
		s.Sum = math.NaN()
		s.Mean = math.NaN()
		return s
	case pos > 0:
		s.Sum, s.Mean = math.Inf(1), math.Inf(1)
		return s
	case neg > 0:
		s.Sum, s.Mean = math.Inf(-1), math.Inf(-1)
		return s
	}
	sum := new(big.Rat)
	for _, v := range vals {
		sum.Add(sum, new(big.Rat).SetFloat64(float64(v)))
	}
	count := big.NewRat(n, 1)
	mean := new(big.Rat).Quo(sum, count)
	variance := new(big.Rat)
	for _, v := range vals {
		d := new(big.Rat).Sub(new(big.Rat).SetFloat64(float64(v)), mean)
		variance.Add(variance, d.Mul(d, d))
	}
	variance.Quo(variance, count)
	s.Sum, _ = sum.Float64()
	s.Mean, _ = mean.Float64()
	if s.Sum == 0 && allNegZero {
		s.Sum, s.Mean = math.Copysign(0, -1), math.Copysign(0, -1)
	}
	f := new(big.Float).SetPrec(512).SetRat(variance)
	s.StdDev, _ = f.Sqrt(f).Float64()
	return s
}

// sameF64 is bit equality, except that any NaN matches any NaN, as same
// is for float32.
func sameF64(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (a != a && b != b)
}

// sameSummary is bit equality of every field.
func sameSummary(a, b reduce.Summary) bool {
	return a.Count == b.Count && sameF64(a.Sum, b.Sum) && sameF64(a.Mean, b.Mean) &&
		sameF64(a.StdDev, b.StdDev) && same(a.Min, b.Min) && same(a.Max, b.Max)
}

// matchesRef compares a Summary with the reference: bit for bit, except
// StdDev, which the package promises only to one ulp.
func matchesRef(got, want reduce.Summary) bool {
	sd := got.StdDev
	if want.StdDev == want.StdDev && math.Abs(sd-want.StdDev) <= ulp(want.StdDev) {
		sd = want.StdDev
	}
	got.StdDev = sd
	return sameSummary(got, want)
}

func ulp(x float64) float64 {
	return math.Nextafter(math.Abs(x), math.Inf(1)) - math.Abs(x)
}

func fmtSummary(s reduce.Summary) string {
	return fmt.Sprintf("{n %d sum %v (%#x) mean %v sd %v min %v max %v}",
		s.Count, s.Sum, math.Float64bits(s.Sum), s.Mean, s.StdDev, s.Min, s.Max)
}

// checkStats runs Sum and Stats through every execution path, requires
// them to agree bit for bit with each other, and the plain ones to agree
// with the reference.
func checkStats(t *testing.T, id string, r raster.Float32Raster) {
	t.Helper()
	want := refStats(r)
	got := reduce.Stats(r)
	if !matchesRef(got, want) {
		t.Fatalf("%s: Stats = %s, want %s", id, fmtSummary(got), fmtSummary(want))
	}
	sum, n := reduce.Sum(r)
	if !sameF64(sum, got.Sum) || n != got.Count {
		t.Fatalf("%s: Sum = %v %d, Stats says %v %d", id, sum, n, got.Sum, got.Count)
	}
	ctx := context.Background()
	src := engine.NewMemorySource(r)
	for _, opts := range tilings() {
		s, err := reduce.StatsTiled(ctx, r, opts)
		if err != nil || !sameSummary(s, got) {
			t.Fatalf("%s StatsTiled %+v = %s, %v, want %s", id, opts, fmtSummary(s), err, fmtSummary(got))
		}
		s, err = reduce.StatsChunked(ctx, src, opts)
		if err != nil || !sameSummary(s, got) {
			t.Fatalf("%s StatsChunked %+v = %s, %v, want %s", id, opts, fmtSummary(s), err, fmtSummary(got))
		}
		sum2, n2, err := reduce.SumTiled(ctx, r, opts)
		if err != nil || !sameF64(sum2, sum) || n2 != n {
			t.Fatalf("%s SumTiled %+v = %v %d, %v, want %v %d", id, opts, sum2, n2, err, sum, n)
		}
		sum2, n2, err = reduce.SumChunked(ctx, src, opts)
		if err != nil || !sameF64(sum2, sum) || n2 != n {
			t.Fatalf("%s SumChunked %+v = %v %d, %v, want %v %d", id, opts, sum2, n2, err, sum, n)
		}
	}
}

// TestStatsValues is TestValues for Sum and Stats, with the cases a sum
// gets wrong that a minimum does not: cancellation, overflow of float32,
// signed zeros summing to zero, and values the vector backend's blocks
// must hand to the scalar loop.
func TestStatsValues(t *testing.T) {
	cases := []struct {
		name  string
		data  []float32
		valid []bool
	}{
		{name: "one cell", data: []float32{7}},
		{name: "ordinary", data: []float32{3, 1, 4, 1, 5, 9, 2, 6}},
		{name: "negatives", data: []float32{-3, -1, -4}},
		{name: "infinities", data: []float32{posInf, 1, negInf}},
		{name: "one infinity", data: []float32{posInf, 1, 2}},
		{name: "nan", data: []float32{1, nan, 3}},
		{name: "nan under an invalid cell", data: []float32{1, nan, 3}, valid: []bool{true, false, true}},
		{name: "signed zeros", data: []float32{0, negZero}},
		{name: "all -0", data: []float32{negZero}},
		{name: "all invalid", data: []float32{1, 2, 3}, valid: []bool{false, false, false}},
		{name: "one valid", data: []float32{1, 2, 3}, valid: []bool{false, true, false}},
		{
			// A float64 accumulator loses the small value entirely.
			name: "cancellation",
			data: []float32{math.MaxFloat32, 1e-30, -math.MaxFloat32},
		},
		{
			// The sum overflows float32 but not float64.
			name: "beyond float32",
			data: []float32{math.MaxFloat32, math.MaxFloat32},
		},
		{
			// 2^33 apart: wider than the vector backend's window, so
			// its blocks go to the scalar loop.
			name: "wide block",
			data: []float32{1, 0x1p-33, -1, 3},
		},
		{name: "subnormals", data: []float32{math.SmallestNonzeroFloat32, -2 * math.SmallestNonzeroFloat32, 0x1p-130}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Widths around the 8-lane vectors and the 64-cell blocks.
			for _, w := range []int{len(tc.data), 1, 7, 8, 9, 63, 64, 65, 130} {
				if w < len(tc.data) {
					continue
				}
				checkStats(t, fmt.Sprintf("%s w=%d", tc.name, w), layOut(t, tc.data, tc.valid, w))
			}
		})
	}
}

// TestStatsDEM runs a raster the size of a few bands, of DEM-like values,
// through every path with and without a mask: the tiling-independence
// guarantee over enough cells that tiles, bands and vector blocks all
// split it in different places.
func TestStatsDEM(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	w, h := 301, 67
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = float32(300 + 2000*rng.Float64())
	}
	checkStats(t, "unmasked", r)
	r.Valid = raster.NewMask(w * h)
	for i := range r.Data {
		raster.MaskSet(r.Valid, i, rng.IntN(10) != 0)
	}
	checkStats(t, "masked", r)
	checkStats(t, "window", r.Window(3, 2, w-7, h-5))
}

// TestStatsBackends requires the scalar and vector accumulators to give
// the same bits through the whole package, not only inside accum.
func TestStatsBackends(t *testing.T) {
	defer accum.UseScalar(false)
	rng := rand.New(rand.NewPCG(23, 24))
	w, h := 257, 33
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = float32(rng.NormFloat64() * 50)
	}
	r.Valid = raster.NewMask(w * h)
	for i := range r.Data {
		raster.MaskSet(r.Valid, i, rng.IntN(7) != 0)
	}
	accum.UseScalar(true)
	want := reduce.Stats(r)
	accum.UseScalar(false)
	for _, opts := range tilings() {
		got, err := reduce.StatsTiled(context.Background(), r, opts)
		if err != nil || !sameSummary(got, want) {
			t.Fatalf("%s backend %+v: %s, %v, scalar gives %s", accum.Backend(), opts, fmtSummary(got), err, fmtSummary(want))
		}
	}
}

// TestStatsNaNIsCanonical is TestNaNIsCanonical for the float64 results.
func TestStatsNaNIsCanonical(t *testing.T) {
	want := math.Float64bits(math.NaN())
	r := layOut(t, []float32{1, otherNaN, math.Float32frombits(0x7fc0_1234), 3}, nil, 130)
	s := reduce.Stats(r)
	sum, _ := reduce.Sum(r)
	for name, v := range map[string]float64{"Sum": sum, "Stats.Sum": s.Sum, "Mean": s.Mean, "StdDev": s.StdDev} {
		if math.Float64bits(v) != want {
			t.Errorf("%s = %#x, want the canonical NaN %#x", name, math.Float64bits(v), want)
		}
	}
	if math.Float32bits(s.Min) != math.Float32bits(nan) || math.Float32bits(s.Max) != math.Float32bits(nan) {
		t.Errorf("Min, Max = %#x %#x, want the canonical NaN", math.Float32bits(s.Min), math.Float32bits(s.Max))
	}
}

// TestStatsInvalidDataNeverRead is TestInvalidDataNeverRead for Stats.
func TestStatsInvalidDataNeverRead(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))
	w, h := 131, 29
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	r.Valid = raster.NewMask(w * h)
	for i := range r.Data {
		r.Data[i] = float32(rng.NormFloat64() * 1000)
		raster.MaskSet(r.Valid, i, rng.IntN(3) != 0)
	}
	want := reduce.Stats(r)
	hazards := []float32{nan, posInf, negInf, math.MaxFloat32, -math.MaxFloat32, 0x1p-140}
	for i := range r.Data {
		if !raster.MaskGet(r.Valid, i) {
			r.Data[i] = hazards[rng.IntN(len(hazards))]
		}
	}
	if got := reduce.Stats(r); !sameSummary(got, want) {
		t.Fatalf("scrambling invalid cells changed %s to %s", fmtSummary(want), fmtSummary(got))
	}
}

// TestStatsCancelledReturnsNoValue is TestCancelledReturnsNoValue for Sum
// and Stats.
func TestStatsCancelledReturnsNoValue(t *testing.T) {
	r := newRaster(64, 64)
	for i := range r.Data {
		r.Data[i] = float32(i) + 1
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	opts := engine.Options{TileWidth: 8, TileHeight: 8, Workers: 2}
	src := engine.NewMemorySource(r)
	if s, n, err := reduce.SumTiled(ctx, r, opts); err == nil || s != 0 || n != 0 {
		t.Fatalf("SumTiled: %v %d, %v, want zeroes and an error", s, n, err)
	}
	if s, n, err := reduce.SumChunked(ctx, src, opts); err == nil || s != 0 || n != 0 {
		t.Fatalf("SumChunked: %v %d, %v, want zeroes and an error", s, n, err)
	}
	if s, err := reduce.StatsTiled(ctx, r, opts); err == nil || s != (reduce.Summary{}) {
		t.Fatalf("StatsTiled: %s, %v, want the zero Summary and an error", fmtSummary(s), err)
	}
	if s, err := reduce.StatsChunked(ctx, src, opts); err == nil || s != (reduce.Summary{}) {
		t.Fatalf("StatsChunked: %s, %v, want the zero Summary and an error", fmtSummary(s), err)
	}
}
