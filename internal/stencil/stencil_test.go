package stencil

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestAtan32Accuracy(t *testing.T) {
	// A stride through every non-negative finite float32 bit pattern, plus
	// the neighbourhoods of the reduction thresholds. The exhaustive run
	// quoted in Atan32's documentation found a maximum of 1.41e-7.
	const bound = 1.5e-7
	check := func(x float32) {
		got := float64(Atan32(x))
		if e := math.Abs(got - math.Atan(float64(x))); e > bound {
			t.Fatalf("Atan32(%g) = %g, math.Atan = %g, error %.3g > %.3g",
				x, got, math.Atan(float64(x)), e, bound)
		}
	}
	for b := uint32(0); b < 0x7F800000; b += 9973 {
		check(math.Float32frombits(b))
	}
	for _, th := range []float32{atanTanPi8, atanTan3Pi8, 1} {
		x := th
		for range 1000 {
			x = math.Nextafter32(x, 0)
		}
		for range 2000 {
			check(x)
			x = math.Nextafter32(x, 10)
		}
	}
	if got := Atan32(0); math.Float32bits(got) != 0 {
		t.Errorf("Atan32(+0) = %g (%#x), want +0", got, math.Float32bits(got))
	}
	if got := Atan32(float32(math.Inf(1))); got != float32(math.Pi/2) {
		t.Errorf("Atan32(+Inf) = %g, want π/2", got)
	}
	if got := Atan32(float32(math.NaN())); got == got {
		t.Errorf("Atan32(NaN) = %g, want NaN", got)
	}
}

func TestRowLengthPanics(t *testing.T) {
	mustPanic(t, "short row", func() {
		HornSlopeRow(make([]float32, 4), make([]float32, 6), make([]float32, 5), make([]float32, 6), 1, 1, 1, false)
	})
	mustPanic(t, "short aspect row", func() {
		HornAspectRow(make([]float32, 4), make([]float32, 6), make([]float32, 6), make([]float32, 5), 1, 1, -1, false)
	})
	mustPanic(t, "short hillshade row", func() {
		HornHillshadeRow(make([]float32, 4), make([]float32, 5), make([]float32, 6), make([]float32, 6), 1, 1, 1, 1, 1)
	})
	mustPanic(t, "short curvature row", func() {
		ZTCurvatureRow(make([]float32, 4), make([]float32, 6), make([]float32, 6), make([]float32, 5), 1, 1, 1, 1, 1, CurvPlan)
	})
	mustPanic(t, "unknown curvature kind", func() {
		ZTCurvatureRow(make([]float32, 4), make([]float32, 6), make([]float32, 6), make([]float32, 6), 1, 1, 1, 1, 1, CurvMean+1)
	})
	mustPanic(t, "short ruggedness row", func() {
		RuggednessRow(make([]float32, 4), make([]float32, 6), make([]float32, 5), make([]float32, 6), RugTPI)
	})
	mustPanic(t, "unknown ruggedness kind", func() {
		RuggednessRow(make([]float32, 4), make([]float32, 6), make([]float32, 6), make([]float32, 6), RugRoughness+1)
	})
	mustPanic(t, "negative ruggedness kind", func() {
		RuggednessRow(make([]float32, 4), make([]float32, 6), make([]float32, 6), make([]float32, 6), -1)
	})
	mustPanic(t, "dx/dy length", func() {
		HornGradientRow(make([]float32, 4), make([]float32, 3), make([]float32, 6), make([]float32, 6), make([]float32, 6), 1, 1)
	})
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: did not panic", name)
		}
	}()
	f()
}

// naiveErode is the per-cell reference for Erode3x3.
func naiveErode(dst []uint64, dstOff, dstStride int, src []uint64, srcOff, srcStride, w, h int) {
	get := func(x, y int) bool {
		i := srcOff + y*srcStride + x
		return src[i>>6]>>uint(i&63)&1 != 0
	}
	for y := range h {
		for x := range w {
			ok := x > 0 && y > 0 && x < w-1 && y < h-1
			for dy := -1; ok && dy <= 1; dy++ {
				for dx := -1; ok && dx <= 1; dx++ {
					ok = get(x+dx, y+dy)
				}
			}
			i := dstOff + y*dstStride + x
			if ok {
				dst[i>>6] |= 1 << uint(i&63)
			} else {
				dst[i>>6] &^= 1 << uint(i&63)
			}
		}
	}
}

func TestErode3x3MatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, w := range []int{1, 2, 3, 4, 7, 62, 63, 64, 65, 66, 127, 128, 129, 200} {
		for _, h := range []int{1, 2, 3, 4, 7} {
			for _, extra := range []int{0, 1, 13, (64 - w%64) % 64} {
				stride := w + extra
				for _, offs := range [][2]int{{0, 0}, {1, 63}, {63, 1}, {64, 100}, {37, 64}} {
					n := offs[0] + offs[1] + (h-1)*stride + w + 64
					words := (n + 63) >> 6
					src := make([]uint64, words)
					for k := range src {
						// Mostly-valid masks so interior cells survive the
						// erosion often enough to be checked.
						src[k] = rng.Uint64() | rng.Uint64() | rng.Uint64()
					}
					dst := make([]uint64, words)
					for k := range dst {
						dst[k] = rng.Uint64()
					}
					want := append([]uint64(nil), dst...)
					naiveErode(want, offs[1], stride, src, offs[0], stride, w, h)
					Erode3x3(dst, offs[1], stride, src, offs[0], stride, w, h)
					for k := range dst {
						if dst[k] != want[k] {
							t.Fatalf("w=%d h=%d stride=%d offs=%v: word %d = %#016x, want %#016x",
								w, h, stride, offs, k, dst[k], want[k])
						}
					}
				}
			}
		}
	}
}

func TestErode3x3DifferentStrides(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, w := range []int{63, 64, 65} {
		h := 6
		srcStride, dstStride := w+5, 64*((w+63)/64)
		src := make([]uint64, 16)
		dst := make([]uint64, 16)
		for k := range src {
			src[k] = rng.Uint64() | rng.Uint64()
			dst[k] = rng.Uint64()
		}
		want := append([]uint64(nil), dst...)
		naiveErode(want, 3, dstStride, src, 70, srcStride, w, h)
		Erode3x3(dst, 3, dstStride, src, 70, srcStride, w, h)
		for k := range dst {
			if dst[k] != want[k] {
				t.Fatalf("w=%d: word %d = %#016x, want %#016x", w, k, dst[k], want[k])
			}
		}
	}
}

// TestErodeBoxMatchesNaive checks ErodeBox against a per-cell reference
// for radii 0 to 3, one to three sources with different strides and
// offsets, and widths around the word boundaries, including radius 0 in
// place.
func TestErodeBoxMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	get := func(m MaskRegion, x, y int) bool {
		i := m.Off + y*m.Stride + x
		return m.Bits[i>>6]>>uint(i&63)&1 != 0
	}
	for _, r := range []int{0, 1, 2, 3} {
		for _, w := range []int{1, 2, 7, 62, 63, 64, 65, 129} {
			for _, h := range []int{1, 2, 5} {
				for nsrc := 1; nsrc <= 3; nsrc++ {
					sw, sh := w+2*r, h+2*r
					mk := func(off, stride int) MaskRegion {
						bits := make([]uint64, (off+(sh-1)*stride+sw+63)>>6+1)
						for k := range bits {
							bits[k] = ^(rng.Uint64() & rng.Uint64() & rng.Uint64())
						}
						return MaskRegion{bits, off, stride}
					}
					srcs := make([]MaskRegion, nsrc)
					for i := range srcs {
						srcs[i] = mk(rng.IntN(130), sw+[]int{0, 1, 64}[rng.IntN(3)])
					}
					dst := mk(rng.IntN(130), w+rng.IntN(70))
					if r == 0 && rng.IntN(2) == 0 {
						dst = srcs[rng.IntN(nsrc)] // in place
					}
					want := make([]bool, w*h)
					for y := range h {
						for x := range w {
							ok := true
							for _, s := range srcs {
								for j := 0; ok && j <= 2*r; j++ {
									for i := 0; ok && i <= 2*r; i++ {
										ok = get(s, x+i, y+j)
									}
								}
							}
							want[y*w+x] = ok
						}
					}
					before := append([]uint64(nil), dst.Bits...)
					ErodeBox(dst, srcs, w, h, r, make([]uint64, ErodeScratch(w, r)))
					inRegion := make(map[int]bool)
					for y := range h {
						for x := range w {
							inRegion[dst.Off+y*dst.Stride+x] = true
							if got := get(dst, x, y); got != want[y*w+x] {
								t.Fatalf("r=%d w=%d h=%d srcs=%d: cell (%d, %d) = %v, want %v", r, w, h, nsrc, x, y, got, want[y*w+x])
							}
						}
					}
					for i := range len(before) * 64 {
						if !inRegion[i] && (dst.Bits[i>>6]^before[i>>6])>>uint(i&63)&1 != 0 {
							t.Fatalf("r=%d w=%d h=%d srcs=%d: bit %d outside the region changed", r, w, h, nsrc, i)
						}
					}
				}
			}
		}
	}
	mustPanic(t, "no sources", func() { ErodeBox(MaskRegion{make([]uint64, 1), 0, 1}, nil, 1, 1, 0, make([]uint64, 1)) })
}

// TestErodeReachMatchesNaive checks ErodeReach against a per-cell
// reference: one to four sources, each with its own radius from 0 to 3
// in any order (so equal radii are sometimes adjacent and share a row,
// and sometimes not), widths around the word boundaries, and a
// destination whose other bits must survive.
func TestErodeReachMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 13))
	get := func(m MaskRegion, x, y int) bool {
		i := m.Off + y*m.Stride + x
		return m.Bits[i>>6]>>uint(i&63)&1 != 0
	}
	mk := func(off, stride, w, h int) MaskRegion {
		bits := make([]uint64, (off+(h-1)*stride+w+63)>>6+1)
		for k := range bits {
			bits[k] = ^(rng.Uint64() & rng.Uint64() & rng.Uint64())
		}
		return MaskRegion{bits, off, stride}
	}
	for _, w := range []int{1, 2, 7, 62, 63, 64, 65, 129} {
		for _, h := range []int{1, 2, 5} {
			for nsrc := 1; nsrc <= 4; nsrc++ {
				for range 4 {
					radii := make([]int, nsrc)
					srcs := make([]MaskRegion, nsrc)
					rmax := 0
					for i := range srcs {
						r := rng.IntN(4)
						radii[i], rmax = r, max(rmax, r)
						sw := w + 2*r
						srcs[i] = mk(rng.IntN(130), sw+[]int{0, 1, 64}[rng.IntN(3)], sw, h+2*r)
					}
					dst := mk(rng.IntN(130), w+rng.IntN(70), w, h)
					want := make([]bool, w*h)
					for y := range h {
						for x := range w {
							ok := true
							for k, s := range srcs {
								r := radii[k]
								for j := 0; ok && j <= 2*r; j++ {
									for i := 0; ok && i <= 2*r; i++ {
										ok = get(s, x+i, y+j)
									}
								}
							}
							want[y*w+x] = ok
						}
					}
					before := append([]uint64(nil), dst.Bits...)
					ErodeReach(dst, srcs, radii, w, h, make([]uint64, ErodeReachScratch(w, rmax)))
					inRegion := make(map[int]bool)
					for y := range h {
						for x := range w {
							inRegion[dst.Off+y*dst.Stride+x] = true
							if got := get(dst, x, y); got != want[y*w+x] {
								t.Fatalf("w=%d h=%d radii=%v: cell (%d, %d) = %v, want %v", w, h, radii, x, y, got, want[y*w+x])
							}
						}
					}
					for i := range len(before) * 64 {
						if !inRegion[i] && (dst.Bits[i>>6]^before[i>>6])>>uint(i&63)&1 != 0 {
							t.Fatalf("w=%d h=%d radii=%v: bit %d outside the region changed", w, h, radii, i)
						}
					}
				}
			}
		}
	}
	one := MaskRegion{make([]uint64, 1), 0, 1}
	mustPanic(t, "no sources", func() { ErodeReach(one, nil, nil, 1, 1, make([]uint64, 2)) })
	mustPanic(t, "radius count", func() { ErodeReach(one, []MaskRegion{one}, nil, 1, 1, make([]uint64, 2)) })
	mustPanic(t, "negative radius", func() { ErodeReach(one, []MaskRegion{one}, []int{-1}, 1, 1, make([]uint64, 2)) })
}

func TestClearBorder(t *testing.T) {
	const w, h, stride, off = 65, 4, 70, 5
	m := make([]uint64, 8)
	for k := range m {
		m[k] = ^uint64(0)
	}
	ClearBorder(m, off, stride, w, h)
	for i := range len(m) * 64 {
		set := m[i>>6]>>uint(i&63)&1 != 0
		want := true
		if j := i - off; j >= 0 && j < (h-1)*stride+w && j%stride < w {
			x, y := j%stride, j/stride
			want = x != 0 && y != 0 && x != w-1 && y != h-1
		}
		if set != want {
			t.Fatalf("bit %d = %v, want %v", i, set, want)
		}
	}
}

// atan2Specials are the argument values whose combinations exercise every
// special case of math.Atan2, plus ordinary values of each sign.
var atan2Specials = []float32{
	0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	1, -1, 0.5, -2, 3e38, -3e38, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-30, -7e20,
}

// atan2Sample returns a random float32 of random sign whose magnitude is
// log-uniform over most of the float32 range, or occasionally uniform in
// [0, 4), so that ratios near 1 and the reduction thresholds are common.
func atan2Sample(rng *rand.Rand) float32 {
	var v float32
	if rng.IntN(4) == 0 {
		v = float32(rng.Float64() * 4)
	} else {
		v = float32(math.Pow(2, rng.Float64()*200-100))
	}
	if rng.IntN(2) == 0 {
		v = -v
	}
	return v
}

func TestAtan2F32Accuracy(t *testing.T) {
	const bound = 3e-7
	var worst float64
	check := func(y, x float32) {
		got := Atan2F32(y, x)
		want := math.Atan2(float64(y), float64(x))
		if math.IsNaN(want) {
			if got == got {
				t.Fatalf("Atan2F32(%g, %g) = %g, want NaN", y, x, got)
			}
			return
		}
		if math.Signbit(float64(got)) != math.Signbit(want) {
			t.Fatalf("Atan2F32(%g, %g) = %g, math.Atan2 = %g: sign differs", y, x, got, want)
		}
		e := math.Abs(float64(got) - want)
		worst = max(worst, e)
		if e > bound {
			t.Fatalf("Atan2F32(%g, %g) = %g, math.Atan2 = %g, error %.3g > %.3g", y, x, got, want, e, bound)
		}
	}
	for _, y := range atan2Specials {
		for _, x := range atan2Specials {
			check(y, x)
		}
	}
	n := 2_000_000
	if testing.Short() {
		n = 100_000
	}
	rng := rand.New(rand.NewPCG(21, 22))
	for range n {
		y, x := atan2Sample(rng), atan2Sample(rng)
		check(y, x)
		// Nearly equal magnitudes, around the π/4 diagonals.
		check(y, math.Nextafter32(y, 0)*float32(1-2*rng.IntN(2)))
	}
	t.Logf("largest error %.3g radians", worst)

	pi := float32(math.Pi)
	for _, tc := range []struct {
		y, x, want float32
	}{
		{0, 0, 0},
		{float32(math.Copysign(0, -1)), 0, float32(math.Copysign(0, -1))},
		{0, float32(math.Copysign(0, -1)), pi},
		{float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1)), -pi},
		{0, -5, pi},
		{float32(math.Copysign(0, -1)), -5, -pi},
		{5, 0, pi / 2},
		{-5, float32(math.Copysign(0, -1)), -pi / 2},
		{float32(math.Inf(1)), float32(math.Inf(1)), pi / 4},
		{float32(math.Inf(-1)), float32(math.Inf(-1)), -(pi - pi/4)},
		{3, float32(math.Inf(-1)), pi},
		{float32(math.Inf(-1)), 3, -pi / 2},
	} {
		got := Atan2F32(tc.y, tc.x)
		if math.Float32bits(got) != math.Float32bits(tc.want) {
			t.Errorf("Atan2F32(%g, %g) = %g (%#x), want %g (%#x)", tc.y, tc.x, got, math.Float32bits(got), tc.want, math.Float32bits(tc.want))
		}
	}
}

// TestHornGradientNearFlat checks the Horn differences on windows whose
// cells lie within a factor of two of each other, as a DEM's neighbours
// do. There each neighbour difference is exact (Sterbenz), so the
// weighted difference must be within one float32 ulp of the sum of their
// magnitudes, however high the terrain. Summing the elevations before
// subtracting instead rounds at the ulp of four times the elevation, up
// to 5.9e-3 at 8800 m, which on gentle slopes swamps the gradient and
// turns aspect by up to 180° (tools/herbie/RESULTS.md).
func TestHornGradientNearFlat(t *testing.T) {
	ulp := func(s float64) float64 {
		if s == 0 {
			return 0
		}
		_, e := math.Frexp(s)
		return math.Ldexp(1, e-24)
	}
	check := func(axis string, base, relief float64, i int, got float32, a, b, d float64) {
		t.Helper()
		want := a + b + 2*d
		tol := ulp(math.Abs(a) + math.Abs(b) + 2*math.Abs(d))
		if err := math.Abs(float64(got) - want); err > tol {
			t.Errorf("base %g relief %g cell %d: %s = %g, want %g (error %.3g, tolerance %.3g)",
				base, relief, i, axis, got, want, err, tol)
		}
	}
	const n = 37 // lanes and a scalar tail on both SIMD backends
	rng := rand.New(rand.NewPCG(31, 32))
	for _, base := range []float64{500, 1000, 4000, 8800} {
		for _, relief := range []float64{0.01, 1, 100} {
			var rows [3][]float32
			for r := range rows {
				rows[r] = make([]float32, n+2)
				for c := range rows[r] {
					rows[r][c] = float32(base + (rng.Float64()*2-1)*relief)
				}
			}
			dx, dy := make([]float32, n), make([]float32, n)
			HornGradientRow(dx, dy, rows[0], rows[1], rows[2], 1, 1)
			for i := range n {
				z := func(r, c int) float64 { return float64(rows[r][i+c]) }
				check("dx", base, relief, i, dx[i], z(0, 2)-z(0, 0), z(2, 2)-z(2, 0), z(1, 2)-z(1, 0))
				check("dy", base, relief, i, dy[i], z(2, 0)-z(0, 0), z(2, 2)-z(0, 2), z(2, 1)-z(0, 1))
			}
		}
	}
}

func TestRuggednessRowKnownWindows(t *testing.T) {
	tiny := float32(math.Ldexp(1, -12))
	cases := []struct {
		name                          string
		w                             [9]float32 // z1..z9, row-major
		riley, wilson, tpi, roughness float32
	}{
		{"flat", [9]float32{7, 7, 7, 7, 7, 7, 7, 7, 7}, 0, 0, 0, 0},
		{"3-4-5", [9]float32{3, 4, 0, 0, 0, 0, 0, 0, 0}, 5, 0.875, -0.875, 4},
		{"pit", [9]float32{2, 2, 2, 2, -6, 2, 2, 2, 2}, float32(math.Sqrt(512)), 8, -8, 8},
		{"spread", [9]float32{-1, 5, 2, 0, 3, -4, 1, 1, 9}, float32(math.Sqrt(16 + 4 + 1 + 9 + 49 + 4 + 4 + 36)), 3.375, 3 - 1.625, 13},
		// Riley sums in float64: in float32, 1 + 7·2^-24 would be 1.
		{"float64 sum", [9]float32{1, tiny, tiny, tiny, 0, tiny, tiny, tiny, tiny}, 1 + float32(math.Ldexp(1, -22)),
			(1 + 7*tiny) * 0.125, -(1 + 7*tiny) * 0.125, 1},
	}
	for _, c := range cases {
		r0, r1, r2 := c.w[0:3], c.w[3:6], c.w[6:9]
		for kind, want := range []float32{c.riley, c.wilson, c.tpi, c.roughness} {
			got := make([]float32, 1)
			RuggednessRow(got, r0, r1, r2, RuggednessKind(kind))
			if math.Float32bits(got[0]) != math.Float32bits(want) {
				t.Errorf("%s kind %d: got %g (%#x), want %g (%#x)",
					c.name, kind, got[0], math.Float32bits(got[0]), want, math.Float32bits(want))
			}
		}
	}
}

func TestRoughnessNaNAndSignedZero(t *testing.T) {
	nan, negZero := float32(math.NaN()), float32(math.Copysign(0, -1))
	got := make([]float32, 1)
	RuggednessRow(got, []float32{1, 2, 3}, []float32{4, nan, 6}, []float32{7, 8, 9}, RugRoughness)
	if got[0] == got[0] {
		t.Errorf("roughness with a NaN in the window = %g, want NaN", got[0])
	}
	RuggednessRow(got, []float32{0, negZero, 0}, []float32{negZero, 0, 0}, []float32{0, 0, negZero}, RugRoughness)
	if math.Float32bits(got[0]) != 0 {
		t.Errorf("roughness of signed zeros = %g (%#x), want +0", got[0], math.Float32bits(got[0]))
	}
}
