package terrain

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: did not panic", name)
		}
	}()
	f()
}

func isBorder(r raster.Float32Raster, x, y int) bool {
	return x == 0 || y == 0 || x == r.Width-1 || y == r.Height-1
}

// plane returns a w×h DEM of z = ax·(x·cs) + ay·(y·csy) + 100.
func plane(w, h int, ax, ay, cs, csy float64) raster.Float32Raster {
	d := make([]float32, w*h)
	for y := range h {
		for x := range w {
			d[y*w+x] = float32(ax*float64(x)*cs + ay*float64(y)*csy + 100)
		}
	}
	return raster.NewFloat32(w, h, d)
}

func checkBorderNaN(t *testing.T, name string, r raster.Float32Raster) {
	t.Helper()
	for y := range r.Height {
		for x := range r.Width {
			if v := r.Data[r.Index(x, y)]; isBorder(r, x, y) && v == v {
				t.Fatalf("%s: border cell (%d, %d) = %g, want NaN", name, x, y, v)
			}
		}
	}
}

func eachInterior(r raster.Float32Raster, f func(x, y int, v float32)) {
	for y := 1; y < r.Height-1; y++ {
		for x := 1; x < r.Width-1; x++ {
			f(x, y, r.Data[r.Index(x, y)])
		}
	}
}

func TestTiltedPlane(t *testing.T) {
	// Rises 0.3 per unit eastward and falls 0.4 per unit southward, on
	// anisotropic 10×5 cells. Horn is exact on a plane, so only float32
	// rounding separates the result from the analytic values.
	const ax, ay, cs, csy = 0.3, -0.4, 10.0, 5.0
	dem := plane(9, 7, ax, ay, cs, csy)
	dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
	Gradient(dx, dy, dem, GradientOptions{CellSize: cs, CellSizeY: csy})
	checkBorderNaN(t, "dx", dx)
	checkBorderNaN(t, "dy", dy)
	const gradTol = 1e-5
	eachInterior(dx, func(x, y int, v float32) {
		if math.Abs(float64(v)-ax) > gradTol {
			t.Errorf("dx(%d, %d) = %g, want %g", x, y, v, ax)
		}
	})
	eachInterior(dy, func(x, y int, v float32) {
		if math.Abs(float64(v)-ay) > gradTol {
			t.Errorf("dy(%d, %d) = %g, want %g", x, y, v, ay)
		}
	})

	m := math.Hypot(ax, ay) // 0.5
	for _, tc := range []struct {
		units SlopeUnits
		want  float64
		tol   float64
	}{
		{SlopeDegrees, math.Atan(m) * 180 / math.Pi, 1e-3},
		{SlopeRadians, math.Atan(m), 2e-5},
		{SlopePercent, 100 * m, 1e-3},
	} {
		dst := raster.NewFloat32Like(dem)
		Slope(dst, dem, SlopeOptions{CellSize: cs, CellSizeY: csy, Units: tc.units})
		checkBorderNaN(t, "slope", dst)
		eachInterior(dst, func(x, y int, v float32) {
			if math.Abs(float64(v)-tc.want) > tc.tol {
				t.Errorf("units %d: slope(%d, %d) = %g, want %g ± %g", tc.units, x, y, v, tc.want, tc.tol)
			}
		})
	}

	// ZFactor scales elevations: halving them halves the gradient.
	dst := raster.NewFloat32Like(dem)
	Slope(dst, dem, SlopeOptions{CellSize: cs, CellSizeY: csy, ZFactor: 2, Units: SlopePercent})
	eachInterior(dst, func(x, y int, v float32) {
		if math.Abs(float64(v)-200*m) > 2e-3 {
			t.Errorf("ZFactor 2: slope(%d, %d) = %g, want %g", x, y, v, 200*m)
		}
	})
}

func TestFlatIsZero(t *testing.T) {
	d := make([]float32, 12*5)
	for i := range d {
		d[i] = 1234.5
	}
	dem := raster.NewFloat32(12, 5, d)
	for _, u := range []SlopeUnits{SlopeDegrees, SlopePercent, SlopeRadians} {
		dst := raster.NewFloat32Like(dem)
		Slope(dst, dem, SlopeOptions{CellSize: 30, Units: u})
		eachInterior(dst, func(x, y int, v float32) {
			if math.Float32bits(v) != 0 {
				t.Errorf("units %d: flat slope(%d, %d) = %g, want +0", u, x, y, v)
			}
		})
	}
}

// TestHandComputed uses the worked example from Esri's "How Slope works"
// (cell size 5): dz/dx = 0.05, dz/dy = -3.8 with y increasing down the
// rows, rise/run = 3.80032, slope 75.25762°.
func TestHandComputed(t *testing.T) {
	dem := raster.NewFloat32(3, 3, []float32{
		50, 45, 50,
		30, 30, 30,
		8, 10, 10,
	})
	dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
	Gradient(dx, dy, dem, GradientOptions{CellSize: 5})
	if got := dx.Data[4]; math.Abs(float64(got)-0.05) > 1e-6 {
		t.Errorf("dx = %g, want 0.05", got)
	}
	if got := dy.Data[4]; math.Abs(float64(got)+3.8) > 1e-6 {
		t.Errorf("dy = %g, want -3.8", got)
	}
	rr := math.Hypot(0.05, 3.8)
	for _, tc := range []struct {
		units SlopeUnits
		want  float64
		tol   float64
	}{
		{SlopeDegrees, 75.25762, 1e-4},
		{SlopePercent, 100 * rr, 1e-3},
		{SlopeRadians, math.Atan(rr), 1e-6},
	} {
		dst := raster.NewFloat32Like(dem)
		Slope(dst, dem, SlopeOptions{CellSize: 5, Units: tc.units})
		if got := dst.Data[4]; math.Abs(float64(got)-tc.want) > tc.tol {
			t.Errorf("units %d: slope = %.6f, want %.6f ± %g", tc.units, got, tc.want, tc.tol)
		}
		checkBorderNaN(t, "slope", dst)
	}
}

func TestGradientSigns(t *testing.T) {
	// Elevation rises with the column (east) and the row (south).
	dem := plane(5, 5, 1, 2, 1, 1)
	dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
	Gradient(dx, dy, dem, GradientOptions{CellSize: 1})
	if dx.Data[dx.Index(2, 2)] != 1 || dy.Data[dy.Index(2, 2)] != 2 {
		t.Errorf("gradient = (%g, %g), want (1, 2)", dx.Data[dx.Index(2, 2)], dy.Data[dy.Index(2, 2)])
	}
}

// singleOutputOps runs each function that has one output, for the edge
// and validity tests.
var singleOutputOps = map[string]func(dst, dem raster.Float32Raster){
	"slope":     func(dst, dem raster.Float32Raster) { Slope(dst, dem, SlopeOptions{CellSize: 1}) },
	"aspect":    func(dst, dem raster.Float32Raster) { Aspect(dst, dem, AspectOptions{CellSize: 1}) },
	"hillshade": func(dst, dem raster.Float32Raster) { Hillshade(dst, dem, HillshadeOptions{CellSize: 1}) },
}

func TestSmallRastersAreAllBorder(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {2, 5}, {5, 2}, {1, 7}} {
		dem := plane(sz[0], sz[1], 1, 1, 1, 1)
		dem.Valid = raster.NewMask(len(dem.Data))
		for name, op := range singleOutputOps {
			dst := raster.NewFloat32Like(dem)
			op(dst, dem)
			for y := range dst.Height {
				for x := range dst.Width {
					if v := dst.Data[dst.Index(x, y)]; v == v || dst.IsValid(x, y) {
						t.Errorf("%s %v: cell (%d, %d) = %g valid=%v, want NaN and invalid", name, sz, x, y, v, dst.IsValid(x, y))
					}
				}
			}
		}
	}
}

// fixture is a dem and output set built as windows into larger parents
// with the given stride, so Stride > Width and non-zero ValidOffset are
// exercised.
type fixture struct {
	dem, out1, out2 raster.Float32Raster
	parentOut       raster.Float32Raster
}

func newFixture(rng *rand.Rand, w, h, extra int, masked bool, special float64) fixture {
	return newFlatFixture(rng, w, h, extra, masked, special, false)
}

// newFlatFixture is newFixture, with flat set making the DEM parent's
// cells equal in runs of columns and rows, so that some gradients are
// zero in one or both components.
func newFlatFixture(rng *rand.Rand, w, h, extra int, masked bool, special float64, flat bool) fixture {
	hazards := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		0, float32(math.Copysign(0, -1)), 3e38}
	// The DEM parent has a one-cell margin around the window (plus two
	// columns), so the window's edge is not the parent's.
	pw, ph, stride := w+3, h+2, w+3+extra
	demP := raster.NewFloat32Stride(pw, ph, stride, make([]float32, (ph-1)*stride+pw))
	for i := range demP.Data {
		demP.Data[i] = float32(rng.NormFloat64()*100 + 1000)
		if flat && (i%stride%7 < 4 || i/stride%5 < 3) {
			demP.Data[i] = 1000
		}
		if rng.Float64() < special {
			demP.Data[i] = hazards[rng.IntN(len(hazards))]
		}
	}
	// Both outputs are disjoint windows of one parent with a different
	// stride and, when masked, one shared mask of random bits.
	oh, ostride := 2*h+3, w+3+2*extra
	outP := raster.NewFloat32Stride(pw, oh, ostride, make([]float32, (oh-1)*ostride+pw))
	if masked {
		demP.Valid = raster.NewMask(len(demP.Data))
		for i := range demP.Data {
			if rng.Float64() < 0.03 {
				raster.MaskSet(demP.Valid, i, false)
			}
		}
		outP.Valid = raster.NewMask(len(outP.Data))
		for k := range outP.Valid {
			outP.Valid[k] = rng.Uint64()
		}
		if r := uint(len(outP.Data) & 63); r != 0 {
			outP.Valid[len(outP.Valid)-1] &= 1<<r - 1
		}
	}
	return fixture{
		dem:       demP.Window(2, 1, w, h),
		out1:      outP.Window(1, 1, w, h),
		out2:      outP.Window(2, h+2, w, h),
		parentOut: outP,
	}
}

func sameBits(a, b float32) bool {
	if a != a && b != b {
		return true
	}
	return math.Float32bits(a) == math.Float32bits(b)
}

// TestSIMDMatchesScalar runs every public function with the scalar and
// the SIMD kernels over windows with Stride > Width, masked and unmasked
// DEMs and hazardous values, and requires identical valid cells and
// identical masks.
func TestSIMDMatchesScalar(t *testing.T) {
	stencil.UseScalar(false)
	if stencil.Backend() == "scalar" {
		t.Skip("no SIMD backend in this build (needs GOEXPERIMENT=simd on an AVX2 amd64 CPU)")
	}
	defer stencil.UseScalar(false)
	type run func(f fixture)
	ops := map[string]run{
		"gradient": func(f fixture) {
			Gradient(f.out1, f.out2, f.dem, GradientOptions{CellSize: 7, CellSizeY: 3, ZFactor: 1.5})
		},
		"degrees": func(f fixture) { Slope(f.out1, f.dem, SlopeOptions{CellSize: 10}) },
		"percent": func(f fixture) { Slope(f.out1, f.dem, SlopeOptions{CellSize: 2, Units: SlopePercent}) },
		"radians": func(f fixture) { Slope(f.out1, f.dem, SlopeOptions{CellSize: 25, CellSizeY: 30, Units: SlopeRadians}) },
		"aspect":  func(f fixture) { Aspect(f.out1, f.dem, AspectOptions{CellSize: 10, CellSizeY: 12}) },
		"aspect-trig-zero": func(f fixture) {
			Aspect(f.out1, f.dem, AspectOptions{CellSize: 3, ZFactor: -2, Trigonometric: true, ZeroForFlat: true})
		},
		"hillshade": func(f fixture) { Hillshade(f.out1, f.dem, HillshadeOptions{CellSize: 30}) },
		"hillshade-low-east": func(f fixture) {
			Hillshade(f.out1, f.dem, HillshadeOptions{CellSize: 5, CellSizeY: 8, ZFactor: 3, Azimuth: 95, Altitude: 12})
		},
	}
	for name, op := range ops {
		for _, w := range []int{3, 4, 9, 10, 17, 63, 64, 65, 100} {
			for _, h := range []int{3, 4, 5, 7} {
				for _, extra := range []int{0, 5} {
					for _, masked := range []bool{false, true} {
						for _, special := range []float64{0, 0.2} {
							for _, flat := range []bool{false, true} {
								id := fmt.Sprintf("%s w=%d h=%d extra=%d masked=%v special=%v flat=%v", name, w, h, extra, masked, special, flat)
								seed := uint64(w*1000 + h*10 + extra)
								a := newFlatFixture(rand.New(rand.NewPCG(seed, 9)), w, h, extra, masked, special, flat)
								b := newFlatFixture(rand.New(rand.NewPCG(seed, 9)), w, h, extra, masked, special, flat)
								stencil.UseScalar(true)
								op(a)
								stencil.UseScalar(false)
								op(b)
								for i := range a.parentOut.Data {
									if !sameBits(a.parentOut.Data[i], b.parentOut.Data[i]) {
										if masked && !raster.MaskGet(a.parentOut.Valid, i) {
											continue
										}
										t.Fatalf("%s: parent cell %d: scalar %g (%#x), SIMD %g (%#x)", id, i,
											a.parentOut.Data[i], math.Float32bits(a.parentOut.Data[i]),
											b.parentOut.Data[i], math.Float32bits(b.parentOut.Data[i]))
									}
								}
								for k := range a.parentOut.Valid {
									if a.parentOut.Valid[k] != b.parentOut.Valid[k] {
										t.Fatalf("%s: mask word %d differs", id, k)
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestValidityMatchesNaive checks the output masks against a per-cell
// reference, at word boundaries, and checks that bits outside the output
// windows (row padding and other cells of the parent) are untouched.
func TestValidityMatchesNaive(t *testing.T) {
	for _, w := range []int{3, 62, 63, 64, 65, 66, 129} {
		for _, extra := range []int{0, 1, 61} {
			rng := rand.New(rand.NewPCG(uint64(w), uint64(extra)))
			for op := range 3 {
				f := newFixture(rng, w, 6, extra, true, 0)
				before := append([]uint64(nil), f.parentOut.Valid...)
				outs := []raster.Float32Raster{f.out1}
				switch op {
				case 0:
					Gradient(f.out1, f.out2, f.dem, GradientOptions{CellSize: 1})
					outs = append(outs, f.out2)
				case 1:
					Aspect(f.out1, f.dem, AspectOptions{CellSize: 1})
				case 2:
					Hillshade(f.out1, f.dem, HillshadeOptions{CellSize: 1})
				}

				inWindow := make(map[int]bool)
				for _, out := range outs {
					for y := range out.Height {
						for x := range out.Width {
							bit := out.ValidOffset + out.Index(x, y)
							inWindow[bit] = true
							want := !isBorder(out, x, y)
							for dy := -1; want && dy <= 1; dy++ {
								for dx := -1; want && dx <= 1; dx++ {
									want = f.dem.IsValid(x+dx, y+dy)
								}
							}
							if got := out.IsValid(x, y); got != want {
								t.Fatalf("op %d w=%d extra=%d: cell (%d, %d) valid=%v, want %v", op, w, extra, x, y, got, want)
							}
						}
					}
				}
				for i := range len(f.parentOut.Valid) * 64 {
					if !inWindow[i] && raster.MaskGet(f.parentOut.Valid, i) != raster.MaskGet(before, i) {
						t.Fatalf("op %d w=%d extra=%d: bit %d outside the outputs changed", op, w, extra, i)
					}
				}
			}
		}
	}
}

func TestUnmaskedDEM(t *testing.T) {
	for name, op := range singleOutputOps {
		t.Run(name, func(t *testing.T) { testUnmaskedDEM(t, op) })
	}
}

func testUnmaskedDEM(t *testing.T, op func(dst, dem raster.Float32Raster)) {
	dem := plane(6, 5, 1, 1, 1, 1)
	// A masked output of an unmasked DEM gets a valid interior and an
	// invalid border, whatever its bits were before. It is a window with
	// Stride > Width, so the parent's other cells and the row padding
	// must keep their bits.
	parent := raster.NewFloat32Stride(9, 7, 11, make([]float32, 6*11+9))
	parent.Valid = raster.NewMask(len(parent.Data))
	rng := rand.New(rand.NewPCG(11, 12))
	for k := range parent.Valid {
		parent.Valid[k] = rng.Uint64()
	}
	if r := uint(len(parent.Data) & 63); r != 0 {
		parent.Valid[len(parent.Valid)-1] &= 1<<r - 1
	}
	before := append([]uint64(nil), parent.Valid...)
	dst := parent.Window(2, 1, 6, 5)
	op(dst, dem)
	inWindow := make(map[int]bool)
	for y := range 5 {
		for x := range 6 {
			inWindow[dst.ValidOffset+dst.Index(x, y)] = true
			if want := !isBorder(dst, x, y); dst.IsValid(x, y) != want {
				t.Errorf("cell (%d, %d) valid=%v, want %v", x, y, dst.IsValid(x, y), want)
			}
		}
	}
	for i := range len(parent.Valid) * 64 {
		if !inWindow[i] && raster.MaskGet(parent.Valid, i) != raster.MaskGet(before, i) {
			t.Fatalf("bit %d outside dst changed", i)
		}
	}
	// An unmasked output stays unmasked.
	plain := raster.NewFloat32Like(dem)
	op(plain, dem)
	if plain.Valid != nil {
		t.Error("unmasked output gained a mask")
	}
	checkBorderNaN(t, "plain", plain)
}

func TestPanics(t *testing.T) {
	dem := plane(8, 8, 1, 1, 1, 1)
	ok := SlopeOptions{CellSize: 1}
	mustPanic(t, "size mismatch", func() { Slope(raster.NewFloat32(8, 7, make([]float32, 56)), dem, ok) })
	mustPanic(t, "dst is dem", func() { Slope(dem, dem, ok) })
	mustPanic(t, "dst overlaps dem", func() {
		big := plane(8, 10, 1, 1, 1, 1)
		Slope(big.Window(0, 2, 8, 8), big.Window(0, 0, 8, 8), ok)
	})
	mustPanic(t, "dx is dy", func() {
		d := raster.NewFloat32Like(dem)
		Gradient(d, d, dem, GradientOptions{CellSize: 1})
	})
	masked := plane(8, 8, 1, 1, 1, 1)
	masked.Valid = raster.NewMask(64)
	mustPanic(t, "missing output mask", func() { Slope(raster.NewFloat32Like(dem), masked, ok) })
	mustPanic(t, "missing dy mask", func() {
		Gradient(raster.NewFloat32Like(masked), raster.NewFloat32Like(dem), masked, GradientOptions{CellSize: 1})
	})
	mustPanic(t, "shared mask bits", func() {
		dst := raster.NewFloat32Like(dem)
		dst.Valid = masked.Valid
		Slope(dst, masked, ok)
	})
	mustPanic(t, "invalid raster", func() { Slope(raster.Float32Raster{}, dem, ok) })
	for _, o := range []SlopeOptions{
		{}, {CellSize: -1}, {CellSize: math.NaN()}, {CellSize: math.Inf(1)},
		{CellSize: 1, CellSizeY: -2}, {CellSize: 1, ZFactor: math.Inf(-1)}, {CellSize: 1, Units: 7},
		// Scale factors that overflow or underflow float32: flat cells
		// would be NaN (0·Inf), or every cell flat.
		{CellSize: 1e-300}, {CellSize: 1e-40}, {CellSize: 1, ZFactor: 1e300}, {CellSize: 1, CellSizeY: 1e-40},
		{CellSize: 1e300}, {CellSize: 1, ZFactor: 1e-50}, {CellSize: math.SmallestNonzeroFloat64},
	} {
		mustPanic(t, fmt.Sprintf("options %+v", o), func() { Slope(raster.NewFloat32Like(dem), dem, o) })
	}

	for name, op := range singleOutputOps {
		mustPanic(t, name+": dst is dem", func() { op(dem, dem) })
		mustPanic(t, name+": size mismatch", func() { op(raster.NewFloat32(7, 8, make([]float32, 56)), dem) })
		mustPanic(t, name+": missing output mask", func() { op(raster.NewFloat32Like(dem), masked) })
	}
	for _, o := range []AspectOptions{
		{}, {CellSize: math.NaN()}, {CellSize: 1, CellSizeY: math.Inf(1)}, {CellSize: 1, ZFactor: math.NaN()},
	} {
		mustPanic(t, fmt.Sprintf("aspect options %+v", o), func() { Aspect(raster.NewFloat32Like(dem), dem, o) })
	}
	for _, o := range []HillshadeOptions{
		{}, {CellSize: -3}, {CellSize: 1, ZFactor: math.Inf(1)},
		{CellSize: 1, Azimuth: math.NaN()}, {CellSize: 1, Azimuth: math.Inf(-1)},
		{CellSize: 1, Altitude: -10}, {CellSize: 1, Altitude: 90.5}, {CellSize: 1, Altitude: math.NaN()},
	} {
		mustPanic(t, fmt.Sprintf("hillshade options %+v", o), func() { Hillshade(raster.NewFloat32Like(dem), dem, o) })
	}
	// Tiny but representable scale factors are fine.
	Slope(raster.NewFloat32Like(dem), dem, SlopeOptions{CellSize: 1e30, ZFactor: 1e-5})
	// Azimuths outside [0, 360] are directions like any other.
	Hillshade(raster.NewFloat32Like(dem), dem, HillshadeOptions{CellSize: 1, Azimuth: -45, Altitude: 90})

	// Disjoint windows of one parent are fine, including a shared mask.
	parent := plane(8, 20, 1, 1, 1, 1)
	parent.Valid = raster.NewMask(160)
	Slope(parent.Window(0, 10, 8, 8), parent.Window(0, 0, 8, 8), ok)
}
