package terrain

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/raster"
)

var curvatureTypes = []CurvatureType{CurvatureProfile, CurvaturePlan, CurvatureMean}

func (c CurvatureType) String() string {
	switch c {
	case CurvatureProfile:
		return "profile"
	case CurvaturePlan:
		return "plan"
	case CurvatureMean:
		return "mean"
	}
	return fmt.Sprintf("CurvatureType(%d)", int(c))
}

// ztDerivs64 is the Zevenbergen–Thorne p, q, r, s and t of the window
// centred on (x, y) of dem, in float64.
func ztDerivs64(dem raster.Float32Raster, x, y int, cx, cy, z float64) [5]float64 {
	at := func(dx, dy int) float64 { return float64(dem.Data[dem.Index(x+dx, y+dy)]) }
	return [5]float64{
		z * (at(1, 0) - at(-1, 0)) / (2 * cx),
		z * (at(0, 1) - at(0, -1)) / (2 * cy),
		z * (at(-1, 0) - 2*at(0, 0) + at(1, 0)) / (cx * cx),
		z * (at(-1, -1) - at(1, -1) - at(-1, 1) + at(1, 1)) / (4 * cx * cy),
		z * (at(0, -1) - 2*at(0, 0) + at(0, 1)) / (cy * cy),
	}
}

// curvature64 is the curvature of type c from d = (p, q, r, s, t), in
// float64, with 0 for profile and plan where p = q = 0.
func curvature64(c CurvatureType, d [5]float64) float64 {
	p, q, r, s, t := d[0], d[1], d[2], d[3], d[4]
	g := p*p + q*q
	w := 1 + g
	switch c {
	case CurvatureProfile:
		if g == 0 {
			return 0
		}
		return -(p*p*r + 2*p*q*s + q*q*t) / (g * w * math.Sqrt(w))
	case CurvaturePlan:
		if g == 0 {
			return 0
		}
		return -(q*q*r - 2*p*q*s + p*p*t) / (g * math.Sqrt(g))
	default:
		return -((1+q*q)*r - 2*p*q*s + (1+p*p)*t) / (2 * w * math.Sqrt(w))
	}
}

// curvatureTol bounds the difference between the float32 curvature and
// curvature64(c, d), given bounds e on the error of each float32
// derivative: e carried to first order through the formula (by numeric
// partial derivatives), plus the rounding of the formula itself, about
// one float32 epsilon per operation on the magnitude of its terms.
func curvatureTol(c CurvatureType, d, e [5]float64) float64 {
	const eps = 0x1p-24
	f := curvature64(c, d)
	var tol float64
	for i := range d {
		if e[i] == 0 {
			continue
		}
		h := 1e-7 * max(math.Abs(d[i]), 1e-30)
		up, dn := d, d
		up[i] += h
		dn[i] -= h
		tol += math.Abs(curvature64(c, up)-curvature64(c, dn)) / (2 * h) * e[i]
	}
	// The formula with every term made positive gives the scale of
	// the rounding of the numerator's sums.
	abs := [5]float64{math.Abs(d[0]), math.Abs(d[1]), math.Abs(d[2]), -math.Abs(d[3]), math.Abs(d[4])}
	if c == CurvatureProfile {
		abs[3] = math.Abs(d[3])
	}
	scale := math.Abs(curvature64(c, abs))
	return tol + 16*eps*(scale+math.Abs(f))
}

// derivErrors bounds the error of the float32 derivatives of the window
// at (x, y): each is a sum of at most four elevations, rounded at most
// three times, then multiplied by a factor that is itself rounded.
func derivErrors(dem raster.Float32Raster, x, y int, cx, cy, z float64) [5]float64 {
	const eps = 0x1p-24
	var m float64
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			m = max(m, math.Abs(float64(dem.Data[dem.Index(x+dx, y+dy)])))
		}
	}
	d := ztDerivs64(dem, x, y, cx, cy, z)
	k := [5]float64{z / (2 * cx), z / (2 * cy), z / (cx * cx), z / (4 * cx * cy), z / (cy * cy)}
	var e [5]float64
	for i := range e {
		e[i] = 3*4*eps*m*math.Abs(k[i]) + 2*eps*math.Abs(d[i])
	}
	return e
}

// TestCurvatureQuadric uses z = aX² + bXY + cY² + dX + eY with integer
// coefficients and power-of-two cell sizes, where every elevation and
// every Zevenbergen–Thorne derivative is exact in float32 and equal to
// the analytic one, so only the formula's rounding separates the result
// from the analytic curvature.
func TestCurvatureQuadric(t *testing.T) {
	for _, tc := range []struct {
		a, b, c, d, e float64
		cx, cy, z     float64
	}{
		{3, -2, 1, 5, -7, 2, 4, 1},
		{-1, 0, -1, 0, 0, 1, 1, 1},      // a dome, flat at its top
		{1, 0, -1, 0.5, 0, 0.5, 0.5, 1}, // a saddle
		{2, 3, -4, -1, 2, 8, 2, -0.25},
	} {
		const w, h = 9, 7
		d := make([]float32, w*h)
		for y := range h {
			for x := range w {
				X, Y := float64(x-w/2)*tc.cx, float64(y-h/2)*tc.cy
				d[y*w+x] = float32(tc.a*X*X + tc.b*X*Y + tc.c*Y*Y + tc.d*X + tc.e*Y)
			}
		}
		dem := raster.NewFloat32(w, h, d)
		for _, ct := range curvatureTypes {
			dst := raster.NewFloat32Like(dem)
			Curvature(dst, dem, CurvatureOptions{CellSize: tc.cx, CellSizeY: tc.cy, ZFactor: tc.z, Type: ct})
			checkBorderNaN(t, "quadric", dst)
			eachInterior(dst, func(x, y int, v float32) {
				X, Y := float64(x-w/2)*tc.cx, float64(y-h/2)*tc.cy
				z := tc.z
				der := [5]float64{
					z * (2*tc.a*X + tc.b*Y + tc.d), z * (tc.b*X + 2*tc.c*Y + tc.e),
					z * 2 * tc.a, z * tc.b, z * 2 * tc.c,
				}
				if got := ztDerivs64(dem, x, y, tc.cx, tc.cy, tc.z); got != der {
					t.Fatalf("%+v: derivatives at (%d, %d) = %v, want %v", tc, x, y, got, der)
				}
				want := curvature64(ct, der)
				if tol := curvatureTol(ct, der, [5]float64{}); math.Abs(float64(v)-want) > tol {
					t.Errorf("%+v %v: (%d, %d) = %g, want %g ± %.3g", tc, ct, x, y, v, want, tol)
				}
			})
		}
	}
}

func TestCurvatureFlatAndPlane(t *testing.T) {
	flat := make([]float32, 8*6)
	for i := range flat {
		flat[i] = 812.25
	}
	for name, dem := range map[string]raster.Float32Raster{
		"flat":  raster.NewFloat32(8, 6, flat),
		"plane": plane(8, 6, 3, -2, 1, 1),
	} {
		dem.Valid = raster.NewMask(len(dem.Data))
		raster.MaskFillRange(dem.Valid, 0, len(dem.Data), true)
		for _, ct := range curvatureTypes {
			for _, o := range []CurvatureOptions{{CellSize: 1, Type: ct}, {CellSize: 3, CellSizeY: 0.5, ZFactor: -2, Type: ct}} {
				dst := raster.NewFloat32Like(dem)
				Curvature(dst, dem, o)
				checkBorderNaN(t, name, dst)
				eachInterior(dst, func(x, y int, v float32) {
					if math.Float32bits(v) != 0 || !dst.IsValid(x, y) {
						t.Errorf("%s %+v: (%d, %d) = %g (%#x) valid=%v, want +0 and valid",
							name, o, x, y, v, math.Float32bits(v), dst.IsValid(x, y))
					}
				})
			}
		}
	}
}

// TestCurvatureSigns checks the sign convention: positive is convex.
func TestCurvatureSigns(t *testing.T) {
	surface := func(f func(X, Y float64) float64) raster.Float32Raster {
		d := make([]float32, 9)
		for y := range 3 {
			for x := range 3 {
				d[y*3+x] = float32(f(float64(x+3), float64(y-1)))
			}
		}
		return raster.NewFloat32(3, 3, d)
	}
	at := func(dem raster.Float32Raster, ct CurvatureType) float32 {
		dst := raster.NewFloat32Like(dem)
		Curvature(dst, dem, CurvatureOptions{CellSize: 1, Type: ct})
		return dst.Data[4]
	}
	// Windows centred on (4, 0), on the flank of each surface.
	dome := surface(func(X, Y float64) float64 { return 100 - X*X - Y*Y })
	bowl := surface(func(X, Y float64) float64 { return X*X + Y*Y })
	saddle := surface(func(X, Y float64) float64 { return X*X - Y*Y })
	for _, ct := range curvatureTypes {
		if v := at(dome, ct); !(v > 0) {
			t.Errorf("dome %v = %g, want > 0", ct, v)
		}
		if v := at(bowl, ct); !(v < 0) {
			t.Errorf("bowl %v = %g, want < 0", ct, v)
		}
	}
	// Along the saddle's rising axis the profile is concave (the slope
	// steepens uphill) and the contours bulge outwards.
	if v := at(saddle, CurvatureProfile); !(v < 0) {
		t.Errorf("saddle profile = %g, want < 0", v)
	}
	if v := at(saddle, CurvaturePlan); !(v > 0) {
		t.Errorf("saddle plan = %g, want > 0", v)
	}
}

// TestCurvatureHandComputed uses the window
//
//	10 12 15
//	11 13 17
//	13 16 20
//
// with unit cells: p = (17 - 11)/2 = 3, q = (16 - 12)/2 = 2,
// r = 11 - 26 + 17 = 2, t = 12 - 26 + 16 = 2, s = (10 + 20 - 15 - 13)/4
// = 0.5, g = 13. Profile is -(9·2 + 2·3·2·0.5 + 4·2)/(13·14^1.5) =
// -32/(13·14^1.5), plan -(4·2 - 6 + 9·2)/13^1.5 = -20/13^1.5 and mean
// -(5·2 - 6 + 10·2)/(2·14^1.5) = -12/14^1.5.
func TestCurvatureHandComputed(t *testing.T) {
	dem := raster.NewFloat32(3, 3, []float32{
		10, 12, 15,
		11, 13, 17,
		13, 16, 20,
	})
	for ct, want := range map[CurvatureType]float64{
		CurvatureProfile: -32 / (13 * math.Pow(14, 1.5)),
		CurvaturePlan:    -20 / math.Pow(13, 1.5),
		CurvatureMean:    -12 / math.Pow(14, 1.5),
	} {
		dst := raster.NewFloat32Like(dem)
		Curvature(dst, dem, CurvatureOptions{CellSize: 1, Type: ct})
		if got := dst.Data[4]; math.Abs(float64(got)-want) > 1e-6*math.Abs(want) {
			t.Errorf("%v = %.8g, want %.8g", ct, got, want)
		}
		checkBorderNaN(t, ct.String(), dst)
	}
}

// TestCurvatureNaNCentre checks that the centre is read: a NaN or
// infinite centre among equal neighbours is not a flat cell, and gives a
// result that is not finite (mean curvature, which has no flat case, is
// ∓Inf for an infinite centre).
func TestCurvatureNaNCentre(t *testing.T) {
	for _, c := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		dem := raster.NewFloat32(3, 3, []float32{5, 5, 5, 5, c, 5, 5, 5, 5})
		for _, ct := range curvatureTypes {
			dst := raster.NewFloat32Like(dem)
			Curvature(dst, dem, CurvatureOptions{CellSize: 1, Type: ct})
			v := dst.Data[4]
			want := "NaN"
			if ct == CurvatureMean && c == c {
				want = "infinite"
			}
			if got := v != v; want == "NaN" && !got || want == "infinite" && !math.IsInf(float64(v), 0) {
				t.Errorf("centre %g: %v = %g, want %s", c, ct, v, want)
			}
		}
	}
}

// TestCurvatureRandomWindows compares Curvature with a float64 reference
// over random 3×3 windows of every orientation and scale, some nearly
// flat, within the derived bound of curvatureTol.
func TestCurvatureRandomWindows(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	const n = 300
	d := make([]float32, n*3)
	for i := range d {
		col := i % n
		scale := math.Pow(10, float64(col%5))
		if col%7 == 0 {
			// Nearly flat: a large base with small relief.
			d[i] = float32(1000 + rng.NormFloat64()*1e-3)
			continue
		}
		d[i] = float32(rng.NormFloat64() * scale)
	}
	dem := raster.NewFloat32(n, 3, d)
	const cx, cy = 3, 4
	for _, ct := range curvatureTypes {
		dst := raster.NewFloat32Like(dem)
		Curvature(dst, dem, CurvatureOptions{CellSize: cx, CellSizeY: cy, Type: ct})
		var worst float64
		eachInterior(dst, func(x, y int, v float32) {
			der := ztDerivs64(dem, x, y, cx, cy, 1)
			want := curvature64(ct, der)
			tol := curvatureTol(ct, der, derivErrors(dem, x, y, cx, cy, 1))
			e := math.Abs(float64(v) - want)
			worst = max(worst, e/tol)
			if !(e <= tol) {
				t.Errorf("%v: (%d, %d) = %g, want %g ± %.3g", ct, x, y, v, want, tol)
			}
		})
		t.Logf("%v: largest error %.3g of the bound", ct, worst)
	}
}
