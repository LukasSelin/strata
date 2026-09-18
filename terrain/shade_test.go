package terrain

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/raster"
)

// angleDiff is the absolute difference of two angles in degrees, modulo
// 360.
func angleDiff(a, b float64) float64 {
	d := math.Mod(math.Abs(a-b), 360)
	return min(d, 360-d)
}

func TestAspectPlanes(t *testing.T) {
	// Each plane falls towards the named direction: elevation rises
	// against it. ax is the rise per unit eastward (column), ay per unit
	// southward (row). Horn is exact on a plane, so only float32 rounding
	// and the arctangent separate the result from the analytic bearing.
	const tol = 1e-4
	for _, tc := range []struct {
		name          string
		ax, ay        float64
		bearing, trig float64
	}{
		{"N", 0, 0.5, 0, 90},
		{"NE", -0.3, 0.3, 45, 45},
		{"E", -0.5, 0, 90, 0},
		{"SE", -0.2, -0.2, 135, 315},
		{"S", 0, -0.5, 180, 270},
		{"SW", 0.4, -0.4, 225, 225},
		{"W", 0.5, 0, 270, 180},
		{"NW", 0.1, 0.1, 315, 135},
		{"NNE", -0.25, 0.25 * math.Sqrt(3), 30, 60},
	} {
		dem := plane(9, 6, tc.ax, tc.ay, 10, 10)
		for _, trig := range []bool{false, true} {
			want := tc.bearing
			if trig {
				want = tc.trig
			}
			dst := raster.NewFloat32Like(dem)
			Aspect(dst, dem, AspectOptions{CellSize: 10, Trigonometric: trig})
			checkBorderNaN(t, "aspect", dst)
			eachInterior(dst, func(x, y int, v float32) {
				if !(v >= 0 && v < 360) || angleDiff(float64(v), want) > tol {
					t.Errorf("%s trig=%v: aspect(%d, %d) = %g, want %g ± %g", tc.name, trig, x, y, v, want, tol)
				}
			})
		}
	}

	// On anisotropic cells the direction is that of the true gradient:
	// rising 1 per 20 units east and 1 per 10 units south falls towards
	// bearing atan2(-1/20, 1/10).
	dem := plane(7, 7, 0.05, 0.1, 20, 10)
	dst := raster.NewFloat32Like(dem)
	Aspect(dst, dem, AspectOptions{CellSize: 20, CellSizeY: 10})
	want := 360 + math.Atan2(-0.05, 0.1)*180/math.Pi
	eachInterior(dst, func(x, y int, v float32) {
		if angleDiff(float64(v), want) > tol {
			t.Errorf("anisotropic: aspect(%d, %d) = %g, want %g", x, y, v, want)
		}
	})
}

// TestAspectExactAxes checks the axis directions, where the folding of
// signed zeros and of results that round to 360 matters: none may be -0
// or 360.
func TestAspectExactAxes(t *testing.T) {
	for _, tc := range []struct {
		ax, ay float64
		want   float32
	}{
		{0, 1, 0}, {-1, 0, 90}, {0, -1, 180}, {1, 0, 270},
	} {
		dem := plane(5, 5, tc.ax, tc.ay, 1, 1)
		dst := raster.NewFloat32Like(dem)
		Aspect(dst, dem, AspectOptions{CellSize: 1})
		eachInterior(dst, func(x, y int, v float32) {
			// 90 and 270 are float32 π/2 and -π/2 in degrees, which need
			// not round to exactly 90 and 270.
			if math.Abs(float64(v-tc.want)) > 2e-5 || math.Float32bits(v) == math.Float32bits(float32(math.Copysign(0, -1))) {
				t.Errorf("plane (%g, %g): aspect(%d, %d) = %g (%#x), want %g", tc.ax, tc.ay, x, y, v, math.Float32bits(v), tc.want)
			}
		})
	}
	// A bearing a hair west of north rounds to 360 and is written as 0; a
	// hair east of north stays tiny and positive.
	dem := raster.NewFloat32(3, 3, []float32{0, 0, 0, 0, 0, 0, 1e6, 1e6, 1e6 + 0.0625})
	for _, ax := range []float32{0.0625, -0.0625} {
		dem.Data[2*3+2] = 1e6 + ax
		dst := raster.NewFloat32Like(dem)
		Aspect(dst, dem, AspectOptions{CellSize: 1})
		v := dst.Data[4]
		if !(v >= 0 && v < 1e-4) || math.Signbit(float64(v)) {
			t.Errorf("dx sign %g: aspect = %g (%#x), want a tiny non-negative value", ax, v, math.Float32bits(v))
		}
	}
}

func TestAspectFlat(t *testing.T) {
	d := make([]float32, 10*6)
	for i := range d {
		d[i] = 812.25
	}
	dem := raster.NewFloat32(10, 6, d)
	dem.Valid = raster.NewMask(len(d))
	raster.MaskFillRange(dem.Valid, 0, len(d), true)
	for _, tc := range []struct {
		opts AspectOptions
		want float32
	}{
		{AspectOptions{CellSize: 5}, -1},
		{AspectOptions{CellSize: 5, Trigonometric: true}, -1},
		{AspectOptions{CellSize: 5, ZeroForFlat: true}, 0},
	} {
		dst := raster.NewFloat32Like(dem)
		Aspect(dst, dem, tc.opts)
		checkBorderNaN(t, "flat aspect", dst)
		eachInterior(dst, func(x, y int, v float32) {
			if math.Float32bits(v) != math.Float32bits(tc.want) || !dst.IsValid(x, y) {
				t.Errorf("%+v: flat aspect(%d, %d) = %g valid=%v, want %g and valid", tc.opts, x, y, v, dst.IsValid(x, y), tc.want)
			}
		})
	}
	// A slope in one axis only is not flat.
	dem = plane(5, 5, 0, 1e-3, 1, 1)
	dst := raster.NewFloat32Like(dem)
	Aspect(dst, dem, AspectOptions{CellSize: 1})
	if v := dst.Data[dst.Index(2, 2)]; v != 0 {
		t.Errorf("gentle north-facing plane: aspect = %g, want 0", v)
	}
}

// TestAspectHandComputed uses the 3×3 window of Esri's "How Aspect works"
// (dz/dx = -8.125, dz/dy = -0.375 with y down the rows), whose published
// aspect is 92.64°. The reference here is computed by hand in float64:
// the downslope direction (8.125 east, -0.375 north) has bearing
// 90° + atan(0.375/8.125) = 92.6425°. gdaldem is not installed, so there
// is no gdaldem comparison.
func TestAspectHandComputed(t *testing.T) {
	dem := raster.NewFloat32(3, 3, []float32{
		101, 92, 85,
		101, 92, 85,
		101, 91, 84,
	})
	dst := raster.NewFloat32Like(dem)
	Aspect(dst, dem, AspectOptions{CellSize: 5})
	want := 90 + math.Atan(0.375/8.125)*180/math.Pi
	if got := dst.Data[4]; math.Abs(float64(got)-want) > 1e-4 || math.Abs(want-92.64) > 5e-3 {
		t.Errorf("aspect = %.5f, want %.5f", got, want)
	}
	checkBorderNaN(t, "aspect", dst)
}

// TestAspectRandomWindows compares Aspect with a float64 reference over
// random 3×3 windows of every orientation and scale.
func TestAspectRandomWindows(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	const n = 200
	d := make([]float32, n*3)
	for i := range d {
		d[i] = float32(rng.NormFloat64() * math.Pow(10, float64(rng.IntN(5))))
	}
	dem := raster.NewFloat32(n, 3, d)
	for _, trig := range []bool{false, true} {
		dst := raster.NewFloat32Like(dem)
		Aspect(dst, dem, AspectOptions{CellSize: 3, CellSizeY: 4, Trigonometric: trig})
		var worst float64
		eachInterior(dst, func(x, _ int, v float32) {
			z := func(dx, dy int) float64 { return float64(d[(1+dy)*n+x+dx]) }
			gx := ((z(1, -1) + 2*z(1, 0) + z(1, 1)) - (z(-1, -1) + 2*z(-1, 0) + z(-1, 1))) / (8 * 3)
			gy := ((z(-1, 1) + 2*z(0, 1) + z(1, 1)) - (z(-1, -1) + 2*z(0, -1) + z(1, -1))) / (8 * 4)
			want := math.Atan2(-gx, gy) * 180 / math.Pi
			if trig {
				want = math.Atan2(gy, -gx) * 180 / math.Pi
			}
			e := angleDiff(float64(v), want)
			worst = max(worst, e)
			if !(v >= 0 && v < 360) || e > 1e-3 {
				t.Errorf("trig=%v: aspect(%d) = %g, want %g", trig, x, v, want)
			}
		})
		t.Logf("trig=%v: largest difference from float64 %.3g degrees", trig, worst)
	}
}

func TestHillshadeFlat(t *testing.T) {
	d := make([]float32, 8*5)
	for i := range d {
		d[i] = -42
	}
	dem := raster.NewFloat32(8, 5, d)
	for _, tc := range []struct {
		opts HillshadeOptions
		alt  float64
	}{
		{HillshadeOptions{CellSize: 30}, 45},
		{HillshadeOptions{CellSize: 30, Azimuth: 90, Altitude: 30}, 30},
		{HillshadeOptions{CellSize: 30, Azimuth: 360, Altitude: 90}, 90},
		{HillshadeOptions{CellSize: 30, Altitude: 1e-3}, 1e-3},
	} {
		dst := raster.NewFloat32Like(dem)
		Hillshade(dst, dem, tc.opts)
		checkBorderNaN(t, "hillshade", dst)
		want := 255 * math.Sin(tc.alt*math.Pi/180)
		eachInterior(dst, func(x, y int, v float32) {
			if math.Abs(float64(v)-want) > 1e-4 {
				t.Errorf("%+v: flat hillshade(%d, %d) = %g, want %g", tc.opts, x, y, v, want)
			}
		})
	}
}

// TestHillshadeHugeAzimuth checks that azimuths of any finite size are
// directions: one past 1e300 degrees once overflowed to Inf in radians and
// made every cell NaN.
func TestHillshadeHugeAzimuth(t *testing.T) {
	dem := plane(6, 5, 3, -1, 1, 1)
	for _, az := range []float64{1e308, -1e308, 360 * (1 << 60), 1e20 + 45} {
		got, want := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
		Hillshade(got, dem, HillshadeOptions{CellSize: 1, Azimuth: az})
		reduced := math.Mod(az, 360)
		if reduced == 0 {
			reduced = 360 // 0 would mean the default
		}
		Hillshade(want, dem, HillshadeOptions{CellSize: 1, Azimuth: reduced})
		eachInterior(got, func(x, y int, v float32) {
			if w := want.Data[want.Index(x, y)]; v != w {
				t.Errorf("azimuth %g: hillshade(%d, %d) = %g, want %g as for %g", az, x, y, v, w, reduced)
			}
		})
	}
}

func TestHillshadeFacingLight(t *testing.T) {
	// A plane whose normal points at the light: its aspect is the azimuth
	// and its slope 90° − altitude, so it falls by tan(90° − alt) per unit
	// towards the light.
	for _, tc := range []struct{ az, alt float64 }{
		{315, 45}, {0, 45}, {90, 30}, {200, 60}, {45, 80}, {270, 10},
	} {
		azr, altr := tc.az*math.Pi/180, tc.alt*math.Pi/180
		fall := math.Tan(math.Pi/2 - altr)
		// Rises per unit east and south: the light is at (sin az, -cos az).
		ax, ay := -fall*math.Sin(azr), fall*math.Cos(azr)
		dem := plane(9, 7, ax, ay, 2, 2)
		dst := raster.NewFloat32Like(dem)
		opts := HillshadeOptions{CellSize: 2, Azimuth: tc.az, Altitude: tc.alt}
		if tc.az == 0 {
			opts.Azimuth = 360
		}
		Hillshade(dst, dem, opts)
		eachInterior(dst, func(x, y int, v float32) {
			if math.Abs(float64(v)-255) > 5e-3 || v > 255 {
				t.Errorf("az=%g alt=%g: hillshade(%d, %d) = %g, want 255", tc.az, tc.alt, x, y, v)
			}
		})

		// The opposite plane faces away from the light, at an angle of
		// 2·slope to it: 255·max(0, sin(2·alt − 90°)), which is clamped
		// to +0 below 45° and only rounding away from 0 at 45°.
		dem = plane(9, 7, -ax, -ay, 2, 2)
		Hillshade(dst, dem, opts)
		away := max(0, 255*math.Sin(2*altr-math.Pi/2))
		eachInterior(dst, func(x, y int, v float32) {
			if tc.alt < 45 && math.Float32bits(v) != 0 || math.Abs(float64(v)-away) > 5e-3 {
				t.Errorf("az=%g alt=%g: away from light, hillshade(%d, %d) = %g, want %g", tc.az, tc.alt, x, y, v, away)
			}
		})
	}
}

// TestHillshadeHandComputed compares Hillshade with the angle form of the
// formula, evaluated in float64 from slope and aspect, over random
// windows, lights and anisotropic cells. It also checks the Esri aspect
// window under the default light by hand: dx = -1.625, dy = -0.075
// (cell size 5), slope = atan(1.62673) = 58.4197°, aspect = 92.6425°,
// 255·(sin 45°·cos 58.4197° + cos 45°·sin 58.4197°·cos(315° − 92.6425°))
// = 255·(0.37031 − 0.44514) < 0, so the cell is 0 (lit from the
// northwest, it faces east). Lit from the east it is
// 255·(0.37031 + 0.60239·cos(90° − 92.6425°)) = 247.874. gdaldem is not
// installed, so there is no gdaldem comparison.
func TestHillshadeHandComputed(t *testing.T) {
	esri := raster.NewFloat32(3, 3, []float32{
		101, 92, 85,
		101, 92, 85,
		101, 91, 84,
	})
	dst := raster.NewFloat32Like(esri)
	Hillshade(dst, esri, HillshadeOptions{CellSize: 5})
	if got := dst.Data[4]; got != 0 {
		t.Errorf("Esri window, default light: hillshade = %g, want 0", got)
	}
	Hillshade(dst, esri, HillshadeOptions{CellSize: 5, Azimuth: 90})
	s := math.Atan(math.Hypot(1.625, 0.075))
	asp := (90 + math.Atan(0.075/1.625)*180/math.Pi) * math.Pi / 180
	want := 255 * (math.Sin(math.Pi/4)*math.Cos(s) + math.Cos(math.Pi/4)*math.Sin(s)*math.Cos(math.Pi/2-asp))
	if got := dst.Data[4]; math.Abs(float64(got)-want) > 1e-3 || math.Abs(want-247.874) > 1e-3 {
		t.Errorf("Esri window, light from the east: hillshade = %.4f, want %.4f", got, want)
	}

	rng := rand.New(rand.NewPCG(41, 42))
	const n = 300
	d := make([]float32, n*3)
	for i := range d {
		d[i] = float32(rng.NormFloat64() * 20)
	}
	dem := raster.NewFloat32(n, 3, d)
	for range 20 {
		az, alt := rng.Float64()*720-360, 0.5+rng.Float64()*89.5
		if az == 0 {
			continue
		}
		cs, csy := 1+rng.Float64()*30, 1+rng.Float64()*30
		dst := raster.NewFloat32Like(dem)
		Hillshade(dst, dem, HillshadeOptions{CellSize: cs, CellSizeY: csy, Azimuth: az, Altitude: alt})
		eachInterior(dst, func(x, _ int, v float32) {
			z := func(dx, dy int) float64 { return float64(d[(1+dy)*n+x+dx]) }
			gx := ((z(1, -1) + 2*z(1, 0) + z(1, 1)) - (z(-1, -1) + 2*z(-1, 0) + z(-1, 1))) / (8 * cs)
			gy := ((z(-1, 1) + 2*z(0, 1) + z(1, 1)) - (z(-1, -1) + 2*z(0, -1) + z(1, -1))) / (8 * csy)
			slope := math.Atan(math.Hypot(gx, gy))
			aspect := math.Atan2(-gx, gy)
			altr, azr := alt*math.Pi/180, az*math.Pi/180
			want := 255 * (math.Sin(altr)*math.Cos(slope) + math.Cos(altr)*math.Sin(slope)*math.Cos(azr-aspect))
			want = max(want, 0)
			if !(v >= 0 && v <= 255) || math.Abs(float64(v)-want) > 2e-3 {
				t.Errorf("az=%g alt=%g cs=%g,%g: hillshade(%d) = %g, want %g", az, alt, cs, csy, x, v, want)
			}
		})
	}
}
