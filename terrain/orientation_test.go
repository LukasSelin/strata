package terrain

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// orientationAngles is northness or eastness as Geomorpho90m writes it,
// in float64 from the gradient (gx, gy): slope by atan, aspect as a
// compass bearing by atan2, then sin(slope)·cos(aspect) or
// sin(slope)·sin(aspect), or the cosine or sine alone, 0 where flat.
func orientationAngles(o OrientationOptions, gx, gy float64) float64 {
	if o.Unweighted && gx == 0 && gy == 0 {
		return 0
	}
	slope := math.Atan(math.Hypot(gx, gy))
	aspect := math.Atan2(-gx, gy)
	w := math.Sin(slope)
	if o.Unweighted {
		w = 1
	}
	if o.Component == Eastness {
		return w * math.Sin(aspect)
	}
	return w * math.Cos(aspect)
}

// TestOrientationPlanes checks planes facing each of the eight compass
// directions at several steepnesses against the closed form: on a plane
// z = a·x + b·y (x east, y south) the gradient is (a, b), so northness is
// b/sqrt(1 + a² + b²) and eastness -a/sqrt(1 + a² + b²), or over
// sqrt(a² + b²) unweighted. Horn is exact on a plane up to the float32
// rounding of the elevations, about 1e-7 of the gradient here.
func TestOrientationPlanes(t *testing.T) {
	const tol = 1e-6
	for _, dir := range [][2]float64{{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}} {
		for _, steep := range []float64{0.05, 0.5, 1, 4} {
			a, b := dir[0]*steep, dir[1]*steep
			dem := plane(7, 6, a, b, 10, 10)
			for _, c := range []OrientationComponent{Northness, Eastness} {
				for _, unweighted := range []bool{false, true} {
					o := OrientationOptions{CellSize: 10, Component: c, Unweighted: unweighted}
					comp := b
					if c == Eastness {
						comp = -a
					}
					want := comp / math.Sqrt(1+a*a+b*b)
					if unweighted {
						want = comp / math.Hypot(a, b)
					}
					dst := raster.NewFloat32Like(dem)
					Orientation(dst, dem, o)
					checkBorderNaN(t, "orientation", dst)
					eachInterior(dst, func(x, y int, v float32) {
						if math.Abs(float64(v)-want) > tol {
							t.Fatalf("plane (%v, %v) %+v: cell (%d, %d) = %v, want %v", a, b, o, x, y, v, want)
						}
					})
				}
			}
		}
	}
}

// TestOrientationCardinal checks the values a slope facing exactly north
// or east has: unweighted, a component of exactly ±1 and one of exactly
// 0; weighted, 0 across it. A plane with an integer gradient on unit
// cells makes Horn's gradient exact.
func TestOrientationCardinal(t *testing.T) {
	for _, c := range []struct {
		a, b          float64
		north, east   float64 // unweighted
		northW, eastW float64 // weighted: ±sin(slope) along, 0 across
	}{
		{0, 3, 1, 0, 3 / math.Sqrt(10), 0},    // rises southward: faces north
		{0, -3, -1, 0, -3 / math.Sqrt(10), 0}, // faces south
		{-2, 0, 0, 1, 0, 2 / math.Sqrt(5)},    // rises westward: faces east
	} {
		dem := plane(5, 5, c.a, c.b, 1, 1)
		for _, unweighted := range []bool{true, false} {
			wantN, wantE := c.north, c.east
			if !unweighted {
				wantN, wantE = c.northW, c.eastW
			}
			for comp, want := range map[OrientationComponent]float64{Northness: wantN, Eastness: wantE} {
				dst := raster.NewFloat32Like(dem)
				Orientation(dst, dem, OrientationOptions{CellSize: 1, Component: comp, Unweighted: unweighted})
				got := float64(dst.Data[dst.Index(2, 2)])
				if got != float64(float32(want)) {
					t.Errorf("plane (%v, %v) component %d unweighted=%v: %v, want %v", c.a, c.b, comp, unweighted, got, float32(want))
				}
			}
		}
	}
}

// roughDEM is a noisy surface with a flat patch, for the checks that
// compare against other definitions cell by cell.
func roughDEM(seed uint64, w, h int) raster.Float32Raster {
	rng := rand.New(rand.NewPCG(seed, 11))
	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range dem.Data {
		dem.Data[i] = float32(300 + 40*rng.NormFloat64())
	}
	for y := 10; y < 16; y++ {
		for x := 5; x < 12; x++ {
			dem.Data[dem.Index(x, y)] = 300
		}
	}
	return dem
}

// TestOrientationIsTheAngleForm checks the rewrite without angles: at
// every cell of a rough DEM, for both components, weighted or not, Horn's
// and the fit's, Orientation is Geomorpho90m's angle form at the gradient
// Gradient writes, to within the float32 rounding of the result. Flat
// cells are exactly 0.
func TestOrientationIsTheAngleForm(t *testing.T) {
	const w, h = 43, 31
	dem := roughDEM(95, w, h)
	for _, fit := range []int{0, 3} {
		g := GradientOptions{CellSize: 10, CellSizeY: 14, ZFactor: 0.8, FitRadius: fit}
		dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
		Gradient(dx, dy, dem, g)
		for _, c := range []OrientationComponent{Northness, Eastness} {
			for _, unweighted := range []bool{false, true} {
				o := OrientationOptions{CellSize: g.CellSize, CellSizeY: g.CellSizeY, ZFactor: g.ZFactor, FitRadius: fit,
					Component: c, Unweighted: unweighted}
				dst := raster.NewFloat32Like(dem)
				Orientation(dst, dem, o)
				flat := 0
				for y := fit + 1; y < h-fit-1; y++ {
					for x := fit + 1; x < w-fit-1; x++ {
						gx, gy := float64(dx.Data[dx.Index(x, y)]), float64(dy.Data[dy.Index(x, y)])
						got := dst.Data[dst.Index(x, y)]
						if gx == 0 && gy == 0 {
							flat++
							if got != 0 {
								t.Fatalf("%+v: flat cell (%d, %d) = %v, want 0", o, x, y, got)
							}
							continue
						}
						want := orientationAngles(o, gx, gy)
						f := float32(math.Abs(want))
						ulp := float64(math.Nextafter32(f, float32(math.Inf(1))) - f)
						if math.Abs(float64(got)-want) > ulp+1e-15 {
							t.Fatalf("%+v: cell (%d, %d) = %v, angle form %v", o, x, y, got, want)
						}
					}
				}
				if fit == 0 && flat == 0 {
					t.Fatalf("%+v: the flat patch has no flat cells", o)
				}
			}
		}
	}
}

// TestOrientationIdentities checks what the two components are together:
// the horizontal part of the unit normal, whose length is sin(slope), or
// with Unweighted a unit vector, except on flat cells. Slope's arctangent
// is float32 (within 1.41e-7 radians), so the weighted identity holds to
// a few 1e-7.
func TestOrientationIdentities(t *testing.T) {
	dem := roughDEM(96, 37, 29)
	for _, fit := range []int{0, 2} {
		comps := func(unweighted bool) (n, e raster.Float32Raster) {
			n, e = raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
			Orientation(n, dem, OrientationOptions{CellSize: 12, FitRadius: fit, Unweighted: unweighted})
			Orientation(e, dem, OrientationOptions{CellSize: 12, FitRadius: fit, Unweighted: unweighted, Component: Eastness})
			return n, e
		}
		slope := raster.NewFloat32Like(dem)
		Slope(slope, dem, SlopeOptions{CellSize: 12, FitRadius: fit, Units: SlopeRadians})
		n, e := comps(false)
		un, ue := comps(true)
		for y := fit + 1; y < dem.Height-fit-1; y++ {
			for x := fit + 1; x < dem.Width-fit-1; x++ {
				i := dem.Index(x, y)
				nv, ev := float64(n.Data[i]), float64(e.Data[i])
				s := math.Sin(float64(slope.Data[i]))
				if math.Abs(math.Hypot(nv, ev)-s) > 5e-7 {
					t.Fatalf("fit=%d (%d, %d): |(northness, eastness)| = %v, sin(slope) = %v", fit, x, y, math.Hypot(nv, ev), s)
				}
				uv := math.Hypot(float64(un.Data[i]), float64(ue.Data[i]))
				if s != 0 && math.Abs(uv-1) > 3e-7 || s == 0 && uv != 0 {
					t.Fatalf("fit=%d (%d, %d): unweighted length %v, slope %v", fit, x, y, uv, s)
				}
			}
		}
	}
}

// TestOrientationRotation checks the components against the compass: a
// DEM turned 90° clockwise faces east where it faced north, so its
// eastness is the original's northness, and its northness the negative of
// the original's eastness. Horn's sums take their terms in another order
// on the turned grid, so the values agree to rounding, not bit for bit.
func TestOrientationRotation(t *testing.T) {
	const w, h = 31, 23
	dem := roughDEM(97, w, h)
	rot := raster.NewFloat32(h, w, make([]float32, w*h))
	for yn := range w {
		for xn := range h {
			rot.Data[rot.Index(xn, yn)] = dem.Data[dem.Index(yn, h-1-xn)]
		}
	}
	get := func(r raster.Float32Raster, c OrientationComponent) raster.Float32Raster {
		out := raster.NewFloat32Like(r)
		Orientation(out, r, OrientationOptions{CellSize: 10, Component: c})
		return out
	}
	n, e := get(dem, Northness), get(dem, Eastness)
	rn, re := get(rot, Northness), get(rot, Eastness)
	for yn := 1; yn < w-1; yn++ {
		for xn := 1; xn < h-1; xn++ {
			i, j := rot.Index(xn, yn), dem.Index(yn, h-1-xn)
			if d := math.Abs(float64(re.Data[i] - n.Data[j])); d > 1e-6 {
				t.Fatalf("turned (%d, %d): eastness %v, original northness %v", xn, yn, re.Data[i], n.Data[j])
			}
			if d := math.Abs(float64(rn.Data[i] + e.Data[j])); d > 1e-6 {
				t.Fatalf("turned (%d, %d): northness %v, original eastness %v", xn, yn, rn.Data[i], e.Data[j])
			}
		}
	}
}

// TestOrientationIsGradientThenComponent checks that Orientation is
// Gradient followed by the from-gradient kernel bit for bit, Horn's and
// the fit's, on both backends, with the hazards of weightedOperand: the
// graph's shared-gradient plan runs exactly that.
func TestOrientationIsGradientThenComponent(t *testing.T) {
	defer stencil.UseScalar(false)
	for _, scalar := range []bool{false, true} {
		stencil.UseScalar(scalar)
		rng := rand.New(rand.NewPCG(98, 1))
		dem := weightedOperand(rng, 300, 13, 900, true, true)
		for _, fit := range []int{0, 1, 4} {
			for _, unweighted := range []bool{false, true} {
				o := OrientationOptions{CellSize: 5, CellSizeY: 7, FitRadius: fit, Component: Eastness, Unweighted: unweighted}
				dx, dy := raster.NewFloat32Like(dem), raster.NewFloat32Like(dem)
				Gradient(dx, dy, dem, GradientOptions{CellSize: o.CellSize, CellSizeY: o.CellSizeY, FitRadius: fit})
				want := raster.NewFloat32Like(dem)
				copy(want.Valid, dx.Valid)
				for y := range dem.Height {
					row := func(r raster.Float32Raster) []float32 { return r.Data[r.Index(0, y) : r.Index(0, y)+dem.Width] }
					stencil.OrientationFromGradientRow(row(want), row(dx), row(dy), true, unweighted)
				}
				got := raster.NewFloat32Like(dem)
				Orientation(got, dem, o)
				sameCells(t, fmt.Sprintf("scalar=%v fit=%d unweighted=%v", scalar, fit, unweighted), got, want)
			}
		}
	}
}

// TestOrientationForms checks that the Tiled and Chunked forms write the
// plain function's bits, and that validity is the gradient's: the same
// erosion as Slope's with the same FitRadius.
func TestOrientationForms(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 3))
	dem := weightedOperand(rng, 70, 41, 600, true, true)
	runs := []engine.Options{
		{TileWidth: 16, TileHeight: 16, Workers: 3},
		{TileWidth: 7, TileHeight: 3, Workers: 2},
	}
	for _, fit := range []int{0, 2} {
		o := OrientationOptions{CellSize: 25, FitRadius: fit, Unweighted: true}
		want := raster.NewFloat32Like(dem)
		Orientation(want, dem, o)
		slope := raster.NewFloat32Like(dem)
		Slope(slope, dem, SlopeOptions{CellSize: 25, FitRadius: fit})
		for y := range dem.Height {
			for x := range dem.Width {
				if want.IsValid(x, y) != slope.IsValid(x, y) {
					t.Fatalf("fit=%d: validity at (%d, %d) = %v, Slope's is %v", fit, x, y, want.IsValid(x, y), slope.IsValid(x, y))
				}
			}
		}
		for _, eo := range runs {
			got := raster.NewFloat32Like(dem)
			if err := OrientationTiled(context.Background(), got, dem, o, eo); err != nil {
				t.Fatal(err)
			}
			sameCells(t, fmt.Sprintf("fit=%d tiled %+v", fit, eo), got, want)
			got = raster.NewFloat32Like(dem)
			if err := OrientationChunked(context.Background(), engine.NewMemorySink(got), engine.NewMemorySource(dem), o, eo); err != nil {
				t.Fatal(err)
			}
			sameCells(t, fmt.Sprintf("fit=%d chunked %+v", fit, eo), got, want)
		}
	}
}

func TestOrientationPanics(t *testing.T) {
	r := raster.NewFloat32(8, 8, make([]float32, 64))
	o := func() raster.Float32Raster { return raster.NewFloat32Like(r) }
	for _, c := range []OrientationComponent{-1, 2} {
		mustPanic(t, fmt.Sprintf("component %d", c), func() { Orientation(o(), r, OrientationOptions{CellSize: 1, Component: c}) })
	}
	mustPanic(t, "cell size", func() { Orientation(o(), r, OrientationOptions{}) })
	mustPanic(t, "fit radius", func() { Orientation(o(), r, OrientationOptions{CellSize: 1, FitRadius: MaxRadius + 1}) })
}
