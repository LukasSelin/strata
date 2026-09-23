package terrain

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

var ruggednessTypes = []RuggednessType{RuggednessTRI, RuggednessTRIWilson, RuggednessTPI, RuggednessRoughness}

// TestRuggednessPlane uses integer planes z = a·x + b·y + 100, on which
// every value the kernels round is an exact small integer or a multiple
// of 1/8, so each measure equals its closed form exactly: over the eight
// neighbours, Σ(a·i + b·j)² = 6a² + 6b², the absolute differences are
// |a|, |b|, |a+b| and |a-b| twice each, the neighbours average to the
// centre, and the window spans 2|a| + 2|b|.
func TestRuggednessPlane(t *testing.T) {
	for _, ab := range [][2]float64{{0, 0}, {1, 0}, {0, -3}, {2, 5}, {-7, 4}, {-100, -1000}} {
		a, b := ab[0], ab[1]
		dem := plane(9, 7, a, b, 1, 1)
		want := map[RuggednessType]float32{
			RuggednessTRI:       float32(math.Sqrt(6*a*a + 6*b*b)),
			RuggednessTRIWilson: float32(2 * (math.Abs(a) + math.Abs(b) + math.Abs(a+b) + math.Abs(a-b)) / 8),
			RuggednessTPI:       0,
			RuggednessRoughness: float32(2*math.Abs(a) + 2*math.Abs(b)),
		}
		for _, rt := range ruggednessTypes {
			dst := raster.NewFloat32Like(dem)
			Ruggedness(dst, dem, RuggednessOptions{Type: rt})
			checkBorderNaN(t, "ruggedness", dst)
			for y := 1; y < dem.Height-1; y++ {
				for x := 1; x < dem.Width-1; x++ {
					if got := dst.Data[dst.Index(x, y)]; math.Float32bits(got) != math.Float32bits(want[rt]) {
						t.Fatalf("plane %v type %d cell (%d, %d) = %v, want %v", ab, rt, x, y, got, want[rt])
					}
				}
			}
		}
	}
}

// TestRuggednessSigns checks what the measures say about a peak and a
// pit: TPI is positive on the peak and negative in the pit, and the
// others, which ignore the sign of each difference, agree on both.
func TestRuggednessSigns(t *testing.T) {
	for _, h := range []float32{5, -5} {
		dem := raster.NewFloat32(3, 3, []float32{10, 10, 10, 10, 10 + h, 10, 10, 10, 10})
		got := make(map[RuggednessType]float32)
		for _, rt := range ruggednessTypes {
			dst := raster.NewFloat32Like(dem)
			Ruggedness(dst, dem, RuggednessOptions{Type: rt})
			got[rt] = dst.Data[dst.Index(1, 1)]
		}
		want := map[RuggednessType]float32{
			RuggednessTRI:       float32(math.Sqrt(8 * 25)),
			RuggednessTRIWilson: 5,
			RuggednessTPI:       h,
			RuggednessRoughness: 5,
		}
		for rt, w := range want {
			if got[rt] != w {
				t.Errorf("centre %+v: type %d = %v, want %v", h, rt, got[rt], w)
			}
		}
	}
}

// TestRuggednessNaNCentre checks that the centre is read by every type.
func TestRuggednessNaNCentre(t *testing.T) {
	nan := float32(math.NaN())
	dem := raster.NewFloat32(3, 3, []float32{1, 2, 3, 4, nan, 6, 7, 8, 9})
	for _, rt := range ruggednessTypes {
		dst := raster.NewFloat32Like(dem)
		Ruggedness(dst, dem, RuggednessOptions{Type: rt})
		if v := dst.Data[dst.Index(1, 1)]; v == v {
			t.Errorf("type %d with a NaN centre = %v, want NaN", rt, v)
		}
	}
}

// TestRuggednessRandomWindows compares every type with a float64
// evaluation of its formula, over random DEMs of real-world elevations.
// The kernels round the differences and sums to float32 (Riley's sum
// excepted), so the float64 value is only near: within a few float32
// epsilons of the largest magnitude summed.
func TestRuggednessRandomWindows(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 14))
	const w, h = 40, 30
	for rep := range 20 {
		dem := raster.NewFloat32(w, h, make([]float32, w*h))
		base, relief := rng.Float64()*8000-500, math.Pow(10, rng.Float64()*4-2)
		for i := range dem.Data {
			dem.Data[i] = float32(base + rng.NormFloat64()*relief)
		}
		for _, rt := range ruggednessTypes {
			dst := raster.NewFloat32Like(dem)
			Ruggedness(dst, dem, RuggednessOptions{Type: rt})
			for y := 1; y < h-1; y++ {
				for x := 1; x < w-1; x++ {
					z := window(dem, x, y)
					c := float64(z[4])
					var sq, ab, sum, zmax float64
					hi, lo := math.Inf(-1), math.Inf(1)
					for i, v := range z {
						f := float64(v)
						hi, lo = max(hi, f), min(lo, f)
						zmax = max(zmax, math.Abs(f))
						if i == 4 {
							continue
						}
						sq += (f - c) * (f - c)
						ab += math.Abs(f - c)
						sum += f
					}
					want := map[RuggednessType]float64{
						RuggednessTRI:       math.Sqrt(sq),
						RuggednessTRIWilson: ab / 8,
						RuggednessTPI:       c - sum/8,
						RuggednessRoughness: hi - lo,
					}[rt]
					// Each difference is rounded once, relative to at most
					// 2·zmax, and each of seven additions once more.
					tol := 16 * 0x1p-24 * zmax
					if got := float64(dst.Data[dst.Index(x, y)]); math.Abs(got-want) > tol {
						t.Fatalf("rep %d type %d cell (%d, %d) = %v, want %v within %.3g", rep, rt, x, y, got, want, tol)
					}
				}
			}
		}
	}
}

// TestRuggednessForms checks that the Tiled and Chunked forms write the
// plain function's bits for several tilings, on a masked DEM.
func TestRuggednessForms(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 92))
	const w, h = 37, 23
	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	dem.Valid = raster.NewMask(w * h)
	for i := range dem.Data {
		dem.Data[i] = float32(rng.NormFloat64()*40 + 300)
		raster.MaskSet(dem.Valid, i, rng.IntN(15) != 0)
	}
	ctx := context.Background()
	for _, rt := range ruggednessTypes {
		o := RuggednessOptions{Type: rt}
		want := raster.NewFloat32Like(dem)
		Ruggedness(want, dem, o)
		for _, eo := range []engine.Options{{TileHeight: 1}, {TileWidth: 5, TileHeight: 4, Workers: 3}, {TileHeight: 64}} {
			tiled := raster.NewFloat32Like(dem)
			if err := RuggednessTiled(ctx, tiled, dem, o, eo); err != nil {
				t.Fatal(err)
			}
			chunked := raster.NewFloat32Like(dem)
			if err := RuggednessChunked(ctx, engine.NewMemorySink(chunked), engine.NewMemorySource(dem), o, eo); err != nil {
				t.Fatal(err)
			}
			for _, got := range []raster.Float32Raster{tiled, chunked} {
				for i := range want.Data {
					if raster.MaskGet(got.Valid, i) != raster.MaskGet(want.Valid, i) {
						t.Fatalf("type %d %+v: cell %d validity differs", rt, eo, i)
					}
					if raster.MaskGet(want.Valid, i) && math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) {
						t.Fatalf("type %d %+v: cell %d = %v, want %v", rt, eo, i, got.Data[i], want.Data[i])
					}
				}
			}
		}
	}
}
