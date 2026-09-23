package focal_test

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/internal/focalrow"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
)

// sizes cover all-edge rasters (no larger than 2r), a single interior
// cell, lengths around the SIMD blocks, and a raster larger than a band
// row of 32 cells plus a vector and a tail.
var sizes = [][2]int{{1, 1}, {3, 3}, {4, 11}, {11, 4}, {12, 12}, {45, 7}, {70, 19}}

var radii = []int{1, 2, 3, 5}

// TestMatchesNaive holds every operation to its per-cell definition,
// bit for bit, over hazard values, masks and window layouts.
func TestMatchesNaive(t *testing.T) {
	d := fuzzdata.New([]byte("TestMatchesNaive"))
	for kind := range numKinds {
		for _, r := range append(radii, focal.MaxRadius) {
			for _, sz := range sizes {
				for _, masked := range []bool{false, true} {
					for _, vals := range []cellValues{anyValues, smoothValues} {
						s := newSpec(d, kind, r, floatWeights)
						src := newSrc(d, sz[0], sz[1], vals, masked)
						dst := rastertest.Output(d, sz[0], sz[1], masked || d.Bool(), true)
						if err := s.run(0, engine.Options{}, dst, src); err != nil {
							t.Fatal(err)
						}
						requireNaive(t, fmt.Sprintf("%d×%d masked %v", sz[0], sz[1], masked), s, dst, src)
					}
				}
			}
		}
	}
}

// tilings is the grid of engine options the paths are compared over: no
// tiling, one-cell tiles, tiles that do not divide the raster, and one,
// several and GOMAXPROCS workers.
func tilings() []engine.Options {
	var out []engine.Options
	for _, tw := range []int{0, 1, 7, 64} {
		for _, th := range []int{0, 1, 5} {
			for _, wk := range []int{1, 3, runtime.GOMAXPROCS(0)} {
				out = append(out, engine.Options{TileWidth: tw, TileHeight: th, Workers: wk})
			}
		}
	}
	return out
}

// TestPathsAgree is the package's central guarantee (DESIGN.md §23): the
// plain, Tiled and Chunked forms write the same bits, Data and validity,
// for every tile size and worker count.
func TestPathsAgree(t *testing.T) {
	d := fuzzdata.New([]byte("TestPathsAgree"))
	for kind := range numKinds {
		for _, r := range radii {
			t.Run(fmt.Sprintf("%s/r=%d", kindNames[kind], r), func(t *testing.T) {
				for _, sz := range [][2]int{{9, 4}, {23, 13}, {70, 12}} {
					for _, masked := range []bool{false, true} {
						s := newSpec(d, kind, r, floatWeights)
						src := newSrc(d, sz[0], sz[1], anyValues, masked)
						want := rastertest.Output(d, sz[0], sz[1], masked, true)
						if err := s.run(0, engine.Options{}, want, src); err != nil {
							t.Fatal(err)
						}
						want = rastertest.Compact(want)
						for _, opts := range tilings() {
							for path := 1; path <= 2; path++ {
								got := rastertest.Output(d, sz[0], sz[1], masked, true)
								if err := s.run(path, opts, got, src); err != nil {
									t.Fatalf("path %d %+v: %v", path, opts, err)
								}
								requireSame(t, fmt.Sprintf("%v %d×%d masked %v path %d %+v", s, sz[0], sz[1], masked, path, opts), got, want)
							}
						}
					}
				}
			})
		}
	}
}

// TestSIMDMatchesScalar runs every operation on both backends over
// hazards, masks and windows, and requires the same bits.
func TestSIMDMatchesScalar(t *testing.T) {
	if focalrow.Backend() == "scalar" {
		t.Skip("no SIMD backend in this build")
	}
	defer focalrow.UseScalar(false)
	d := fuzzdata.New([]byte("TestSIMDMatchesScalar"))
	for kind := range numKinds {
		for _, r := range append(radii, focal.MaxRadius) {
			for _, w := range []int{17, 33, 40, 64, 65, 100} {
				masked := w%2 == 1
				s := newSpec(d, kind, r, floatWeights)
				src := newSrc(d, w, 2*r+3, anyValues, masked)
				want := raster.NewFloat32Like(src)
				focalrow.UseScalar(true)
				if err := s.run(0, engine.Options{}, want, src); err != nil {
					t.Fatal(err)
				}
				got := raster.NewFloat32Like(src)
				focalrow.UseScalar(false)
				if err := s.run(0, engine.Options{}, got, src); err != nil {
					t.Fatal(err)
				}
				requireSame(t, fmt.Sprintf("%v width %d on %s", s, w, focalrow.Backend()), got, want)
			}
		}
	}
}

// TestAliasedStride runs every operation on windows of a raster 16384
// cells wide, whose rows are 64 KiB apart. There the AVX2 column passes
// over more than seven rows fold them a group at a time over chunks of
// the row (internal/focalrow, DESIGN.md §53), which must not change a
// bit: both backends, plain and Tiled, must match the per-cell
// definition, on a window narrower than a chunk and one wider.
func TestAliasedStride(t *testing.T) {
	const stride = 16384
	defer focalrow.UseScalar(false)
	d := fuzzdata.New([]byte("TestAliasedStride"))
	for kind := range numKinds {
		for _, r := range []int{3, 4, 5, focal.MaxRadius} {
			for _, w := range []int{100, 4096 + 2*r + 37} {
				h, masked := 2*r+4, w == 100
				s := newSpec(d, kind, r, floatWeights)
				parent := newSrc(d, stride, h, anyValues, masked)
				if parent.Stride != stride {
					parent = rastertest.Compact(parent)
				}
				src := parent.Window(5, 0, w, h)
				for _, scalar := range []bool{false, true} {
					focalrow.UseScalar(scalar)
					for path, opts := range []engine.Options{{}, {TileWidth: 1000, TileHeight: 3, Workers: 2}} {
						dst := rastertest.Output(d, w, h, masked, true)
						if err := s.run(path, opts, dst, src); err != nil {
							t.Fatal(err)
						}
						requireNaive(t, fmt.Sprintf("width %d path %d on %s", w, path, focalrow.Backend()), s, dst, src)
					}
				}
			}
		}
	}
}

// TestHandComputed checks small cases worked on paper, including the
// orientation of the weights and Convolve's rotation.
func TestHandComputed(t *testing.T) {
	// src(x, y) = 10·y + x on a 3×3 raster: one interior cell, (1, 1).
	src := raster.NewFloat32(3, 3, []float32{0, 1, 2, 10, 11, 12, 20, 21, 22})
	// Only the weight at row 0, column 2 (dy = -1, dx = +1) is set.
	w := []float32{0, 0, 1, 0, 0, 0, 0, 0, 0}
	cases := []struct {
		name string
		run  func(dst raster.Float32Raster)
		want float32
	}{
		{"Correlate reads (x+1, y-1)", func(dst raster.Float32Raster) {
			focal.Correlate(dst, src, focal.WeightsOptions{Radius: 1, Weights: w})
		}, 2},
		{"Convolve reads (x-1, y+1)", func(dst raster.Float32Raster) {
			focal.Convolve(dst, src, focal.WeightsOptions{Radius: 1, Weights: w})
		}, 20},
		{"CorrelateSeparable row x+1, col y-1", func(dst raster.Float32Raster) {
			focal.CorrelateSeparable(dst, src, focal.SeparableOptions{Radius: 1, Row: []float32{0, 0, 1}, Col: []float32{1, 0, 0}})
		}, 2},
		{"CorrelateSeparable sums", func(dst raster.Float32Raster) {
			focal.CorrelateSeparable(dst, src, focal.SeparableOptions{Radius: 1, Row: []float32{1, 1, 1}, Col: []float32{1, 2, 1}})
		}, 0 + 1 + 2 + 2*(10+11+12) + 20 + 21 + 22},
		{"Mean", func(dst raster.Float32Raster) { focal.Mean(dst, src, focal.BoxOptions{Radius: 1}) }, 11},
		{"Min", func(dst raster.Float32Raster) { focal.Min(dst, src, focal.BoxOptions{Radius: 1}) }, 0},
		{"Max", func(dst raster.Float32Raster) { focal.Max(dst, src, focal.BoxOptions{Radius: 1}) }, 22},
	}
	for _, c := range cases {
		dst := raster.NewFloat32(3, 3, make([]float32, 9))
		c.run(dst)
		if got := dst.Data[4]; got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
		for i, v := range dst.Data {
			if i != 4 && v == v {
				t.Errorf("%s: edge cell %d = %v, want NaN", c.name, i, v)
			}
		}
	}
}

// TestConvolveIsRotatedCorrelate checks Convolve against Correlate with
// the weights rotated by 180°, bit for bit.
func TestConvolveIsRotatedCorrelate(t *testing.T) {
	d := fuzzdata.New([]byte("TestConvolveIsRotatedCorrelate"))
	for _, r := range radii {
		src := newSrc(d, 40, 17, anyValues, true)
		s := newSpec(d, kConvolve, r, floatWeights)
		got, want := raster.NewFloat32Like(src), raster.NewFloat32Like(src)
		focal.Convolve(got, src, focal.WeightsOptions{Radius: r, Weights: s.w})
		focal.Correlate(want, src, focal.WeightsOptions{Radius: r, Weights: rot180(s.w)})
		requireSame(t, fmt.Sprintf("r=%d", r), got, want)
	}
}

// TestSignedZero checks that a sum of -0 products is -0: the fold starts
// from the first product, not from +0.
func TestSignedZero(t *testing.T) {
	negz := float32(math.Copysign(0, -1))
	src := raster.NewFloat32(40, 3, make([]float32, 120))
	for i := range src.Data {
		src.Data[i] = negz
	}
	for _, kind := range []int{kCorrelate, kSeparable, kMean, kMin, kMax} {
		s := newSpec(fuzzdata.New(nil), kind, 1, intWeights)
		for i := range s.w {
			s.w[i] = 1
		}
		for i := range s.row {
			s.row[i], s.col[i] = 1, 1
		}
		dst := raster.NewFloat32Like(src)
		if err := s.run(0, engine.Options{}, dst, src); err != nil {
			t.Fatal(err)
		}
		for x := 1; x < 39; x++ {
			if v := dst.Row(1)[x]; math.Float32bits(v) != math.Float32bits(negz) {
				t.Fatalf("%v: cell %d = %v, want -0", s, x, v)
			}
		}
	}
}

// TestUnmaskedInputMaskedOutput checks that an output mask gets its
// interior marked valid and its edge cleared when the input has none,
// whatever was in it before.
func TestUnmaskedInputMaskedOutput(t *testing.T) {
	d := fuzzdata.New([]byte("TestUnmaskedInputMaskedOutput"))
	for kind := range numKinds {
		s := newSpec(d, kind, 2, floatWeights)
		src := newSrc(d, 20, 9, smoothValues, false)
		dst := rastertest.Output(d, 20, 9, true, true)
		if err := s.run(0, engine.Options{}, dst, src); err != nil {
			t.Fatal(err)
		}
		requireNaive(t, "unmasked input", s, dst, src)
	}
}

func TestGaussian(t *testing.T) {
	for _, r := range []int{1, 3, focal.MaxRadius} {
		for _, sigma := range []float64{0.5, 1, 2.5, 100} {
			g := focal.Gaussian(r, sigma)
			if len(g) != 2*r+1 {
				t.Fatalf("Gaussian(%d, %v) has %d taps", r, sigma, len(g))
			}
			var sum float64
			for i, v := range g {
				sum += float64(v)
				if v != g[len(g)-1-i] {
					t.Errorf("Gaussian(%d, %v) is not symmetric: %v", r, sigma, g)
				}
				if i > 0 && i <= r && !(v >= g[i-1]) {
					t.Errorf("Gaussian(%d, %v) does not rise to the centre: %v", r, sigma, g)
				}
			}
			if math.Abs(sum-1) > 1e-6 {
				t.Errorf("Gaussian(%d, %v) sums to %v", r, sigma, sum)
			}
		}
	}
	for _, sigma := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		mustPanic(t, fmt.Sprintf("sigma %v", sigma), "focal: ", func() { focal.Gaussian(2, sigma) })
	}
	mustPanic(t, "radius 0", "focal: ", func() { focal.Gaussian(0, 1) })
	mustPanic(t, "radius too large", "focal: ", func() { focal.Gaussian(focal.MaxRadius+1, 1) })
}

// TestPanics checks that invalid options panic with "focal:" and invalid
// operands with "engine:", before anything is written.
func TestPanics(t *testing.T) {
	src := raster.NewFloat32(8, 8, make([]float32, 64))
	nan := float32(math.NaN())
	options := []struct {
		name string
		run  func(dst raster.Float32Raster)
	}{
		{"radius 0", func(dst raster.Float32Raster) { focal.Mean(dst, src, focal.BoxOptions{}) }},
		{"radius -1", func(dst raster.Float32Raster) { focal.Min(dst, src, focal.BoxOptions{Radius: -1}) }},
		{"radius too large", func(dst raster.Float32Raster) {
			focal.Max(dst, src, focal.BoxOptions{Radius: focal.MaxRadius + 1})
		}},
		{"too few weights", func(dst raster.Float32Raster) {
			focal.Correlate(dst, src, focal.WeightsOptions{Radius: 1, Weights: make([]float32, 8)})
		}},
		{"weights for another radius", func(dst raster.Float32Raster) {
			focal.Convolve(dst, src, focal.WeightsOptions{Radius: 2, Weights: make([]float32, 9)})
		}},
		{"NaN weight", func(dst raster.Float32Raster) {
			focal.Correlate(dst, src, focal.WeightsOptions{Radius: 1, Weights: []float32{0, 0, 0, 0, nan, 0, 0, 0, 0}})
		}},
		{"Inf tap", func(dst raster.Float32Raster) {
			focal.CorrelateSeparable(dst, src, focal.SeparableOptions{Radius: 1,
				Row: []float32{1, 1, 1}, Col: []float32{1, float32(math.Inf(-1)), 1}})
		}},
		{"row and col lengths differ", func(dst raster.Float32Raster) {
			focal.CorrelateSeparable(dst, src, focal.SeparableOptions{Radius: 1, Row: []float32{1, 1, 1}, Col: []float32{1, 1, 1, 1, 1}})
		}},
		{"missing weights, Tiled", func(dst raster.Float32Raster) {
			_ = focal.CorrelateTiled(context.Background(), dst, src, focal.WeightsOptions{Radius: 1}, engine.Options{})
		}},
		{"radius 0, Chunked", func(dst raster.Float32Raster) {
			_ = focal.MeanChunked(context.Background(), engine.NewMemorySink(dst), engine.NewMemorySource(src), focal.BoxOptions{}, engine.Options{})
		}},
	}
	for _, o := range options {
		dst := raster.NewFloat32(8, 8, make([]float32, 64))
		for i := range dst.Data {
			dst.Data[i] = 7
		}
		mustPanic(t, o.name, "focal: ", func() { o.run(dst) })
		for _, v := range dst.Data {
			if v != 7 {
				t.Fatalf("%s: wrote before panicking", o.name)
			}
		}
	}

	operands := []struct {
		name string
		run  func()
	}{
		{"size mismatch", func() { focal.Mean(raster.NewFloat32(7, 8, make([]float32, 56)), src, focal.BoxOptions{Radius: 1}) }},
		{"in place", func() { focal.Min(src, src, focal.BoxOptions{Radius: 1}) }},
		{"masked input, unmasked output", func() {
			in := raster.NewFloat32(8, 8, make([]float32, 64))
			in.Valid = raster.NewMask(64)
			focal.Max(raster.NewFloat32(8, 8, make([]float32, 64)), in, focal.BoxOptions{Radius: 1})
		}},
	}
	for _, o := range operands {
		mustPanic(t, o.name, "engine: ", o.run)
	}
}
