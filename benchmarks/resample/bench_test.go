package resample_test

import (
	"math"
	"testing"

	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/internal/resamprow"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// Run the suite with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/resample -run '^$' -bench . -count 5 -timeout 3h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt

var resampKernels = suite.Kernels{Name: "resamp", Backend: resamp.Backend, UseScalar: resamp.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, resampKernels) }

// scale is one ratio of output to source resolution: s source cells per
// output cell along each axis.
type scale struct {
	name  string
	s     float64
	sizes []int
}

var (
	up4       = scale{"Up4", 0.25, nil}
	up2       = scale{"Up2", 0.5, nil}
	down2     = scale{"Down2", 2, []int{256, 1024, 4096}}
	down1p37  = scale{"Down1p37", 1.37, []int{256, 1024, 4096}}
	allScales = []scale{up2, up4, down2, down1p37}
)

// fixture is a source and an output for one output size.
type fixture struct {
	size     int
	sg, dg   raster.Grid
	src, dst []float32
	srcValid []uint64
	dstValid []uint64
}

func newFixture(size int, sc scale) *fixture {
	sn := int(math.Round(float64(size) * sc.s))
	f := &fixture{
		size: size,
		sg:   raster.Grid{Width: sn, Height: sn, ResolutionX: 1, ResolutionY: -1, OriginY: float64(sn)},
		src:  make([]float32, sn*sn),
		dst:  make([]float32, size*size),
	}
	// The output covers the source exactly, so every output cell is valid
	// without a mask.
	res := float64(sn) / float64(size)
	f.dg = raster.Grid{Width: size, Height: size, ResolutionX: res, ResolutionY: -res, OriginY: float64(sn)}
	suite.FillDEM(f.src, sn, 1)
	suite.FillUniform(f.dst, 3, 0, 1) // fault the output's pages in before timing
	f.srcValid = suite.RandomMask(sn*sn, 4, 0.1)
	f.dstValid = raster.NewMask(size * size)
	return f
}

func (f *fixture) rasters(masked bool) (dst, src raster.Dataset) {
	s := raster.NewFloat32(f.sg.Width, f.sg.Height, f.src)
	d := raster.NewFloat32(f.size, f.size, f.dst)
	if masked {
		s.Valid, d.Valid = f.srcValid, f.dstValid
	}
	return raster.NewDataset(f.dg, d), raster.NewDataset(f.sg, s)
}

// op is one method at one scale, through Resample or, if direct,
// through resamp.Direct2D.
type op struct {
	m      resample.Method
	sc     scale
	direct bool
	masks  bool
}

func (o op) workload(f *fixture, c suite.Case) suite.Workload {
	if (c.Masked && !o.masks) || (o.direct && (c.Masked || c.Backend == suite.SIMD)) {
		return suite.Workload{}
	}
	dst, src := f.rasters(c.Masked)
	bytes := 4 * (1 + o.sc.s*o.sc.s)
	if c.Masked {
		bytes += (1 + o.sc.s*o.sc.s) / 8
	}
	run := func() { resample.Resample(dst, src, resample.Options{Method: o.m}) }
	if o.direct {
		x := resamp.Spec{N: f.dg.Width, Origin: f.dg.OriginX, Res: f.dg.ResolutionX, SrcN: f.sg.Width, SrcOrigin: f.sg.OriginX, SrcRes: f.sg.ResolutionX}
		y := resamp.Spec{N: f.dg.Height, Origin: f.dg.OriginY, Res: f.dg.ResolutionY, SrcN: f.sg.Height, SrcOrigin: f.sg.OriginY, SrcRes: f.sg.ResolutionY}
		run = func() {
			// A plan per call, as Resample builds one.
			p := resamp.NewPlan(resamp.Method(o.m), x, y)
			resamprow.Direct2D(dst.Raster.Data, f.size, src.Raster.Data, f.sg.Width, 0, 0, &p.Axes, 0, f.size, 0, f.size)
		}
	}
	return suite.Workload{Run: run, BytesPerCell: bytes}
}

func (o op) bench(b *testing.B) {
	suite.Run(b, suite.Matrix[*fixture]{
		Kernels:  resampKernels,
		Fixture:  func(size int) *fixture { return newFixture(size, o.sc) },
		Workload: o.workload,
		Sizes:    o.sc.sizes,
	})
}

func BenchmarkNearestUp2(b *testing.B)      { op{m: resample.Nearest, sc: up2}.bench(b) }
func BenchmarkNearestUp4(b *testing.B)      { op{m: resample.Nearest, sc: up4}.bench(b) }
func BenchmarkNearestDown2(b *testing.B)    { op{m: resample.Nearest, sc: down2}.bench(b) }
func BenchmarkNearestDown1p37(b *testing.B) { op{m: resample.Nearest, sc: down1p37}.bench(b) }

func BenchmarkBilinearUp2(b *testing.B)   { op{m: resample.Bilinear, sc: up2, masks: true}.bench(b) }
func BenchmarkBilinearUp4(b *testing.B)   { op{m: resample.Bilinear, sc: up4, masks: true}.bench(b) }
func BenchmarkBilinearDown2(b *testing.B) { op{m: resample.Bilinear, sc: down2, masks: true}.bench(b) }
func BenchmarkBilinearDown1p37(b *testing.B) {
	op{m: resample.Bilinear, sc: down1p37, masks: true}.bench(b)
}

func BenchmarkCubicUp2(b *testing.B)      { op{m: resample.Cubic, sc: up2}.bench(b) }
func BenchmarkCubicUp4(b *testing.B)      { op{m: resample.Cubic, sc: up4}.bench(b) }
func BenchmarkCubicDown2(b *testing.B)    { op{m: resample.Cubic, sc: down2}.bench(b) }
func BenchmarkCubicDown1p37(b *testing.B) { op{m: resample.Cubic, sc: down1p37}.bench(b) }

func BenchmarkLanczosUp2(b *testing.B)   { op{m: resample.Lanczos, sc: up2, masks: true}.bench(b) }
func BenchmarkLanczosUp4(b *testing.B)   { op{m: resample.Lanczos, sc: up4, masks: true}.bench(b) }
func BenchmarkLanczosDown2(b *testing.B) { op{m: resample.Lanczos, sc: down2, masks: true}.bench(b) }
func BenchmarkLanczosDown1p37(b *testing.B) {
	op{m: resample.Lanczos, sc: down1p37, masks: true}.bench(b)
}

func BenchmarkAverageUp2(b *testing.B)      { op{m: resample.Average, sc: up2}.bench(b) }
func BenchmarkAverageUp4(b *testing.B)      { op{m: resample.Average, sc: up4}.bench(b) }
func BenchmarkAverageDown2(b *testing.B)    { op{m: resample.Average, sc: down2}.bench(b) }
func BenchmarkAverageDown1p37(b *testing.B) { op{m: resample.Average, sc: down1p37}.bench(b) }

func BenchmarkCubicDirectUp2(b *testing.B)   { op{m: resample.Cubic, sc: up2, direct: true}.bench(b) }
func BenchmarkCubicDirectDown2(b *testing.B) { op{m: resample.Cubic, sc: down2, direct: true}.bench(b) }
func BenchmarkLanczosDirectUp2(b *testing.B) { op{m: resample.Lanczos, sc: up2, direct: true}.bench(b) }
func BenchmarkLanczosDirectDown2(b *testing.B) {
	op{m: resample.Lanczos, sc: down2, direct: true}.bench(b)
}

// TestAllocs checks every method and scale on a small raster against a
// bound that does not grow with the raster: a call builds its weight
// tables, a few slices per axis, and takes its workspaces from a pool.
func TestAllocs(t *testing.T) {
	defer resampKernels.UseScalar(false)
	const limit = 96
	for _, sc := range allScales {
		f := newFixture(64, sc)
		for _, m := range []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic, resample.Lanczos, resample.Average} {
			for _, masked := range []bool{false, true} {
				for _, backend := range suite.Backends {
					c := suite.Case{Size: f.size, Masked: masked, Backend: backend, Workers: 1}
					resampKernels.UseScalar(backend == suite.Scalar)
					w := op{m: m, sc: sc, masks: true}.workload(f, c)
					w.Run() // warm the workspace pool
					if allocs := testing.AllocsPerRun(10, w.Run); allocs > limit {
						t.Errorf("%v %s/%s: %v allocs/op, want at most %d", m, sc.name, c.Name(), allocs, limit)
					}
				}
			}
		}
	}
}
