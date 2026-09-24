package terrain

import (
	"context"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// SurfaceOptions configures Surface: the cell geometry the products
// share, and each product's own settings, which mean what they mean in
// GradientOptions, SlopeOptions, AspectOptions and HillshadeOptions.
type SurfaceOptions struct {
	// CellSize, CellSizeY and ZFactor are as in SlopeOptions.
	CellSize  float64
	CellSizeY float64
	ZFactor   float64
	// Units is Slope's unit. The zero value is SlopeDegrees.
	Units SlopeUnits
	// ZeroForFlat and Trigonometric are Aspect's.
	ZeroForFlat   bool
	Trigonometric bool
	// Azimuth and Altitude are Hillshade's light.
	Azimuth  float64
	Altitude float64
}

// SurfaceOutputs names the rasters Surface writes. A zero raster (nil
// Data) is a product not wanted; at least one must be set.
type SurfaceOutputs struct {
	// Dx and Dy are Gradient's two outputs.
	Dx, Dy    raster.Float32Raster
	Slope     raster.Float32Raster
	Aspect    raster.Float32Raster
	Hillshade raster.Float32Raster
}

// SurfaceSinks is SurfaceOutputs for SurfaceChunked: a nil sink is a
// product not wanted.
type SurfaceSinks struct {
	Dx, Dy    engine.RasterSink
	Slope     engine.RasterSink
	Aspect    engine.RasterSink
	Hillshade engine.RasterSink
}

// Surface computes any of dem's Horn gradient, slope, aspect and
// hillshade in one pass over it, from one gradient: each product it
// writes is bit for bit what Gradient, Slope, Aspect or Hillshade would
// write with the same settings, Data and validity, because those compute
// their gradient and then their product with the same arithmetic
// (DESIGN.md §52). Computing several products this way reads the DEM
// once, instead of once per product, and computes its gradient once.
//
// Every output must have dem's dimensions and must not overlap dem or
// another output. Edges and validity are the products' own, as in the
// package documentation. It panics if no output is set, and on the
// checks and option errors of the products it computes.
func Surface(out SurfaceOutputs, dem raster.Float32Raster, opts SurfaceOptions) {
	_ = SurfaceTiled(context.Background(), out, dem, opts, engine.Options{Workers: 1})
}

// SurfaceTiled is Surface run by the engine: it takes the same operands,
// applies the same checks and writes the same bits for every
// engine.Options, and returns ctx.Err() if ctx is done before every cell
// is written.
func SurfaceTiled(ctx context.Context, out SurfaceOutputs, dem raster.Float32Raster, opts SurfaceOptions, eopts engine.Options) error {
	want := [surfaceProducts]bool{
		out.Dx.Data != nil, out.Dy.Data != nil,
		out.Slope.Data != nil, out.Aspect.Data != nil, out.Hillshade.Data != nil,
	}
	all := [surfaceProducts]raster.Float32Raster{out.Dx, out.Dy, out.Slope, out.Aspect, out.Hillshade}
	p, order := newSurface(want, opts)
	dst := make([]raster.Float32Raster, len(order))
	for i, k := range order {
		dst[i] = all[k]
	}
	return exec.ProcessN(ctx, dst, []raster.Float32Raster{dem}, p, eopts)
}

// SurfaceChunked is Surface run by the engine over a source and sinks
// with bounded memory. It writes the bits Surface would write into
// in-memory rasters, for every engine.Options.
func SurfaceChunked(ctx context.Context, out SurfaceSinks, dem engine.RasterSource, opts SurfaceOptions, eopts engine.Options) error {
	all := [surfaceProducts]engine.RasterSink{out.Dx, out.Dy, out.Slope, out.Aspect, out.Hillshade}
	var want [surfaceProducts]bool
	for k, s := range all {
		want[k] = s != nil
	}
	p, order := newSurface(want, opts)
	dst := make([]engine.RasterSink, len(order))
	for i, k := range order {
		dst[i] = all[k]
	}
	return exec.ProcessChunked(ctx, dst, []engine.RasterSource{dem}, p, eopts)
}

// The products, in the order of SurfaceOutputs.
const (
	surfaceDx = iota
	surfaceDy
	surfaceSlope
	surfaceAspect
	surfaceHillshade
	surfaceProducts
)

// newSurface returns the pipeline that writes the wanted products, and
// the product each of its outputs is. Value 0 is the DEM, 1 and 2 the
// gradient, and each wanted product after dx and dy is one pointwise
// stage over them. Each product's parameters come from its own
// constructor, so they are the ones the standalone function uses.
func newSurface(want [surfaceProducts]bool, opts SurfaceOptions) (*exec.Pipeline, []int) {
	g := newGradientKernel(GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor})
	stages := []exec.Stage{{Kernel: g, In: []int{0}}}
	var outs, order []int
	for k, w := range want[:surfaceSlope] {
		if w {
			outs, order = append(outs, 1+k), append(order, k)
		}
	}
	add := func(k int, kernel exec.Kernel) {
		if !want[k] {
			return
		}
		stages = append(stages, exec.Stage{Kernel: kernel, In: []int{1, 2}})
		outs, order = append(outs, 2+len(stages)-1), append(order, k)
	}
	add(surfaceSlope, func() exec.Kernel {
		s := newSlopeKernel(SlopeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor, Units: opts.Units})
		return slopeFromGradient{s.scale, s.atan}
	}())
	add(surfaceAspect, func() exec.Kernel {
		a := newAspectKernel(AspectOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
			ZeroForFlat: opts.ZeroForFlat, Trigonometric: opts.Trigonometric})
		return aspectFromGradient{a.flat, a.trig}
	}())
	add(surfaceHillshade, func() exec.Kernel {
		h := newHillshadeKernel(HillshadeOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor,
			Azimuth: opts.Azimuth, Altitude: opts.Altitude})
		return hillshadeFromGradient{h.c, h.bx, h.by}
	}())
	if len(outs) == 0 {
		panic("terrain: Surface has no outputs; set at least one of Dx, Dy, Slope, Aspect and Hillshade")
	}
	return exec.NewPipeline(1, stages, outs), order
}

// slopeFromGradient, aspectFromGradient and hillshadeFromGradient are
// the pointwise stages: dx and dy in, one product out, through the
// stencil kernels that finish the fused ones.
type slopeFromGradient struct {
	scale float32
	atan  bool
}

func (slopeFromGradient) Radius() int                  { return 0 }
func (slopeFromGradient) Arity() (inputs, outputs int) { return 2, 1 }

func (k slopeFromGradient) Process(dst exec.Span, src exec.Window) {
	out, gx, gy := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		stencil.SlopeFromGradientRow(out.Row(y), gx.Row(y), gy.Row(y), k.scale, k.atan)
	}
}

type aspectFromGradient struct {
	flat float32
	trig bool
}

func (aspectFromGradient) Radius() int                  { return 0 }
func (aspectFromGradient) Arity() (inputs, outputs int) { return 2, 1 }

func (k aspectFromGradient) Process(dst exec.Span, src exec.Window) {
	out, gx, gy := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		stencil.AspectFromGradientRow(out.Row(y), gx.Row(y), gy.Row(y), k.flat, k.trig)
	}
}

type hillshadeFromGradient struct{ c, bx, by float32 }

func (hillshadeFromGradient) Radius() int                  { return 0 }
func (hillshadeFromGradient) Arity() (inputs, outputs int) { return 2, 1 }

func (k hillshadeFromGradient) Process(dst exec.Span, src exec.Window) {
	out, gx, gy := dst.Dst[0], src.Src[0], src.Src[1]
	for y := range dst.Height {
		stencil.HillshadeFromGradientRow(out.Row(y), gx.Row(y), gy.Row(y), k.c, k.bx, k.by)
	}
}
