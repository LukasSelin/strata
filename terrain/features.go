package terrain

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/raster"
)

// FeatureOp is the operation behind one output of Features. It is one of
// SlopeOptions, AspectOptions, HillshadeOptions, CurvatureOptions and
// RuggednessOptions, each meaning what it means to its own function; no
// other type can implement it.
type FeatureOp interface {
	featureKernel() exec.Kernel
}

func (o SlopeOptions) featureKernel() exec.Kernel      { return newSlopeKernel(o) }
func (o AspectOptions) featureKernel() exec.Kernel     { return newAspectKernel(o) }
func (o HillshadeOptions) featureKernel() exec.Kernel  { return newHillshadeKernel(o) }
func (o CurvatureOptions) featureKernel() exec.Kernel  { return newCurvatureKernel(o) }
func (o RuggednessOptions) featureKernel() exec.Kernel { return newRuggednessKernel(o) }

// Feature is one raster Features writes: an operation, and the raster
// its result goes to.
type Feature struct {
	Op  FeatureOp
	Dst raster.Float32Raster
}

// FeatureSink is Feature for FeaturesChunked.
type FeatureSink struct {
	Op  FeatureOp
	Dst engine.RasterSink
}

// Features computes several terrain features of one DEM in one pass over
// it: any mix of Slope, Aspect, Hillshade, Curvature and Ruggedness,
// each with its own options, so the same measure can be taken at several
// window sizes (RuggednessOptions.Radius) next to the 3×3 derivatives.
// This is the shape of a feature stack for a model or a geomorphometric
// survey, where every product reads the same neighbourhoods.
//
//	terrain.Features([]terrain.Feature{
//		{Op: terrain.SlopeOptions{CellSize: 10}, Dst: slope},
//		{Op: terrain.RuggednessOptions{Type: terrain.RuggednessTPI, Radius: 1}, Dst: tpi3},
//		{Op: terrain.RuggednessOptions{Type: terrain.RuggednessTPI, Radius: 4}, Dst: tpi9},
//		{Op: terrain.RuggednessOptions{Type: terrain.RuggednessTPI, Radius: 8}, Dst: tpi17},
//	}, dem)
//
// Each output is bit for bit what its own function writes with the same
// options, Data and validity: its border is its own window's radius, and
// its validity is the DEM's mask eroded by its own window, not by the
// largest one (DESIGN.md §52). What is shared is the reading of the DEM,
// which for data on disk or in a compressed file is most of the cost
// (benchmarks/gdalsuite/WORKFLOW.md); the operations do not share
// arithmetic, so asking for slope, aspect and hillshade here computes
// the Horn gradient three times, where Surface computes it once.
//
// Every output must have dem's dimensions and must not overlap dem or
// another output. It panics if out is empty, if an Op is nil, and on
// the checks and option errors of each operation.
func Features(out []Feature, dem raster.Float32Raster) {
	_ = FeaturesTiled(context.Background(), out, dem, engine.Options{Workers: 1})
}

// FeaturesTiled is Features run by the engine: it takes the same
// operands, applies the same checks and writes the same bits for every
// engine.Options, and returns ctx.Err() if ctx is done before every cell
// is written.
func FeaturesTiled(ctx context.Context, out []Feature, dem raster.Float32Raster, eopts engine.Options) error {
	ops := make([]FeatureOp, len(out))
	dst := make([]raster.Float32Raster, len(out))
	for i, f := range out {
		ops[i], dst[i] = f.Op, f.Dst
	}
	return exec.ProcessN(ctx, dst, []raster.Float32Raster{dem}, newFeatures(ops), eopts)
}

// FeaturesChunked is Features run by the engine over a source and sinks
// with bounded memory. It writes the bits Features would write into
// in-memory rasters, for every engine.Options. The tiles it reads are
// grown by the largest radius asked for.
func FeaturesChunked(ctx context.Context, out []FeatureSink, dem engine.RasterSource, eopts engine.Options) error {
	ops := make([]FeatureOp, len(out))
	dst := make([]engine.RasterSink, len(out))
	for i, f := range out {
		ops[i], dst[i] = f.Op, f.Dst
	}
	return exec.ProcessChunked(ctx, dst, []engine.RasterSource{dem}, newFeatures(ops), eopts)
}

// newFeatures is the pipeline that fans the DEM, value 0, out to one
// stage per feature; stage i writes value 1+i, which is output i. The
// stages read nothing but the DEM, so the pipeline's radius is the
// largest of theirs, and the engine gives each output the ring and the
// erosion of its own stage's radius.
func newFeatures(ops []FeatureOp) *exec.Pipeline {
	if len(ops) == 0 {
		panic("terrain: Features has no outputs")
	}
	stages := make([]exec.Stage, len(ops))
	outs := make([]int, len(ops))
	for i, op := range ops {
		if op == nil {
			panic(fmt.Sprintf("terrain: Features output %d has no Op", i))
		}
		stages[i] = exec.Stage{Kernel: op.featureKernel(), In: []int{0}}
		outs[i] = 1 + i
	}
	return exec.NewPipeline(1, stages, outs)
}
