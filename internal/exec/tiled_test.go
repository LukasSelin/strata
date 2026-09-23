package exec_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// adapter pairs a plain public function with its tiled entry point.
type adapter struct {
	name    string
	inputs  int
	outputs int
	inPlace bool // dst may be the first input
	direct  func(dst, src []raster.Float32Raster)
	tiled   func(ctx context.Context, dst, src []raster.Float32Raster, opts engine.Options) error
}

// focalWeights are asymmetric 7×7 weights for focal.Correlate at radius
// 3, so that a kernel reading its window mirrored or transposed fails.
var focalWeights = func() []float32 {
	w := make([]float32, 49)
	for i := range w {
		w[i] = float32(i*7%11-5) / 4
	}
	return w
}()

// focalSeparable are asymmetric taps for focal.CorrelateSeparable at
// radius 2, a ScratchKernel: under the poisoned scratch of TestMain, a
// kernel reading scratch it has not written fails.
var focalSeparable = focal.SeparableOptions{Radius: 2, Row: []float32{1, -2, 0.5, 3, 0.25}, Col: []float32{-1, 0.5, 2, 0.75, 1.5}}

var adapters = []adapter{
	{
		name: "focal-correlate-r3", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			focal.Correlate(dst[0], src[0], focal.WeightsOptions{Radius: 3, Weights: focalWeights})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return focal.CorrelateTiled(ctx, dst[0], src[0], focal.WeightsOptions{Radius: 3, Weights: focalWeights}, o)
		},
	},
	{
		name: "focal-separable-r2", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) { focal.CorrelateSeparable(dst[0], src[0], focalSeparable) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return focal.CorrelateSeparableTiled(ctx, dst[0], src[0], focalSeparable, o)
		},
	},
	{
		// On the 300×9 wide window, radius 4 leaves one interior row.
		name: "focal-max-r4", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) { focal.Max(dst[0], src[0], focal.BoxOptions{Radius: 4}) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return focal.MaxTiled(ctx, dst[0], src[0], focal.BoxOptions{Radius: 4}, o)
		},
	},
	{
		name: "clamp", inputs: 1, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Clamp(dst[0], src[0], 950, 1050) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.ClampTiled(ctx, dst[0], src[0], 950, 1050, o)
		},
	},
	{
		name: "add", inputs: 2, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Add(dst[0], src[0], src[1]) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.AddTiled(ctx, dst[0], src[0], src[1], o)
		},
	},
	{
		name: "sub", inputs: 2, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Sub(dst[0], src[0], src[1]) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.SubTiled(ctx, dst[0], src[0], src[1], o)
		},
	},
	{
		name: "mul", inputs: 2, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Mul(dst[0], src[0], src[1]) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.MulTiled(ctx, dst[0], src[0], src[1], o)
		},
	},
	{
		name: "min", inputs: 2, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Min(dst[0], src[0], src[1]) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.MinTiled(ctx, dst[0], src[0], src[1], o)
		},
	},
	{
		name: "max", inputs: 2, outputs: 1, inPlace: true,
		direct: func(dst, src []raster.Float32Raster) { algebra.Max(dst[0], src[0], src[1]) },
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return algebra.MaxTiled(ctx, dst[0], src[0], src[1], o)
		},
	},
	{
		name: "slope-degrees", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Slope(dst[0], src[0], terrain.SlopeOptions{CellSize: 10, CellSizeY: 12})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.SlopeTiled(ctx, dst[0], src[0], terrain.SlopeOptions{CellSize: 10, CellSizeY: 12}, o)
		},
	},
	{
		name: "slope-percent", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Slope(dst[0], src[0], terrain.SlopeOptions{CellSize: 3, ZFactor: 2, Units: terrain.SlopePercent})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.SlopeTiled(ctx, dst[0], src[0], terrain.SlopeOptions{CellSize: 3, ZFactor: 2, Units: terrain.SlopePercent}, o)
		},
	},
	{
		name: "hillshade", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Hillshade(dst[0], src[0], terrain.HillshadeOptions{CellSize: 30, Azimuth: 100, Altitude: 20})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.HillshadeTiled(ctx, dst[0], src[0], terrain.HillshadeOptions{CellSize: 30, Azimuth: 100, Altitude: 20}, o)
		},
	},
	{
		name: "aspect", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Aspect(dst[0], src[0], terrain.AspectOptions{CellSize: 5, Trigonometric: true})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.AspectTiled(ctx, dst[0], src[0], terrain.AspectOptions{CellSize: 5, Trigonometric: true}, o)
		},
	},
	{
		name: "curvature-plan", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Curvature(dst[0], src[0], terrain.CurvatureOptions{CellSize: 5, CellSizeY: 4, Type: terrain.CurvaturePlan})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.CurvatureTiled(ctx, dst[0], src[0], terrain.CurvatureOptions{CellSize: 5, CellSizeY: 4, Type: terrain.CurvaturePlan}, o)
		},
	},
	{
		name: "ruggedness-tri", inputs: 1, outputs: 1,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Ruggedness(dst[0], src[0], terrain.RuggednessOptions{})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.RuggednessTiled(ctx, dst[0], src[0], terrain.RuggednessOptions{}, o)
		},
	},
	{
		name: "gradient", inputs: 1, outputs: 2,
		direct: func(dst, src []raster.Float32Raster) {
			terrain.Gradient(dst[0], dst[1], src[0], terrain.GradientOptions{CellSize: 7})
		},
		tiled: func(ctx context.Context, dst, src []raster.Float32Raster, o engine.Options) error {
			return terrain.GradientTiled(ctx, dst[0], dst[1], src[0], terrain.GradientOptions{CellSize: 7}, o)
		},
	},
}

var adapterSizes = [][2]int{
	{1, 1}, {2, 2}, {3, 3}, {1, 5}, {6, 1}, {4, 3}, {5, 7}, {17, 9}, {63, 4}, {65, 6}, {130, 5},
}

// TestTiledMatchesPlain runs every tiled entry point whole and in tiles
// and bands of many shapes, and requires the same bits as the plain
// function on the whole raster: same validity words across each output's
// root (the border, and bits outside the output, included) and the same
// Data on valid cells, any NaN matching any NaN.
func TestTiledMatchesPlain(t *testing.T) {
	for _, a := range adapters {
		t.Run(a.name, func(t *testing.T) {
			for _, sz := range adapterSizes {
				for _, lay := range layouts {
					// masks.in says whether inputs have masks, masks.out
					// whether outputs do (required if any input does).
					for _, masks := range []struct{ in, out bool }{{false, false}, {false, true}, {true, true}} {
						for _, inPlace := range []bool{false, a.inPlace} {
							testAdapter(t, a, sz[0], sz[1], lay, masks.in, masks.out, inPlace)
							if !a.inPlace {
								break
							}
						}
					}
				}
			}
		})
	}
}

func testAdapter(t *testing.T, a adapter, w, h int, lay layout, inMask, outMask, inPlace bool) {
	seed := uint64(w*1000 + h)
	rng := rand.New(rand.NewPCG(seed, 3))
	var ins, outs []operand
	for i := range a.inputs {
		// With masks, the first input always has one and a second input
		// half the time.
		masked := inMask && (i == 0 || rng.IntN(2) == 0)
		ins = append(ins, newOperand(rng, w, h, lay.in, masked))
	}
	for range a.outputs {
		outs = append(outs, newOperand(rng, w, h, lay.out, outMask))
	}
	if inPlace {
		outs[0] = ins[0]
		if outMask && !inMask {
			return // same fixture as the unmasked in-place case
		}
	}
	for _, run := range engineRuns {
		id := fmt.Sprintf("%dx%d %v inMask=%v outMask=%v inPlace=%v %v", w, h, lay, inMask, outMask, inPlace, run)
		want, got := cloneAll(ins, outs, inPlace), cloneAll(ins, outs, inPlace)
		a.direct(want.dst(), want.src())
		processWith(t, run, func(ctx context.Context, o engine.Options) error {
			return a.tiled(ctx, got.dst(), got.src(), o)
		})
		for i := range outs {
			requireSameRoots(t, fmt.Sprintf("%s dst[%d]", id, i), got.outs[i].root, want.outs[i].root)
		}
		for i := range ins {
			requireSameRoots(t, fmt.Sprintf("%s src[%d]", id, i), got.ins[i].root, want.ins[i].root)
		}
	}
}

type operands struct{ ins, outs []operand }

// cloneAll deep-copies a fixture. With inPlace the first output is the
// first input's copy.
func cloneAll(ins, outs []operand, inPlace bool) operands {
	var o operands
	for _, in := range ins {
		o.ins = append(o.ins, in.clone())
	}
	for i, out := range outs {
		if i == 0 && inPlace {
			o.outs = append(o.outs, o.ins[0])
			continue
		}
		o.outs = append(o.outs, out.clone())
	}
	return o
}

func (o operands) src() []raster.Float32Raster { return views(o.ins) }
func (o operands) dst() []raster.Float32Raster { return views(o.outs) }

func views(ops []operand) []raster.Float32Raster {
	rs := make([]raster.Float32Raster, len(ops))
	for i, op := range ops {
		rs[i] = op.r
	}
	return rs
}

// TestTiledCancelled checks that every tiled entry point reports a done
// context as its error and writes nothing.
func TestTiledCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, a := range adapters {
		rng := rand.New(rand.NewPCG(5, 5))
		var ins, outs []operand
		for range a.inputs {
			ins = append(ins, newOperand(rng, 9, 7, true, true))
		}
		for range a.outputs {
			outs = append(outs, newOperand(rng, 9, 7, true, true))
		}
		got := cloneAll(ins, outs, false)
		err := a.tiled(ctx, got.dst(), got.src(), engine.Options{TileWidth: 4, TileHeight: 4})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("%s: err = %v, want context.Canceled", a.name, err)
		}
		for i := range outs {
			requireSameRoots(t, fmt.Sprintf("%s dst[%d]", a.name, i), got.outs[i].root, outs[i].root)
		}
	}
}
