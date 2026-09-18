package exec

import (
	"context"
	"fmt"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/overlap"
	"github.com/LukasSelin/strata/raster"
)

// Process runs a kernel with one input and one output: dst = k(src). See
// ProcessN.
func Process(ctx context.Context, dst, src raster.Float32Raster, k Kernel, opts engine.Options) error {
	return ProcessN(ctx, []raster.Float32Raster{dst}, []raster.Float32Raster{src}, k, opts)
}

// ProcessN runs k over every cell of the outputs dst from the inputs src,
// which must have k's arity and all the same Width and Height. It returns
// nil when every cell is written, or ctx.Err() if ctx is done before
// then; see the package documentation for what a cancelled call leaves
// behind. ctx must not be nil.
//
// It panics on programming errors: a nil kernel or negative radius, a
// wrong number of rasters, rasters that fail Validate or differ in size,
// an input with a validity mask and an output without one, overlapping
// operands (see the package documentation), and negative Options.
func ProcessN(ctx context.Context, dst, src []raster.Float32Raster, k Kernel, opts engine.Options) error {
	if k == nil {
		panic("engine: nil kernel")
	}
	r := k.Radius()
	check(dst, src, k, r, opts)
	return newJob(dst, src, k, r, opts).run(ctx)
}

func check(dst, src []raster.Float32Raster, k Kernel, r int, opts engine.Options) {
	nout := checkKernel(len(dst), len(src), k, r, opts)

	for i, d := range dst {
		requireRaster("dst", i, d, dst[0])
	}
	masked := false
	for i, s := range src {
		requireRaster("src", i, s, dst[0])
		masked = masked || s.Valid != nil
	}
	for i, d := range dst {
		if masked && d.Valid == nil {
			panic(fmt.Sprintf("engine: an input has a validity mask but dst[%d].Valid is nil; allocate it "+
				"with raster.NewFloat32Like, or set Valid on its root raster before windowing", i))
		}
	}

	for i, d := range dst {
		for j := i + 1; j < len(dst); j++ {
			if dataMeet(d, dst[j], r) {
				panic(fmt.Sprintf("engine: dst[%d] and dst[%d] share Data", i, j))
			}
			if bitsMeet(d, dst[j], r) {
				panic(fmt.Sprintf("engine: dst[%d] and dst[%d] share validity bits", i, j))
			}
		}
		for j, s := range src {
			if r > 0 {
				if overlap.DataSpans(d, s) {
					panic(fmt.Sprintf("engine: dst[%d] and src[%d] share Data; a kernel with radius %d "+
						"would read neighbours it has already overwritten", i, j, r))
				}
				if overlap.BitSpans(d, s) {
					panic(fmt.Sprintf("engine: dst[%d] and src[%d] share validity bits", i, j))
				}
				continue
			}
			switch overlap.Data(d, s) {
			case overlap.Partial:
				panic(fmt.Sprintf("engine: dst[%d] overlaps src[%d] at a different offset or stride", i, j))
			case overlap.Same:
				if nout > 1 {
					panic(fmt.Sprintf("engine: dst[%d] is src[%d]; only a kernel with one output runs in place", i, j))
				}
			}
			if overlap.Bits(d, s) == overlap.Partial {
				panic(fmt.Sprintf("engine: dst[%d] validity bits overlap src[%d]'s at a different offset or stride", i, j))
			}
		}
	}
}

// checkKernel checks a kernel's radius and arity against the operand
// counts, and the Options, and returns the number of outputs.
func checkKernel(ndst, nsrc int, k Kernel, r int, opts engine.Options) (nout int) {
	if r < 0 {
		panic(fmt.Sprintf("engine: kernel radius %d is negative", r))
	}
	nin, nout := k.Arity()
	if nin < 0 || nout < 1 {
		panic(fmt.Sprintf("engine: kernel arity (%d inputs, %d outputs) needs at least one output", nin, nout))
	}
	if nsrc != nin || ndst != nout {
		panic(fmt.Sprintf("engine: kernel takes %d inputs and %d outputs, got %d and %d",
			nin, nout, nsrc, ndst))
	}
	if opts.TileWidth < 0 || opts.TileHeight < 0 || opts.Workers < 0 {
		panic(fmt.Sprintf("engine: negative Options %+v", opts))
	}
	return nout
}

func requireRaster(name string, i int, r, ref raster.Float32Raster) {
	if err := r.Validate(); err != nil {
		panic(fmt.Sprintf("engine: %s[%d]: %v", name, i, err))
	}
	if r.Width != ref.Width || r.Height != ref.Height {
		panic(fmt.Sprintf("engine: %s[%d] is %d×%d, dst[0] is %d×%d",
			name, i, r.Width, r.Height, ref.Width, ref.Height))
	}
}

// dataMeet and bitsMeet report whether two outputs share memory: exactly
// for radius 0, like package algebra, and by span for larger radii, like
// package terrain.
func dataMeet(a, b raster.Float32Raster, r int) bool {
	if r == 0 {
		return overlap.Data(a, b) != overlap.Disjoint
	}
	return overlap.DataSpans(a, b)
}

func bitsMeet(a, b raster.Float32Raster, r int) bool {
	if r == 0 {
		return overlap.Bits(a, b) != overlap.Disjoint
	}
	return overlap.BitSpans(a, b)
}
