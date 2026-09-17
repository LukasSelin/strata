package engine

import (
	"context"
	"fmt"

	"strata/internal/overlap"
	"strata/raster"
)

// Options configures how Process divides and schedules the work. The
// zero value runs the whole raster as one tile. Options never change the
// result.
type Options struct {
	// TileWidth and TileHeight are the tile size in cells. 0 means the
	// raster's width or height. Tiles are processed in row-major order.
	TileWidth  int
	TileHeight int
	// Workers is the number of goroutines that process tiles. 0 lets the
	// engine choose. This version runs every tile on the calling
	// goroutine whatever the value, so callers can already pass the
	// setting STRATA-9's worker pool will use.
	Workers int
}

// Process runs a kernel with one input and one output: dst = k(src). See
// ProcessN.
func Process(ctx context.Context, dst, src raster.Float32Raster, k Kernel, opts Options) error {
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
func ProcessN(ctx context.Context, dst, src []raster.Float32Raster, k Kernel, opts Options) error {
	if k == nil {
		panic("engine: nil kernel")
	}
	r := k.Radius()
	check(dst, src, k, r, opts)
	return newExec(dst, src, k, r, opts).run(ctx)
}

func check(dst, src []raster.Float32Raster, k Kernel, r int, opts Options) {
	if r < 0 {
		panic(fmt.Sprintf("engine: kernel radius %d is negative", r))
	}
	nin, nout := k.Arity()
	if nin < 0 || nout < 1 {
		panic(fmt.Sprintf("engine: kernel arity (%d inputs, %d outputs) needs at least one output", nin, nout))
	}
	if len(src) != nin || len(dst) != nout {
		panic(fmt.Sprintf("engine: kernel takes %d inputs and %d outputs, got %d and %d",
			nin, nout, len(src), len(dst)))
	}
	if opts.TileWidth < 0 || opts.TileHeight < 0 || opts.Workers < 0 {
		panic(fmt.Sprintf("engine: negative Options %+v", opts))
	}

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
