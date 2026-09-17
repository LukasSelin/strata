package engine

import (
	"context"
	"math"

	"strata/internal/overlap"
	"strata/internal/stencil"
	"strata/raster"
)

// bandCells is the target number of cells in a band, the unit of work
// between cancellation checks: bands are whole rows of a tile, at least
// one. 1<<16 cells keeps a check within a millisecond or so of work for
// the slowest kernels while amortising the per-call cost of pointwise
// ones. It is a variable so tests can force one-row bands.
var bandCells = 1 << 16

// exec is one ProcessN call after its checks. Everything a band needs is
// allocated here, once per call, so bands allocate nothing.
type exec struct {
	k    Kernel
	r    int
	edge float32
	w, h int

	tileW, tileH, bandRows int

	dst, src []raster.Float32Raster
	// dstViews and srcViews are the span and window views handed to the
	// kernel, overwritten for every band.
	dstViews, srcViews []raster.Float32Raster

	// masked lists the inputs that have a validity mask, dstMasked
	// whether any output has one.
	masked    []int
	dstMasked bool
	// sameBits[i] is the masked input whose bits are dst[i]'s own (radius
	// 0 in place), or -1.
	sameBits []int
	// regions and scratch are ErodeBox's arguments for radius > 0.
	regions []stencil.MaskRegion
	scratch []uint64
}

func newExec(dst, src []raster.Float32Raster, k Kernel, r int, opts Options) *exec {
	e := &exec{
		k:        k,
		r:        r,
		edge:     float32(math.NaN()),
		w:        dst[0].Width,
		h:        dst[0].Height,
		dst:      dst,
		src:      src,
		dstViews: make([]raster.Float32Raster, len(dst)),
		srcViews: make([]raster.Float32Raster, len(src)),
	}
	if ek, ok := k.(EdgeKernel); ok {
		e.edge = ek.Edge()
	}
	e.tileW, e.tileH = e.w, e.h
	if opts.TileWidth > 0 {
		e.tileW = min(opts.TileWidth, e.w)
	}
	if opts.TileHeight > 0 {
		e.tileH = min(opts.TileHeight, e.h)
	}
	e.bandRows = max(1, bandCells/e.tileW)

	for _, d := range dst {
		e.dstMasked = e.dstMasked || d.Valid != nil
	}
	for j, s := range src {
		if s.Valid != nil {
			e.masked = append(e.masked, j)
		}
	}
	if !e.dstMasked || len(e.masked) == 0 {
		return e
	}
	if r > 0 {
		e.regions = make([]stencil.MaskRegion, len(e.masked))
		e.scratch = make([]uint64, stencil.ErodeScratch(e.tileW, r))
		return e
	}
	e.sameBits = make([]int, len(dst))
	for i, d := range dst {
		e.sameBits[i] = -1
		for _, j := range e.masked {
			if overlap.Bits(d, src[j]) == overlap.Same {
				e.sameBits[i] = j
				break
			}
		}
	}
	return e
}

// run plans tiles in row-major order and bands of rows within each tile,
// checking ctx before every band.
func (e *exec) run(ctx context.Context) error {
	done := ctx.Done()
	for ty := 0; ty < e.h; ty += e.tileH {
		ty1 := min(ty+e.tileH, e.h)
		for tx := 0; tx < e.w; tx += e.tileW {
			tx1 := min(tx+e.tileW, e.w)
			for y := ty; y < ty1; y += e.bandRows {
				if done != nil {
					select {
					case <-done:
						return ctx.Err()
					default:
					}
				}
				e.band(tx, y, tx1, min(y+e.bandRows, ty1))
			}
		}
	}
	return nil
}
