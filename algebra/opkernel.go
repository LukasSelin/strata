package algebra

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
	"github.com/LukasSelin/strata/internal/vec"
)

// The kernels package graph lowers algebra's operations onto (DESIGN.md
// §55): the ones the Tiled and Chunked entry points run, so a graph writes
// their bits and fuses them as they fuse.
func init() {
	opkernel.Register("algebra.Binary", func(o any) exec.Kernel { return newBinaryOp(o.(vec.Op)) })
	opkernel.Register("algebra.Clamp", func(o any) exec.Kernel {
		k := o.([2]float32)
		return clampKernel{k[0], k[1]}
	})
	opkernel.Register("algebra.Mask", func(any) exec.Kernel { return maskOp{} })
	// Normalize's second pass, given the bounds its first pass found: lo
	// and hi-lo, subtracted in float32 as NormalizeTiled subtracts them.
	opkernel.Register("algebra.Normalize", func(o any) exec.Kernel {
		k := o.([2]float32)
		return normalizeKernel{k[0], k[1] - k[0]}
	})
}
