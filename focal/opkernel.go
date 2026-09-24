package focal

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
)

// The kernels package graph lowers focal's operations onto (DESIGN.md
// §55). Mean, Min, Max and CorrelateSeparable are ScratchKernels, which a
// Pipeline does not take as stages yet (§52), so the planner gives each a
// pass of its own; Correlate and Convolve fuse.
func init() {
	opkernel.Register("focal.Mean", func(o any) exec.Kernel { return newBox(o.(BoxOptions), boxMean) })
	opkernel.Register("focal.Min", func(o any) exec.Kernel { return newBox(o.(BoxOptions), boxMin) })
	opkernel.Register("focal.Max", func(o any) exec.Kernel { return newBox(o.(BoxOptions), boxMax) })
	opkernel.Register("focal.Correlate", func(o any) exec.Kernel { return newWeightsKernel(o.(WeightsOptions), false) })
	opkernel.Register("focal.Convolve", func(o any) exec.Kernel { return newWeightsKernel(o.(WeightsOptions), true) })
	opkernel.Register("focal.CorrelateSeparable", func(o any) exec.Kernel { return newSeparableKernel(o.(SeparableOptions)) })
}
