package terrain

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
)

// The kernels package graph lowers terrain's operations onto (DESIGN.md
// §55). Each is the kernel the entry point of the same name runs, and
// the FromGradient ones are Surface's stages, so a graph that shares one
// gradient between several products writes what Surface writes.
func init() {
	opkernel.Register("terrain.Gradient", func(o any) exec.Kernel { return newGradientKernel(o.(GradientOptions)) })
	opkernel.Register("terrain.Slope", func(o any) exec.Kernel { return newSlopeKernel(o.(SlopeOptions)) })
	opkernel.Register("terrain.Aspect", func(o any) exec.Kernel { return newAspectKernel(o.(AspectOptions)) })
	opkernel.Register("terrain.Hillshade", func(o any) exec.Kernel { return newHillshadeKernel(o.(HillshadeOptions)) })
	opkernel.Register("terrain.Curvature", func(o any) exec.Kernel { return newCurvatureKernel(o.(CurvatureOptions)) })
	opkernel.Register("terrain.Ruggedness", func(o any) exec.Kernel { return newRuggednessKernel(o.(RuggednessOptions)) })
	opkernel.Register("terrain.SlopeFromGradient", func(o any) exec.Kernel {
		s := newSlopeKernel(o.(SlopeOptions))
		return slopeFromGradient{s.scale, s.atan}
	})
	opkernel.Register("terrain.AspectFromGradient", func(o any) exec.Kernel {
		a := newAspectKernel(o.(AspectOptions))
		return aspectFromGradient{a.flat, a.trig}
	})
	opkernel.Register("terrain.HillshadeFromGradient", func(o any) exec.Kernel {
		h := newHillshadeKernel(o.(HillshadeOptions))
		return hillshadeFromGradient{h.c, h.bx, h.by}
	})
}
