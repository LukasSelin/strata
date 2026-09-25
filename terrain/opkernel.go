package terrain

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
)

// The kernels package graph lowers terrain's operations onto (DESIGN.md
// §55). Each is the kernel the entry point of the same name runs, Horn's
// or the quadratic fit's by FitRadius, and the FromGradient ones are
// Surface's stages, so a graph that shares one gradient between several
// products writes what Surface writes. They take no cell geometry: the
// gradient they read has it.
func init() {
	opkernel.Register("terrain.Gradient", func(o any) exec.Kernel { return gradientOp(o.(GradientOptions)) })
	opkernel.Register("terrain.Slope", func(o any) exec.Kernel { return slopeOp(o.(SlopeOptions)) })
	opkernel.Register("terrain.Aspect", func(o any) exec.Kernel { return aspectOp(o.(AspectOptions)) })
	opkernel.Register("terrain.Hillshade", func(o any) exec.Kernel { return hillshadeOp(o.(HillshadeOptions)) })
	opkernel.Register("terrain.HeatLoad", func(o any) exec.Kernel { return heatLoadOp(o.(HeatLoadOptions)) })
	opkernel.Register("terrain.Orientation", func(o any) exec.Kernel { return orientationOp(o.(OrientationOptions)) })
	opkernel.Register("terrain.Curvature", func(o any) exec.Kernel { return curvatureOp(o.(CurvatureOptions)) })
	opkernel.Register("terrain.Ruggedness", func(o any) exec.Kernel { return newRuggednessKernel(o.(RuggednessOptions)) })
	opkernel.Register("terrain.SlopeFromGradient", func(o any) exec.Kernel {
		scale, atan := slopeScale(o.(SlopeOptions).Units)
		return slopeFromGradient{scale, atan}
	})
	opkernel.Register("terrain.AspectFromGradient", func(o any) exec.Kernel {
		a := o.(AspectOptions)
		return aspectFromGradient{aspectFlat(a.ZeroForFlat), a.Trigonometric}
	})
	opkernel.Register("terrain.HillshadeFromGradient", func(o any) exec.Kernel {
		h := o.(HillshadeOptions)
		c, bx, by := hillshadeLight(h.Azimuth, h.Altitude)
		return hillshadeFromGradient{c, bx, by}
	})
	opkernel.Register("terrain.HeatLoadFromGradient", func(o any) exec.Kernel {
		return heatLoadFromGradient{heatLoadTerms(o.(HeatLoadOptions))}
	})
	opkernel.Register("terrain.OrientationFromGradient", func(o any) exec.Kernel {
		r := o.(OrientationOptions)
		return orientationFromGradient{orientationEast(r.Component), r.Unweighted}
	})
}
