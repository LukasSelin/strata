package algebra

import (
	"strata/engine"
	"strata/internal/vec"
)

// ClampKernel returns Clamp as an engine kernel with radius 0, one input
// and one output, for engine.Process.
func ClampKernel(lo, hi float32) engine.Kernel { return clampKernel{lo, hi} }

// AddKernel returns Add as an engine kernel with radius 0, two inputs (a
// then b) and one output, for engine.ProcessN.
func AddKernel() engine.Kernel { return binaryOp{vec.Add} }

// SubKernel returns Sub as an engine kernel, like AddKernel.
func SubKernel() engine.Kernel { return binaryOp{vec.Sub} }

// MulKernel returns Mul as an engine kernel, like AddKernel.
func MulKernel() engine.Kernel { return binaryOp{vec.Mul} }

// MinKernel returns Min as an engine kernel, like AddKernel.
func MinKernel() engine.Kernel { return binaryOp{vec.Min} }

// MaxKernel returns Max as an engine kernel, like AddKernel.
func MaxKernel() engine.Kernel { return binaryOp{vec.Max} }

type clampKernel struct{ lo, hi float32 }

func (clampKernel) Radius() int                  { return 0 }
func (clampKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (k clampKernel) Process(dst engine.Span, src engine.Window) {
	d, s := dst.Dst[0], src.Src[0]
	clampValues(d, s, k.lo, k.hi, compact(d) && compact(s))
}

type binaryOp struct{ kernel binaryKernel }

func (binaryOp) Radius() int                  { return 0 }
func (binaryOp) Arity() (inputs, outputs int) { return 2, 1 }

func (k binaryOp) Process(dst engine.Span, src engine.Window) {
	d, a, b := dst.Dst[0], src.Src[0], src.Src[1]
	binaryValues(d, a, b, k.kernel, compact(d) && compact(a) && compact(b))
}
