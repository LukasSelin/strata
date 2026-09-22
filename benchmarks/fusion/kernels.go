package fusion

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/pointwise"
	"github.com/LukasSelin/strata/internal/vec"
)

// Inputs is the number of factors in the product every form computes:
// dst = ((((a0·a1)·a2)·a3)·a4)·a5, five multiplies.
const Inputs = 6

// mulStage is one multiply of the chain: the same work algebra.Mul's
// kernel does, one internal/vec call per span when every view is compact
// and one per row otherwise. It is here rather than borrowed because
// algebra's kernel is unexported; Pipeline needs a Kernel value.
type mulStage struct{}

func (mulStage) Radius() int                  { return 0 }
func (mulStage) Arity() (inputs, outputs int) { return 2, 1 }

func (mulStage) Process(dst exec.Span, src exec.Window) {
	d, a, b := dst.Dst[0], src.Src[0], src.Src[1]
	if pointwise.Compact(d) && pointwise.Compact(a) && pointwise.Compact(b) {
		n := d.Width * d.Height
		vec.Mul(d.Data[:n], a.Data[:n], b.Data[:n])
		return
	}
	for y := range d.Height {
		vec.Mul(d.Row(y), a.Row(y), b.Row(y))
	}
}

// NewPipeline returns the chain as one tile-level fused kernel
// (DESIGN.md §52): five mulStage stages, each folding the running product
// with the next input. The four intermediates live in the worker's
// scratch; only the last stage writes the output.
func NewPipeline() *exec.Pipeline {
	stages := make([]exec.Stage, 0, Inputs-1)
	acc := 0
	for i := 1; i < Inputs; i++ {
		stages = append(stages, exec.Stage{Kernel: mulStage{}, In: []int{acc, i}})
		acc = Inputs + len(stages) - 1
	}
	return exec.NewPipeline(Inputs, stages, acc)
}

// Fused is the chain fused by hand at register level (DESIGN.md §29): one
// loop that loads the six inputs, multiplies them in the chain's order
// and stores once. No intermediate exists in memory, not even in scratch.
// It is what a fusion generator would have to emit for this chain, and
// the benchmark's reason to exist is to measure whether that is worth
// generating.
type Fused struct {
	mul6 func(dst, a, b, c, d, e, f []float32)
}

// NewFused returns the fused kernel for the internal/vec backend in use
// when it is called: AVX2 if vec.Backend reports it, scalar otherwise, so
// the suite's backend switch applies to it as to the other two forms.
func NewFused() Fused {
	if vec.Backend() == "avx2" && haveAVX2 {
		return Fused{mul6: mul6AVX2}
	}
	return Fused{mul6: mul6Scalar}
}

func (Fused) Radius() int                  { return 0 }
func (Fused) Arity() (inputs, outputs int) { return Inputs, 1 }

func (k Fused) Process(dst exec.Span, src exec.Window) {
	d, s := dst.Dst[0], src.Src
	whole := pointwise.Compact(d)
	for _, r := range s {
		whole = whole && pointwise.Compact(r)
	}
	if whole {
		n := d.Width * d.Height
		k.mul6(d.Data[:n], s[0].Data[:n], s[1].Data[:n], s[2].Data[:n],
			s[3].Data[:n], s[4].Data[:n], s[5].Data[:n])
		return
	}
	for y := range d.Height {
		k.mul6(d.Row(y), s[0].Row(y), s[1].Row(y), s[2].Row(y),
			s[3].Row(y), s[4].Row(y), s[5].Row(y))
	}
}

// mul6Scalar is the fused loop in plain Go. Each product is converted to
// float32 so that it is rounded before the next multiply, exactly as the
// chain rounds when it stores an intermediate: the Go spec lets an
// implementation fuse floating-point operations otherwise, and the fused
// form must equal the chain bit for bit.
func mul6Scalar(dst, a, b, c, d, e, f []float32) {
	a, b, c = a[:len(dst)], b[:len(dst)], c[:len(dst)]
	d, e, f = d[:len(dst)], e[:len(dst)], f[:len(dst)]
	for i := range dst {
		v := float32(a[i] * b[i])
		v = float32(v * c[i])
		v = float32(v * d[i])
		v = float32(v * e[i])
		dst[i] = v * f[i]
	}
}

// Check that the kernels satisfy the engine's contract.
var (
	_ exec.Kernel = mulStage{}
	_ exec.Kernel = Fused{}
)
