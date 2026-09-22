package vec

import "fmt"

// This file is the fused chain evaluator (DESIGN.md §29). A Chain is a
// left-deep sequence of this package's own pointwise kernels, run as one
// pass over the cells instead of one pass per kernel, so the value
// between two operations is never written to a buffer.
//
// It computes exactly what the same operations compute one at a time.
// That is the whole contract: a Chain is an optimisation, and the staged
// form is its reference, so every Op means precisely what its kernel
// means — the NaN and signed-zero conventions of Min and Max, the
// unfused multiply-add of Affine, the sign-bit clear of Abs.

// Op names one operation of a Chain.
//
// A binary op combines the running value with a slice named by Step.Src;
// every other op transforms the running value alone, using the
// immediates in Step.K.
type Op uint8

const (
	OpAdd       Op = iota // acc + src
	OpSub                 // acc - src
	OpMul                 // acc * src
	OpDiv                 // acc / src
	OpMin                 // min(acc, src)
	OpMax                 // max(acc, src)
	OpAddScalar           // acc + K[0]
	OpMulScalar           // acc * K[0]
	OpAffine              // float32(acc*K[0]) + K[1]
	OpClamp               // min(max(acc, K[0]), K[1])
	OpAbs                 // |acc|
	OpSqrt                // sqrt(acc)
	numOps
)

// opBinary is the last op that reads a source slice. The binary ops are
// first in the list so that this is a comparison rather than a table.
const opBinary = OpMax

var opNames = [numOps]string{
	OpAdd: "Add", OpSub: "Sub", OpMul: "Mul", OpDiv: "Div",
	OpMin: "Min", OpMax: "Max",
	OpAddScalar: "AddScalar", OpMulScalar: "MulScalar",
	OpAffine: "Affine", OpClamp: "Clamp", OpAbs: "Abs", OpSqrt: "Sqrt",
}

func (o Op) String() string {
	if o >= numOps {
		return fmt.Sprintf("Op(%d)", uint8(o))
	}
	return opNames[o]
}

// Binary reports whether o reads a source slice, and so needs a Step.Src.
func (o Op) Binary() bool { return o <= opBinary }

// Step is one operation of a Chain.
type Step struct {
	// Op is the operation.
	Op Op
	// Src names the slice a binary op reads, as an index into the srcs
	// Run is given. It is -1 for every other op.
	Src int
	// K holds the immediates of the ops that take them, and is ignored
	// by the ops that do not.
	K [2]float32
}

// MaxSteps is the longest chain NewChain accepts. The evaluators lay out
// one operation's working state per step in a stack array, so the bound
// is what keeps that array off the heap and out of the engine's scratch;
// no chain the pipeline lowering builds comes close to it.
const MaxSteps = 16

// Chain is a left-deep chain of pointwise operations, evaluated in one
// pass with the running value carried between the operations rather than
// through memory (DESIGN.md §29).
//
// It is left-deep: the running value starts as one input and is always
// the *first* operand of every step, so a chain has exactly one live
// intermediate and the vector backend can hold it in a register for the
// whole chain. Operands are never swapped, even for an operation that
// would commute, because a caller that wants src-op-acc is asking for a
// different chain and should say so.
//
// A Chain is immutable once built, so one value serves every worker.
type Chain struct {
	inputs int
	first  int
	steps  []Step
}

// NewChain returns a Chain that starts from srcs[first] and applies
// steps to it, over inputs source slices.
//
// It panics on programming errors, as the rest of the kernels do: no
// steps or more than MaxSteps, a first that is not an input, an unknown
// operation, a binary step whose Src is not an input, or a step that
// names a Src it does not read. steps is copied, so a caller may reuse it.
func NewChain(inputs, first int, steps []Step) *Chain {
	if inputs < 1 {
		panic(fmt.Sprintf("vec: chain has %d inputs; it needs at least one", inputs))
	}
	if first < 0 || first >= inputs {
		panic(fmt.Sprintf("vec: chain starts from input %d, which is not one of its %d", first, inputs))
	}
	if len(steps) == 0 {
		panic("vec: chain has no steps")
	}
	if len(steps) > MaxSteps {
		panic(fmt.Sprintf("vec: chain has %d steps, at most %d fit", len(steps), MaxSteps))
	}
	for i, s := range steps {
		switch {
		case s.Op >= numOps:
			panic(fmt.Sprintf("vec: chain step %d has unknown operation %s", i, s.Op))
		case s.Op.Binary() && (s.Src < 0 || s.Src >= inputs):
			panic(fmt.Sprintf("vec: chain step %d (%s) reads input %d, which is not one of its %d",
				i, s.Op, s.Src, inputs))
		case !s.Op.Binary() && s.Src != -1:
			panic(fmt.Sprintf("vec: chain step %d (%s) reads no input, so its Src must be -1, not %d",
				i, s.Op, s.Src))
		}
	}
	return &Chain{inputs: inputs, first: first, steps: append([]Step(nil), steps...)}
}

// Inputs is how many slices Run expects.
func (c *Chain) Inputs() int { return c.inputs }

// Steps is how many operations the chain runs.
func (c *Chain) Steps() int { return len(c.steps) }

// Run computes dst from srcs, which must hold Inputs slices each as long
// as dst. Inputs the chain never reads are still required, and still
// checked, so that a caller may pass the operands it has in one slice.
func (c *Chain) Run(dst []float32, srcs [][]float32) {
	if len(srcs) != c.inputs {
		panic(fmt.Sprintf("vec: chain takes %d inputs, Run got %d", c.inputs, len(srcs)))
	}
	for i, s := range srcs {
		if len(s) != len(dst) {
			panic(fmt.Sprintf("vec: chain input %d has length %d, dst has %d", i, len(s), len(dst)))
		}
	}
	if len(dst) == 0 {
		return
	}
	chainFloat32(c, dst, srcs)
}
