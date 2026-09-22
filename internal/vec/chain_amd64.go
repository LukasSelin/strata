//go:build goexperiment.simd && amd64

package vec

import "simd/archsimd"

// This file is the AVX2 chain evaluator: register-level fusion
// (DESIGN.md §29). Where scalarChainFloat32 carries a block of cells
// from one step to the next, this carries eight cells in one YMM
// register for the whole chain, so an intermediate value is never
// written anywhere at all — not to a buffer, not to L2, not to the
// stack. Each input is read once and dst is written once, which is the
// floor a chain of pointwise operations can reach.
//
// It must agree bit for bit with scalarChainFloat32, which means every
// step is the same instruction sequence its own kernel would emit:
// min8/max8 rather than VMINPS/VMAXPS, and VMULPS then VADDPS rather
// than VFMADD for Affine.

// chainOp is one step of a chain as chainLanes needs it. The wrapper
// fills op, src and kf, which costs no vector instruction; chainLanes
// fills k0 and k1 from kf, which is the first 256-bit work either does.
type chainOp struct {
	op Op
	// src is the slice a binary op reads, nil for every other op. It is
	// advanced one lane at a time, in step with dst.
	src []float32
	// kf are the step's immediates, and k0 and k1 the same values
	// broadcast to every lane.
	kf     [2]float32
	k0, k1 archsimd.Float32x8
}

func chainFloat32AVX2(c *Chain, dst []float32, srcs [][]float32) {
	if len(dst) < avxLane {
		scalarChainFloat32(c, dst, srcs)
		return
	}
	// Declared and filled here, before any 256-bit instruction, so that
	// zeroing it and storing its slice headers — either of which the
	// compiler may do with legacy SSE — cannot land inside the AVX
	// region chainLanes holds. This is the rule of reduceMinFloat32AVX2
	// and internal/stencil/simd_amd64.go.
	var ops [MaxSteps]chainOp
	for i, s := range c.steps {
		ops[i].op, ops[i].kf = s.Op, s.K
		if s.Op.Binary() {
			ops[i].src = srcs[s.Src]
		}
	}
	n := chainLanes(ops[:len(c.steps)], srcs[c.first], dst)
	scalarChainFrom(c, dst, srcs, n)
}

// chainLanes runs the chain over the whole lanes of dst, reading first
// as the chain's starting value, and returns how many cells it consumed.
// Every slice it is given is as long as dst, so advancing them together
// keeps them so. It clears the upper AVX bits before it returns, so its
// caller's scalar tail pays no SSE/AVX transition.
//
// The loop over ops is a loop over the chain's *operations*, not its
// cells: it runs at most MaxSteps times per eight cells, so the bounds
// checks it cannot prove away are amortised over a vector rather than
// paid per element (see internal/vec/bce_test.go).
func chainLanes(ops []chainOp, first, dst []float32) int {
	for j := range ops {
		o := &ops[j]
		o.k0 = archsimd.BroadcastFloat32x8(o.kf[0])
		o.k1 = archsimd.BroadcastFloat32x8(o.kf[1])
	}
	n := len(dst)
	for len(dst) >= avxLane && len(first) >= avxLane {
		acc := load8(first)
		for j := range ops {
			o := &ops[j]
			switch o.op {
			case OpAdd:
				acc = acc.Add(load8(o.src))
				o.src = o.src[avxLane:]
			case OpSub:
				acc = acc.Sub(load8(o.src))
				o.src = o.src[avxLane:]
			case OpMul:
				acc = acc.Mul(load8(o.src))
				o.src = o.src[avxLane:]
			case OpDiv:
				acc = acc.Div(load8(o.src))
				o.src = o.src[avxLane:]
			case OpMin:
				acc = min8(acc, load8(o.src))
				o.src = o.src[avxLane:]
			case OpMax:
				acc = max8(acc, load8(o.src))
				o.src = o.src[avxLane:]
			case OpAddScalar:
				acc = acc.Add(o.k0)
			case OpMulScalar:
				acc = acc.Mul(o.k0)
			case OpAffine:
				acc = acc.Mul(o.k0).Add(o.k1)
			case OpSubDiv:
				acc = acc.Sub(o.k0).Div(o.k1)
			case OpClamp:
				acc = min8(max8(acc, o.k0), o.k1)
			case OpAbs:
				acc = acc.Abs()
			case OpSqrt:
				acc = acc.Sqrt()
			}
		}
		store8(acc, dst)
		dst, first = dst[avxLane:], first[avxLane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}
