//go:build goexperiment.simd && arm64

package vec

import "simd/archsimd"

// This file is the NEON chain evaluator: register-level fusion
// (DESIGN.md §29). Where scalarChainFloat32 carries a block of cells
// from one step to the next, this carries four cells in one NEON
// register for the whole chain, so an intermediate value is never
// written anywhere at all — not to a buffer, not to L2, not to the
// stack. Each input is read once and dst is written once, which is the
// floor a chain of pointwise operations can reach.
//
// It is chain_amd64.go's loop at four lanes, with this backend's own
// rules: min4/max4 are NEON's FMIN and FMAX, which already are Go's
// builtins, and there is no SSE/AVX transition to clear before the
// scalar tail. It must agree bit for bit with scalarChainFloat32, which
// means every step is the instruction sequence its own kernel would
// emit — FMUL then FADD rather than FMADD for Affine.

// chainOp is one step of a chain as chainLanes needs it. The wrapper
// fills op, src and kf, which costs no vector instruction; chainLanes
// fills k0 and k1 from kf.
type chainOp struct {
	op Op
	// src is the slice a binary op reads, nil for every other op. It is
	// advanced one lane at a time, in step with dst.
	src []float32
	// kf are the step's immediates, and k0 and k1 the same values
	// broadcast to every lane.
	kf     [2]float32
	k0, k1 archsimd.Float32x4
}

func chainFloat32NEON(c *Chain, dst []float32, srcs [][]float32) {
	if len(dst) < neonLane {
		scalarChainFloat32(c, dst, srcs)
		return
	}
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
// keeps them so.
//
// The loop over ops is a loop over the chain's *operations*, not its
// cells: it runs at most MaxSteps times per four cells, so the bounds
// checks it cannot prove away are amortised over a vector rather than
// paid per element (see internal/vec/bce_test.go).
func chainLanes(ops []chainOp, first, dst []float32) int {
	for j := range ops {
		o := &ops[j]
		o.k0 = archsimd.BroadcastFloat32x4(o.kf[0])
		o.k1 = archsimd.BroadcastFloat32x4(o.kf[1])
	}
	n := len(dst)
	for len(dst) >= neonLane && len(first) >= neonLane {
		acc := load4(first)
		for j := range ops {
			o := &ops[j]
			switch o.op {
			case OpAdd:
				acc = acc.Add(load4(o.src))
				o.src = o.src[neonLane:]
			case OpSub:
				acc = acc.Sub(load4(o.src))
				o.src = o.src[neonLane:]
			case OpMul:
				acc = acc.Mul(load4(o.src))
				o.src = o.src[neonLane:]
			case OpDiv:
				acc = acc.Div(load4(o.src))
				o.src = o.src[neonLane:]
			case OpMin:
				acc = min4(acc, load4(o.src))
				o.src = o.src[neonLane:]
			case OpMax:
				acc = max4(acc, load4(o.src))
				o.src = o.src[neonLane:]
			case OpAddScalar:
				acc = acc.Add(o.k0)
			case OpMulScalar:
				acc = acc.Mul(o.k0)
			case OpAffine:
				acc = acc.Mul(o.k0).Add(o.k1)
			case OpSubDiv:
				acc = acc.Sub(o.k0).Div(o.k1)
			case OpClamp:
				acc = min4(max4(acc, o.k0), o.k1)
			case OpAbs:
				acc = acc.Abs()
			case OpSqrt:
				acc = acc.Sqrt()
			}
		}
		store4(acc, dst)
		dst, first = dst[neonLane:], first[neonLane:]
	}
	return n - len(dst)
}
