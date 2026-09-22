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

// chainWide is how many vectors chainLanes carries at once. Four keeps
// the accumulators well inside the register file and divides the
// per-operation dispatch — a switch, a slice advance and the bounds
// checks under them — by four, which is what makes the loop faster than
// the staged kernels it replaces rather than slower (DESIGN.md §29).
const chainWide = 4

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
	if len(dst) < chainWide*neonLane {
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

// chainLanes runs the chain over whole groups of chainWide vectors of
// dst, reading first as the chain's starting value, and returns how many
// cells it consumed. Every slice it is given is as long as dst, so
// advancing them together keeps them so.
//
// The group, rather than one vector, is what makes the loop worth
// entering. Its body is a loop over the chain's *operations*, and each
// pass of that loop costs a switch, a slice advance and the bounds
// checks neither can prove away; over one vector that is as much work as
// the arithmetic it dispatches, and the whole point of carrying the
// value in a register is lost. Over chainWide vectors it is a quarter of
// that, and the accumulators are four of the thirty-two registers.
//
// What is left over goes to scalarChainFrom rather than a second copy of
// this switch: at most chainWide·neonLane - 1 cells of a span, and the
// block evaluator is already the reference for every step.
func chainLanes(ops []chainOp, first, dst []float32) int {
	for j := range ops {
		o := &ops[j]
		o.k0 = archsimd.BroadcastFloat32x4(o.kf[0])
		o.k1 = archsimd.BroadcastFloat32x4(o.kf[1])
	}
	const wide = chainWide * neonLane
	n := len(dst)
	for len(dst) >= wide && len(first) >= wide {
		a0, a1 := load4(first), load4(first[neonLane:])
		a2, a3 := load4(first[2*neonLane:]), load4(first[3*neonLane:])
		for j := range ops {
			o := &ops[j]
			s := o.src
			switch o.op {
			case OpAdd:
				a0, a1 = a0.Add(load4(s)), a1.Add(load4(s[neonLane:]))
				a2, a3 = a2.Add(load4(s[2*neonLane:])), a3.Add(load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpSub:
				a0, a1 = a0.Sub(load4(s)), a1.Sub(load4(s[neonLane:]))
				a2, a3 = a2.Sub(load4(s[2*neonLane:])), a3.Sub(load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpMul:
				a0, a1 = a0.Mul(load4(s)), a1.Mul(load4(s[neonLane:]))
				a2, a3 = a2.Mul(load4(s[2*neonLane:])), a3.Mul(load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpDiv:
				a0, a1 = a0.Div(load4(s)), a1.Div(load4(s[neonLane:]))
				a2, a3 = a2.Div(load4(s[2*neonLane:])), a3.Div(load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpMin:
				a0, a1 = min4(a0, load4(s)), min4(a1, load4(s[neonLane:]))
				a2, a3 = min4(a2, load4(s[2*neonLane:])), min4(a3, load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpMax:
				a0, a1 = max4(a0, load4(s)), max4(a1, load4(s[neonLane:]))
				a2, a3 = max4(a2, load4(s[2*neonLane:])), max4(a3, load4(s[3*neonLane:]))
				o.src = s[wide:]
			case OpAddScalar:
				a0, a1, a2, a3 = a0.Add(o.k0), a1.Add(o.k0), a2.Add(o.k0), a3.Add(o.k0)
			case OpMulScalar:
				a0, a1, a2, a3 = a0.Mul(o.k0), a1.Mul(o.k0), a2.Mul(o.k0), a3.Mul(o.k0)
			case OpAffine:
				a0, a1 = a0.Mul(o.k0).Add(o.k1), a1.Mul(o.k0).Add(o.k1)
				a2, a3 = a2.Mul(o.k0).Add(o.k1), a3.Mul(o.k0).Add(o.k1)
			case OpSubDiv:
				a0, a1 = a0.Sub(o.k0).Div(o.k1), a1.Sub(o.k0).Div(o.k1)
				a2, a3 = a2.Sub(o.k0).Div(o.k1), a3.Sub(o.k0).Div(o.k1)
			case OpClamp:
				a0, a1 = min4(max4(a0, o.k0), o.k1), min4(max4(a1, o.k0), o.k1)
				a2, a3 = min4(max4(a2, o.k0), o.k1), min4(max4(a3, o.k0), o.k1)
			case OpAbs:
				a0, a1, a2, a3 = a0.Abs(), a1.Abs(), a2.Abs(), a3.Abs()
			case OpSqrt:
				a0, a1, a2, a3 = a0.Sqrt(), a1.Sqrt(), a2.Sqrt(), a3.Sqrt()
			}
		}
		store4(a0, dst)
		store4(a1, dst[neonLane:])
		store4(a2, dst[2*neonLane:])
		store4(a3, dst[3*neonLane:])
		dst, first = dst[wide:], first[wide:]
	}
	return n - len(dst)
}
