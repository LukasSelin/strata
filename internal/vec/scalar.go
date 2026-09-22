// Package vec provides flat-slice numeric kernels for float32 data.
//
// This file holds the scalar backend: the correctness reference and
// fallback implementation for every kernel. Every SIMD backend added
// later must produce identical results to these functions, including
// on NaN, infinities, and zero-length or odd-length inputs.
//
// Each kernel first reslices its operands to the length of the slice it
// ranges over. The exported functions have already checked the lengths
// are equal; the reslice proves it to the compiler, which then drops the
// bounds check on every element (TestNoBoundsChecksInLoops).
package vec

import "math"

func scalarAddFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scalarSubFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func scalarMulFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func scalarDivFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scalarAddScalarFloat32(dst, src []float32, value float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = v + value
	}
}

func scalarMulScalarFloat32(dst, src []float32, value float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = v * value
	}
}

// scalarAffineFloat32 computes dst[i] = a*src[i] + b, the kernel of
// transfer.Rescale. The product is wrapped in an explicit float32
// conversion, which stops the compiler fusing the multiply and the add
// into one FMA: arm64 emits FMADD for a*v + b and amd64 does not, so
// without the conversion the canonical scalar result would differ
// between architectures, and the AVX2 kernel (a separate VMULPS and
// VADDPS) could not match it either (docs/adr/0001-simd-backend.md).
func scalarAffineFloat32(dst, src []float32, a, b float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = float32(v*a) + b
	}
}

// scalarSubDivFloat32 computes dst[i] = (src[i] - lo) / span, the kernel
// of algebra.Normalize. Subtracting before dividing, rather than
// multiplying by a reciprocal, is what makes src == lo give exactly 0 and
// src == lo + span exactly 1.
func scalarSubDivFloat32(dst, src []float32, lo, span float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = (v - lo) / span
	}
}

func scalarMinFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = min(a[i], b[i])
	}
}

func scalarMaxFloat32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

func scalarClampFloat32(dst, src []float32, lo, hi float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = min(max(v, lo), hi)
	}
}

// scalarAbsFloat32 clears the sign bit directly rather than branching on
// v < 0, so that -0 maps to +0 and NaN payloads are preserved unchanged.
func scalarAbsFloat32(dst, src []float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
	}
}

func scalarSqrtFloat32(dst, src []float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = float32(math.Sqrt(float64(v)))
	}
}

// scalarReduceMinFloat32 and scalarReduceMaxFloat32 fold src into acc
// with Go's builtin min and max, which make NaN absorbing and order -0
// below +0. Both are associative and commutative under those semantics,
// so a vector backend may fold its lanes in any order and still agree
// bit for bit (which NaN's payload survives aside, as everywhere in this
// package). An empty src returns acc, so acc is the identity across
// calls and a caller folds a whole band without an empty-slice case.
func scalarReduceMinFloat32(acc float32, src []float32) float32 {
	for _, v := range src {
		acc = min(acc, v)
	}
	return acc
}

func scalarReduceMaxFloat32(acc float32, src []float32) float32 {
	for _, v := range src {
		acc = max(acc, v)
	}
	return acc
}

// chainCells is how many cells scalarChainFrom carries from one step to
// the next. A whole band — the span the engine hands a kernel — is up to
// 65536 cells, or 256 KiB, which is where an unfused chain's buffers go
// and why they cost anything at all (DESIGN.md §29, §51).
//
// 2048 cells is 8 KiB: past the width at which BenchmarkWidth's per-call
// cost stops mattering, and a quarter of the 32 KiB L1 data cache of the
// smallest machine this project measures on, so the six or seven operand
// streams passing through keep the rest of it. Larger blocks keep
// measuring slightly faster on a machine with a 128 KiB L1, which is the
// per-call cost still shrinking rather than a cache effect, and is not a
// reason to spend another machine's L1.
const chainCells = 2048

// scalarChainFloat32 is the scalar chain evaluator: for each block of
// cells it applies every step of the chain before moving on, so an
// intermediate value never leaves the block.
//
// The block, rather than one cell at a time, is what keeps this the
// canonical reference cheaply. Each step is the very kernel the unfused
// chain would have called, over the same cells in the same order, so the
// bits are the staged chain's bits by construction and the tight loops
// stay the bounds-check-free ones this file already has. Carrying the
// value in a register instead is worth doing where a register holds
// eight cells at once; that is the vector backend's job (chain_amd64.go).
//
// The block is never copied into or out of: the first step reads the
// chain's starting input straight into it and the last writes straight
// out to dst, so a chain of n steps makes exactly n passes over the
// block and none over anything else.
func scalarChainFloat32(c *Chain, dst []float32, srcs [][]float32) {
	scalarChainFrom(c, dst, srcs, 0)
}

// scalarChainFrom runs the chain over dst[from:], reading srcs[i][from:].
// The offset is what lets the vector backend hand it a tail without
// building a second slice of operands to do it (chain_amd64.go).
func scalarChainFrom(c *Chain, dst []float32, srcs [][]float32, from int) {
	var block [chainCells]float32
	last := len(c.steps) - 1
	for off := from; off < len(dst); off += chainCells {
		n := min(chainCells, len(dst)-off)
		// acc is where the running value is read from: the chain's
		// starting input for the first step, the block after that.
		acc := srcs[c.first][off : off+n]
		for i, s := range c.steps {
			out := block[:n]
			if i == last {
				out = dst[off : off+n]
			}
			var src []float32
			if s.Op.Binary() {
				src = srcs[s.Src][off : off+n]
			}
			switch s.Op {
			case OpAdd:
				scalarAddFloat32(out, acc, src)
			case OpSub:
				scalarSubFloat32(out, acc, src)
			case OpMul:
				scalarMulFloat32(out, acc, src)
			case OpDiv:
				scalarDivFloat32(out, acc, src)
			case OpMin:
				scalarMinFloat32(out, acc, src)
			case OpMax:
				scalarMaxFloat32(out, acc, src)
			case OpAddScalar:
				scalarAddScalarFloat32(out, acc, s.K[0])
			case OpMulScalar:
				scalarMulScalarFloat32(out, acc, s.K[0])
			case OpAffine:
				scalarAffineFloat32(out, acc, s.K[0], s.K[1])
			case OpSubDiv:
				scalarSubDivFloat32(out, acc, s.K[0], s.K[1])
			case OpClamp:
				scalarClampFloat32(out, acc, s.K[0], s.K[1])
			case OpAbs:
				scalarAbsFloat32(out, acc)
			case OpSqrt:
				scalarSqrtFloat32(out, acc)
			}
			acc = out
		}
	}
}
