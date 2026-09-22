package vec_test

import (
	"testing"

	"github.com/LukasSelin/strata/internal/bcecheck"
)

// TestNoBoundsChecksInLoops compiles this package, in the build
// configuration the test runs in (GOEXPERIMENT included), and fails if a
// bounds check survives inside a kernel's loop: in the loop itself, or
// in code inlined into one, such as the array loads of the AVX2 kernels
// or the scalar tails they call. A bounds check per element costs as
// much as the arithmetic it guards. Checks outside loops, such as the
// reslices that prove the operands as long as dst, run once per call and
// are allowed.
//
// The chain evaluators are named, because their tightest loop runs over
// a chain's operations rather than over its cells: at most MaxSteps
// iterations per block of chainCells cells (scalar), per vector of one
// lane width (the SIMD backends), or once per call (the wrappers, which
// lay a chain's steps out before handing them to the lane loop). Every
// check they leave is one step's operand lookup — including the reslice
// prologue of a kernel inlined into them — so it is amortised over a
// whole block, lane or call instead of being paid per element, which is
// what this test exists to prevent. The element loops inside those
// kernels stay checked wherever else they are called.
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/vec",
		"scalarChainFrom", "chainLanes", "chainFloat32AVX2", "chainFloat32NEON")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range inLoops {
		t.Error(c)
	}
	if total == 0 {
		t.Fatal("the optimization log has no bounds checks at all; it was probably not written")
	}
}
