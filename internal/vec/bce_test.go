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
// The two chain evaluators are named, because their tightest loop runs
// over a chain's operations rather than over its cells: at most MaxSteps
// iterations per block of chainCells cells (scalar) or per vector of
// avxLane cells (AVX2). Every check they leave is one step's operand
// lookup — including the reslice prologue of a kernel inlined into them
// — so it is amortised over a whole block or lane instead of being paid
// per element, which is what this test exists to prevent. The element
// loops inside those kernels stay checked wherever else they are called.
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/vec",
		"scalarChainFrom", "chainLanes")
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
