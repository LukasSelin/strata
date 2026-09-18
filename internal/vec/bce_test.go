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
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/vec")
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
