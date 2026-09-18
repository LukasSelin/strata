package stencil_test

import (
	"testing"

	"github.com/LukasSelin/strata/internal/bcecheck"
)

// TestNoBoundsChecksInLoops fails if a bounds check survives inside a
// per-cell loop of the row kernels, in the loop itself or in code
// inlined into one, in the build configuration the test runs in. The
// kernels read their 3×3 window through the shifted row views hornViews
// gives them for that reason: the compiler charged two checks per cell
// for r0[i+1] and r0[i+2], which it cannot prove in bounds from a range
// over a slice two cells shorter (DESIGN.md §39).
//
// Two exceptions, both measured rather than assumed:
//
//   - scalarHornHillshadeRow keeps its two checks per cell because the
//     views cost it more in spilled registers than the checks cost in
//     branches, 33 to 56% over BenchmarkRowWidth. Its own comment has
//     the numbers.
//   - mask.go works a word at a time, so a check there costs a 64th of
//     what one costs in a cell loop, and its word indices come from bit
//     offsets a caller chose, which the compiler cannot prove in bounds
//     without restructuring the bit shuffling around them.
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/stencil", "mask.go", "scalarHornHillshadeRow")
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
