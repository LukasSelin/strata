package resamp_test

import (
	"testing"

	"github.com/LukasSelin/strata/internal/bcecheck"
)

// TestNoBoundsChecksInLoops fails if a bounds check survives inside a
// per-cell or per-tap loop of the kernels, in the build configuration the
// test runs in (DESIGN.md §39). Allowed, each with the reason its
// checks cannot go:
//
//   - table.go: the tables are built once per call;
//   - Direct2D: the reference, which reads every tap through the tables;
//   - columnAt: an output column's first tap and weights, looked up in the
//     tables, once per column for every four or eight rows, not per tap;
//   - hRowsNEON and hRowsAVX2: the offsets of each block of four or eight
//     rows, once per block.
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/resamp",
		"table.go", "Direct2D", "columnAt", "hRowsNEON", "hRowsAVX2")
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
