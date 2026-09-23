package focalrow_test

import (
	"testing"

	"github.com/LukasSelin/strata/internal/bcecheck"
)

// TestNoBoundsChecksInLoops fails if a bounds check survives inside a
// per-cell loop of the row kernels, in the build configuration the test
// runs in (DESIGN.md §39). The scalar kernels loop over terms outside and
// cells inside, each cell loop over slices resliced to its length once
// per term, so they carry none.
//
// The SIMD lane functions are the exception, by construction rather than
// by measurement: their innermost loop is over a cell's terms, and each
// term reads a block (or vector) at an offset the compiler cannot bound,
// so converting it to an array pointer costs one check. That is one
// compare per term per 32 cells on AVX2 and 16 on NEON (a vector in the
// tail loops), against a load, a multiply and an add per 8 or 4 cells;
// the loads inside a block are constant slices of that pointer and carry
// none. The alternative, a pass per term over the whole row, removes the
// check and adds a load and a store of the accumulator per term.
// foldColumn, which splits the AVX2 column passes over many rows into
// groups of rows, is exempt too: its loops run once per chunk of 1024
// cells and group of rows, and slice src once for each.
func TestNoBoundsChecksInLoops(t *testing.T) {
	inLoops, total, err := bcecheck.Check("github.com/LukasSelin/strata/internal/focalrow",
		"weightedLanes", "correlate2DLanes", "sumLanes", "extremeLanes", "foldColumn")
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
