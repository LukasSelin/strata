package cog

import (
	"math"

	"github.com/LukasSelin/strata/cog/internal/kern"
	"github.com/LukasSelin/strata/raster"
)

// A block of one band is decoded a row at a time: the row's samples are
// written to the block's values, and the validity of every 64 cells
// written so far is worked out straight after, while they are still in
// the first-level cache, rather than in a second pass over the whole
// block once it has left it. The per-row kernels are in internal/kern,
// with AVX2 forms in GOEXPERIMENT=simd builds.

// nodataTest is a block's NoData test on the bits of its float32
// values, prepared once per block.
type nodataTest struct {
	mode       int
	want, care uint32 // testExact: NoData where bits&care == want
	lo         uint32 // testRange: NoData where bits-lo <= span
	span       uint64
}

const (
	testNone  = 0              // no NoData: every cell is valid
	testExact = kern.ModeExact // one bit pattern, or ±0
	testRange = kern.ModeRange // a run of patterns: ARE_REAL_EQUAL on float32
	testNaN   = kern.ModeNaN   // NoData is NaN: every NaN
)

// float32Test is float32 samples' test: GDAL's ARE_REAL_EQUAL, which
// for a finite non-zero NoData is a short run of bit patterns.
func float32Test(nd noData) nodataTest {
	switch {
	case !nd.set:
		return nodataTest{mode: testNone}
	case nd.nan:
		return nodataTest{mode: testNaN}
	}
	if lo, span, ok := realEqualRun32(nd.cmp32); ok {
		return nodataTest{mode: testRange, lo: lo, span: span}
	}
	return exactTest(nd.cmp32)
}

// intTest is 8- and 16-bit integers' test: they and their NoData are
// exact in float32, so a cell is NoData if and only if it equals it.
func intTest(nd noData) nodataTest {
	if !nd.set {
		return nodataTest{mode: testNone}
	}
	return exactTest(float32(nd.cmp))
}

func exactTest(v float32) nodataTest {
	if v == 0 {
		return nodataTest{mode: testExact, want: 0, care: 0x7fffffff} // either zero
	}
	return nodataTest{mode: testExact, want: math.Float32bits(v), care: ^uint32(0)}
}

// word returns the validity bits of chunk, at most 64 cells.
func (t *nodataTest) word(chunk []float32) uint64 {
	return kern.Word(chunk, t.mode, t.want, t.care, t.lo, t.span)
}

// validator builds a block's validity bits as its rows are written.
type validator struct {
	t     nodataTest
	valid []uint64 // nil for testNone
	done  int      // cells whose words are done, a multiple of 64
	all   bool     // every cell so far is valid
}

func newValidator(t nodataTest, n int) validator {
	v := validator{t: t, all: true}
	if t.mode != testNone {
		v.valid = make([]uint64, raster.MaskWords(n))
	}
	return v
}

// upto works out the words of the first n cells of vals that it has not
// yet, as far as whole words go.
func (v *validator) upto(vals []float32, n int) {
	if v.valid == nil {
		return
	}
	for ; v.done+64 <= n; v.done += 64 {
		w := v.t.word(vals[v.done : v.done+64])
		v.valid[v.done>>6] = w
		v.all = v.all && w == ^uint64(0)
	}
}

// finish works out the rest of vals, a last partial word, and returns the
// validity bits, or nil if every cell is valid.
func (v *validator) finish(vals []float32) []uint64 {
	v.upto(vals, len(vals))
	if v.valid == nil {
		return nil
	}
	if rest := vals[v.done:]; len(rest) > 0 {
		w := v.t.word(rest)
		v.valid[v.done>>6] = w
		v.all = v.all && w == ^uint64(0)>>(64-uint(len(rest)))
	}
	if v.all {
		return nil
	}
	return v.valid
}
