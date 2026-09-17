// Package rapidsource implements fuzzdata.Source with rapid's
// generators (DESIGN.md §39), so a test body written for a fuzz input
// also runs as a property test: rapid searches for a failing case and
// then shrinks it, printing the draws that remain instead of a blob of
// bytes, and reruns it from the seed it reports.
//
// Values follow fuzzdata.Reader's, which decodes a fuzz input, so the
// same body sees the same kinds of operand either way: shapes and
// options in range, and float32s biased towards the values kernels get
// wrong. What differs is the search. A Reader takes each value from the
// next bytes of one input, while a Source draws each one separately, and
// rapid simplifies the draws of a failing case one at a time: a 40×9 DEM
// of arbitrary bits shrinks towards the smallest raster and the plainest
// values that still fail.
package rapidsource

import (
	"fmt"
	"math"

	"pgregory.net/rapid"

	"strata/internal/fuzzdata"
)

// Source draws the values of fuzzdata.Source from rapid.
type Source struct {
	t     *rapid.T
	draws int
}

// New returns a Source drawing from t.
func New(t *rapid.T) *Source { return &Source{t: t} }

var _ fuzzdata.Source = (*Source)(nil)

// label names a draw in rapid's report of a failing case. Each is
// numbered, because a test draws in the order its own branches take and
// the same name comes up again.
func (s *Source) label(name string) string {
	s.draws++
	return fmt.Sprintf("%s#%d", name, s.draws)
}

// Uint64 returns an arbitrary word.
func (s *Source) Uint64() uint64 { return rapid.Uint64().Draw(s.t, s.label("uint64")) }

// Dense returns a word whose bits are each clear with probability about
// 2⁻ⁿ: a validity mask with most cells valid.
func (s *Source) Dense(n int) uint64 {
	var v uint64
	for range n {
		v |= rapid.Uint64().Draw(s.t, s.label("dense"))
	}
	return v
}

// Byte returns an arbitrary byte.
func (s *Source) Byte() byte { return rapid.Byte().Draw(s.t, s.label("byte")) }

// Bool returns an arbitrary boolean.
func (s *Source) Bool() bool { return rapid.Bool().Draw(s.t, s.label("bool")) }

// IntN returns a value in [0, n). It panics if n <= 0.
func (s *Source) IntN(n int) int {
	if n <= 0 {
		panic("rapidsource: IntN needs n > 0")
	}
	return rapid.IntRange(0, n-1).Draw(s.t, s.label("intn"))
}

// Range returns a value in [lo, hi].
func (s *Source) Range(lo, hi int) int {
	return rapid.IntRange(lo, hi).Draw(s.t, s.label("range"))
}

// float32Gen draws the values fuzzdata.Reader.Float32 decodes: one of
// the specials kernels get wrong, an arbitrary bit pattern, or a small
// DEM-like value. Simplest first, so shrinking lands on a plain number
// rather than a NaN payload.
var float32Gen = rapid.OneOf(
	rapid.Custom(func(t *rapid.T) float32 {
		return 1000 + float32(rapid.IntRange(-128, 127).Draw(t, "quarters"))/4
	}),
	rapid.SampledFrom(fuzzdata.Specials),
	rapid.Custom(func(t *rapid.T) float32 {
		return math.Float32frombits(rapid.Uint32().Draw(t, "bits"))
	}),
)

// Float32 returns a special value, an arbitrary bit pattern, or a small
// DEM-like value.
func (s *Source) Float32() float32 { return float32Gen.Draw(s.t, s.label("float32")) }

// specials64 are float64 values past float32's range.
var specials64 = []float64{
	math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64,
	1e300, 1e-300, 1e39, 1e-39, 1e-46, 1e20,
}

var float64Gen = rapid.OneOf(
	rapid.Custom(func(t *rapid.T) float64 {
		return float64(rapid.IntRange(-128, 127).Draw(t, "quarters")) / 4
	}),
	rapid.SampledFrom(specials64),
	rapid.Custom(func(t *rapid.T) float64 {
		return float64(rapid.SampledFrom(fuzzdata.Specials).Draw(t, "special32"))
	}),
	rapid.Custom(func(t *rapid.T) float64 {
		return math.Float64frombits(rapid.Uint64().Draw(t, "bits"))
	}),
)

// Float64 is Float32 for float64: a small value, one of the specials of
// either width, or an arbitrary bit pattern.
func (s *Source) Float64() float64 { return float64Gen.Draw(s.t, s.label("float64")) }
