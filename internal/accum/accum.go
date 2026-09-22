// Package accum accumulates float32 values exactly, so that a sum, a mean
// or a variance over a raster does not depend on the order the cells were
// added in (DESIGN.md §49).
//
// # Representation
//
// A finite float32 is a signed 24-bit integer significand times a power
// of two fixed by its exponent field. Sum keeps one int64 bin per
// exponent and adds each value's significand to its bin: integer
// addition, so exact, associative and commutative, and the sum of the
// values is the sum over the bins of bin·2^(k−150). That makes the order
// of Adds, the split into partials and the order of Combines irrelevant
// to the bits of every result, which is the whole point.
//
// Moments adds the squares the same way. A float32 squared is exact in 48
// bits, so each square is split into two 24-bit halves added to the bins
// of its two weights.
//
// The scalar loop keeps its bins in four interleaved sets, one per
// position modulo four, which measured 25% faster than one set. The sets
// are merged when the bins are normalised and when a result is read.
//
// # Backends
//
// On arm64, and on amd64 where the CPU has AVX2, a GOEXPERIMENT=simd
// build adds 64-cell blocks in registers instead (blocks.go) and puts the
// same integers into the bins, so every result is bit for bit the scalar
// loop's. The choice
// between backends, and what was measured against them, is in
// benchmarks/reduce/RESULTS.md.
//
// # Limits
//
// A bin grows by less than 2^24 per value. Bins are normalised — carried
// so that every bin but the top one holds 0 or 1 — before they could
// overflow, and the top bin holds less than 2^23 per value ever added, so
// an accumulator takes up to MaxCount values. More panics: at 2^38 cells a
// float32 raster is a terabyte, and the limit is a guard, not a policy.
//
// # Values
//
// NaN and the infinities are counted, not binned, and the results follow
// IEEE arithmetic on them: any NaN, or both infinities, make the sum NaN,
// and one infinity makes it that infinity. A sum of zeros is -0 only when
// every value was -0, as IEEE addition gives. Results that are NaN are the
// canonical quiet NaN, whatever payload the inputs carried.
//
//strata:kernel
package accum

import (
	"fmt"
	"math"
)

// MaxCount is the number of values an accumulator, together with every
// accumulator combined into it, may take. Adding more panics.
const MaxCount = 1 << 38

const (
	// lanes is the number of interleaved bin sets.
	lanes = 4
	// sumBins is one bin per float32 exponent field, and 48 more. Field
	// 255 is NaN and the infinities, which are never binned. The bins
	// above 254 take the carries, and the upper pieces of the block sums a
	// vector backend adds 24 and 48 bins above a block's base.
	sumBins = 256 + 48
	// sqBins covers the squares: the low half of a square with exponent
	// field e goes to bin 2e and the high half to bin 2e+24. It is sized
	// for field 255, which is never binned, so that an index computed from
	// any uint8 field is provably in range, and 48 more take the upper
	// pieces of a vector backend's block sums and the carries.
	sqBins = 2*255 + 24 + 48 + 1
	// sumBias and sqBias are the negated exponents of bin 0's weight: bin
	// k of the sums weighs 2^(k−sumBias), bin j of the squares 2^(j−sqBias).
	sumBias = 150
	sqBias  = 2 * sumBias
	// normaliseEvery bounds how many values the bins take between
	// normalisations: each adds less than 2^24 to a bin, so 2^30 of them
	// leave a bin far below 2^63. It fits an int on every platform.
	normaliseEvery = 1 << 30
	// signBits is the bit pattern of -0.
	signBits = 1 << 31
)

// specials counts what is never binned.
type specials struct {
	// n is the number of values added, special or not.
	n int64
	// nan, posInf and negInf count those values.
	nan, posInf, negInf int64
	// pending counts the values added since the bins were last
	// normalised.
	pending int64
	// notNegZero is non-zero once any value other than -0 was added.
	notNegZero uint32
}

// Sum is an exact running sum of float32 values. The zero value is an
// empty sum, and the identity of Combine.
type Sum struct {
	bins [lanes][sumBins]int64
	specials
}

// Count returns the number of values added, NaN and infinities included.
func (s *Sum) Count() int64 { return s.n }

// Add adds every value of xs.
func (s *Sum) Add(xs []float32) {
	for len(xs) > 0 {
		k := min(len(xs), normaliseEvery)
		if s.admit(k) {
			s.normalise()
		}
		s.notNegZero |= sumKernel(&s.bins, &s.specials, xs[:k])
		xs = xs[k:]
	}
}

// Add1 adds one value. Add is the faster way to add a run of them.
func (s *Sum) Add1(v float32) {
	if s.admit(1) {
		s.normalise()
	}
	s.notNegZero |= addSum1(&s.bins[0], &s.specials, v)
}

// Combine adds everything added to b into s. b is not changed.
func (s *Sum) Combine(b *Sum) {
	s.combineSpecials(&b.specials)
	for l := range lanes {
		addBins(s.bins[0][:], b.bins[l][:])
	}
	s.normalise()
}

// normalise merges the bin sets into the first and carries it.
func (s *Sum) normalise() {
	for l := 1; l < lanes; l++ {
		addBins(s.bins[0][:], s.bins[l][:])
		clear(s.bins[l][:])
	}
	carry(s.bins[0][:])
	s.pending = 0
}

// addBins adds src to dst bin by bin.
func addBins(dst, src []int64) {
	n := min(len(dst), len(src))
	dst, src = dst[:n], src[:n]
	for k := range dst {
		dst[k] += src[k]
	}
}

// admit counts k more values, enforcing MaxCount, and reports whether the
// bins must be normalised before they take them.
func (s *specials) admit(k int) bool {
	if s.n+int64(k) > MaxCount {
		panic(fmt.Sprintf("accum: more than %d values", int64(MaxCount)))
	}
	s.n += int64(k)
	full := s.pending+int64(k) > normaliseEvery
	if full {
		s.pending = 0
	}
	s.pending += int64(k)
	return full
}

func (s *specials) combineSpecials(b *specials) {
	if s.n+b.n > MaxCount {
		panic(fmt.Sprintf("accum: more than %d values", int64(MaxCount)))
	}
	s.n += b.n
	s.nan += b.nan
	s.posInf += b.posInf
	s.negInf += b.negInf
	s.notNegZero |= b.notNegZero
}

// sumKernel is the function Sum.Add runs: addSums, or a vector backend's
// kernel where the CPU has one (blocks.go). Every backend adds the
// same integers to the bins in some order, so all give the same bits.
var sumKernel = addSums

// backend names the loops sumKernel and momentsKernel hold.
var backend = "scalar"

// Backend names the loops in use: "avx2", "neon" or "scalar".
func Backend() string { return backend }

// addSums is Sum's hot loop: four values per iteration, one into each
// bin set. It returns the OR of every value's bits with -0's cleared, for
// notNegZero.
func addSums(bins *[lanes][sumBins]int64, sp *specials, xs []float32) uint32 {
	var nz uint32
	b0, b1, b2, b3 := &bins[0], &bins[1], &bins[2], &bins[3]
	for ; len(xs) >= 4; xs = xs[4:] {
		u0 := math.Float32bits(xs[0])
		u1 := math.Float32bits(xs[1])
		u2 := math.Float32bits(xs[2])
		u3 := math.Float32bits(xs[3])
		nz |= (u0 ^ signBits) | (u1 ^ signBits) | (u2 ^ signBits) | (u3 ^ signBits)
		if special(u0, u1, u2, u3) {
			addSum1(b0, sp, xs[0])
			addSum1(b1, sp, xs[1])
			addSum1(b2, sp, xs[2])
			addSum1(b3, sp, xs[3])
			continue
		}
		e0, m0 := split(u0)
		e1, m1 := split(u1)
		e2, m2 := split(u2)
		e3, m3 := split(u3)
		b0[e0] += m0
		b1[e1] += m1
		b2[e2] += m2
		b3[e3] += m3
	}
	for _, v := range xs {
		nz |= addSum1(b0, sp, v)
	}
	return nz
}

// special reports whether any of the four bit patterns is NaN or an
// infinity: with the sign shifted out, those are exactly the patterns at
// or above +Inf's, so one max and one compare cover all four.
func special(u0, u1, u2, u3 uint32) bool {
	return max(u0<<1, u1<<1, u2<<1, u3<<1) >= 0x7f800000<<1
}

// addSum1 adds one value to one bin set and returns its bits with -0's
// cleared.
func addSum1(b *[sumBins]int64, sp *specials, v float32) uint32 {
	u := math.Float32bits(v)
	if u&0x7f800000 == 0x7f800000 {
		sp.addSpecial(u)
		return u ^ signBits
	}
	e, m := split(u)
	b[e] += m
	return u ^ signBits
}

func (s *specials) addSpecial(u uint32) {
	switch {
	case u&0x7fffff != 0:
		s.nan++
	case u&signBits != 0:
		s.negInf++
	default:
		s.posInf++
	}
}

// split returns the bin and the signed significand of a finite float32's
// bit pattern. A subnormal has exponent field 0 and no hidden bit, but
// weighs what field 1 does, so it goes to bin 1.
func split(u uint32) (bin uint8, sig int64) {
	e := (u >> 23) & 0xff
	m := int64(u&0x7fffff) | int64(min(e, 1))<<23
	neg := -int64(u >> 31) // 0 or -1
	return uint8(max(e, 1)), (m ^ neg) - neg
}

// carry normalises bins so that every bin but the last holds 0 or 1, by
// moving each bin's excess over its parity into the next bin, which
// weighs twice as much. The value the bins represent does not change.
func carry(bins []int64) {
	for ; len(bins) >= 2; bins = bins[1:] {
		c := bins[0] >> 1
		bins[0] -= c << 1
		bins[1] += c
	}
}
