package array

import "math"

// exactSum is an exact running sum of float64 values: a list of
// non-overlapping partials whose sum is the sum of every finite value
// added, kept by Shewchuk's error-free additions, and rounded once when
// read, the algorithm of Python's math.fsum. Since the partials hold the
// exact sum, the result does not depend on the order values were added
// in, and it is correctly rounded.
//
// It is the axis reductions' counterpart of internal/accum, which keeps
// ~10 KB of bins per accumulator: right for one sum over a whole raster,
// too much for one per output element. Both return the correctly rounded
// sum, so they agree bit for bit on float32 input.
//
// The zero value is an empty sum.
type exactSum struct {
	p []float64
	// n counts every value added, finite or not.
	n int64
	// nan, posInf and negInf record non-finite inputs; overflow is the
	// sign of a finite sum that left the float64 range, or 0.
	nan, posInf, negInf bool
	overflow            float64
	// notNegZero is set once any value other than -0 was added.
	notNegZero bool
}

// reset empties s, keeping its partials' storage.
func (s *exactSum) reset() {
	*s = exactSum{p: s.p[:0]}
}

// add adds x.
func (s *exactSum) add(x float64) {
	s.n++
	switch {
	case x != x:
		s.nan = true
		return
	case math.IsInf(x, 1):
		s.posInf = true
		return
	case math.IsInf(x, -1):
		s.negInf = true
		return
	}
	if x != 0 || !math.Signbit(x) {
		s.notNegZero = true
	}
	if s.overflow != 0 {
		return
	}
	var ok bool
	s.p, ok = grow(s.p, x)
	if !ok {
		s.overflow = math.Copysign(1, x)
		s.p = s.p[:0]
	}
}

// grow adds x to the expansion p exactly and returns it, with ok false if
// a partial overflowed.
func grow(p []float64, x float64) ([]float64, bool) {
	i := 0
	for _, y := range p {
		if math.Abs(x) < math.Abs(y) {
			x, y = y, x
		}
		hi := x + y
		if math.IsInf(hi, 0) {
			return p, false
		}
		lo := y - (hi - x)
		if lo != 0 {
			p[i] = lo
			i++
		}
		x = hi
	}
	return append(p[:i], x), true
}

// special returns the sum when a non-finite value or an overflow decides
// it, with ok false otherwise. It follows IEEE addition: any NaN, or both
// infinities, give NaN; one infinity gives that infinity.
func (s *exactSum) special() (v float64, ok bool) {
	switch {
	case s.nan || s.posInf && s.negInf:
		return math.NaN(), true
	case s.posInf:
		return math.Inf(1), true
	case s.negInf:
		return math.Inf(-1), true
	case s.overflow != 0:
		return math.Inf(int(s.overflow)), true
	}
	return 0, false
}

// value returns the sum, correctly rounded. With nothing added it is 0.
// A sum of zeros is -0 only when every value was -0.
func (s *exactSum) value() float64 {
	if v, ok := s.special(); ok {
		return v
	}
	if s.n > 0 && !s.notNegZero {
		return math.Copysign(0, -1)
	}
	v := round(s.p)
	if v == 0 {
		return 0
	}
	return v
}

// round returns the sum of the non-overlapping, increasing partials p,
// correctly rounded to nearest, ties to even: CPython's math.fsum
// finish. It walks down from the largest partial until a sum is inexact,
// then corrects a result that sits on a rounding tie the lower partials
// break.
func round(p []float64) float64 {
	n := len(p)
	if n == 0 {
		return 0
	}
	n--
	hi := p[n]
	var lo float64
	for n > 0 {
		x := hi
		n--
		y := p[n]
		hi = x + y
		yr := hi - x
		lo = y - yr
		if lo != 0 {
			break
		}
	}
	if n > 0 && (lo < 0 && p[n-1] < 0 || lo > 0 && p[n-1] > 0) {
		y := lo * 2
		x := hi + y
		yr := x - hi
		if y == yr {
			hi = x
		}
	}
	return hi
}

// mean returns the sum divided by the number of values added, correctly
// rounded: not the rounded sum divided and rounded again, but the
// float64 nearest the exact quotient, ties to even. scratch is reused
// storage for the exact comparisons, returned grown. With nothing added
// the mean is NaN.
func (s *exactSum) mean(scratch []float64) (float64, []float64) {
	if s.n == 0 {
		return math.NaN(), scratch
	}
	if v, ok := s.special(); ok {
		return v, scratch
	}
	sum := s.value()
	k := float64(s.n) // exact: n is far below 2^53
	q := sum / k
	if a := math.Abs(q); sum == 0 || a > math.MaxFloat64/(4*k) || a < 0x1p-960 {
		// Zero is exact. Near the ends of the range the products below
		// could overflow or lose bits; float32 input never gets there.
		return q, scratch
	}
	// The exact quotient is within a few ulps of q. Step q toward it
	// while it lies beyond the midpoint to a neighbour; the sign of
	// sum − mid·k is exact, computed on the partials.
	for {
		up := math.Nextafter(q, math.Inf(1))
		var sg int
		sg, scratch = s.cmpMid(q, up, k, scratch)
		if sg > 0 {
			q = up
			continue
		}
		if sg == 0 {
			return even(q, up), scratch
		}
		break
	}
	for {
		down := math.Nextafter(q, math.Inf(-1))
		var sg int
		sg, scratch = s.cmpMid(down, q, k, scratch)
		if sg < 0 {
			q = down
			continue
		}
		if sg == 0 {
			return even(down, q), scratch
		}
		return q, scratch
	}
}

// cmpMid returns the sign of S − ((lo+hi)/2)·k, where S is the exact sum
// and lo < hi are neighbouring float64s. The midpoint is lo + h with h
// half their difference, a power of two, so both products are exact as
// two-term expansions and the comparison is exact.
func (s *exactSum) cmpMid(lo, hi, k float64, scratch []float64) (int, []float64) {
	h := (hi - lo) / 2
	e := append(scratch[:0], s.p...)
	ph := float64(lo * k)
	pl := math.FMA(lo, k, -ph)
	hk := float64(h * k) // exact: a power of two times an integer below 2^53
	var ok bool
	for _, t := range [...]float64{-ph, -pl, -hk} {
		if e, ok = grow(e, t); !ok {
			panic("array: exact mean overflowed") // mean keeps q far from the limit
		}
	}
	v := round(e)
	switch {
	case v > 0:
		return 1, e
	case v < 0:
		return -1, e
	}
	return 0, e
}

// even returns whichever of two neighbouring float64s has an even
// significand.
func even(a, b float64) float64 {
	if math.Float64bits(a)&1 == 0 {
		return a
	}
	return b
}
