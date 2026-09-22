package accum

import (
	"math"
	"math/big"
)

// Value returns the sum of the values added, correctly rounded to
// float64: the float64 nearest the exact sum, ties to even. An empty sum
// is +0.
func (s *Sum) Value() float64 {
	if v, ok := s.special(); ok {
		return v
	}
	x := exact(s.bins[0][:], s.bins[1][:], s.bins[2][:], s.bins[3][:])
	if x.Sign() == 0 {
		return s.zero()
	}
	r, _ := new(big.Rat).SetFrac(x, weight(1, sumBias)).Float64()
	return r
}

// Sum is Sum.Value over the values added to m.
func (m *Moments) Sum() float64 { return m.sum.Value() }

// Mean returns the exact sum divided by Count, correctly rounded, or NaN
// with no values.
func (m *Moments) Mean() float64 {
	s := &m.sum
	if s.n == 0 {
		return math.NaN()
	}
	if v, ok := s.special(); ok {
		return v
	}
	x := exact(s.bins[0][:], s.bins[1][:], s.bins[2][:], s.bins[3][:])
	if x.Sign() == 0 {
		return s.zero()
	}
	r, _ := new(big.Rat).SetFrac(x, weight(s.n, sumBias)).Float64()
	return r
}

// Variance returns the population variance of the values, Σ(x−mean)²/n,
// correctly rounded from its exact value. It is NaN with no values and
// when any value is NaN or infinite.
func (m *Moments) Variance() float64 {
	num, den, ok := m.variance()
	if !ok {
		return math.NaN()
	}
	r, _ := new(big.Rat).SetFrac(num, den).Float64()
	return r
}

// StdDev returns the square root of Variance's exact value. The root is
// taken at 256 bits and rounded to float64, which is within one ulp of
// the true root and, like every result here, a function of the values
// alone, not of the order they were added in. It is NaN when Variance is.
func (m *Moments) StdDev() float64 {
	num, den, ok := m.variance()
	if !ok {
		return math.NaN()
	}
	if num.Sign() == 0 {
		return 0
	}
	const prec = 256
	f := new(big.Float).SetPrec(prec).SetInt(num)
	f.Quo(f, new(big.Float).SetPrec(prec).SetInt(den))
	f.Sqrt(f)
	r, _ := f.Float64()
	return r
}

// variance returns the exact variance as num/den, or ok false when it is
// NaN. With S1 the sum and S2 the sum of squares,
// Σ(x−mean)²/n = (n·S2 − S1²)/n², and with the bins' scales
// S1 = X1·2^−150 and S2 = X2·2^−300 that is (n·X2 − X1²)/(n²·2^300), a
// ratio of integers.
func (m *Moments) variance() (num, den *big.Int, ok bool) {
	s := &m.sum
	if s.n == 0 || s.nan > 0 || s.posInf > 0 || s.negInf > 0 {
		return nil, nil, false
	}
	x1 := exact(s.bins[0][:], s.bins[1][:], s.bins[2][:], s.bins[3][:])
	x2 := exact(m.sq[0][:], m.sq[1][:], m.sq[2][:], m.sq[3][:])
	n := big.NewInt(s.n)
	num = new(big.Int).Mul(n, x2)
	num.Sub(num, x1.Mul(x1, x1))
	den = new(big.Int).Mul(n, n)
	den.Lsh(den, sqBias)
	return num, den, true
}

// special returns the sum when a NaN or an infinity decides it.
func (s *specials) special() (float64, bool) {
	switch {
	case s.nan > 0 || s.posInf > 0 && s.negInf > 0:
		return math.NaN(), true
	case s.posInf > 0:
		return math.Inf(1), true
	case s.negInf > 0:
		return math.Inf(-1), true
	}
	return 0, false
}

// zero returns the zero IEEE addition gives an exactly zero sum: -0 only
// when there were values and every one was -0.
func (s *specials) zero() float64 {
	if s.n > 0 && s.notNegZero == 0 {
		return math.Copysign(0, -1)
	}
	return 0
}

// weight returns n·2^bias.
func weight(n int64, bias uint) *big.Int {
	return new(big.Int).Lsh(big.NewInt(n), bias)
}

// exact returns the integer Σ bins[l][k]·2^k over every lane l, so that
// the value the bins hold is exact(...)·2^−bias. It works on a copy and
// leaves the bins as they are, so reading a result changes nothing.
func exact(bins ...[]int64) *big.Int {
	m := append([]int64(nil), bins[0]...)
	for _, b := range bins[1:] {
		addBins(m, b)
	}
	carry(m)
	// Every bin below the top now holds 0 or 1, so the low part is a bit
	// pattern and only the top bin, which may be negative, is a multiple.
	top := len(m) - 1
	x := new(big.Int).Lsh(big.NewInt(m[top]), uint(top))
	low := new(big.Int)
	for k := range top {
		if m[k] != 0 {
			low.SetBit(low, k, 1)
		}
	}
	return x.Add(x, low)
}
