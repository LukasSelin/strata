package accum

import "math"

// Moments is an exact running sum of float32 values and of their squares,
// which is what a mean and a variance need. The zero value is empty, and
// the identity of Combine.
type Moments struct {
	sum Sum
	sq  [lanes][sqBins]int64
}

// Count returns the number of values added, NaN and infinities included.
func (m *Moments) Count() int64 { return m.sum.n }

// Add adds every value of xs.
func (m *Moments) Add(xs []float32) {
	for len(xs) > 0 {
		k := min(len(xs), normaliseEvery)
		if m.sum.admit(k) {
			m.normalise()
		}
		m.sum.notNegZero |= momentsKernel(&m.sum.bins, &m.sq, &m.sum.specials, xs[:k])
		xs = xs[k:]
	}
}

// Add1 adds one value. Add is the faster way to add a run of them.
func (m *Moments) Add1(v float32) {
	if m.sum.admit(1) {
		m.normalise()
	}
	m.sum.notNegZero |= addMoment1(&m.sum.bins[0], &m.sq[0], &m.sum.specials, v)
}

// Combine adds everything added to b into m. b is not changed.
func (m *Moments) Combine(b *Moments) {
	m.sum.Combine(&b.sum)
	for l := range lanes {
		addBins(m.sq[0][:], b.sq[l][:])
	}
	m.normalise()
}

func (m *Moments) normalise() {
	m.sum.normalise()
	for l := 1; l < lanes; l++ {
		addBins(m.sq[0][:], m.sq[l][:])
		clear(m.sq[l][:])
	}
	carry(m.sq[0][:])
}

// momentsKernel is the function Moments.Add runs: addMoments, or a vector
// backend's kernel, as sumKernel.
var momentsKernel = addMoments

// addMoments is Moments' hot loop, addSums with the squares beside the
// values.
func addMoments(bins *[lanes][sumBins]int64, sq *[lanes][sqBins]int64, sp *specials, xs []float32) uint32 {
	var nz uint32
	b0, b1, b2, b3 := &bins[0], &bins[1], &bins[2], &bins[3]
	q0, q1, q2, q3 := &sq[0], &sq[1], &sq[2], &sq[3]
	for ; len(xs) >= 4; xs = xs[4:] {
		u0 := math.Float32bits(xs[0])
		u1 := math.Float32bits(xs[1])
		u2 := math.Float32bits(xs[2])
		u3 := math.Float32bits(xs[3])
		nz |= (u0 ^ signBits) | (u1 ^ signBits) | (u2 ^ signBits) | (u3 ^ signBits)
		if special(u0, u1, u2, u3) {
			addMoment1(b0, q0, sp, xs[0])
			addMoment1(b1, q1, sp, xs[1])
			addMoment1(b2, q2, sp, xs[2])
			addMoment1(b3, q3, sp, xs[3])
			continue
		}
		addFinite(b0, q0, u0)
		addFinite(b1, q1, u1)
		addFinite(b2, q2, u2)
		addFinite(b3, q3, u3)
	}
	for _, v := range xs {
		nz |= addMoment1(b0, q0, sp, v)
	}
	return nz
}

// addMoment1 adds one value and its square to one bin set and returns its
// bits with -0's cleared.
func addMoment1(b *[sumBins]int64, q *[sqBins]int64, sp *specials, v float32) uint32 {
	u := math.Float32bits(v)
	if u&0x7f800000 == 0x7f800000 {
		sp.addSpecial(u)
		return u ^ signBits
	}
	addFinite(b, q, u)
	return u ^ signBits
}

// addFinite adds a finite value's bit pattern and its square. The square
// of the 24-bit significand is exact in 48 bits; its low and high 24 bits
// go to the bins of their weights, 2e and 2e+24, so that neither addend
// exceeds a sum bin's.
func addFinite(b *[sumBins]int64, q *[sqBins]int64, u uint32) {
	e, m := split(u)
	b[e] += m
	a := uint64(u&0x7fffff) | uint64(min((u>>23)&0xff, 1))<<23 // |significand|
	s := a * a
	j := 2 * int(e)
	q[j] += int64(s & (1<<24 - 1))
	q[j+24] += int64(s >> 24)
}
