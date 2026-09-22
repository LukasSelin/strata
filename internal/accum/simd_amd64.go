//go:build goexperiment.simd && amd64

package accum

import "simd/archsimd"

// This file is the AVX2 backend of Sum and Moments. It adds 64-cell
// blocks in registers instead of in bins.
//
// Within a block, every value is shifted onto the block's base exponent
// field and added to a 64-bit lane. A significand is below 2^24, so with
// the base at most sumWindow fields below the block's largest a shifted
// one is below 2^56, and 64 of them stay below 2^62. The block's sum is
// then added to three bins in 24-bit pieces: the same integer the scalar
// loop would have added cell by cell. So the bins, and every result, are
// bit for bit the scalar backend's by construction rather than by
// matching an evaluation order.
//
// Squares go the same way. A squared significand is below 2^48 and is
// split into two 24-bit halves as in addFinite; each is shifted by twice
// the value's distance from the base, so the base may only be
// squareWindow fields below the largest.
//
// A block holding a NaN or an infinity, or a non-zero value further below
// its largest than the window, goes to the scalar loop whole. So does the
// tail after the last whole block. Zeros shift harmlessly wherever they
// are.

const (
	block        = 64
	sumWindow    = 32
	squareWindow = 16
	vlanes       = 8
	pieceMask    = 1<<24 - 1
)

func init() { useScalar(false) }

// useScalar switches Sum and Moments to the scalar loops, or back, for
// tests that compare the two in one binary.
func useScalar(scalar bool) {
	if scalar || !archsimd.X86.AVX2() {
		sumKernel, momentsKernel = addSums, addMoments
	} else {
		sumKernel, momentsKernel = addSumsAVX2, addMomentsAVX2
	}
}

func addSumsAVX2(bins *[lanes][sumBins]int64, sp *specials, xs []float32) uint32 {
	var nzAll uint32
	for ; len(xs) >= block; xs = xs[block:] {
		blk := (*[block]float32)(xs)
		base, nz, ok := blockBase(blk, sumWindow)
		nzAll |= nz
		if !ok {
			addSums(bins, sp, blk[:])
			continue
		}
		s, _, _ := blockSums(blk, base, false)
		addSumPieces(&bins[0], uint8(base), s) // #nosec G115 -- blockBase's base is in [1, 222]
	}
	archsimd.ClearAVXUpperBits()
	return nzAll | addSums(bins, sp, xs)
}

func addMomentsAVX2(bins *[lanes][sumBins]int64, sq *[lanes][sqBins]int64, sp *specials, xs []float32) uint32 {
	var nzAll uint32
	for ; len(xs) >= block; xs = xs[block:] {
		blk := (*[block]float32)(xs)
		base, nz, ok := blockBase(blk, squareWindow)
		nzAll |= nz
		if !ok {
			addMoments(bins, sq, sp, blk[:])
			continue
		}
		s, lo, hi := blockSums(blk, base, true)
		addSumPieces(&bins[0], uint8(base), s)      // #nosec G115 -- blockBase's base is in [1, 238]
		addSquarePieces(&sq[0], uint8(base), lo, hi) // #nosec G115 -- as above
	}
	archsimd.ClearAVXUpperBits()
	return nzAll | addMoments(bins, sq, sp, xs)
}

// addSumPieces adds a, a multiple of the weight of sum bin base with
// |a| < 2^62, to bins base, base+24 and base+48 in 24-bit pieces, so that
// no bin takes more than a cell would add. base is a uint8 so that every
// index is provably in range.
func addSumPieces(b *[sumBins]int64, base uint8, a int64) {
	p0, p1, p2 := pieces(a)
	k := int(base)
	b[k] += p0
	b[k+24] += p1
	b[k+48] += p2
}

// addSquarePieces adds lo and hi, the block's sums of square halves, as
// multiples of the weights of square bins 2·base and 2·base+24.
func addSquarePieces(q *[sqBins]int64, base uint8, lo, hi int64) {
	l0, l1, l2 := pieces(lo)
	h0, h1, h2 := pieces(hi)
	j := 2 * int(base)
	q[j] += l0
	q[j+24] += l1 + h0
	q[j+48] += l2 + h1
	q[j+72] += h2
}

// pieces splits a, |a| < 2^62, into three signed 24-bit pieces with
// a = p0 + p1·2^24 + p2·2^48.
func pieces(a int64) (p0, p1, p2 int64) {
	neg := a >> 63
	m := (a ^ neg) - neg
	return (m&pieceMask ^ neg) - neg, ((m>>24)&pieceMask ^ neg) - neg, (m>>48 ^ neg) - neg
}

// blockBase returns the base exponent field of a block for the given
// window, the OR of its values' bits with -0's cleared, and whether the
// block can be added in registers at all.
func blockBase(blk *[block]float32, window int32) (base int32, nz uint32, ok bool) {
	one := archsimd.BroadcastUint32x8(1)
	sign := archsimd.BroadcastUint32x8(signBits)
	var mx, nzv archsimd.Uint32x8
	mn := archsimd.BroadcastUint32x8(^uint32(0))
	for i := 0; i < block; i += vlanes {
		u := archsimd.LoadFloat32x8Array((*[vlanes]float32)(blk[i:])).ToBits()
		a := u.ShiftAllLeft(1) // magnitude bits, exponent field on top
		mx = mx.Max(a)
		mn = mn.Min(a.Sub(one)) // a zero wraps to the largest value
		nzv = nzv.Or(u.Xor(sign))
	}
	var mxs, mns, nzs [vlanes]uint32
	mx.StoreArray(&mxs)
	mn.StoreArray(&mns)
	nzv.StoreArray(&nzs)
	var hmx, hmn uint32 = 0, ^uint32(0)
	for k := range vlanes {
		hmx, hmn = max(hmx, mxs[k]), min(hmn, mns[k])
		nz |= nzs[k]
	}
	if hmx >= 0x7f800000<<1 {
		return 0, nz, false // a NaN or an infinity
	}
	emax := int32(hmx >> 24)
	// The smallest non-zero magnitude's field, with subnormals weighing
	// what field 1 does. An all-zero block wraps to field 0 here.
	emin := max(int32((hmn+1)>>24), 1)
	base = max(emax, window+1) - window
	return base, nz, emin >= base
}

// blockSums returns the block's sum as a multiple of the weight of sum
// bin base and, with squares, the sums of the squares' low and high
// halves as multiples of the weight of square bin 2·base.
func blockSums(blk *[block]float32, base int32, squares bool) (s, lo, hi int64) {
	field := archsimd.BroadcastUint32x8(0xff)
	mant := archsimd.BroadcastUint32x8(0x7fffff)
	one := archsimd.BroadcastInt32x8(1)
	zero := archsimd.BroadcastInt32x8(0)
	bv := archsimd.BroadcastInt32x8(base)
	half := archsimd.BroadcastUint64x4(pieceMask)
	var acc0, acc1 archsimd.Int64x4
	var lo0, lo1, hi0, hi1 archsimd.Uint64x4
	for i := 0; i < block; i += vlanes {
		u := archsimd.LoadFloat32x8Array((*[vlanes]float32)(blk[i:])).ToBits()
		e := u.ShiftAllRight(23).And(field).BitsToInt32()
		a := u.And(mant).BitsToInt32().Or(e.Min(one).ShiftAllLeft(23)) // |significand|
		sg := u.BitsToInt32().ShiftAllRight(31)                        // 0 or -1
		m := a.Xor(sg).Sub(sg)
		sh := e.Max(one).Sub(bv).Max(zero).ToBits()
		shLo, shHi := sh.GetLo().ExtendToUint64(), sh.GetHi().ExtendToUint64()
		acc0 = acc0.Add(m.GetLo().ExtendToInt64().ShiftLeft(shLo))
		acc1 = acc1.Add(m.GetHi().ExtendToInt64().ShiftLeft(shHi))
		if squares {
			au := a.ToBits()
			q0 := au.GetLo().ExtendToUint64().ReshapeToUint32s()
			q1 := au.GetHi().ExtendToUint64().ReshapeToUint32s()
			q0s, q1s := q0.MulWidenEven(q0), q1.MulWidenEven(q1) // below 2^48
			sh0, sh1 := shLo.Add(shLo), shHi.Add(shHi)
			lo0 = lo0.Add(q0s.And(half).ShiftLeft(sh0))
			lo1 = lo1.Add(q1s.And(half).ShiftLeft(sh1))
			hi0 = hi0.Add(q0s.ShiftAllRight(24).ShiftLeft(sh0))
			hi1 = hi1.Add(q1s.ShiftAllRight(24).ShiftLeft(sh1))
		}
	}
	var t [4]int64
	acc0.Add(acc1).StoreArray(&t)
	s = t[0] + t[1] + t[2] + t[3]
	if squares {
		var l, h [4]uint64
		lo0.Add(lo1).StoreArray(&l)
		hi0.Add(hi1).StoreArray(&h)
		lo = int64(l[0] + l[1] + l[2] + l[3]) // #nosec G115 -- the block's sums are below 2^62
		hi = int64(h[0] + h[1] + h[2] + h[3]) // #nosec G115 -- as above
	}
	return s, lo, hi
}
