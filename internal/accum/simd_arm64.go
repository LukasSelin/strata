//go:build goexperiment.simd && arm64

package accum

import "simd/archsimd"

// This file is the NEON half of the vector backend of Sum and Moments;
// blocks.go has the rest and explains the method. It is simd_amd64.go
// four lanes at a time, with arm64's spellings: HiToLo in place of GetHi,
// a signed Shift in place of ShiftLeft (every count here is
// non-negative), and MulWidenLo in place of MulWidenEven.

const vlanes = 4

const vectorBackend = "neon"

// haveVector is always true: NEON is part of the arm64 baseline.
func haveVector() bool { return true }

// clearUpper has nothing to do on arm64.
func clearUpper() {}

// blockBase returns the base exponent field of a block for the given
// window, the OR of its values' bits with -0's cleared, and whether the
// block can be added in registers at all.
func blockBase(blk *[block]float32, window int32) (base int32, nz uint32, ok bool) {
	one := archsimd.BroadcastUint32x4(1)
	sign := archsimd.BroadcastUint32x4(signBits)
	var mx, nzv archsimd.Uint32x4
	mn := archsimd.BroadcastUint32x4(^uint32(0))
	for i := 0; i < block; i += vlanes {
		u := archsimd.LoadFloat32x4Array((*[vlanes]float32)(blk[i:])).ToBits()
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
	field := archsimd.BroadcastUint32x4(0xff)
	mant := archsimd.BroadcastUint32x4(0x7fffff)
	one := archsimd.BroadcastInt32x4(1)
	zero := archsimd.BroadcastInt32x4(0)
	bv := archsimd.BroadcastInt32x4(base)
	half := archsimd.BroadcastUint64x2(pieceMask)
	var acc0, acc1 archsimd.Int64x2
	var lo0, lo1, hi0, hi1 archsimd.Uint64x2
	for i := 0; i < block; i += vlanes {
		u := archsimd.LoadFloat32x4Array((*[vlanes]float32)(blk[i:])).ToBits()
		e := u.ShiftAllRight(23).And(field).BitsToInt32()
		a := u.And(mant).BitsToInt32().Or(e.Min(one).ShiftAllLeft(23)) // |significand|
		sg := u.BitsToInt32().ShiftAllRight(31)                        // 0 or -1
		m := a.Xor(sg).Sub(sg)
		sh := e.Max(one).Sub(bv).Max(zero)
		shLo, shHi := sh.ExtendLo2ToInt64(), sh.HiToLo().ExtendLo2ToInt64()
		acc0 = acc0.Add(m.ExtendLo2ToInt64().Shift(shLo))
		acc1 = acc1.Add(m.HiToLo().ExtendLo2ToInt64().Shift(shHi))
		if squares {
			au := a.ToBits()
			ah := au.HiToLo()
			q0s, q1s := au.MulWidenLo(au), ah.MulWidenLo(ah) // below 2^48
			sh0, sh1 := shLo.Add(shLo), shHi.Add(shHi)
			lo0 = lo0.Add(q0s.And(half).Shift(sh0))
			lo1 = lo1.Add(q1s.And(half).Shift(sh1))
			hi0 = hi0.Add(q0s.ShiftAllRight(24).Shift(sh0))
			hi1 = hi1.Add(q1s.ShiftAllRight(24).Shift(sh1))
		}
	}
	var t [2]int64
	acc0.Add(acc1).StoreArray(&t)
	s = t[0] + t[1]
	if squares {
		var l, h [2]uint64
		lo0.Add(lo1).StoreArray(&l)
		hi0.Add(hi1).StoreArray(&h)
		lo = int64(l[0] + l[1]) // #nosec G115 -- the block's sums are below 2^62
		hi = int64(h[0] + h[1]) // #nosec G115 -- as above
	}
	return s, lo, hi
}
