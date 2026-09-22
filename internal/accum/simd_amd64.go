//go:build goexperiment.simd && amd64

package accum

import "simd/archsimd"

// This file is the AVX2 half of the vector backend of Sum and Moments;
// blocks.go has the rest and explains the method.

const vlanes = 8

const vectorBackend = "avx2"

func haveVector() bool { return archsimd.X86.AVX2() }

// clearUpper runs after the block loop, before the scalar tail.
func clearUpper() { archsimd.ClearAVXUpperBits() }

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
