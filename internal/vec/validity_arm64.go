//go:build goexperiment.simd && arm64

package vec

import "simd/archsimd"

// validBitsNEON is scalarValidBits a word at a time. NEON has no
// VMOVMSKPS, so each compare's all-ones lanes are ANDed with their bit's
// weight, sixteen cells' weights are ORed into one vector, and two
// pairwise adds sum its lanes into the sixteen bits (the weights are
// disjoint, so the sum is their OR). FCMEQ is false for a NaN, so a NaN
// fill compares each cell with itself, and any other fill takes the
// complement of the cells equal to it.
func validBitsNEON(dst []uint64, src []float32, fill float32) {
	f := archsimd.BroadcastFloat32x4(fill)
	b := laneWeights()
	nan := fill != fill
	d, s := dst, src
	for len(s) >= 64 && len(d) > 0 {
		c := (*[64]float32)(s)
		var w uint64
		if nan {
			w = notNaN16((*[16]float32)(c[0:16]), &b) | notNaN16((*[16]float32)(c[16:32]), &b)<<16 |
				notNaN16((*[16]float32)(c[32:48]), &b)<<32 | notNaN16((*[16]float32)(c[48:64]), &b)<<48
		} else {
			w = ^(equal16((*[16]float32)(c[0:16]), f, &b) | equal16((*[16]float32)(c[16:32]), f, &b)<<16 |
				equal16((*[16]float32)(c[32:48]), f, &b)<<32 | equal16((*[16]float32)(c[48:64]), f, &b)<<48)
		}
		d[0] = w
		d, s = d[1:], s[64:]
	}
	scalarValidBits(d, s, fill)
}

// laneBits are the weights of the first four cells' bits.
var laneBits = [4]uint32{1, 2, 4, 8}

// laneWeights returns the weights of sixteen cells' bits, four lanes to a
// vector.
func laneWeights() [4]archsimd.Uint32x4 {
	w := archsimd.LoadUint32x4Array(&laneBits)
	return [4]archsimd.Uint32x4{w, w.ShiftAllLeft(4), w.ShiftAllLeft(8), w.ShiftAllLeft(12)}
}

// gather16 returns the sixteen bits of four compares' lanes.
func gather16(m0, m1, m2, m3 archsimd.Mask32x4, b *[4]archsimd.Uint32x4) uint64 {
	v := m0.ToInt32x4().ToBits().And(b[0]).
		Or(m1.ToInt32x4().ToBits().And(b[1])).
		Or(m2.ToInt32x4().ToBits().And(b[2])).
		Or(m3.ToInt32x4().ToBits().And(b[3]))
	v = v.ConcatAddPairs(v)
	v = v.ConcatAddPairs(v)
	return uint64(v.GetElem(0))
}

// equal16 returns the bits of the cells of c equal to f's lanes.
func equal16(c *[16]float32, f archsimd.Float32x4, b *[4]archsimd.Uint32x4) uint64 {
	return gather16(load4(c[0:4]).Equal(f), load4(c[4:8]).Equal(f),
		load4(c[8:12]).Equal(f), load4(c[12:16]).Equal(f), b)
}

// notNaN16 returns the bits of the cells of c that are not NaN.
func notNaN16(c *[16]float32, b *[4]archsimd.Uint32x4) uint64 {
	x0, x1, x2, x3 := load4(c[0:4]), load4(c[4:8]), load4(c[8:12]), load4(c[12:16])
	return gather16(x0.Equal(x0), x1.Equal(x1), x2.Equal(x2), x3.Equal(x3), b)
}
