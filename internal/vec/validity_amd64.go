//go:build goexperiment.simd && amd64

package vec

import "simd/archsimd"

// validBitsAVX2 is scalarValidBits a word at a time: eight compares of
// eight cells, each gathered into a byte of the word with VMOVMSKPS.
// Equal is VCMPPS's ordered equality, false for a NaN, so a NaN fill
// compares each cell with itself, and any other fill takes the
// complement of the cells equal to it.
//
// The two tests are separate loops, and nothing in either is a legacy
// SSE instruction: a test of fill inside the loop compiled to UCOMISS,
// which with the upper YMM bits dirty ran the NaN loop at the scalar
// kernel's speed (docs/adr/0001-simd-backend.md).
func validBitsAVX2(dst []uint64, src []float32, fill float32) {
	var w int
	if fill != fill {
		w = notNaNWords(dst, src)
	} else {
		w = unequalWords(dst, src, archsimd.BroadcastFloat32x8(fill))
	}
	archsimd.ClearAVXUpperBits()
	scalarValidBits(dst[w:], src[64*w:], fill)
}

// unequalWords writes the words of the whole words of src, the bits of
// the cells not equal to f's lanes, and returns how many it wrote.
func unequalWords(dst []uint64, src []float32, f archsimd.Float32x8) int {
	d, s := dst, src
	for len(s) >= 64 && len(d) > 0 {
		c := (*[64]float32)(s)
		e0 := load8(c[0:8]).Equal(f).ToBits()
		e1 := load8(c[8:16]).Equal(f).ToBits()
		e2 := load8(c[16:24]).Equal(f).ToBits()
		e3 := load8(c[24:32]).Equal(f).ToBits()
		e4 := load8(c[32:40]).Equal(f).ToBits()
		e5 := load8(c[40:48]).Equal(f).ToBits()
		e6 := load8(c[48:56]).Equal(f).ToBits()
		e7 := load8(c[56:64]).Equal(f).ToBits()
		d[0] = ^(uint64(e0) | uint64(e1)<<8 | uint64(e2)<<16 | uint64(e3)<<24 |
			uint64(e4)<<32 | uint64(e5)<<40 | uint64(e6)<<48 | uint64(e7)<<56)
		d, s = d[1:], s[64:]
	}
	return len(dst) - len(d)
}

// notNaNWords is unequalWords for a NaN fill: the bits of the cells that
// are not NaN.
func notNaNWords(dst []uint64, src []float32) int {
	d, s := dst, src
	for len(s) >= 64 && len(d) > 0 {
		c := (*[64]float32)(s)
		x0, x1, x2, x3 := load8(c[0:8]), load8(c[8:16]), load8(c[16:24]), load8(c[24:32])
		x4, x5, x6, x7 := load8(c[32:40]), load8(c[40:48]), load8(c[48:56]), load8(c[56:64])
		d[0] = uint64(x0.Equal(x0).ToBits()) | uint64(x1.Equal(x1).ToBits())<<8 |
			uint64(x2.Equal(x2).ToBits())<<16 | uint64(x3.Equal(x3).ToBits())<<24 |
			uint64(x4.Equal(x4).ToBits())<<32 | uint64(x5.Equal(x5).ToBits())<<40 |
			uint64(x6.Equal(x6).ToBits())<<48 | uint64(x7.Equal(x7).ToBits())<<56
		d, s = d[1:], s[64:]
	}
	return len(dst) - len(d)
}
