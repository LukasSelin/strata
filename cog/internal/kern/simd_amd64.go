//go:build goexperiment.simd && amd64

package kern

import (
	"math"
	"simd/archsimd"
)

// This file is the AVX2 backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). Every result is bit for bit the scalar
// kernel's, which simd_amd64_test.go checks. Lanes are loaded through
// array pointers, and each lane function clears the upper AVX bits
// before the scalar tail.

func init() {
	// AVX2, not just AVX, as in strata's internal/vec.
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{planesRow: planesRowAVX2, uint16Row: uint16RowAVX2, word64: wordAVX2}
	simdName = "avx2"
	simdKernels.install()
}

// planesRowAVX2 is scalarPlanesRow: the byte sum stays scalar, as it
// is serial, and the planes are put back together sixteen samples at a
// time. Each plane's bytes are widened to 16 bits, paired into
// p3|p2<<8 and p1|p0<<8, and those interleaved into 32-bit samples.
func planesRowAVX2(vals []float32, row []byte) {
	var acc byte
	for i, b := range row {
		acc += b
		row[i] = acc
	}
	n := len(vals)
	p0, p1, p2, p3 := row[:n], row[n:2*n], row[2*n:3*n], row[3*n:4*n]
	i := planesLanes(vals, p0, p1, p2, p3)
	for ; i < n; i++ {
		vals[i] = math.Float32frombits(uint32(p3[i]) | uint32(p2[i])<<8 | uint32(p1[i])<<16 | uint32(p0[i])<<24)
	}
}

func planesLanes(vals []float32, p0, p1, p2, p3 []byte) int {
	n := len(vals)
	i := 0
	for ; i+16 <= n; i += 16 {
		b0 := archsimd.LoadUint8x16Array((*[16]uint8)(p0[i:]))
		b1 := archsimd.LoadUint8x16Array((*[16]uint8)(p1[i:]))
		b2 := archsimd.LoadUint8x16Array((*[16]uint8)(p2[i:]))
		b3 := archsimd.LoadUint8x16Array((*[16]uint8)(p3[i:]))
		// Samples 0-7, then 8-15.
		lo := b3.ExtendLo8ToUint16().Or(b2.ExtendLo8ToUint16().ShiftAllLeft(8))
		hi := b1.ExtendLo8ToUint16().Or(b0.ExtendLo8ToUint16().ShiftAllLeft(8))
		store4(lo.InterleaveLo(hi), vals[i:])
		store4(lo.InterleaveHi(hi), vals[i+4:])
		b0, b1, b2, b3 = upper(b0), upper(b1), upper(b2), upper(b3)
		lo = b3.ExtendLo8ToUint16().Or(b2.ExtendLo8ToUint16().ShiftAllLeft(8))
		hi = b1.ExtendLo8ToUint16().Or(b0.ExtendLo8ToUint16().ShiftAllLeft(8))
		store4(lo.InterleaveLo(hi), vals[i+8:])
		store4(lo.InterleaveHi(hi), vals[i+12:])
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// upper moves the upper eight bytes of v to the lower eight.
func upper(v archsimd.Uint8x16) archsimd.Uint8x16 {
	q := v.ReshapeToUint64s()
	return q.InterleaveHi(q).ReshapeToUint8s()
}

// store4 stores four 32-bit samples, built as eight 16-bit halves, low
// half first, as float32 bits.
func store4(v archsimd.Uint16x8, dst []float32) {
	v.ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(dst))
}

// uint16RowAVX2 is scalarUint16Row: the horizontal sum stays scalar, in
// place, and the widening to float32 is eight at a time.
func uint16RowAVX2(vals []float32, row []byte, signed, pred bool) {
	if pred {
		r := row[:2*len(vals)]
		var acc uint16
		for i := 0; i < len(r); i += 2 {
			acc += uint16(r[i]) | uint16(r[i+1])<<8
			r[i], r[i+1] = byte(acc), byte(acc>>8) // #nosec G115 -- the low and high bytes
		}
	}
	i := uint16Lanes(vals, row, signed)
	scalarUint16Row(vals[i:], row[2*i:], signed, false)
}

func uint16Lanes(vals []float32, row []byte, signed bool) int {
	n := len(vals)
	i := 0
	for ; i+8 <= n; i += 8 {
		x := archsimd.LoadUint8x16Array((*[16]uint8)(row[2*i:])).ReshapeToUint16s()
		var f archsimd.Float32x8
		if signed {
			f = x.BitsToInt16().ExtendToInt32().ConvertToFloat32()
		} else {
			f = x.ExtendToUint32().BitsToInt32().ConvertToFloat32()
		}
		f.StoreArray((*[8]float32)(vals[i:]))
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// wordAVX2 is scalarWord on 64 cells, eight at a time: each test is a
// compare on the bits, and the lanes' results are gathered into the word
// with VMOVMSKPS.
func wordAVX2(chunk []float32, mode int, want, care, lo uint32, span uint64) uint64 {
	c := (*[64]float32)(chunk)
	var word uint64
	switch mode {
	case ModeNaN:
		// Valid unless the magnitude's bits exceed +Inf's.
		abs := archsimd.BroadcastUint32x8(0x7fffffff)
		inf := archsimd.BroadcastInt32x8(0x7f800000)
		for k := 0; k < 8; k++ {
			x := archsimd.LoadFloat32x8Array((*[8]float32)(c[8*k:])).ToBits()
			nan := x.And(abs).BitsToInt32().Greater(inf)
			word |= uint64(^nan.ToBits()) << (8 * k)
		}
	case ModeRange:
		// Valid outside the run: bits-lo > span, unsigned, compared
		// signed with both sides' sign bits flipped.
		vlo := archsimd.BroadcastUint32x8(lo)
		sign := archsimd.BroadcastUint32x8(0x80000000)
		vspan := archsimd.BroadcastInt32x8(int32(uint32(span) ^ 0x80000000)) // #nosec G115 -- span is a few ulps
		for k := 0; k < 8; k++ {
			x := archsimd.LoadFloat32x8Array((*[8]float32)(c[8*k:])).ToBits()
			out := x.Sub(vlo).Xor(sign).BitsToInt32().Greater(vspan)
			word |= uint64(out.ToBits()) << (8 * k)
		}
	default:
		vwant := archsimd.BroadcastUint32x8(want)
		vcare := archsimd.BroadcastUint32x8(care)
		for k := 0; k < 8; k++ {
			x := archsimd.LoadFloat32x8Array((*[8]float32)(c[8*k:])).ToBits()
			nd := x.And(vcare).Equal(vwant)
			word |= uint64(^nd.ToBits()) << (8 * k)
		}
	}
	archsimd.ClearAVXUpperBits()
	return word
}
