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
	simdKernels = &kernelSet{planesRow: planesRowAVX2, uint16Row: uint16RowAVX2, word64: wordAVX2, copyRow: copyRowAVX2}
	simdName = "avx2"
	simdKernels.install()
}

// planesRowAVX2 is scalarPlanesRow. The byte sum is sixteen bytes at a
// time (sumBytes), and the planes are put back together sixteen samples
// at a time: each plane's bytes are widened to 16 bits, paired into
// p3|p2<<8 and p1|p0<<8, and those interleaved into 32-bit samples.
func planesRowAVX2(vals []float32, row []byte) {
	i := sumBytes(row)
	if i < len(row) {
		var acc byte
		if i > 0 {
			acc = row[i-1]
		}
		for ; i < len(row); i++ {
			acc += row[i]
			row[i] = acc
		}
	}
	n := len(vals)
	p0, p1, p2, p3 := row[:n], row[n:2*n], row[2*n:3*n], row[3*n:4*n]
	i = planesLanes(vals, p0, p1, p2, p3)
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
		lo.InterleaveLo(hi).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(vals[i:]))
		lo.InterleaveHi(hi).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(vals[i+4:]))
		b0, b1, b2, b3 = upper(b0), upper(b1), upper(b2), upper(b3)
		lo = b3.ExtendLo8ToUint16().Or(b2.ExtendLo8ToUint16().ShiftAllLeft(8))
		hi = b1.ExtendLo8ToUint16().Or(b0.ExtendLo8ToUint16().ShiftAllLeft(8))
		lo.InterleaveLo(hi).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(vals[i+8:]))
		lo.InterleaveHi(hi).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(vals[i+12:]))
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// upper moves the upper eight bytes of v to the lower eight.
func upper(v archsimd.Uint8x16) archsimd.Uint8x16 {
	q := v.ReshapeToUint64s()
	return q.InterleaveHi(q).ReshapeToUint8s()
}

// uint16RowAVX2 is scalarUint16Row, eight samples at a time: summed, if
// predicted, as sumBytes sums bytes, then widened to float32.
func uint16RowAVX2(vals []float32, row []byte, signed, pred bool) {
	i := uint16Lanes(vals, row, signed, pred)
	if i == len(vals) {
		return
	}
	// The tail, carrying on from the last whole vector's sum.
	var acc uint16
	if pred && i > 0 {
		acc = uint16(row[2*i-2]) | uint16(row[2*i-1])<<8
	}
	for ; i < len(vals); i++ {
		v := uint16(row[2*i]) | uint16(row[2*i+1])<<8
		if pred {
			acc += v
			v = acc
		}
		if signed {
			vals[i] = float32(int16(v)) // #nosec G115 -- reinterpreting the bits is the point
		} else {
			vals[i] = float32(v)
		}
	}
}

// Byte-shuffle indices: halfWords takes 16-bit word 3 to the upper four
// words and zeros the lower; lastWord takes word 7 to all eight. halfBytes
// and lastByte are the same for bytes 7 and 15. A negative index zeros.
var (
	halfWords = [16]int8{-1, -1, -1, -1, -1, -1, -1, -1, 6, 7, 6, 7, 6, 7, 6, 7}
	lastWord  = [16]int8{14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15}
	halfBytes = [16]int8{-1, -1, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, 7, 7, 7, 7}
	lastByte  = [16]int8{15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15}
)

// sumBytes replaces row's bytes with their running sum, sixteen at a time,
// as far as whole vectors go, and returns how far that was. Within each
// eight bytes the sum is three shifted adds; the upper eight then add the
// lower's last, and all sixteen the previous vector's last, so the only
// dependency from one vector to the next is one shuffle and one add.
func sumBytes(row []byte) int {
	half := archsimd.LoadInt8x16Array(&halfBytes)
	last := archsimd.LoadInt8x16Array(&lastByte)
	var carry archsimd.Uint8x16
	i := 0
	for ; i+16 <= len(row); i += 16 {
		x := archsimd.LoadUint8x16Array((*[16]uint8)(row[i:]))
		x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(8).ReshapeToUint8s())
		x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(16).ReshapeToUint8s())
		x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(32).ReshapeToUint8s())
		x = x.Add(x.PermuteOrZero(half)).Add(carry)
		carry = x.PermuteOrZero(last)
		x.StoreArray((*[16]uint8)(row[i:]))
	}
	archsimd.ClearAVXUpperBits()
	return i
}

func uint16Lanes(vals []float32, row []byte, signed, pred bool) int {
	half := archsimd.LoadInt8x16Array(&halfWords)
	last := archsimd.LoadInt8x16Array(&lastWord)
	var carry archsimd.Uint16x8
	n := len(vals)
	i := 0
	for ; i+8 <= n; i += 8 {
		x := archsimd.LoadUint8x16Array((*[16]uint8)(row[2*i:])).ReshapeToUint16s()
		if pred {
			x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(16).ReshapeToUint16s())
			x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(32).ReshapeToUint16s())
			x = x.Add(x.ReshapeToUint8s().PermuteOrZero(half).ReshapeToUint16s()).Add(carry)
			carry = x.ReshapeToUint8s().PermuteOrZero(last).ReshapeToUint16s()
			// The tail carries on from the last sample, so keep it.
			x.ReshapeToUint8s().StoreArray((*[16]uint8)(row[2*i:]))
		}
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

// copyRowAVX2 is scalarCopyRow, eight samples at a time.
func copyRowAVX2(vals []float32, row []byte) {
	n := len(vals)
	i := 0
	for ; i+8 <= n; i += 8 {
		archsimd.LoadUint8x32Array((*[32]uint8)(row[4*i:])).ReshapeToUint32s().BitsToFloat32().
			StoreArray((*[8]float32)(vals[i:]))
	}
	archsimd.ClearAVXUpperBits()
	scalarCopyRow(vals[i:], row[4*i:])
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
