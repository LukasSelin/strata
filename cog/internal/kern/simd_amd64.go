//go:build goexperiment.simd && amd64

package kern

import (
	"math"
	"simd/archsimd"
)

// This file is the AVX2 backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). Every result is bit for bit the scalar
// kernel's, which simd_amd64_test.go and simd_test.go check. Lanes are
// loaded through array pointers, and each lane function clears the upper
// AVX bits before the scalar tail.

func init() {
	// AVX2, not just AVX, as in strata's internal/vec.
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{
		planesRow: planesRowAVX2, sumBytes: sumBytesAVX2, uint8Row: uint8RowAVX2,
		uint16Row: uint16RowAVX2, word64: wordAVX2, copyRow: copyRowAVX2,
	}
	simdName = "avx2"
	simdKernels.install()
}

// planesRowAVX2 is scalarPlanesRow in one pass over the planes that never
// writes the summed bytes back. Each 256-bit register holds two streams:
// its lower half walks the first half of the row's samples and its upper
// half the second, sixteen samples each a step, so each half sums on its
// own and nothing crosses between them. Each stream's sum starts from
// the total of every byte before it, which a quick pass of wrapping byte
// adds finds first (streamTotals). Then the four planes are summed side by
// side (prefix16x2), four independent carry chains of one add each, and
// interleaved into samples (samples16x2). Only a ragged tail goes through
// row, scalar.
func planesRowAVX2(vals []float32, row []byte) {
	n := len(vals)
	h := n / 2 &^ 31 // samples per stream
	planes := [4][]byte{row[:n], row[n : 2*n], row[2*n : 3*n], row[3*n : 4*n]}
	// Where each plane's two streams start: after every byte before.
	tot := streamTotals(planes, h)
	var lo, hi [4]byte
	var acc byte
	for k, p := range planes {
		lo[k], hi[k] = acc, acc+tot[2*k]
		acc = hi[k] + tot[2*k+1]
		for _, v := range p[2*h:] {
			acc += v
		}
	}
	if h > 0 {
		planesLanes(vals, planes, h, lo, hi)
	}
	if 2*h == n {
		return
	}
	// The tail: each plane's sum carries on from its second stream.
	for k, p := range planes {
		acc := hi[k] + tot[2*k+1]
		for j := 2 * h; j < n; j++ {
			acc += p[j]
			p[j] = acc
		}
	}
	p0, p1, p2, p3 := planes[0], planes[1], planes[2], planes[3]
	for i := 2 * h; i < n; i++ {
		vals[i] = math.Float32frombits(uint32(p3[i]) | uint32(p2[i])<<8 | uint32(p1[i])<<16 | uint32(p0[i])<<24)
	}
}

// streamTotals returns the sum of the bytes of each plane's two streams,
// wrapping: its first h bytes, then its next h, in one pass of wrapping
// byte adds into eight registers. h is a multiple of 32. The registers'
// bytes are added up together at the end: VPSADBW against zero sums each
// eight, and 64-bit interleaves and adds finish two registers at a time.
func streamTotals(planes [4][]byte, h int) [8]byte {
	if h == 0 {
		return [8]byte{}
	}
	a0, b0 := planes[0][:h:h], planes[0][h:2*h:2*h]
	a1, b1 := planes[1][:h:h], planes[1][h:2*h:2*h]
	a2, b2 := planes[2][:h:h], planes[2][h:2*h:2*h]
	a3, b3 := planes[3][:h:h], planes[3][h:2*h:2*h]
	// In memory order, so the row streams in as one run of lines.
	s0, s1 := sum32(a0), sum32(b0)
	s2, s3 := sum32(a1), sum32(b1)
	s4, s5 := sum32(a2), sum32(b2)
	s6, s7 := sum32(a3), sum32(b3)
	var sums [8]uint64
	sum2(s0, s1).StoreArray((*[2]uint64)(sums[0:]))
	sum2(s2, s3).StoreArray((*[2]uint64)(sums[2:]))
	sum2(s4, s5).StoreArray((*[2]uint64)(sums[4:]))
	sum2(s6, s7).StoreArray((*[2]uint64)(sums[6:]))
	archsimd.ClearAVXUpperBits()
	var t [8]byte
	for k, v := range sums {
		t[k] = byte(v) // #nosec G115 -- the wrapping sum is the low byte
	}
	return t
}

// sum32 returns b's bytes added up, wrapping, lane by lane, 32 at a
// time. len(b) is a multiple of 32.
func sum32(b []byte) archsimd.Uint8x32 {
	var acc archsimd.Uint8x32
	for i := 0; i+32 <= len(b); i += 32 {
		acc = acc.Add(archsimd.LoadUint8x32Array((*[32]uint8)(b[i : i+32 : i+32])))
	}
	return acc
}

// sum2 returns the sums of x's bytes and of y's.
func sum2(x, y archsimd.Uint8x32) archsimd.Uint64x2 {
	var zero archsimd.Uint8x32
	sx, sy := x.SumOf8AbsDiff(zero), y.SumOf8AbsDiff(zero)
	// x's eights 0+1 and y's 0+1 in the lower half, 2+3 in the upper.
	q := sx.InterleaveLoGrouped(sy).Add(sx.InterleaveHiGrouped(sy))
	return q.GetLo().Add(q.GetHi())
}

// Byte-shuffle indices, applied to each 128-bit half on its own:
// halfByte takes byte 7 to the upper eight bytes and zeros the lower,
// lastByte takes byte 15 to all sixteen. A negative index zeros.
var (
	halfByte = [32]int8{
		-1, -1, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, 7, 7, 7, 7,
		-1, -1, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, 7, 7, 7, 7,
	}
	lastByte = [32]int8{
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	}
)

// prefix16x2 returns the running sum, wrapping, of each 128-bit half of
// x's bytes on its own, and each half's total in each of its bytes.
// Within each eight bytes the sum is three shifted adds, and each upper
// eight then add the lower eight's last. None of it depends on the
// vector before, so a caller's carry from vector to vector is one add.
func prefix16x2(x archsimd.Uint8x32, half, last archsimd.Int8x32) (sum, total archsimd.Uint8x32) {
	x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(8).ReshapeToUint8s())
	x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(16).ReshapeToUint8s())
	x = x.Add(x.ReshapeToUint64s().ShiftAllLeft(32).ReshapeToUint8s())
	x = x.Add(x.PermuteOrZeroGrouped(half))
	return x, x.PermuteOrZeroGrouped(last)
}

// prefix32 is prefix16x2 carried across the halves: the running sum of
// x's 32 bytes, and their total in every byte.
func prefix32(x archsimd.Uint8x32, half, last archsimd.Int8x32) (sum, total archsimd.Uint8x32) {
	x, t := prefix16x2(x, half, last)
	// The lower half's total goes to the upper half, and the two
	// totals added are the whole.
	q := t.ReshapeToUint64s()
	var zero archsimd.Uint64x4
	x = x.Add(zero.ConcatPermute128Scalars(0, 2, q).ReshapeToUint8s())
	return x, t.Add(q.ConcatPermute128Scalars(1, 0, q).ReshapeToUint8s())
}

// load2 loads sixteen bytes from each of a and b into the lower and upper
// halves of one register.
func load2(a, b []byte) archsimd.Uint8x32 {
	var x archsimd.Uint8x32
	return x.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(a))).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(b)))
}

// broadcast2 is lo in each byte of the lower half and hi in each of the
// upper.
func broadcast2(lo, hi byte) archsimd.Uint8x32 {
	var x archsimd.Uint8x32
	return x.SetLo(archsimd.BroadcastUint8x16(lo)).SetHi(archsimd.BroadcastUint8x16(hi))
}

func planesLanes(vals []float32, planes [4][]byte, h int, lo, hi [4]byte) {
	half := archsimd.LoadInt8x32Array(&halfByte)
	last := archsimd.LoadInt8x32Array(&lastByte)
	lo8 := archsimd.BroadcastUint16x16(0x00ff)
	c0 := broadcast2(lo[0], hi[0])
	c1 := broadcast2(lo[1], hi[1])
	c2 := broadcast2(lo[2], hi[2])
	c3 := broadcast2(lo[3], hi[3])
	// One index for all ten streams, each a slice of length h, so the
	// loop's condition proves every load and store in bounds.
	va, vb := vals[:h:h], vals[h:2*h:2*h]
	a0, b0 := planes[0][:h:h], planes[0][h:2*h:2*h]
	a1, b1 := planes[1][:h:h], planes[1][h:2*h:2*h]
	a2, b2 := planes[2][:h:h], planes[2][h:2*h:2*h]
	a3, b3 := planes[3][:h:h], planes[3][h:2*h:2*h]
	for i := 0; i+16 <= h; i += 16 {
		j := i + 16
		s0, t0 := prefix16x2(load2(a0[i:j:j], b0[i:j:j]), half, last)
		s1, t1 := prefix16x2(load2(a1[i:j:j], b1[i:j:j]), half, last)
		s2, t2 := prefix16x2(load2(a2[i:j:j], b2[i:j:j]), half, last)
		s3, t3 := prefix16x2(load2(a3[i:j:j], b3[i:j:j]), half, last)
		samples16x2((*[16]float32)(va[i:j:j]), (*[16]float32)(vb[i:j:j]), s0.Add(c0), s1.Add(c1), s2.Add(c2), s3.Add(c3), lo8)
		c0, c1, c2, c3 = c0.Add(t0), c1.Add(t1), c2.Add(t2), c3.Add(t3)
	}
	archsimd.ClearAVXUpperBits()
}

// samples16x2 puts sixteen samples of each stream together from their
// four planes' bytes, most significant plane first, and stores the lower
// half's to va and the upper's to vb. archsimd has no AVX2 byte
// interleave, so bytes are paired by masks and shifts on 16-bit words,
// which splits the even samples from the odd, and 16- and then 32-bit
// interleaves put them back in order.
func samples16x2(va, vb *[16]float32, b0, b1, b2, b3 archsimd.Uint8x32, lo8 archsimd.Uint16x16) {
	w0, w1 := b0.ReshapeToUint16s(), b1.ReshapeToUint16s()
	w2, w3 := b2.ReshapeToUint16s(), b3.ReshapeToUint16s()
	// The low and high 16 bits of the even samples, then of the odd.
	evLo := w3.And(lo8).Or(w2.ShiftAllLeft(8))
	evHi := w1.And(lo8).Or(w0.ShiftAllLeft(8))
	odLo := w3.ShiftAllRight(8).Or(w2.AndNot(lo8))
	odHi := w1.ShiftAllRight(8).Or(w0.AndNot(lo8))
	// In each half, samples 0,2,4,6, then 8,10,12,14, of it.
	ev0 := evLo.InterleaveLoGrouped(evHi).ReshapeToUint32s()
	ev1 := evLo.InterleaveHiGrouped(evHi).ReshapeToUint32s()
	od0 := odLo.InterleaveLoGrouped(odHi).ReshapeToUint32s()
	od1 := odLo.InterleaveHiGrouped(odHi).ReshapeToUint32s()
	// In each half, samples 0-3, 4-7, 8-11 and 12-15 of it.
	a := ev0.InterleaveLoGrouped(od0).ReshapeToUint64s()
	b := ev0.InterleaveHiGrouped(od0).ReshapeToUint64s()
	c := ev1.InterleaveLoGrouped(od1).ReshapeToUint64s()
	d := ev1.InterleaveHiGrouped(od1).ReshapeToUint64s()
	a.ConcatPermute128Scalars(0, 2, b).ReshapeToUint32s().BitsToFloat32().StoreArray((*[8]float32)(va[0:]))
	c.ConcatPermute128Scalars(0, 2, d).ReshapeToUint32s().BitsToFloat32().StoreArray((*[8]float32)(va[8:]))
	a.ConcatPermute128Scalars(1, 3, b).ReshapeToUint32s().BitsToFloat32().StoreArray((*[8]float32)(vb[0:]))
	c.ConcatPermute128Scalars(1, 3, d).ReshapeToUint32s().BitsToFloat32().StoreArray((*[8]float32)(vb[8:]))
}

// sumBytesAVX2 is scalarSumBytes, 32 bytes at a time (prefix32).
func sumBytesAVX2(row []byte) {
	i := sumLanes(row)
	var acc byte
	if i > 0 {
		acc = row[i-1]
	}
	for ; i < len(row); i++ {
		acc += row[i]
		row[i] = acc
	}
}

func sumLanes(row []byte) int {
	half := archsimd.LoadInt8x32Array(&halfByte)
	last := archsimd.LoadInt8x32Array(&lastByte)
	var carry archsimd.Uint8x32
	i := 0
	for ; i+32 <= len(row); i += 32 {
		s, t := prefix32(archsimd.LoadUint8x32Array((*[32]uint8)(row[i:])), half, last)
		s.Add(carry).StoreArray((*[32]uint8)(row[i:]))
		carry = carry.Add(t)
	}
	archsimd.ClearAVXUpperBits()
	return i
}

// uint8RowAVX2 is scalarUint8Row: the bytes summed, if predicted, by
// sumBytesAVX2, then widened and converted sixteen at a time.
func uint8RowAVX2(vals []float32, row []byte, signed, pred bool) {
	if pred {
		sumBytesAVX2(row)
	}
	i := uint8Lanes(vals, row, signed)
	scalarUint8Row(vals[i:], row[i:], signed, false)
}

func uint8Lanes(vals []float32, row []byte, signed bool) int {
	n := len(vals)
	i := 0
	for ; i+16 <= n; i += 16 {
		lo := archsimd.LoadUint8x16Array((*[16]uint8)(row[i:]))
		hi := upper(lo)
		if signed {
			lo.BitsToInt8().ExtendLo8ToInt32().ConvertToFloat32().StoreArray((*[8]float32)(vals[i:]))
			hi.BitsToInt8().ExtendLo8ToInt32().ConvertToFloat32().StoreArray((*[8]float32)(vals[i+8:]))
		} else {
			lo.ExtendLo8ToUint32().BitsToInt32().ConvertToFloat32().StoreArray((*[8]float32)(vals[i:]))
			hi.ExtendLo8ToUint32().BitsToInt32().ConvertToFloat32().StoreArray((*[8]float32)(vals[i+8:]))
		}
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
// predicted, as prefix32 sums bytes, then widened to float32.
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
// words and zeros the lower; lastWord takes word 7 to all eight. A
// negative index zeros.
var (
	halfWords = [16]int8{-1, -1, -1, -1, -1, -1, -1, -1, 6, 7, 6, 7, 6, 7, 6, 7}
	lastWord  = [16]int8{14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15}
)

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
