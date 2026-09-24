//go:build goexperiment.simd && arm64

package kern

import (
	"math"
	"simd/archsimd"
)

// This file is the NEON backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). Every result is bit for bit the scalar
// kernel's, which simd_test.go checks. The loops have the shape of
// simd_amd64.go's, sixteen bytes a step. There is no SSE/AVX transition
// on arm64, so nothing needs clearing before a scalar tail.
//
// Word stays scalar: NEON has no instruction that gathers a compare's
// lanes into a bit mask, as VMOVMSKPS does, and building one from
// shifts and adds was not worth its cost for 64 cells.

func init() {
	// NEON is part of the arm64 baseline: no feature check.
	simdKernels = &kernelSet{
		planesRow: planesRowNEON, sumBytes: sumBytesNEON, uint8Row: uint8RowNEON,
		uint16Row: uint16RowNEON, word64: scalarWord, copyRow: copyRowNEON,
	}
	simdName = "neon"
	simdKernels.install()
}

// shl moves x's bytes k places up, zero filling: EXT with zero below x.
// (x.ConcatShiftBytesRight(y, s) is y's bytes from s on, then x's.)
// Callers pass k as a constant, which EXT needs to be fast.
func shl(x archsimd.Uint8x16, k uint64) archsimd.Uint8x16 {
	var zero archsimd.Uint8x16
	return x.ConcatShiftBytesRight(zero, 16-k)
}

// prefix16 returns the running sum of x's sixteen bytes, wrapping, in
// four shifted adds, and their total in every byte. None of it depends
// on the vector before, so a caller's carry from vector to vector is one
// add.
func prefix16(x archsimd.Uint8x16) (sum, total archsimd.Uint8x16) {
	x = x.Add(shl(x, 1))
	x = x.Add(shl(x, 2))
	x = x.Add(shl(x, 4))
	x = x.Add(shl(x, 8))
	return x, archsimd.BroadcastUint8x16(x.GetElem(15))
}

// planesRowNEON is scalarPlanesRow, sixteen samples at a time, in one
// pass that never writes the summed bytes back. Each plane's running sum
// starts from the total of the planes before it, which a pass of
// wrapping byte adds finds first (total16). Then the four planes are
// summed side by side (prefix16), four independent carry chains of one
// add each, and zipped into samples. Only a ragged tail goes through row,
// scalar.
func planesRowNEON(vals []float32, row []byte) {
	n := len(vals)
	p0, p1, p2, p3 := row[:n:n], row[n:2*n:2*n], row[2*n:3*n:3*n], row[3*n:4*n:4*n]
	t1 := total16(p0)
	t2 := t1 + total16(p1)
	t3 := t2 + total16(p2)
	c0, c1, c2, c3 := archsimd.BroadcastUint8x16(0), archsimd.BroadcastUint8x16(t1),
		archsimd.BroadcastUint8x16(t2), archsimd.BroadcastUint8x16(t3)
	i := 0
	for ; i+16 <= n; i += 16 {
		j := i + 16
		s0, u0 := prefix16(archsimd.LoadUint8x16Array((*[16]uint8)(p0[i:j:j])))
		s1, u1 := prefix16(archsimd.LoadUint8x16Array((*[16]uint8)(p1[i:j:j])))
		s2, u2 := prefix16(archsimd.LoadUint8x16Array((*[16]uint8)(p2[i:j:j])))
		s3, u3 := prefix16(archsimd.LoadUint8x16Array((*[16]uint8)(p3[i:j:j])))
		s0, s1, s2, s3 = s0.Add(c0), s1.Add(c1), s2.Add(c2), s3.Add(c3)
		c0, c1, c2, c3 = c0.Add(u0), c1.Add(u1), c2.Add(u2), c3.Add(u3)
		// Each sample's two low bytes, then its two high, as 16-bit
		// words; then the words zipped into samples.
		lo0, lo1 := s3.InterleaveLo(s2).ReshapeToUint16s(), s3.InterleaveHi(s2).ReshapeToUint16s()
		hi0, hi1 := s1.InterleaveLo(s0).ReshapeToUint16s(), s1.InterleaveHi(s0).ReshapeToUint16s()
		out := vals[i:j:j]
		lo0.InterleaveLo(hi0).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(out[0:4]))
		lo0.InterleaveHi(hi0).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(out[4:8]))
		lo1.InterleaveLo(hi1).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(out[8:12]))
		lo1.InterleaveHi(hi1).ReshapeToUint32s().BitsToFloat32().StoreArray((*[4]float32)(out[12:16]))
	}
	if i == n {
		return
	}
	// The tail: each plane's sum carries on from its carry.
	for k, p := range [4][]byte{p0, p1, p2, p3} {
		acc := [4]archsimd.Uint8x16{c0, c1, c2, c3}[k].GetElem(0)
		for j := i; j < n; j++ {
			acc += p[j]
			p[j] = acc
		}
	}
	for ; i < n; i++ {
		vals[i] = math.Float32frombits(uint32(p3[i]) | uint32(p2[i])<<8 | uint32(p1[i])<<16 | uint32(p0[i])<<24)
	}
}

// total16 returns the sum of b's bytes, wrapping: wrapping byte adds
// sixteen at a time, whose lanes ADDV adds up at the end.
func total16(b []byte) byte {
	var acc archsimd.Uint8x16
	i := 0
	for ; i+16 <= len(b); i += 16 {
		j := i + 16
		acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(b[i:j:j])))
	}
	t := acc.ReduceSum()
	for _, v := range b[i:] {
		t += v
	}
	return t
}

// sumBytesNEON is scalarSumBytes, sixteen bytes at a time (prefix16).
func sumBytesNEON(row []byte) {
	var carry archsimd.Uint8x16
	i := 0
	for ; i+16 <= len(row); i += 16 {
		j := i + 16
		s, t := prefix16(archsimd.LoadUint8x16Array((*[16]uint8)(row[i:j:j])))
		s.Add(carry).StoreArray((*[16]uint8)(row[i:j:j]))
		carry = carry.Add(t)
	}
	acc := carry.GetElem(0)
	for ; i < len(row); i++ {
		acc += row[i]
		row[i] = acc
	}
}

// uint8RowNEON is scalarUint8Row: the bytes summed, if predicted, by
// sumBytesNEON, then widened and converted sixteen at a time.
func uint8RowNEON(vals []float32, row []byte, signed, pred bool) {
	if pred {
		sumBytesNEON(row)
	}
	n := len(vals)
	i := 0
	for ; i+16 <= n; i += 16 {
		j := i + 16
		b := archsimd.LoadUint8x16Array((*[16]uint8)(row[i:j:j]))
		out := vals[i:j:j]
		if signed {
			lo, hi := b.BitsToInt8().ExtendLo8ToInt16(), b.BitsToInt8().HiToLo().ExtendLo8ToInt16()
			lo.ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[0:4]))
			lo.HiToLo().ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[4:8]))
			hi.ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[8:12]))
			hi.HiToLo().ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[12:16]))
		} else {
			lo, hi := b.ExtendLo8ToUint16(), b.HiToLo().ExtendLo8ToUint16()
			lo.ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[0:4]))
			lo.HiToLo().ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[4:8]))
			hi.ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[8:12]))
			hi.HiToLo().ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[12:16]))
		}
	}
	scalarUint8Row(vals[i:], row[i:], signed, false)
}

// uint16RowNEON is scalarUint16Row, eight samples at a time: summed, if
// predicted, in three shifted adds of 16-bit words with the carry from
// vector to vector one add, then widened to float32.
func uint16RowNEON(vals []float32, row []byte, signed, pred bool) {
	n := len(vals)
	var carry archsimd.Uint16x8
	i := 0
	for ; i+8 <= n; i += 8 {
		j := i + 8
		x := archsimd.LoadUint8x16Array((*[16]uint8)(row[2*i : 2*j : 2*j]))
		w := x.ReshapeToUint16s()
		if pred {
			w = prefix8x16(x).Add(carry)
			carry = archsimd.BroadcastUint16x8(w.GetElem(7))
		}
		out := vals[i:j:j]
		if signed {
			s := w.BitsToInt16()
			s.ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[0:4]))
			s.HiToLo().ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(out[4:8]))
		} else {
			w.ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[0:4]))
			w.HiToLo().ExtendLo4ToUint32().ConvertToFloat32().StoreArray((*[4]float32)(out[4:8]))
		}
	}
	acc := carry.GetElem(0)
	for ; i < n; i++ {
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

// prefix8x16 returns the running sum of x's eight little-endian 16-bit
// words, wrapping, in three shifted adds.
func prefix8x16(x archsimd.Uint8x16) archsimd.Uint16x8 {
	w := x.ReshapeToUint16s()
	w = w.Add(shl(w.ReshapeToUint8s(), 2).ReshapeToUint16s())
	w = w.Add(shl(w.ReshapeToUint8s(), 4).ReshapeToUint16s())
	return w.Add(shl(w.ReshapeToUint8s(), 8).ReshapeToUint16s())
}

// copyRowNEON is scalarCopyRow, four samples at a time.
func copyRowNEON(vals []float32, row []byte) {
	n := len(vals)
	i := 0
	for ; i+4 <= n; i += 4 {
		j := i + 4
		archsimd.LoadUint8x16Array((*[16]uint8)(row[4*i : 4*j : 4*j])).ReshapeToUint32s().BitsToFloat32().
			StoreArray((*[4]float32)(vals[i:j:j]))
	}
	scalarCopyRow(vals[i:], row[4*i:])
}
