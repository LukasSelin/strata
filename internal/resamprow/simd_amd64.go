//go:build goexperiment.simd && amd64

package resamprow

import "simd/archsimd"

// This file is the AVX2 backend (docs/adr/0001-simd-backend.md), eight
// lanes at a time, and simd_arm64.go's twin: the vertical pass is
// lane-parallel across output columns, and the horizontal pass
// transposes eight source rows into a column-major scratch so that every
// tap is one contiguous vector of eight rows, with no gathers (AVX2's
// VPGATHERDD is slow on Zen 2, and archsimd has none), then transposes
// each block of eight output columns back. Each lane performs the scalar
// kernels' operations in their order: VMULPS then VADDPS, never
// VFMADD. Every kernel calls ClearAVXUpperBits before its scalar tail
// (golang/go#80835), and the lane loops keep to VEX instructions.

const lane = 8

func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{hRows: hRowsAVX2, vRow: vRowAVX2}
	simdName = "avx2"
	UseScalar(false)
}

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// transpose8 transposes the 8×8 block whose rows are r0…r7: 32-bit then
// 64-bit interleaves within each 128-bit half, then a swap of halves. It
// takes and returns vectors by value, never an array of them, so no
// vector is ever zeroed or copied through memory with legacy SSE (ADR
// 0001).
func transpose8(r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Float32x8) (c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Float32x8) {
	// t0 = r0[0] r1[0] r0[1] r1[1] | r0[4] r1[4] r0[5] r1[5]; t1 the same for
	// elements 2, 3, 6, 7; t2…t7 likewise for rows 2…7.
	t0, t1 := interleave32(r0, r1)
	t2, t3 := interleave32(r2, r3)
	t4, t5 := interleave32(r4, r5)
	t6, t7 := interleave32(r6, r7)
	// q0 = column 0 | column 4 of rows 0…3, q1 = 1 | 5, q2 = 2 | 6, q3 =
	// 3 | 7; q4…q7 the same for rows 4…7.
	q0, q1 := interleave64(t0, t2)
	q2, q3 := interleave64(t1, t3)
	q4, q5 := interleave64(t4, t6)
	q6, q7 := interleave64(t5, t7)
	return q0.ConcatPermute128Scalars(0, 2, q4), q1.ConcatPermute128Scalars(0, 2, q5),
		q2.ConcatPermute128Scalars(0, 2, q6), q3.ConcatPermute128Scalars(0, 2, q7),
		q0.ConcatPermute128Scalars(1, 3, q4), q1.ConcatPermute128Scalars(1, 3, q5),
		q2.ConcatPermute128Scalars(1, 3, q6), q3.ConcatPermute128Scalars(1, 3, q7)
}

func interleave32(a, b archsimd.Float32x8) (lo, hi archsimd.Uint32x8) {
	x, y := a.ToBits(), b.ToBits()
	return x.InterleaveLoGrouped(y), x.InterleaveHiGrouped(y)
}

func interleave64(a, b archsimd.Uint32x8) (lo, hi archsimd.Float32x8) {
	x, y := a.ReshapeToUint64s(), b.ReshapeToUint64s()
	return x.InterleaveLoGrouped(y).ReshapeToUint32s().BitsToFloat32(),
		x.InterleaveHiGrouped(y).ReshapeToUint32s().BitsToFloat32()
}

func vRowAVX2(dst, t []float32, tStride int, w []float32) {
	i := vRowLanes(dst, t, tStride, w)
	archsimd.ClearAVXUpperBits()
	if i < len(dst) {
		scalarVRow(dst[i:], t[i:], tStride, w)
	}
}

// vRowLanes runs the vertical pass eight output columns at a time and
// returns how many it wrote. It accumulates in dst one tap row at a
// time, so every loop runs over shrinking slices with no bounds checks;
// each lane still adds its taps in order, as the scalar pass does.
func vRowLanes(dst, t []float32, tStride int, w []float32) int {
	n := len(dst) &^ (lane - 1)
	d, r := dst[:n], t[:n]
	w0 := archsimd.BroadcastFloat32x8(w[0])
	for len(d) >= lane && len(r) >= lane {
		store8(w0.Mul(load8(r)), d)
		d, r = d[lane:], r[lane:]
	}
	for k := 1; k < len(w); k++ {
		wk := archsimd.BroadcastFloat32x8(w[k])
		d, r = dst[:n], t[k*tStride:k*tStride+n]
		for len(d) >= lane && len(r) >= lane {
			store8(load8(d).Add(wk.Mul(load8(r))), d)
			d, r = d[lane:], r[lane:]
		}
	}
	return n
}

func hRowsAVX2(t []float32, tStride int, src []float32, sStride, rows int, a *Axis, c0, c1, sx0 int, scratch []float32) {
	lo, hi := span(a, c0, c1)
	lo -= sx0
	hi -= sx0
	cols := scratch[:(hi-lo)*lane]
	j := 0
	for ; j+lane <= rows; j += lane {
		transposeRows(cols, src[j*sStride:], sStride, lo, hi)
		hBlock(t[j*tStride:], tStride, cols, a, c0, c1, sx0+lo)
	}
	archsimd.ClearAVXUpperBits()
	if j < rows {
		scalarHRows(t[j*tStride:], tStride, src[j*sStride:], sStride, rows-j, a, c0, c1, sx0, nil)
	}
}

// transposeRows writes source columns [lo, hi) of the eight rows at src
// into cols, column-major: cols[(i-lo)*8 + l] is row l's cell i. A last
// block of fewer than eight columns loads with masked loads, so the loop
// stays in VEX instructions to the end.
func transposeRows(cols, src []float32, sStride, lo, hi int) {
	n := hi - lo
	s0 := src[lo:hi]
	s1 := src[sStride+lo : sStride+hi][:n]
	s2 := src[2*sStride+lo : 2*sStride+hi][:n]
	s3 := src[3*sStride+lo : 3*sStride+hi][:n]
	s4 := src[4*sStride+lo : 4*sStride+hi][:n]
	s5 := src[5*sStride+lo : 5*sStride+hi][:n]
	s6 := src[6*sStride+lo : 6*sStride+hi][:n]
	s7 := src[7*sStride+lo : 7*sStride+hi][:n]
	out := cols[:n*lane]
	for len(s0) > 0 && len(s1) > 0 && len(s2) > 0 && len(s3) > 0 &&
		len(s4) > 0 && len(s5) > 0 && len(s6) > 0 && len(s7) > 0 {
		k := min(lane, len(s0))
		c0, c1, c2, c3, c4, c5, c6, c7 := transpose8(part(s0), part(s1), part(s2), part(s3),
			part(s4), part(s5), part(s6), part(s7))
		// Column q goes to out[q*8:]; a last block stores only its k.
		o := out[:min(k*lane, len(out))]
		c0.StorePart(o)
		c1.StorePart(o[min(lane, len(o)):])
		c2.StorePart(o[min(2*lane, len(o)):])
		c3.StorePart(o[min(3*lane, len(o)):])
		c4.StorePart(o[min(4*lane, len(o)):])
		c5.StorePart(o[min(5*lane, len(o)):])
		c6.StorePart(o[min(6*lane, len(o)):])
		c7.StorePart(o[min(7*lane, len(o)):])
		out = out[len(o):]
		k = min(k, len(s1), len(s2), len(s3), len(s4), len(s5), len(s6), len(s7))
		s0, s1, s2, s3, s4, s5, s6, s7 = s0[k:], s1[k:], s2[k:], s3[k:], s4[k:], s5[k:], s6[k:], s7[k:]
	}
}

// part loads up to eight cells from s, zero-filled past its end.
func part(s []float32) archsimd.Float32x8 {
	if len(s) >= lane {
		return load8(s)
	}
	v, _ := archsimd.LoadFloat32x8Part(s)
	return v
}

// column is one output column's horizontal sum for eight rows at once,
// over the column-major cells from its first tap.
func column(cols []float32, w []float32) archsimd.Float32x8 {
	acc := archsimd.BroadcastFloat32x8(w[0]).Mul(load8(cols))
	w = w[1:]
	cols = cols[lane:]
	for len(w) > 0 && len(cols) >= lane {
		acc = acc.Add(archsimd.BroadcastFloat32x8(w[0]).Mul(load8(cols)))
		w, cols = w[1:], cols[lane:]
	}
	return acc
}

// columnAt is output column c's sum over cols, whose first column is
// source cell base. A column past c1 repeats c1-1, so a last partial
// block computes a full block and stores part of it.
func columnAt(cols []float32, a *Axis, c, c1, base int) archsimd.Float32x8 {
	c = min(c, c1-1)
	f := (int(a.First[c]) - base) * lane
	n := int(a.Taps[c])
	off := int(a.Off[c])
	return column(cols[f:f+n*lane], a.W[off:off+n])
}

// hBlock computes output columns [c0, c1) for the eight rows transposed
// into cols, whose first column is source cell base, into t. A last
// block of fewer than eight columns stores with masked stores.
func hBlock(t []float32, tStride int, cols []float32, a *Axis, c0, c1, base int) {
	cw := c1 - c0
	t0 := t[:cw]
	t1 := t[tStride : tStride+cw][:cw]
	t2 := t[2*tStride : 2*tStride+cw][:cw]
	t3 := t[3*tStride : 3*tStride+cw][:cw]
	t4 := t[4*tStride : 4*tStride+cw][:cw]
	t5 := t[5*tStride : 5*tStride+cw][:cw]
	t6 := t[6*tStride : 6*tStride+cw][:cw]
	t7 := t[7*tStride : 7*tStride+cw][:cw]
	for c := c0; len(t0) > 0 && len(t1) > 0 && len(t2) > 0 && len(t3) > 0 &&
		len(t4) > 0 && len(t5) > 0 && len(t6) > 0 && len(t7) > 0; c += lane {
		r0, r1, r2, r3, r4, r5, r6, r7 := transpose8(
			columnAt(cols, a, c, c1, base), columnAt(cols, a, c+1, c1, base),
			columnAt(cols, a, c+2, c1, base), columnAt(cols, a, c+3, c1, base),
			columnAt(cols, a, c+4, c1, base), columnAt(cols, a, c+5, c1, base),
			columnAt(cols, a, c+6, c1, base), columnAt(cols, a, c+7, c1, base))
		r0.StorePart(t0)
		r1.StorePart(t1)
		r2.StorePart(t2)
		r3.StorePart(t3)
		r4.StorePart(t4)
		r5.StorePart(t5)
		r6.StorePart(t6)
		r7.StorePart(t7)
		k := min(lane, len(t0), len(t1), len(t2), len(t3), len(t4), len(t5), len(t6), len(t7))
		t0, t1, t2, t3, t4, t5, t6, t7 = t0[k:], t1[k:], t2[k:], t3[k:], t4[k:], t5[k:], t6[k:], t7[k:]
	}
}
