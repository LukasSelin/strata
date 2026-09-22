//go:build goexperiment.simd && arm64

package resamprow

import "simd/archsimd"

// This file is the NEON backend (docs/adr/0001-simd-backend.md), four
// lanes at a time. Each lane performs the scalar kernels' operations in
// their order, so the results match bit for bit; the compiler does not
// fuse archsimd's Mul and Add (ADR 0001, arm64 outcome).
//
// The vertical pass is lane-parallel across output columns. The
// horizontal pass would need a gather per tap, since neighbouring output
// columns read unrelated source offsets, so instead it transposes four
// source rows into a column-major scratch, where every tap of every
// output column is one contiguous vector of four rows, accumulates with
// the rows in the lanes, and transposes each block of four output
// columns back (DESIGN.md §54). Loops use array-pointer loads over
// shrinking slices so the loads carry no bounds checks.

const lane = 4

func init() {
	// NEON is part of the arm64 baseline: no feature check.
	simdKernels = &kernelSet{hRows: hRowsNEON, vRow: vRowNEON}
	simdName = "neon"
	UseScalar(false)
}

func load4(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[lane]float32)(s))
}

func store4(v archsimd.Float32x4, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// transpose4 transposes the 4×4 block whose rows are a, b, c, d.
func transpose4(a, b, c, d archsimd.Float32x4) (archsimd.Float32x4, archsimd.Float32x4, archsimd.Float32x4, archsimd.Float32x4) {
	ab0 := a.ToBits().InterleaveLo(b.ToBits()).ReshapeToUint64s() // a0 b0 a1 b1
	ab1 := a.ToBits().InterleaveHi(b.ToBits()).ReshapeToUint64s() // a2 b2 a3 b3
	cd0 := c.ToBits().InterleaveLo(d.ToBits()).ReshapeToUint64s()
	cd1 := c.ToBits().InterleaveHi(d.ToBits()).ReshapeToUint64s()
	return ab0.InterleaveLo(cd0).ReshapeToUint32s().BitsToFloat32(),
		ab0.InterleaveHi(cd0).ReshapeToUint32s().BitsToFloat32(),
		ab1.InterleaveLo(cd1).ReshapeToUint32s().BitsToFloat32(),
		ab1.InterleaveHi(cd1).ReshapeToUint32s().BitsToFloat32()
}

func vRowNEON(dst, t []float32, tStride int, w []float32) {
	i := vRowLanes(dst, t, tStride, w)
	if i < len(dst) {
		scalarVRow(dst[i:], t[i:], tStride, w)
	}
}

// vRowLanes runs the vertical pass four output columns at a time and
// returns how many it wrote. It accumulates in dst one tap row at a
// time, so every loop runs over shrinking slices with no bounds checks;
// each lane still adds its taps in order, as the scalar pass does.
func vRowLanes(dst, t []float32, tStride int, w []float32) int {
	n := len(dst) &^ (lane - 1)
	d, r := dst[:n], t[:n]
	w0 := archsimd.BroadcastFloat32x4(w[0])
	for len(d) >= lane && len(r) >= lane {
		store4(w0.Mul(load4(r)), d)
		d, r = d[lane:], r[lane:]
	}
	for k := 1; k < len(w); k++ {
		wk := archsimd.BroadcastFloat32x4(w[k])
		d, r = dst[:n], t[k*tStride:k*tStride+n]
		for len(d) >= lane && len(r) >= lane {
			store4(load4(d).Add(wk.Mul(load4(r))), d)
			d, r = d[lane:], r[lane:]
		}
	}
	return n
}

func hRowsNEON(t []float32, tStride int, src []float32, sStride, rows int, a *Axis, c0, c1, sx0 int, scratch []float32) {
	lo, hi := span(a, c0, c1)
	lo -= sx0
	hi -= sx0
	cols := scratch[:(hi-lo)*lane]
	j := 0
	for ; j+lane <= rows; j += lane {
		transposeRows(cols, src[j*sStride:], sStride, lo, hi)
		hBlock(t[j*tStride:], tStride, cols, a, c0, c1, sx0+lo)
	}
	if j < rows {
		scalarHRows(t[j*tStride:], tStride, src[j*sStride:], sStride, rows-j, a, c0, c1, sx0, nil)
	}
}

// transposeRows writes source columns [lo, hi) of the four rows at src
// into cols, column-major: cols[(i-lo)*4 + l] is row l's cell i. A last
// block of fewer than four columns loads zero-filled parts.
func transposeRows(cols, src []float32, sStride, lo, hi int) {
	n := hi - lo
	s0 := src[lo:hi]
	s1 := src[sStride+lo : sStride+hi][:n]
	s2 := src[2*sStride+lo : 2*sStride+hi][:n]
	s3 := src[3*sStride+lo : 3*sStride+hi][:n]
	out := cols[:n*lane]
	for len(s0) > 0 && len(s1) > 0 && len(s2) > 0 && len(s3) > 0 {
		k := min(lane, len(s0), len(s1), len(s2), len(s3))
		c0, c1, c2, c3 := transpose4(part(s0), part(s1), part(s2), part(s3))
		o := out[:min(k*lane, len(out))]
		c0.StorePart(o)
		c1.StorePart(o[min(lane, len(o)):])
		c2.StorePart(o[min(2*lane, len(o)):])
		c3.StorePart(o[min(3*lane, len(o)):])
		out = out[len(o):]
		s0, s1, s2, s3 = s0[k:], s1[k:], s2[k:], s3[k:]
	}
}

// part loads up to four cells from s, zero-filled past its end.
func part(s []float32) archsimd.Float32x4 {
	if len(s) >= lane {
		return load4(s)
	}
	v, _ := archsimd.LoadFloat32x4Part(s)
	return v
}

// column is one output column's horizontal sum for four rows at once,
// over the column-major cells from its first tap.
func column(cols []float32, w []float32) archsimd.Float32x4 {
	acc := archsimd.BroadcastFloat32x4(w[0]).Mul(load4(cols))
	w = w[1:]
	cols = cols[lane:]
	for len(w) > 0 && len(cols) >= lane {
		acc = acc.Add(archsimd.BroadcastFloat32x4(w[0]).Mul(load4(cols)))
		w, cols = w[1:], cols[lane:]
	}
	return acc
}

// columnAt is output column c's sum over cols, whose first column is
// source cell base. A column past c1 repeats c1-1, so a last partial
// block computes a full block and stores part of it. Its table lookups
// and the slice of cols they select are the pass's only bounds checks,
// once per output column per four rows.
func columnAt(cols []float32, a *Axis, c, c1, base int) archsimd.Float32x4 {
	c = min(c, c1-1)
	f := (int(a.First[c]) - base) * lane
	n := int(a.Taps[c])
	off := int(a.Off[c])
	return column(cols[f:f+n*lane], a.W[off:off+n])
}

// hBlock computes output columns [c0, c1) for the four rows transposed
// into cols, whose first column is source cell base, into t.
func hBlock(t []float32, tStride int, cols []float32, a *Axis, c0, c1, base int) {
	cw := c1 - c0
	t0 := t[:cw]
	t1 := t[tStride : tStride+cw][:cw]
	t2 := t[2*tStride : 2*tStride+cw][:cw]
	t3 := t[3*tStride : 3*tStride+cw][:cw]
	for c := c0; len(t0) > 0 && len(t1) > 0 && len(t2) > 0 && len(t3) > 0; c += lane {
		r0, r1, r2, r3 := transpose4(
			columnAt(cols, a, c, c1, base), columnAt(cols, a, c+1, c1, base),
			columnAt(cols, a, c+2, c1, base), columnAt(cols, a, c+3, c1, base))
		r0.StorePart(t0)
		r1.StorePart(t1)
		r2.StorePart(t2)
		r3.StorePart(t3)
		k := min(lane, len(t0), len(t1), len(t2), len(t3))
		t0, t1, t2, t3 = t0[k:], t1[k:], t2[k:], t3[k:]
	}
}
