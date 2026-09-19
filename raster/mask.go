package raster

import "fmt"

// A validity mask is a []uint64 bitmap: bit i is Valid[i>>6]>>(i&63)&1
// (LSB first), and a set bit means the cell holds data. The layout
// matches Arrow validity bitmaps. Bits past the last cell of a mask
// created by NewMask are zero and must stay zero.

// MaskWords returns the number of words a mask needs for n cells.
func MaskWords(n int) int {
	return (n + 63) >> 6
}

// NewMask allocates a mask for n cells with every cell valid and the
// unused bits of the last word cleared. It panics if n is negative.
func NewMask(n int) []uint64 {
	if n < 0 {
		panic(fmt.Sprintf("raster: negative mask size %d", n))
	}
	m := make([]uint64, MaskWords(n))
	for k := range m {
		m[k] = ^uint64(0)
	}
	if r := uint(n & 63); r != 0 {
		m[len(m)-1] = 1<<r - 1
	}
	return m
}

// MaskGet reports whether bit i of m is set (cell i is valid).
func MaskGet(m []uint64, i int) bool {
	return m[i>>6]>>uint(i&63)&1 != 0
}

// MaskSet sets bit i of m to valid.
func MaskSet(m []uint64, i int, valid bool) {
	bit := uint64(1) << uint(i&63)
	if valid {
		m[i>>6] |= bit
	} else {
		m[i>>6] &^= bit
	}
}

// MaskAnd computes dst[k] = a[k] & b[k], the validity of a pointwise
// operation on two inputs that share the same bit layout. dst may alias a
// or b. It panics if the lengths differ, like internal/vec. Masks of
// windows at different offsets must be aligned before they can be ANDed
// word by word.
func MaskAnd(dst, a, b []uint64) {
	if len(a) != len(dst) || len(b) != len(dst) {
		panic("raster: dst, a, and b masks must have equal length")
	}
	for k := range dst {
		dst[k] = a[k] & b[k]
	}
}

// The Range functions below work on bit ranges at arbitrary offsets, for
// windows whose masks do not line up word for word: a window's bits start
// at its ValidOffset and follow its Stride. They touch only the n bits of
// dst starting at dstOff and leave every other bit alone, because a
// window's mask is shared with its parent. They process up to 64 bits per
// step and drop to plain word loops once every offset is word-aligned.
//
// Like MaskAnd, they panic if a range falls outside its mask or an offset
// or n is negative. dst may be the same mask as a source only at the same
// offset; ranges of one mask that overlap at different offsets give
// undefined results.

// MaskFillRange sets bits [off, off+n) of m to valid.
func MaskFillRange(m []uint64, off, n int, valid bool) {
	requireRange("m", m, off, n)
	var fill uint64
	if valid {
		fill = ^uint64(0)
	}
	for n > 0 {
		if off&63 == 0 && n >= 64 {
			w := off >> 6
			words := m[w : w+n>>6]
			for k := range words {
				words[k] = fill
			}
			off += len(words) * 64
			n -= len(words) * 64
			continue
		}
		k := min(n, 64-off&63)
		putBits(m, off, k, fill)
		off += k
		n -= k
	}
}

// MaskCopyRange copies n bits of src starting at srcOff into dst starting
// at dstOff.
func MaskCopyRange(dst []uint64, dstOff int, src []uint64, srcOff, n int) {
	requireRange("dst", dst, dstOff, n)
	requireRange("src", src, srcOff, n)
	for n > 0 {
		if dstOff&63 == 0 && srcOff&63 == 0 && n >= 64 {
			w, ws, words := dstOff>>6, srcOff>>6, n>>6
			copy(dst[w:w+words], src[ws:ws+words])
			dstOff += words * 64
			srcOff += words * 64
			n -= words * 64
			continue
		}
		k := min(n, 64-dstOff&63)
		putBits(dst, dstOff, k, getBits(src, srcOff, k))
		dstOff += k
		srcOff += k
		n -= k
	}
}

// MaskAndRange sets bit dstOff+i of dst to the AND of bit aOff+i of a and
// bit bOff+i of b, for i in [0, n): the validity of a pointwise operation
// whose operands are windows at different offsets.
func MaskAndRange(dst []uint64, dstOff int, a []uint64, aOff int, b []uint64, bOff, n int) {
	requireRange("dst", dst, dstOff, n)
	requireRange("a", a, aOff, n)
	requireRange("b", b, bOff, n)
	for n > 0 {
		if dstOff&63 == 0 && aOff&63 == 0 && bOff&63 == 0 && n >= 64 {
			words := n >> 6
			d := dst[dstOff>>6 : dstOff>>6+words]
			as := a[aOff>>6 : aOff>>6+words]
			bs := b[bOff>>6 : bOff>>6+words]
			for k := range d {
				d[k] = as[k] & bs[k]
			}
			dstOff += words * 64
			aOff += words * 64
			bOff += words * 64
			n -= words * 64
			continue
		}
		k := min(n, 64-dstOff&63)
		putBits(dst, dstOff, k, getBits(a, aOff, k)&getBits(b, bOff, k))
		dstOff += k
		aOff += k
		bOff += k
		n -= k
	}
}

func requireRange(name string, m []uint64, off, n int) {
	if off < 0 || n < 0 || off > len(m)*64-n {
		panic(fmt.Sprintf("raster: %d bits at offset %d outside %d-bit mask %s",
			n, off, len(m)*64, name))
	}
}

// MaskBits returns the k bits of m starting at bit off, LSB first, as the
// low k bits of the result: how a reader takes a run of validity from a
// window whose bits do not start on a word boundary, such as a reduction
// walking the valid cells of a row (DESIGN.md §49). The Range functions
// above use it too, through an unchecked form.
//
// It panics if k is not in [1, 64] or the range falls outside m. There is
// no exported write counterpart: masks are written through the Range
// functions, which keep a window from disturbing bits it does not own.
func MaskBits(m []uint64, off, k int) uint64 {
	if k < 1 || k > 64 {
		panic(fmt.Sprintf("raster: MaskBits takes 1 to 64 bits, got %d", k))
	}
	requireRange("m", m, off, k)
	return getBits(m, off, k)
}

// getBits is MaskBits without the checks, for callers that have already
// made them.
func getBits(m []uint64, off, k int) uint64 {
	w, s := off>>6, uint(off&63)
	v := m[w] >> s
	if s != 0 && int(s)+k > 64 {
		v |= m[w+1] << (64 - s)
	}
	if k < 64 {
		v &= 1<<uint(k) - 1
	}
	return v
}

// putBits stores the low k bits of v into m starting at bit off. The
// range must fit in one word: off&63 + k <= 64.
func putBits(m []uint64, off, k int, v uint64) {
	w, s := off>>6, uint(off&63)
	mask := ^uint64(0)
	if k < 64 {
		mask = 1<<uint(k) - 1
	}
	m[w] = m[w]&^(mask<<s) | (v&mask)<<s
}
