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
