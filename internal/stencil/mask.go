package stencil

// Validity masks here use the raster package's layout: bit i lives in
// word i>>6, LSB first, and a set bit means valid. A region's cell (x, y)
// is bit off + y*stride + x, so windows (non-zero offsets) and padded
// rows (stride > width) are addressed directly. Only the bits of a
// region's own cells are read or written; row padding bits belong to
// someone else (for a window, to cells of its parent) and are left alone.

// Erode3x3 writes the output validity of a radius-1 stencil over a w×h
// region. Destination cell (x, y) becomes valid iff it is not on the
// region's one-cell border and all nine source cells around (x, y) are
// valid. It works on whole words: per row it ANDs the three source rows,
// then erodes horizontally with shifted copies.
//
// dst and src may be the same slice only if the two regions' bits do not
// overlap. It panics if a region needs bits past the end of its mask.
func Erode3x3(dst []uint64, dstOff, dstStride int, src []uint64, srcOff, srcStride int, w, h int) {
	requireBits(dst, dstOff, dstStride, w, h)
	requireBits(src, srcOff, srcStride, w, h)
	ClearBorder(dst, dstOff, dstStride, w, h)
	if w < 3 || h < 3 {
		return
	}
	nw := wordsFor(w)
	scratch := make([]uint64, 2*nw)
	acc, tmp := scratch[:nw], scratch[nw:]
	for y := 1; y < h-1; y++ {
		extractBits(acc, src, srcOff+(y-1)*srcStride, w)
		andBits(acc, src, srcOff+y*srcStride, w)
		andBits(acc, src, srcOff+(y+1)*srcStride, w)
		erodeRow(tmp, acc)
		depositBits(dst, dstOff+y*dstStride, tmp, w)
	}
}

// ClearBorder clears the validity bits of the one-cell border of a w×h
// region: its first and last rows and columns.
func ClearBorder(m []uint64, off, stride, w, h int) {
	requireBits(m, off, stride, w, h)
	clearBits(m, off, w)
	clearBits(m, off+(h-1)*stride, w)
	for y := 1; y < h-1; y++ {
		i := off + y*stride
		m[i>>6] &^= 1 << uint(i&63)
		i += w - 1
		m[i>>6] &^= 1 << uint(i&63)
	}
}

func requireBits(m []uint64, off, stride, w, h int) {
	if off < 0 || len(m)*64 < off+(h-1)*stride+w {
		panic("stencil: mask too short for region")
	}
}

func wordsFor(n int) int { return (n + 63) >> 6 }

// word returns the 64 bits of src starting at bit off,
// with zeros past the end of src.
func word(src []uint64, off int) uint64 {
	w, s := off>>6, uint(off&63)
	v := src[w] >> s
	if s != 0 && w+1 < len(src) {
		v |= src[w+1] << (64 - s)
	}
	return v
}

// lastWordMask keeps the bits of the final word of an n-bit row.
func lastWordMask(n int) uint64 {
	if r := uint(n & 63); r != 0 {
		return 1<<r - 1
	}
	return ^uint64(0)
}

// extractBits copies n bits of src starting at bit off into dst, which
// has wordsFor(n) words, clearing the bits past n.
func extractBits(dst, src []uint64, off, n int) {
	nw := wordsFor(n)
	dst = dst[:nw]
	if off&63 == 0 {
		copy(dst, src[off>>6:off>>6+nw])
	} else {
		for k := range dst {
			dst[k] = word(src, off+k<<6)
		}
	}
	dst[nw-1] &= lastWordMask(n)
}

// andBits is extractBits that ANDs into dst instead of overwriting it.
// Bits of dst past n must already be clear.
func andBits(dst, src []uint64, off, n int) {
	dst = dst[:wordsFor(n)]
	if off&63 == 0 {
		src = src[off>>6 : off>>6+len(dst)]
		for k := range dst {
			dst[k] &= src[k]
		}
		return
	}
	for k := range dst {
		dst[k] &= word(src, off+k<<6)
	}
}

// depositBits overwrites n bits of dst starting at bit off with src,
// leaving every other bit of dst unchanged.
func depositBits(dst []uint64, off int, src []uint64, n int) {
	for k := 0; n > 0; k++ {
		take := min(n, 64)
		m := ^uint64(0)
		if take < 64 {
			m = 1<<uint(take) - 1
		}
		v := src[k] & m
		w, s := off>>6, uint(off&63)
		dst[w] = dst[w]&^(m<<s) | v<<s
		if s != 0 && take > int(64-s) {
			dst[w+1] = dst[w+1]&^(m>>(64-s)) | v>>(64-s)
		}
		off += take
		n -= take
	}
}

// clearBits clears n bits of dst starting at bit off.
func clearBits(dst []uint64, off, n int) {
	for n > 0 {
		w, s := off>>6, uint(off&63)
		take := min(n, int(64-s))
		m := ^uint64(0)
		if take < 64 {
			m = 1<<uint(take) - 1
		}
		dst[w] &^= m << s
		off += take
		n -= take
	}
}

// erodeRow sets out bit x = v[x-1] & v[x] & v[x+1]. v must have zero bits
// past the row width, which clears the last column; the zero carry-in
// clears the first.
func erodeRow(out, v []uint64) {
	out = out[:len(v)]
	var carry uint64
	last := len(v) - 1
	for k, cur := range v {
		left := cur<<1 | carry
		carry = cur >> 63
		var next uint64
		if k < last {
			next = v[k+1] & 1
		}
		right := cur>>1 | next<<63
		out[k] = cur & left & right
	}
}
