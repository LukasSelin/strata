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
	ErodeBox(MaskRegion{dst, dstOff + dstStride + 1, dstStride},
		[]MaskRegion{{src, srcOff, srcStride}}, w-2, h-2, 1, make([]uint64, ErodeScratch(w-2, 1)))
}

// MaskRegion addresses a rectangle of cells in a validity mask: cell
// (x, y) is bit Off + y*Stride + x of Bits.
type MaskRegion struct {
	Bits   []uint64
	Off    int
	Stride int
}

// ErodeScratch returns the number of scratch words ErodeBox needs for a
// destination w cells wide and radius r.
func ErodeScratch(w, r int) int { return wordsFor(w + 2*r) }

// ErodeBox writes the validity of a radius-r neighbourhood operation with
// one or more inputs into a w×h destination region. Each source region
// is its input's halo-expanded region: (w+2r)×(h+2r) cells, whose cell
// (x+r, y+r) is centred on destination cell (x, y). Destination cell
// (x, y) becomes valid iff source cells (x+i, y+j), 0 <= i, j <= 2r, are
// valid in every source. Radius 0 is the AND of the sources. Unlike
// Erode3x3 it has no border: the halo supplies every neighbour.
//
// It works on whole words: per destination row it ANDs the 2r+1 rows of
// every source into scratch, which must hold at least ErodeScratch(w, r)
// words, then shrinks that row by 2r cells, two per pass of shifted
// copies. It allocates nothing. The destination's bits must not overlap a source's, except at
// the same address and stride with radius 0 (in place). It panics if srcs
// is empty or a region needs bits past the end of its mask.
func ErodeBox(dst MaskRegion, srcs []MaskRegion, w, h, r int, scratch []uint64) {
	if len(srcs) == 0 {
		panic("stencil: ErodeBox needs at least one source")
	}
	sw, sh := w+2*r, h+2*r
	requireBits(dst.Bits, dst.Off, dst.Stride, w, h)
	for _, s := range srcs {
		requireBits(s.Bits, s.Off, s.Stride, sw, sh)
	}
	acc := scratch[:wordsFor(sw)]
	for y := range h {
		for i, s := range srcs {
			off := s.Off + y*s.Stride
			for j := range 2*r + 1 {
				if i == 0 && j == 0 {
					extractBits(acc, s.Bits, off, sw)
				} else {
					andBits(acc, s.Bits, off, sw)
				}
				off += s.Stride
			}
		}
		for range r {
			shrink2Row(acc)
		}
		depositBits(dst.Bits, dst.Off+y*dst.Stride, acc, w)
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
	k := 0
	if s := uint(off & 63); s == 0 {
		k = n >> 6
		copy(dst[off>>6:], src[:k])
		off += k << 6
		n -= k << 6
	} else if n >= 64 {
		// Whole source words, each split across two destination words:
		// its low 64-s bits fill the top of one, its high s bits carry
		// into the bottom of the next.
		k = n >> 6
		d := dst[off>>6 : off>>6+k+1]
		carry := d[0] & (1<<s - 1)
		for i, v := range src[:k] {
			d[i] = carry | v<<s
			carry = v >> (64 - s)
		}
		d[k] = d[k]&^(1<<s-1) | carry
		off += k << 6
		n -= k << 6
	}
	// The remaining n < 64 bits, if any.
	for ; n > 0; k++ {
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

// shrink2Row sets bit x of v to v[x] & v[x+1] & v[x+2], in place: two
// shrinks in one pass. Bits past the row width must be zero, so the
// row's last bits read zeros and they stay zero.
func shrink2Row(v []uint64) {
	cur := v[0]
	for k := 1; k < len(v); k++ {
		next := v[k]
		v[k-1] = cur & (cur>>1 | next<<63) & (cur>>2 | next<<62)
		cur = next
	}
	v[len(v)-1] = cur & (cur >> 1) & (cur >> 2)
}
