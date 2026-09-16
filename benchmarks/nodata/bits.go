package nodata

import "math/bits"

// MaskWords returns the number of uint64 words needed for n cells.
func MaskWords(n int) int { return (n + 63) >> 6 }

// MaskFromInvalid builds a validity mask (bit set = valid) from flags.
func MaskFromInvalid(invalid []bool) []uint64 {
	m := make([]uint64, MaskWords(len(invalid)))
	for i, bad := range invalid {
		if !bad {
			m[i>>6] |= 1 << uint(i&63)
		}
	}
	return m
}

// MaskBit reports whether cell i is valid.
func MaskBit(m []uint64, i int) bool { return m[i>>6]>>uint(i&63)&1 != 0 }

func b2u(b bool) uint64 {
	var u uint64
	if b {
		u = 1
	}
	return u
}

// AndMasks computes dst = a & b word by word. Bits past the cell count
// are zero in well-formed masks, so they stay zero.
func AndMasks(dst, a, b []uint64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for k := range dst {
		dst[k] = a[k] & b[k]
	}
}

// FillInvalid writes fill into every cell whose validity bit is clear.
// This is the cost of materialising a defined value under NoData, e.g.
// when exporting a mask-represented raster to a fill-value format.
// All-valid words are skipped with one compare.
func FillInvalid(data []float32, valid []uint64, fill float32) {
	n := len(data)
	for k, word := range valid {
		inv := ^word
		if inv == 0 {
			continue
		}
		base := k << 6
		for inv != 0 {
			i := base + bits.TrailingZeros64(inv)
			if i >= n {
				break
			}
			data[i] = fill
			inv &= inv - 1
		}
	}
}

// MaskFromSentinel is the ingest conversion from a fill value to a mask.
func MaskFromSentinel(dst []uint64, data []float32, fill float32) {
	n := len(data)
	for k := range dst {
		base := k << 6
		end := min(base+64, n)
		var word uint64
		for i := base; i < end; i++ {
			word |= b2u(data[i] != fill) << uint(i-base)
		}
		dst[k] = word
	}
}

// SentinelToNaN is the ingest conversion from a fill value to NaN.
func SentinelToNaN(dst, src []float32, fill float32) {
	src = src[:len(dst)]
	for i, v := range src {
		if v == fill {
			v = NaN32
		}
		dst[i] = v
	}
}

// extractBits copies n bits of src starting at bit off into dst[0:],
// clearing bits of the last word beyond n.
func extractBits(dst, src []uint64, off, n int) {
	w := off >> 6
	s := uint(off & 63)
	nw := MaskWords(n)
	if s == 0 {
		copy(dst[:nw], src[w:w+nw])
	} else {
		for k := 0; k < nw; k++ {
			v := src[w+k] >> s
			if w+k+1 < len(src) {
				v |= src[w+k+1] << (64 - s)
			}
			dst[k] = v
		}
	}
	if r := uint(n & 63); r != 0 {
		dst[nw-1] &= 1<<r - 1
	}
}

// depositBits overwrites n bits of dst starting at bit off with src.
func depositBits(dst []uint64, off int, src []uint64, n int) {
	for k := 0; n > 0; k++ {
		take := min(n, 64)
		m := ^uint64(0)
		if take < 64 {
			m = 1<<uint(take) - 1
		}
		v := src[k] & m
		w := off >> 6
		s := uint(off & 63)
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
		w := off >> 6
		s := uint(off & 63)
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

// SlopeMaskScratchWords is the scratch size SlopeMask needs for width w.
func SlopeMaskScratchWords(w int) int { return 2 * MaskWords(w) }

// SlopeMask computes the output validity of a 3×3 kernel: a cell is
// valid iff all nine input cells are valid and it is not on the border.
// It works a row at a time on whole words: AND the three input rows,
// then erode horizontally with shifted copies.
func SlopeMask(dstValid, valid []uint64, w, h int, scratch []uint64) {
	nw := MaskWords(w)
	acc := scratch[:nw]
	tmp := scratch[nw : 2*nw]
	clearBits(dstValid, 0, w)
	clearBits(dstValid, (h-1)*w, w)
	for y := 1; y < h-1; y++ {
		extractBits(acc, valid, (y-1)*w, w)
		extractBits(tmp, valid, y*w, w)
		for k := range acc {
			acc[k] &= tmp[k]
		}
		extractBits(tmp, valid, (y+1)*w, w)
		for k := range acc {
			acc[k] &= tmp[k]
		}
		erodeRow(tmp, acc)
		depositBits(dstValid, y*w, tmp, w)
	}
	if r := uint(w * h & 63); r != 0 {
		dstValid[len(dstValid)-1] &= 1<<r - 1
	}
}

// erodeRow sets out bit x = v[x-1] & v[x] & v[x+1]. v must have zero
// bits past the row width, which makes the last column invalid; the
// zero carry-in makes the first column invalid.
func erodeRow(out, v []uint64) {
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
