package vec

import "math"

// ValidBits and its kernels: the validity of a row of cells against a
// fill value, the NoData test of a raw file (engine.RawOptions, DESIGN.md
// §31 rule 5). It lives here rather than in the engine because the SIMD
// forms must (DESIGN.md §14), and it is the same shape as every other
// kernel in the package: one flat slice in, one out.

// ValidBits writes the validity of src against the fill value fill into
// dst, a validity mask of len(src) cells from bit 0 (raster's layout:
// LSB first, a set bit valid). Cell i is invalid where src[i] == fill,
// compared as floats, so -0 matches 0; when fill is a NaN, cell i is
// invalid where src[i] is any NaN. The bits of dst's last word past
// len(src) are cleared.
//
// dst must have exactly (len(src)+63)/64 words; it panics otherwise, with
// a "vec: " message.
func ValidBits(dst []uint64, src []float32, fill float32) {
	if len(dst) != (len(src)+63)>>6 {
		panic("vec: ValidBits: dst must have (len(src)+63)/64 words")
	}
	validBits(dst, src, fill)
}

// ValidWord returns the validity of at most 64 cells against fill, as
// ValidBits computes it, as the low len(src) bits of a word. It is for
// the ends of a run that do not fill a word of their mask, which a
// caller merges into it; it runs the scalar kernel, called directly, so
// that nothing it is given escapes. It panics if src has more than 64
// cells.
func ValidWord(src []float32, fill float32) uint64 {
	if len(src) > 64 {
		panic("vec: ValidWord: more than 64 cells")
	}
	var w [1]uint64
	scalarValidBits(w[:(len(src)+63)>>6], src, fill)
	return w[0]
}

// scalarValidBits is the reference. It tests the bits of each cell as
// an integer, which gives the float comparison's answers: a fill other
// than zero or NaN equals exactly the cells with its own bits, zero
// equals the cells whose bits are zero but for the sign, and a NaN fill
// matches the cells whose magnitude's bits exceed +Inf's. Whole words go
// eight cells at a time, each group's bits shifted into place by
// constants. On the benchmark machine that is 1.4 to 1.8 times the speed
// of a loop of float compares with a variable shift per cell, depending
// on where the loop lands in the binary; the spread is code placement,
// not the data. It is what builds without SIMD kernels run.
func scalarValidBits(dst []uint64, src []float32, fill float32) {
	var w int
	if fill != fill {
		w = scalarNotNaNWords(dst, src)
	} else {
		w = scalarUnequalWords(dst, src, fill)
	}
	d, s := dst[w:], src[64*w:]
	if len(d) == 0 {
		return
	}
	// The last, partial word.
	want, care := unequalTest(fill)
	var word uint64
	for j, c := range s {
		var bit uint64
		if fill != fill {
			bit = notNaNBit(c)
		} else {
			bit = unequalBit(c, want, care)
		}
		word |= bit << (uint(j) & 63)
	}
	d[0] = word
}

// unequalTest returns the bits a cell equal to fill, not a NaN, has
// where care is set.
func unequalTest(fill float32) (want, care uint32) {
	if fill == 0 {
		return 0, 0x7fff_ffff // either zero
	}
	return math.Float32bits(fill), ^uint32(0)
}

// unequalBit is 1 where c's bits under care are not want.
func unequalBit(c float32, want, care uint32) uint64 {
	d := (math.Float32bits(c) ^ want) & care
	return uint64((d | -d) >> 31)
}

// notNaNBit is 1 where c is not NaN: its magnitude's bits are at most
// +Inf's.
func notNaNBit(c float32) uint64 {
	m := math.Float32bits(c) & 0x7fff_ffff
	return uint64((0x7f80_0000-m)>>31 ^ 1)
}

// scalarUnequalWords writes the words of the whole words of src, the
// bits of the cells not equal to fill, a non-NaN, and returns how many
// it wrote.
func scalarUnequalWords(dst []uint64, src []float32, fill float32) int {
	want, care := unequalTest(fill)
	d, s := dst, src
	for len(s) >= 64 && len(d) > 0 {
		c := (*[64]float32)(s)
		var word uint64
		for g := 0; g < 64; g += 8 {
			x := (*[8]float32)(c[g : g+8])
			word |= (unequalBit(x[0], want, care) | unequalBit(x[1], want, care)<<1 |
				unequalBit(x[2], want, care)<<2 | unequalBit(x[3], want, care)<<3 |
				unequalBit(x[4], want, care)<<4 | unequalBit(x[5], want, care)<<5 |
				unequalBit(x[6], want, care)<<6 | unequalBit(x[7], want, care)<<7) << uint(g)
		}
		d[0] = word
		d, s = d[1:], s[64:]
	}
	return len(dst) - len(d)
}

// scalarNotNaNWords is scalarUnequalWords for a NaN fill: the bits of the
// cells that are not NaN.
func scalarNotNaNWords(dst []uint64, src []float32) int {
	d, s := dst, src
	for len(s) >= 64 && len(d) > 0 {
		c := (*[64]float32)(s)
		var word uint64
		for g := 0; g < 64; g += 8 {
			x := (*[8]float32)(c[g : g+8])
			word |= (notNaNBit(x[0]) | notNaNBit(x[1])<<1 | notNaNBit(x[2])<<2 | notNaNBit(x[3])<<3 |
				notNaNBit(x[4])<<4 | notNaNBit(x[5])<<5 | notNaNBit(x[6])<<6 | notNaNBit(x[7])<<7) << uint(g)
		}
		d[0] = word
		d, s = d[1:], s[64:]
	}
	return len(dst) - len(d)
}
