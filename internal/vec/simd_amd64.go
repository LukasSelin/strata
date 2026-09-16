//go:build amd64

package vec

// avxLane is the number of float32 lanes in a YMM register.
const avxLane = 8

// Each *AVX2Asm function is implemented in simd_amd64.s. It processes
// exactly n elements (n must be a multiple of avxLane) starting at the
// given pointers; the wrapper below handles the remainder with the
// scalar backend, which is also the correctness reference these must
// agree with bit-for-bit.
//
//go:noescape
func addAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func subAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func mulAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func divAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func minAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func maxAVX2Asm(dst, a, b *float32, n int)

//go:noescape
func addScalarAVX2Asm(dst, src *float32, n int, value float32)

//go:noescape
func mulScalarAVX2Asm(dst, src *float32, n int, value float32)

//go:noescape
func clampAVX2Asm(dst, src *float32, n int, lo, hi float32)

//go:noescape
func absAVX2Asm(dst, src *float32, n int)

//go:noescape
func sqrtAVX2Asm(dst, src *float32, n int)

func init() {
	if !hasAVX2() {
		return
	}
	addFloat32 = addFloat32AVX2
	subFloat32 = subFloat32AVX2
	mulFloat32 = mulFloat32AVX2
	divFloat32 = divFloat32AVX2
	addScalarFloat32 = addScalarFloat32AVX2
	mulScalarFloat32 = mulScalarFloat32AVX2
	minFloat32 = minFloat32AVX2
	maxFloat32 = maxFloat32AVX2
	clampFloat32 = clampFloat32AVX2
	absFloat32 = absFloat32AVX2
	sqrtFloat32 = sqrtFloat32AVX2
}

func avxSplit(n int) int {
	return n - n%avxLane
}

func addFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		addAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarAddFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func subFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		subAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarSubFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func mulFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		mulAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarMulFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func divFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		divAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarDivFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func minFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		minAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarMinFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func maxFloat32AVX2(dst, a, b []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		maxAVX2Asm(&dst[0], &a[0], &b[0], n8)
	}
	if n8 < len(dst) {
		scalarMaxFloat32(dst[n8:], a[n8:], b[n8:])
	}
}

func addScalarFloat32AVX2(dst, src []float32, value float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		addScalarAVX2Asm(&dst[0], &src[0], n8, value)
	}
	if n8 < len(dst) {
		scalarAddScalarFloat32(dst[n8:], src[n8:], value)
	}
}

func mulScalarFloat32AVX2(dst, src []float32, value float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		mulScalarAVX2Asm(&dst[0], &src[0], n8, value)
	}
	if n8 < len(dst) {
		scalarMulScalarFloat32(dst[n8:], src[n8:], value)
	}
}

func clampFloat32AVX2(dst, src []float32, lo, hi float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		clampAVX2Asm(&dst[0], &src[0], n8, lo, hi)
	}
	if n8 < len(dst) {
		scalarClampFloat32(dst[n8:], src[n8:], lo, hi)
	}
}

func absFloat32AVX2(dst, src []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		absAVX2Asm(&dst[0], &src[0], n8)
	}
	if n8 < len(dst) {
		scalarAbsFloat32(dst[n8:], src[n8:])
	}
}

func sqrtFloat32AVX2(dst, src []float32) {
	n8 := avxSplit(len(dst))
	if n8 > 0 {
		sqrtAVX2Asm(&dst[0], &src[0], n8)
	}
	if n8 < len(dst) {
		scalarSqrtFloat32(dst[n8:], src[n8:])
	}
}
