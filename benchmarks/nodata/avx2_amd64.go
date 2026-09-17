//go:build amd64

package nodata

// HaveAVX2 reports whether the AVX2 variants can run on this machine.
var HaveAVX2 = detectAVX2()

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// detectAVX2 mirrors internal/vec's hasAVX2 (which is unexported).
func detectAVX2() bool {
	_, _, ecx1, _ := cpuid(1, 0)
	const osxsaveAVX = 1<<27 | 1<<28
	if ecx1&osxsaveAVX != osxsaveAVX {
		return false
	}
	if eax, _ := xgetbv(); eax&0x6 != 0x6 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}

//go:noescape
func addSentinelAVX2Asm(dst, a, b *float32, n int, nd float32)

//go:noescape
func slopeRowAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy float32)

//go:noescape
func slopeRowNaNAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy float32)

//go:noescape
func slopeRowSentinelAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy, nd float32)

// AddSentinelAVX2 is compute + compare + blend in 8-lane AVX2, with the
// select form for the tail.
func AddSentinelAVX2(dst, a, b []float32, nd float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	n8 := len(dst) &^ 7
	if n8 > 0 {
		addSentinelAVX2Asm(&dst[0], &a[0], &b[0], n8, nd)
	}
	AddSentinelSelect(dst[n8:], a[n8:], b[n8:], nd)
}

func slopeRowAVX2(dst, r0, r1, r2 []float32, invx, invy float32) {
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	n8 := n &^ 7
	if n8 > 0 {
		slopeRowAVX2Asm(&dst[0], &r0[0], &r1[0], &r2[0], n8, invx, invy)
	}
	slopeRowScalar(dst[n8:], r0[n8:], r1[n8:], r2[n8:], invx, invy)
}

func slopeRowNaNAVX2(dst, r0, r1, r2 []float32, invx, invy float32) {
	n := len(dst)
	r0, r1, r2 = r0[:n+2], r1[:n+2], r2[:n+2]
	n8 := n &^ 7
	if n8 > 0 {
		slopeRowNaNAVX2Asm(&dst[0], &r0[0], &r1[0], &r2[0], n8, invx, invy)
	}
	slopeRowNaNScalar(dst[n8:], r0[n8:], r1[n8:], r2[n8:], invx, invy)
}

// SlopeNaNAVX2: fused AVX2 Horn kernel; NaN propagates through the
// arithmetic, with the extra z5·0 term (see hornNaN).
func SlopeNaNAVX2(dst, src []float32, w, h int, cellSize float32) {
	fillBorder(dst, w, h, NaN32)
	slopeRows(dst, src, w, h, cellSize, slopeRowNaNAVX2)
}

// SlopeSentinelAVX2: fused AVX2 Horn kernel with nine compares and a blend.
func SlopeSentinelAVX2(dst, src []float32, w, h int, cellSize, nd float32) {
	fillBorder(dst, w, h, nd)
	invx, invy := hornScales(cellSize)
	for y := 1; y < h-1; y++ {
		out := dst[y*w+1 : y*w+w-1]
		r0 := src[(y-1)*w : y*w]
		r1 := src[y*w : (y+1)*w]
		r2 := src[(y+1)*w : (y+2)*w]
		n8 := len(out) &^ 7
		if n8 > 0 {
			slopeRowSentinelAVX2Asm(&out[0], &r0[0], &r1[0], &r2[0], n8, invx, invy, nd)
		}
		slopeRowSentinelSelect(out[n8:], r0[n8:], r1[n8:], r2[n8:], invx, invy, nd)
	}
}

// SlopeMaskAVX2: the same fused kernel as SlopeNaNAVX2 over every
// interior cell, plus the word-level SlopeMask pass.
func SlopeMaskAVX2(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64) {
	slopeRows(dst, src, w, h, cellSize, slopeRowAVX2)
	SlopeMask(dstValid, valid, w, h, scratch)
}

// SlopeMaskAVX2Fill additionally writes fill under invalid output cells.
func SlopeMaskAVX2Fill(dst, src []float32, dstValid, valid []uint64, w, h int, cellSize float32, scratch []uint64, fill float32) {
	SlopeMaskAVX2(dst, src, dstValid, valid, w, h, cellSize, scratch)
	FillInvalid(dst, dstValid, fill)
}
