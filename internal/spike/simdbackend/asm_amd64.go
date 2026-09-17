//go:build amd64

package simdbackend

// slopeRowAVX2Asm processes exactly n output cells (n a multiple of 8).
//
//go:noescape
func slopeRowAVX2Asm(dst, up, mid, down *float32, n int, invDx8, invDy8 float32)

func slopeRowAsm(dst, up, mid, down []float32, invDx8, invDy8 float32) {
	n8 := len(dst) - len(dst)%8
	if n8 > 0 {
		slopeRowAVX2Asm(&dst[0], &up[0], &mid[0], &down[0], n8, invDx8, invDy8)
	}
	if n8 < len(dst) {
		scalarSlopeRow(dst[n8:], up[n8:], mid[n8:], down[n8:], invDx8, invDy8)
	}
}
