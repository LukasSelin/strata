//go:build amd64

#include "textflag.h"

// func slopeRowAVX2Asm(dst, up, mid, down *float32, n int, invDx8, invDy8 float32)
TEXT ·slopeRowAVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ up+8(FP), SI
	MOVQ mid+16(FP), R8
	MOVQ down+24(FP), R9
	MOVQ n+32(FP), CX
	VBROADCASTSS invDx8+40(FP), Y14
	VBROADCASTSS invDy8+44(FP), Y15
	XORQ AX, AX
loop:
	CMPQ AX, CX
	JGE  done
	VMOVUPS (SI)(AX*4), Y0  // a
	VMOVUPS 4(SI)(AX*4), Y1 // b
	VMOVUPS 8(SI)(AX*4), Y2 // c
	VMOVUPS (R8)(AX*4), Y3  // d
	VMOVUPS 8(R8)(AX*4), Y4 // f
	VMOVUPS (R9)(AX*4), Y5  // g
	VMOVUPS 4(R9)(AX*4), Y6 // h
	VMOVUPS 8(R9)(AX*4), Y7 // i

	// right = (c + (f+f)) + i
	VADDPS Y4, Y4, Y8
	VADDPS Y8, Y2, Y8
	VADDPS Y7, Y8, Y8
	// left = (a + (d+d)) + g
	VADDPS Y3, Y3, Y9
	VADDPS Y9, Y0, Y9
	VADDPS Y5, Y9, Y9
	// gx = (right - left) * invDx8
	VSUBPS Y9, Y8, Y8
	VMULPS Y14, Y8, Y8

	// bottom = (g + (h+h)) + i
	VADDPS Y6, Y6, Y10
	VADDPS Y10, Y5, Y10
	VADDPS Y7, Y10, Y10
	// top = (a + (b+b)) + c
	VADDPS Y1, Y1, Y11
	VADDPS Y11, Y0, Y11
	VADDPS Y2, Y11, Y11
	// gy = (bottom - top) * invDy8
	VSUBPS Y11, Y10, Y10
	VMULPS Y15, Y10, Y10

	// sqrt(gx*gx + gy*gy)
	VMULPS  Y8, Y8, Y8
	VMULPS  Y10, Y10, Y10
	VADDPS  Y10, Y8, Y8
	VSQRTPS Y8, Y8
	VMOVUPS Y8, (DI)(AX*4)

	ADDQ $8, AX
	JMP  loop
done:
	VZEROUPPER
	RET
