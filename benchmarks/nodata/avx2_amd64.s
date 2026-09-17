//go:build amd64

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// func addSentinelAVX2Asm(dst, a, b *float32, n int, nd float32)
// dst = (a == nd || b == nd) ? nd : a + b, as compute + compare + blend.
TEXT ·addSentinelAVX2Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	VBROADCASTSS nd+32(FP), Y8
	XORQ AX, AX
loop_adds:
	CMPQ AX, CX
	JGE  done_adds
	VMOVUPS   (SI)(AX*4), Y0
	VMOVUPS   (BX)(AX*4), Y1
	VADDPS    Y1, Y0, Y2
	VCMPPS    $0, Y8, Y0, Y3
	VCMPPS    $0, Y8, Y1, Y4
	VORPS     Y4, Y3, Y3
	VBLENDVPS Y3, Y8, Y2, Y2
	VMOVUPS   Y2, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_adds
done_adds:
	VZEROUPPER
	RET

// Horn row kernels. r0, r1, r2 point at column x-1 of rows y-1, y, y+1;
// dst points at column x. n must be a multiple of 8. Evaluation order
// matches horn() in slope.go exactly.
//
// Register use: Y0..Y7 = z1 z2 z3 z4 z6 z7 z8 z9, Y14/Y15 = invx/invy,
// Y8 = x-accumulator, Y10 = scratch accumulator, Y11 = y-accumulator,
// Y9 = temporary.

// func slopeRowAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy float32)
TEXT ·slopeRowAVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ r0+8(FP), SI
	MOVQ r1+16(FP), BX
	MOVQ r2+24(FP), DX
	MOVQ n+32(FP), CX
	VBROADCASTSS invx+40(FP), Y14
	VBROADCASTSS invy+44(FP), Y15
	XORQ AX, AX
loop_slope:
	CMPQ AX, CX
	JGE  done_slope
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS 4(SI)(AX*4), Y1
	VMOVUPS 8(SI)(AX*4), Y2
	VMOVUPS (BX)(AX*4), Y3
	VMOVUPS 8(BX)(AX*4), Y4
	VMOVUPS (DX)(AX*4), Y5
	VMOVUPS 4(DX)(AX*4), Y6
	VMOVUPS 8(DX)(AX*4), Y7

	// gx = (((z3+z9) + (z6+z6)) - ((z1+z7) + (z4+z4))) * invx
	VADDPS Y7, Y2, Y8
	VADDPS Y4, Y4, Y9
	VADDPS Y9, Y8, Y8
	VADDPS Y5, Y0, Y10
	VADDPS Y3, Y3, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y8, Y8
	VMULPS Y14, Y8, Y8

	// gy = (((z7+z9) + (z8+z8)) - ((z1+z3) + (z2+z2))) * invy
	VADDPS Y7, Y5, Y11
	VADDPS Y6, Y6, Y9
	VADDPS Y9, Y11, Y11
	VADDPS Y2, Y0, Y10
	VADDPS Y1, Y1, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y11, Y11
	VMULPS Y15, Y11, Y11

	// sqrt(gx*gx + gy*gy)
	VMULPS  Y8, Y8, Y8
	VMULPS  Y11, Y11, Y11
	VADDPS  Y11, Y8, Y8
	VSQRTPS Y8, Y8
	VMOVUPS Y8, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_slope
done_slope:
	VZEROUPPER
	RET

// func slopeRowNaNAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy float32)
// As slopeRowAVX2Asm plus the z5*0 term of hornNaN. Y12 = 0.
TEXT ·slopeRowNaNAVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ r0+8(FP), SI
	MOVQ r1+16(FP), BX
	MOVQ r2+24(FP), DX
	MOVQ n+32(FP), CX
	VBROADCASTSS invx+40(FP), Y14
	VBROADCASTSS invy+44(FP), Y15
	VXORPS Y12, Y12, Y12
	XORQ AX, AX
loop_slopen:
	CMPQ AX, CX
	JGE  done_slopen
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS 4(SI)(AX*4), Y1
	VMOVUPS 8(SI)(AX*4), Y2
	VMOVUPS (BX)(AX*4), Y3
	VMOVUPS 8(BX)(AX*4), Y4
	VMOVUPS (DX)(AX*4), Y5
	VMOVUPS 4(DX)(AX*4), Y6
	VMOVUPS 8(DX)(AX*4), Y7

	// gx = (((z3+z9) + (z6+z6)) - ((z1+z7) + (z4+z4))) * invx
	VADDPS Y7, Y2, Y8
	VADDPS Y4, Y4, Y9
	VADDPS Y9, Y8, Y8
	VADDPS Y5, Y0, Y10
	VADDPS Y3, Y3, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y8, Y8
	VMULPS Y14, Y8, Y8

	// gy = (((z7+z9) + (z8+z8)) - ((z1+z3) + (z2+z2))) * invy
	VADDPS Y7, Y5, Y11
	VADDPS Y6, Y6, Y9
	VADDPS Y9, Y11, Y11
	VADDPS Y2, Y0, Y10
	VADDPS Y1, Y1, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y11, Y11
	VMULPS Y15, Y11, Y11

	// sqrt(gx*gx + gy*gy)
	VMULPS  Y8, Y8, Y8
	VMULPS  Y11, Y11, Y11
	VADDPS  Y11, Y8, Y8
	VMOVUPS 4(BX)(AX*4), Y9
	VMULPS  Y12, Y9, Y9
	VADDPS  Y9, Y8, Y8
	VSQRTPS Y8, Y8
	VMOVUPS Y8, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_slopen
done_slopen:
	VZEROUPPER
	RET

// func slopeRowSentinelAVX2Asm(dst, r0, r1, r2 *float32, n int, invx, invy, nd float32)
// Same arithmetic, plus nine EQ compares against nd ORed into Y12 and a
// final blend. Y13 = nd.
TEXT ·slopeRowSentinelAVX2Asm(SB), NOSPLIT, $0-52
	MOVQ dst+0(FP), DI
	MOVQ r0+8(FP), SI
	MOVQ r1+16(FP), BX
	MOVQ r2+24(FP), DX
	MOVQ n+32(FP), CX
	VBROADCASTSS invx+40(FP), Y14
	VBROADCASTSS invy+44(FP), Y15
	VBROADCASTSS nd+48(FP), Y13
	XORQ AX, AX
loop_slopes:
	CMPQ AX, CX
	JGE  done_slopes
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS 4(SI)(AX*4), Y1
	VMOVUPS 8(SI)(AX*4), Y2
	VMOVUPS (BX)(AX*4), Y3
	VMOVUPS 8(BX)(AX*4), Y4
	VMOVUPS (DX)(AX*4), Y5
	VMOVUPS 4(DX)(AX*4), Y6
	VMOVUPS 8(DX)(AX*4), Y7

	VCMPPS $0, Y13, Y0, Y12
	VCMPPS $0, Y13, Y1, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y2, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y3, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y4, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y5, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y6, Y9
	VORPS  Y9, Y12, Y12
	VCMPPS $0, Y13, Y7, Y9
	VORPS  Y9, Y12, Y12
	VMOVUPS 4(BX)(AX*4), Y9
	VCMPPS $0, Y13, Y9, Y9
	VORPS  Y9, Y12, Y12

	VADDPS Y7, Y2, Y8
	VADDPS Y4, Y4, Y9
	VADDPS Y9, Y8, Y8
	VADDPS Y5, Y0, Y10
	VADDPS Y3, Y3, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y8, Y8
	VMULPS Y14, Y8, Y8

	VADDPS Y7, Y5, Y11
	VADDPS Y6, Y6, Y9
	VADDPS Y9, Y11, Y11
	VADDPS Y2, Y0, Y10
	VADDPS Y1, Y1, Y9
	VADDPS Y9, Y10, Y10
	VSUBPS Y10, Y11, Y11
	VMULPS Y15, Y11, Y11

	VMULPS    Y8, Y8, Y8
	VMULPS    Y11, Y11, Y11
	VADDPS    Y11, Y8, Y8
	VSQRTPS   Y8, Y8
	VBLENDVPS Y12, Y13, Y8, Y8
	VMOVUPS   Y8, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_slopes
done_slopes:
	VZEROUPPER
	RET
