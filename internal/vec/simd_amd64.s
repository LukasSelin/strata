//go:build amd64

#include "textflag.h"

// absMask clears the sign bit of a float32 when ANDed lanewise.
GLOBL absMask<>(SB), RODATA, $4
DATA absMask<>+0(SB)/4, $0x7fffffff

// nanConst is a canonical quiet NaN, used to force NaN results in Min,
// Max, and Clamp lanes where hardware MINPS/MAXPS would otherwise favor
// whichever operand is not NaN. That hardware quirk would disagree with
// the scalar backend's "either operand NaN implies NaN result" rule.
GLOBL nanConst<>(SB), RODATA, $4
DATA nanConst<>+0(SB)/4, $0x7fc00000

// func addAVX2Asm(dst, a, b *float32, n int)
TEXT ·addAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	XORQ AX, AX
loop_add:
	CMPQ AX, CX
	JGE  done_add
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS (BX)(AX*4), Y1
	VADDPS  Y1, Y0, Y2
	VMOVUPS Y2, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_add
done_add:
	VZEROUPPER
	RET

// func subAVX2Asm(dst, a, b *float32, n int)
TEXT ·subAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	XORQ AX, AX
loop_sub:
	CMPQ AX, CX
	JGE  done_sub
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS (BX)(AX*4), Y1
	VSUBPS  Y1, Y0, Y2
	VMOVUPS Y2, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_sub
done_sub:
	VZEROUPPER
	RET

// func mulAVX2Asm(dst, a, b *float32, n int)
TEXT ·mulAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	XORQ AX, AX
loop_mul:
	CMPQ AX, CX
	JGE  done_mul
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS (BX)(AX*4), Y1
	VMULPS  Y1, Y0, Y2
	VMOVUPS Y2, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_mul
done_mul:
	VZEROUPPER
	RET

// func divAVX2Asm(dst, a, b *float32, n int)
TEXT ·divAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	XORQ AX, AX
loop_div:
	CMPQ AX, CX
	JGE  done_div
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS (BX)(AX*4), Y1
	VDIVPS  Y1, Y0, Y2
	VMOVUPS Y2, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_div
done_div:
	VZEROUPPER
	RET

// func minAVX2Asm(dst, a, b *float32, n int)
TEXT ·minAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	VBROADCASTSS nanConst<>(SB), Y8
	XORQ AX, AX
loop_min:
	CMPQ AX, CX
	JGE  done_min
	VMOVUPS   (SI)(AX*4), Y0
	VMOVUPS   (BX)(AX*4), Y1
	VMINPS    Y1, Y0, Y2
	VCMPPS    $3, Y1, Y0, Y3
	VBLENDVPS Y3, Y8, Y2, Y4
	VMOVUPS   Y4, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_min
done_min:
	VZEROUPPER
	RET

// func maxAVX2Asm(dst, a, b *float32, n int)
TEXT ·maxAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ n+24(FP), CX
	VBROADCASTSS nanConst<>(SB), Y8
	XORQ AX, AX
loop_max:
	CMPQ AX, CX
	JGE  done_max
	VMOVUPS   (SI)(AX*4), Y0
	VMOVUPS   (BX)(AX*4), Y1
	VMAXPS    Y1, Y0, Y2
	VCMPPS    $3, Y1, Y0, Y3
	VBLENDVPS Y3, Y8, Y2, Y4
	VMOVUPS   Y4, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_max
done_max:
	VZEROUPPER
	RET

// func addScalarAVX2Asm(dst, src *float32, n int, value float32)
TEXT ·addScalarAVX2Asm(SB), NOSPLIT, $0-28
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS value+24(FP), Y8
	XORQ AX, AX
loop_addscalar:
	CMPQ AX, CX
	JGE  done_addscalar
	VMOVUPS (SI)(AX*4), Y0
	VADDPS  Y8, Y0, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_addscalar
done_addscalar:
	VZEROUPPER
	RET

// func mulScalarAVX2Asm(dst, src *float32, n int, value float32)
TEXT ·mulScalarAVX2Asm(SB), NOSPLIT, $0-28
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS value+24(FP), Y8
	XORQ AX, AX
loop_mulscalar:
	CMPQ AX, CX
	JGE  done_mulscalar
	VMOVUPS (SI)(AX*4), Y0
	VMULPS  Y8, Y0, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_mulscalar
done_mulscalar:
	VZEROUPPER
	RET

// func clampAVX2Asm(dst, src *float32, n int, lo, hi float32)
TEXT ·clampAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS lo+24(FP), Y8
	VBROADCASTSS hi+28(FP), Y9
	VBROADCASTSS nanConst<>(SB), Y10
	XORQ AX, AX
loop_clamp:
	CMPQ AX, CX
	JGE  done_clamp
	VMOVUPS   (SI)(AX*4), Y0
	VMAXPS    Y8, Y0, Y1
	VMINPS    Y9, Y1, Y2
	VCMPPS    $3, Y0, Y0, Y3
	VBLENDVPS Y3, Y10, Y2, Y4
	VMOVUPS   Y4, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_clamp
done_clamp:
	VZEROUPPER
	RET

// func absAVX2Asm(dst, src *float32, n int)
TEXT ·absAVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS absMask<>(SB), Y8
	XORQ AX, AX
loop_abs:
	CMPQ AX, CX
	JGE  done_abs
	VMOVUPS (SI)(AX*4), Y0
	VANDPS  Y8, Y0, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_abs
done_abs:
	VZEROUPPER
	RET

// func sqrtAVX2Asm(dst, src *float32, n int)
TEXT ·sqrtAVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	XORQ AX, AX
loop_sqrt:
	CMPQ AX, CX
	JGE  done_sqrt
	VMOVUPS (SI)(AX*4), Y0
	VSQRTPS Y0, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ $8, AX
	JMP  loop_sqrt
done_sqrt:
	VZEROUPPER
	RET
