//go:build amd64

#include "textflag.h"

// func cpuidAMD64(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuidAMD64(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbvAMD64() (eax, edx uint32)
TEXT ·xgetbvAMD64(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET
