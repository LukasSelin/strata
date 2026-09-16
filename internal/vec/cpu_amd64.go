//go:build amd64

package vec

// cpuidAMD64 and xgetbvAMD64 are implemented in cpu_amd64.s.
func cpuidAMD64(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbvAMD64() (eax, edx uint32)

// hasAVX2 reports whether the CPU supports AVX2 and the OS has enabled
// the extended (YMM) register state needed to use it. Both checks are
// required: a CPU can support AVX2 while the OS has not opted in via
// XSETBV, in which case using YMM registers would fault.
func hasAVX2() bool {
	_, _, ecx1, _ := cpuidAMD64(1, 0)
	const (
		osxsaveBit = 1 << 27
		avxBit     = 1 << 28
	)
	if ecx1&(osxsaveBit|avxBit) != osxsaveBit|avxBit {
		return false
	}

	eax, _ := xgetbvAMD64()
	const xmmAndYmmState = 0x6
	if eax&xmmAndYmmState != xmmAndYmmState {
		return false
	}

	_, ebx7, _, _ := cpuidAMD64(7, 0)
	const avx2Bit = 1 << 5
	return ebx7&avx2Bit != 0
}
