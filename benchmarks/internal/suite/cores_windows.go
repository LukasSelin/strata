package suite

import (
	"math/bits"
	"syscall"
	"unsafe"
)

// CPUs returns the machine's physical cores and logical processors, or
// zeros if unknown. Unlike runtime.NumCPU it ignores the process's CPU
// affinity. It walks the RelationProcessorCore records of
// GetLogicalProcessorInformationEx: one per core, each with group affinity
// masks holding one bit per logical processor.
func CPUs() (physical, logical int) {
	const relationProcessorCore = 0
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GetLogicalProcessorInformationEx")
	if proc.Find() != nil {
		return 0, 0
	}
	var size uint32
	_, _, _ = proc.Call(relationProcessorCore, 0, uintptr(unsafe.Pointer(&size))) // fails by design: it reports the size
	if size == 0 {
		return 0, 0
	}
	buf := make([]byte, size)
	r, _, _ := proc.Call(relationProcessorCore, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)))
	if r == 0 {
		return 0, 0
	}
	// Record: Relationship uint32, Size uint32, then PROCESSOR_RELATIONSHIP:
	// Flags, EfficiencyClass, Reserved[20] bytes, GroupCount uint16 at
	// offset 30, and GroupCount GROUP_AFFINITY (Mask uintptr, Group uint16,
	// Reserved [3]uint16) from offset 32.
	for off := 0; off+32 <= int(size); physical++ {
		recSize := int(*(*uint32)(unsafe.Pointer(&buf[off+4])))
		if recSize < 32 || off+recSize > int(size) {
			return 0, 0
		}
		groups := int(*(*uint16)(unsafe.Pointer(&buf[off+30])))
		for g := range groups {
			m := off + 32 + g*16
			if m+8 > off+recSize {
				break
			}
			logical += bits.OnesCount64(uint64(*(*uintptr)(unsafe.Pointer(&buf[m]))))
		}
		off += recSize
	}
	return physical, logical
}
