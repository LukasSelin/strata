package suite

import (
	"syscall"
	"unsafe"
)

// ProcessMemory returns the process's current and peak memory, from
// K32GetProcessMemoryInfo, and whether it is known.
func ProcessMemory() (Memory, bool) {
	// PROCESS_MEMORY_COUNTERS_EX: cb and PageFaultCount (DWORD), then
	// SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage,
	// QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
	// QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage and
	// PrivateUsage. PagefileUsage is the commit charge, private bytes.
	var c struct {
		cb, pageFaults                   uint32
		peakWorkingSet, workingSet       uintptr
		_, _, _, _                       uintptr
		pagefileUsage, peakPagefileUsage uintptr
		privateUsage                     uintptr
	}
	c.cb = uint32(unsafe.Sizeof(c))
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("K32GetProcessMemoryInfo")
	if proc.Find() != nil {
		return Memory{}, false
	}
	self, err := syscall.GetCurrentProcess()
	if err != nil {
		return Memory{}, false
	}
	if r, _, _ := proc.Call(uintptr(self), uintptr(unsafe.Pointer(&c)), uintptr(c.cb)); r == 0 {
		return Memory{}, false
	}
	return Memory{
		Private:        uint64(c.pagefileUsage),
		PeakPrivate:    uint64(c.peakPagefileUsage),
		WorkingSet:     uint64(c.workingSet),
		PeakWorkingSet: uint64(c.peakWorkingSet),
	}, true
}
