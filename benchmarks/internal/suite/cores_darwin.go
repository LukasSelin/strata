package suite

import "syscall"

// CPUs returns the machine's physical cores and logical processors, or
// zeros if unknown.
func CPUs() (physical, logical int) {
	p, err := syscall.SysctlUint32("hw.physicalcpu")
	if err != nil {
		return 0, 0
	}
	l, err := syscall.SysctlUint32("hw.logicalcpu")
	if err != nil {
		return int(p), 0
	}
	return int(p), int(l)
}
