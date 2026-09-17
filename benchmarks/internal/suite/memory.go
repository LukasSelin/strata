package suite

// Memory is the process's memory use as the operating system reports it,
// in bytes. A zero field is unknown on this platform.
type Memory struct {
	// Private is the memory committed to the process alone (Windows
	// private bytes), and PeakPrivate its maximum so far.
	Private, PeakPrivate uint64
	// WorkingSet is the process's resident memory, and PeakWorkingSet
	// its maximum so far (Linux VmRSS and VmHWM).
	WorkingSet, PeakWorkingSet uint64
}
