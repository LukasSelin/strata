//go:build !linux && !windows

package suite

// ProcessMemory reports that memory use is unknown on this platform.
func ProcessMemory() (Memory, bool) { return Memory{}, false }
