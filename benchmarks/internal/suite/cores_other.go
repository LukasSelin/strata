//go:build !linux && !windows && !darwin

package suite

// CPUs returns zeros: this platform has no detection yet.
func CPUs() (physical, logical int) { return 0, 0 }
