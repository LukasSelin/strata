package suite

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// ProcessMemory returns the process's current and peak resident memory,
// VmRSS and VmHWM from /proc/self/status, and whether they are known.
// Private memory is not reported.
func ProcessMemory() (Memory, bool) {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return Memory{}, false
	}
	defer f.Close()
	var m Memory
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, value, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		kb, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimSpace(value), " kB"), 10, 64)
		if err != nil {
			continue
		}
		switch key {
		case "VmRSS":
			m.WorkingSet = kb << 10
		case "VmHWM":
			m.PeakWorkingSet = kb << 10
		}
	}
	return m, m.PeakWorkingSet > 0
}
