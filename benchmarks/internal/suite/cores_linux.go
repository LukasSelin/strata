package suite

import (
	"bufio"
	"os"
	"strings"
)

// CPUs returns the machine's physical cores and logical processors, or
// zeros if unknown. Unlike runtime.NumCPU it ignores the process's CPU
// affinity. It reads /proc/cpuinfo: logical processors are "processor"
// entries, physical cores distinct (physical id, core id) pairs.
func CPUs() (physical, logical int) {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	cores := map[[2]string]bool{}
	var id [2]string
	flush := func() {
		if id[1] != "" {
			cores[id] = true
		}
		id = [2]string{}
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, value, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			flush()
			continue
		}
		switch strings.TrimSpace(key) {
		case "processor":
			logical++
		case "physical id":
			id[0] = strings.TrimSpace(value)
		case "core id":
			id[1] = strings.TrimSpace(value)
		}
	}
	flush()
	return len(cores), logical
}
