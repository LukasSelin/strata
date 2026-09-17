// Command nodatatable turns `go test -bench` output from benchmarks/nodata
// into Markdown tables: one per workload and size, one row per variant,
// one column per NoData pattern. Each cell is the median ns/cell over all
// -count runs, with cells/sec in millions in parentheses.
//
//	go test ./benchmarks/nodata -run '^$' -bench . -benchmem -count 6 -timeout 3h | tee bench.txt
//	go run ./benchmarks/nodata/cmd/nodatatable < bench.txt
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

var nameRE = regexp.MustCompile(`^Benchmark(\w+)/size=(\d+)/nodata=([\w-]+)/(\S+?)(?:-\d+)?$`)

type key struct{ workload, size, pattern, variant string }

type samples struct {
	nsPerCell []float64
	bytes     float64
	allocs    float64
}

func main() {
	data := map[key]*samples{}
	var workloads, sizes, patterns []string
	variants := map[string][]string{} // workload -> variants in order
	addUnique := func(list *[]string, v string) {
		if !slices.Contains(*list, v) {
			*list = append(*list, v)
		}
	}

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "cpu:") || strings.HasPrefix(line, "goos:") || strings.HasPrefix(line, "goarch:") {
			fmt.Println(line + "  ")
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		m := nameRE.FindStringSubmatch(fields[0])
		if m == nil {
			continue
		}
		k := key{m[1], m[2], m[3], m[4]}
		s := data[k]
		if s == nil {
			s = &samples{}
			data[k] = s
			addUnique(&workloads, k.workload)
			addUnique(&sizes, k.size)
			addUnique(&patterns, k.pattern)
			vs := variants[k.workload]
			addUnique(&vs, k.variant)
			variants[k.workload] = vs
		}
		// fields: name iters (value unit)...
		for i := 2; i+1 < len(fields); i += 2 {
			v, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				continue
			}
			switch fields[i+1] {
			case "ns/cell":
				s.nsPerCell = append(s.nsPerCell, v)
			case "B/op":
				s.bytes = math.Max(s.bytes, v)
			case "allocs/op":
				s.allocs = math.Max(s.allocs, v)
			}
		}
	}
	if err := sc.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	for _, w := range workloads {
		for _, size := range sizes {
			fmt.Printf("\n#### %s, %s×%s\n\n", w, size, size)
			fmt.Print("| variant |")
			for _, p := range patterns {
				fmt.Printf(" %s |", p)
			}
			fmt.Println(" B/op | allocs/op |")
			fmt.Print("|---|")
			for range patterns {
				fmt.Print("---:|")
			}
			fmt.Println("---:|---:|")

			var spreads []float64
			for _, v := range variants[w] {
				fmt.Printf("| %s |", v)
				var bytes, allocs float64
				for _, p := range patterns {
					s := data[key{w, size, p, v}]
					if s == nil || len(s.nsPerCell) == 0 {
						fmt.Print(" – |")
						continue
					}
					med := median(s.nsPerCell)
					spreads = append(spreads, (slices.Max(s.nsPerCell)-slices.Min(s.nsPerCell))/med)
					fmt.Printf(" %s (%s) |", fmtNs(med), fmtRate(1e3/med))
					bytes = math.Max(bytes, s.bytes)
					allocs = math.Max(allocs, s.allocs)
				}
				fmt.Printf(" %g | %g |\n", bytes, allocs)
			}
			if len(spreads) > 0 {
				fmt.Printf("\nCells: median ns/cell (million cells/s). Run-to-run spread (max−min)/median: median %.0f%%, worst %.0f%%.\n",
					100*median(spreads), 100*slices.Max(spreads))
			}
		}
	}
}

func median(v []float64) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func fmtNs(v float64) string {
	switch {
	case v < 1:
		return strconv.FormatFloat(v, 'f', 3, 64)
	case v < 10:
		return strconv.FormatFloat(v, 'f', 2, 64)
	default:
		return strconv.FormatFloat(v, 'f', 1, 64)
	}
}

func fmtRate(v float64) string {
	if v >= 100 {
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
	return strconv.FormatFloat(v, 'f', 1, 64)
}
