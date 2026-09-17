// Command stratabench turns `go test -bench` output from the benchmark
// suite (benchmarks/internal/suite) into a Markdown summary in the terms
// of DESIGN.md §42: million cells per second for the scalar and SIMD
// backends at each raster size, and the SIMD/scalar speedup.
//
//	GOEXPERIMENT=simd go test ./benchmarks/algebra -run '^$' -bench . -count 5 -timeout 2h > bench.txt
//	go run ./benchmarks/cmd/stratabench < bench.txt
//
// It reads standard input, or the files named as arguments, and needs no
// other tools. Every figure is the median over the -count runs, as
// benchstat reports it.
//
// For each package it prints the machine configuration, a §42 headline for
// 4096 × 4096, and one table per operation. Under each table every mask
// setting is classified by the workload classes of DESIGN.md §28, from the
// measured throughput of the SIMD backend (the scalar one if the build has
// no SIMD):
//
//   - compute-bound: throughput at every size stays within 20% of the
//     smallest size's, so the working set does not matter;
//   - memory-bandwidth-bound from the first size where throughput drops
//     below 80% of the smallest size's, if the SIMD/scalar speedup at the
//     largest size is under two thirds of the speedup at the smallest:
//     both backends wait on memory, so extra lanes stop paying;
//   - cache-bound from that size otherwise: throughput falls with the
//     working set, but SIMD keeps most of its lead.
//
// Without SIMD results a drop is reported as working-set-bound, since the
// speedup is what tells cache from memory bandwidth.
//
// Packages whose benchmark names end in a tiles=<shape> level (the engine
// category) get a different table per operation, with a row per size,
// mask and tile shape and a column per worker count; see renderTiled.
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

func main() {
	var in []io.Reader
	for _, name := range os.Args[1:] {
		f, err := os.Open(name)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer f.Close()
		in = append(in, f)
	}
	if len(in) == 0 {
		in = append(in, os.Stdin)
	}
	res, err := parse(io.MultiReader(in...))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(res.entries) == 0 {
		fmt.Fprintln(os.Stderr, "stratabench: no suite benchmark results in input")
		os.Exit(1)
	}
	render(os.Stdout, res)
}

// key identifies one leaf of the suite matrix.
type key struct {
	pkg, op string
	size    int
	mask    string // "off" or "on"
	backend string // "scalar" or "simd"
	workers int
	tiles   string // "" without a tiles level
}

type results struct {
	config  map[string]string // first value seen per key
	pkgs    []string          // short package names in input order
	ops     map[string][]string
	tiled   map[string]bool              // packages with a tiles level
	entries map[key]map[string][]float64 // unit -> one value per run
}

var (
	configRE = regexp.MustCompile(`^([a-z][a-z0-9-]*):\s*(.*?)\s*$`)
	benchRE  = regexp.MustCompile(`^Benchmark(\w+)/size=(\d+)/mask=(on|off)/backend=(\w+)/workers=(\d+)(?:/tiles=(\w+))?(?:-\d+)?$`)
)

func parse(r io.Reader) (*results, error) {
	res := &results{
		config:  map[string]string{},
		ops:     map[string][]string{},
		tiled:   map[string]bool{},
		entries: map[key]map[string][]float64{},
	}
	pkg := ""
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if m := configRE.FindStringSubmatch(line); m != nil {
			if m[1] == "pkg" {
				pkg = m[2][strings.LastIndex(m[2], "/")+1:]
			}
			if _, ok := res.config[m[1]]; !ok {
				res.config[m[1]] = m[2]
			}
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		m := benchRE.FindStringSubmatch(fields[0])
		if m == nil {
			continue
		}
		size, _ := strconv.Atoi(m[2])
		workers, _ := strconv.Atoi(m[5])
		k := key{pkg, m[1], size, m[3], m[4], workers, m[6]}
		units := res.entries[k]
		if units == nil {
			units = map[string][]float64{}
			res.entries[k] = units
			if !slices.Contains(res.pkgs, pkg) {
				res.pkgs = append(res.pkgs, pkg)
			}
			if !slices.Contains(res.ops[pkg], k.op) {
				res.ops[pkg] = append(res.ops[pkg], k.op)
			}
			res.tiled[pkg] = res.tiled[pkg] || k.tiles != ""
		}
		// fields: name iterations (value unit)...
		for i := 2; i+1 < len(fields); i += 2 {
			v, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				continue
			}
			units[fields[i+1]] = append(units[fields[i+1]], v)
		}
	}
	return res, sc.Err()
}

// median of the unit's runs, and whether there were any.
func (res *results) median(k key, unit string) (float64, bool) {
	v := res.entries[k][unit]
	if len(v) == 0 {
		return 0, false
	}
	return median(v), true
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

func render(w io.Writer, res *results) {
	c := res.config
	fmt.Fprintf(w, "| | |\n|---|---|\n")
	fmt.Fprintf(w, "| CPU | %s |\n", orUnknown(c["cpu"]))
	fmt.Fprintf(w, "| Cores | %s physical, %s logical; %s usable by the process, GOMAXPROCS %s |\n",
		orUnknown(c["physicalcores"]), orUnknown(c["logicalcpus"]), orUnknown(c["usablecpus"]), orUnknown(c["gomaxprocs"]))
	goLine := fmt.Sprintf("%s %s/%s", orUnknown(c["goversion"]), c["goos"], c["goarch"])
	if v := c["goamd64"]; v != "" {
		goLine += ", GOAMD64=" + v
	}
	goLine += ", GOEXPERIMENT=" + orUnknown(c["goexperiment"])
	fmt.Fprintf(w, "| Go | %s |\n", goLine)
	var kernels []string
	for k, v := range c {
		if name, ok := strings.CutPrefix(k, "kernels-"); ok {
			kernels = append(kernels, name+": "+v)
		}
	}
	slices.Sort(kernels)
	fmt.Fprintf(w, "| Kernels | %s |\n", orUnknown(strings.Join(kernels, ", ")))
	runs := 0
	for _, units := range res.entries {
		runs = max(runs, len(units["Mcells/s"]))
	}
	fmt.Fprintf(w, "| Runs | %d per benchmark, medians shown |\n", runs)

	for _, pkg := range res.pkgs {
		fmt.Fprintf(w, "\n## %s\n", pkg)
		if res.tiled[pkg] {
			renderTiled(w, res, pkg)
			continue
		}
		renderHeadline(w, res, pkg, 4096)
		for _, op := range res.ops[pkg] {
			renderOp(w, res, pkg, op)
		}
	}
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

// renderHeadline prints the §42 block for one size, unmasked, one worker.
func renderHeadline(w io.Writer, res *results, pkg string, size int) {
	var lines []string
	for _, op := range res.ops[pkg] {
		k := key{pkg, op, size, "off", "scalar", 1, ""}
		scalar, okS := res.median(k, "Mcells/s")
		k.backend = "simd"
		simd, okV := res.median(k, "Mcells/s")
		if !okS && !okV {
			continue
		}
		speedup := "–"
		if okS && okV {
			speedup = fmtSpeedup(simd / scalar)
		}
		lines = append(lines, fmt.Sprintf("%-8s %9s %9s %12s   %s", op,
			fmtRateOpt(scalar, okS), fmtRateOpt(simd, okV), speedup, workersNote))
	}
	if len(lines) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%d × %d raster, no mask, M cells/sec:\n\n```text\n", size, size)
	fmt.Fprintf(w, "%-8s %9s %9s %12s   %s\n", "", "scalar", "SIMD", "SIMD/scalar", "SIMD + workers")
	for _, l := range lines {
		fmt.Fprintln(w, l)
	}
	fmt.Fprint(w, "```\n")
}

const workersNote = "not measured yet (tile engine, STRATA-8/9)"

func renderOp(w io.Writer, res *results, pkg, op string) {
	var sizes []int
	var workerCounts []int // counts above 1 with SIMD results
	for k := range res.entries {
		if k.pkg != pkg || k.op != op {
			continue
		}
		if !slices.Contains(sizes, k.size) {
			sizes = append(sizes, k.size)
		}
		if k.workers > 1 && k.backend == "simd" && !slices.Contains(workerCounts, k.workers) {
			workerCounts = append(workerCounts, k.workers)
		}
	}
	slices.Sort(sizes)
	slices.Sort(workerCounts)

	fmt.Fprintf(w, "\n### %s\n\n", op)
	fmt.Fprint(w, "| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar |")
	for _, n := range workerCounts {
		fmt.Fprintf(w, " SIMD + %d workers M cells/s |", n)
	}
	fmt.Fprint(w, " SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |\n|---|---|---:|---:|---:|")
	for range workerCounts {
		fmt.Fprint(w, "---:|")
	}
	fmt.Fprint(w, "---:|---:|---:|---:|\n")

	var spreads []float64
	for _, mask := range []string{"off", "on"} {
		for _, size := range sizes {
			k := key{pkg, op, size, mask, "scalar", 1, ""}
			if res.entries[k] == nil {
				k.backend = "simd"
				if res.entries[k] == nil {
					continue
				}
			}
			ks := key{pkg, op, size, mask, "scalar", 1, ""}
			kv := key{pkg, op, size, mask, "simd", 1, ""}
			scalar, okS := res.median(ks, "Mcells/s")
			simd, okV := res.median(kv, "Mcells/s")
			speedup := "–"
			if okS && okV {
				speedup = fmtSpeedup(simd / scalar)
			}
			fmt.Fprintf(w, "| %d × %d | %s | %s | %s | %s |", size, size, mask,
				fmtRateOpt(scalar, okS), fmtRateOpt(simd, okV), speedup)
			for _, n := range workerCounts {
				v, ok := res.median(key{pkg, op, size, mask, "simd", n, ""}, "Mcells/s")
				fmt.Fprintf(w, " %s |", fmtRateOpt(v, ok))
			}
			nsCell, okN := res.median(kv, "ns/cell")
			gbS, okGS := res.median(ks, "GB/s")
			gbV, okGV := res.median(kv, "GB/s")
			allocs := 0.0
			for _, kk := range []key{ks, kv} {
				for _, v := range res.entries[kk]["allocs/op"] {
					allocs = max(allocs, v)
				}
				if v := res.entries[kk]["Mcells/s"]; len(v) > 1 {
					spreads = append(spreads, (slices.Max(v)-slices.Min(v))/median(v))
				}
			}
			fmt.Fprintf(w, " %s | %s | %s | %g |\n", fmtNsOpt(nsCell, okN),
				fmtGBOpt(gbS, okGS), fmtGBOpt(gbV, okGV), allocs)
		}
	}
	fmt.Fprintln(w)
	if len(spreads) > 0 {
		fmt.Fprintf(w, "Run-to-run spread of M cells/s, (max − min)/median: median %.0f%%, worst %.0f%%.\n\n",
			100*median(spreads), 100*slices.Max(spreads))
	}
	for _, mask := range []string{"off", "on"} {
		if note := classify(res, pkg, op, mask, "", sizes); note != "" {
			fmt.Fprintf(w, "- mask=%s: %s\n", mask, note)
		}
	}
}

// Thresholds of the §28 classification; see the package comment.
const (
	holdFraction    = 0.80
	speedupFraction = 2.0 / 3
)

// classify labels one operation, mask setting and tile shape by §28
// workload class, from its one-worker results.
func classify(res *results, pkg, op, mask, tiles string, sizes []int) string {
	type point struct {
		size                 int
		scalar, simd, gbSIMD float64
		hasSIMD              bool
	}
	var pts []point
	allSIMD := true
	for _, size := range sizes {
		ks := key{pkg, op, size, mask, "scalar", 1, tiles}
		kv := key{pkg, op, size, mask, "simd", 1, tiles}
		scalar, okS := res.median(ks, "Mcells/s")
		if !okS {
			continue
		}
		simd, okV := res.median(kv, "Mcells/s")
		gb, _ := res.median(kv, "GB/s")
		allSIMD = allSIMD && okV
		pts = append(pts, point{size, scalar, simd, gb, okV})
	}
	if len(pts) < 2 {
		return ""
	}
	primary := func(p point) float64 {
		if allSIMD {
			return p.simd
		}
		return p.scalar
	}
	backend := "SIMD"
	if !allSIMD {
		backend = "scalar"
	}
	first, last := pts[0], pts[len(pts)-1]
	knee := -1
	for i, p := range pts {
		if primary(p) < holdFraction*primary(first) {
			knee = i
			break
		}
	}
	sq := func(n int) string { return fmt.Sprintf("%d²", n) }
	if knee < 0 {
		note := fmt.Sprintf("compute-bound: %s throughput stays within %.0f%% of %s's up to %s (%s → %s M cells/s)",
			backend, 100*(1-holdFraction), sq(first.size), sq(last.size),
			fmtRate(primary(first)), fmtRate(primary(last)))
		if allSIMD {
			note += fmt.Sprintf(", SIMD/scalar %s → %s", fmtSpeedup(first.simd/first.scalar), fmtSpeedup(last.simd/last.scalar))
		}
		return note + "."
	}
	drop := fmt.Sprintf("%s throughput falls to %.0f%% of %s's by %s (%s → %s M cells/s)",
		backend, 100*primary(last)/primary(first), sq(first.size), sq(last.size),
		fmtRate(primary(first)), fmtRate(primary(last)))
	if !allSIMD {
		return fmt.Sprintf("working-set-bound from %s: %s; SIMD results are needed to tell cache- from memory-bandwidth-bound.",
			sq(pts[knee].size), drop)
	}
	sp0, spN := first.simd/first.scalar, last.simd/last.scalar
	if spN >= speedupFraction*sp0 {
		return fmt.Sprintf("cache-bound from %s: %s, but SIMD keeps its lead (SIMD/scalar %s → %s).",
			sq(pts[knee].size), drop, fmtSpeedup(sp0), fmtSpeedup(spN))
	}
	note := fmt.Sprintf("memory-bandwidth-bound from %s: %s and the SIMD/scalar speedup flattens (%s → %s)",
		sq(pts[knee].size), drop, fmtSpeedup(sp0), fmtSpeedup(spN))
	var gbs []string
	for _, p := range pts[knee:] {
		gbs = append(gbs, fmt.Sprintf("%s GB/s at %s", fmtGB(p.gbSIMD), sq(p.size)))
	}
	return note + "; SIMD moves " + strings.Join(gbs, ", ") + "."
}

func fmtRate(v float64) string {
	if v >= 100 {
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
	return strconv.FormatFloat(v, 'f', 1, 64)
}

func fmtRateOpt(v float64, ok bool) string {
	if !ok {
		return "–"
	}
	return fmtRate(v)
}

func fmtSpeedup(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) + "×" }

func fmtNsOpt(v float64, ok bool) string {
	if !ok {
		return "–"
	}
	switch {
	case v < 1:
		return strconv.FormatFloat(v, 'f', 3, 64)
	case v < 10:
		return strconv.FormatFloat(v, 'f', 2, 64)
	default:
		return strconv.FormatFloat(v, 'f', 1, 64)
	}
}

func fmtGB(v float64) string {
	if v >= 10 {
		return strconv.FormatFloat(v, 'f', 1, 64)
	}
	return strconv.FormatFloat(v, 'f', 2, 64)
}

func fmtGBOpt(v float64, ok bool) string {
	if !ok {
		return "–"
	}
	return fmtGB(v)
}
