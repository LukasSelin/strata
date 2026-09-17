package main

import (
	"fmt"
	"io"
	"slices"
	"strings"
)

// Tile shapes of the engine category (benchmarks/engine): plain is the
// plain function, the one-worker reference; strips and 256x256 run
// through the engine.
const (
	tilesPlain  = "plain"
	tilesStrips = "strips"
)

// renderTiled prints a package with a tiles level: a §42 headline whose
// worker columns are the strips shape, and one table per operation.
func renderTiled(w io.Writer, res *results, pkg string) {
	workers := res.workerCounts(pkg, "")
	renderTiledHeadline(w, res, pkg, 4096, workers)
	for _, op := range res.ops[pkg] {
		renderTiledOp(w, res, pkg, op)
	}
}

// workerCounts returns the SIMD worker counts above 1 of a package, or of
// one operation if op is not empty.
func (res *results) workerCounts(pkg, op string) []int {
	var out []int
	for k := range res.entries {
		if k.pkg == pkg && (op == "" || k.op == op) && k.workers > 1 && k.backend == "simd" && !slices.Contains(out, k.workers) {
			out = append(out, k.workers)
		}
	}
	slices.Sort(out)
	return out
}

// tileShapes returns an operation's tile shapes: plain, strips, then the
// others by name.
func (res *results) tileShapes(pkg, op string) []string {
	var out []string
	for k := range res.entries {
		if k.pkg == pkg && k.op == op && !slices.Contains(out, k.tiles) {
			out = append(out, k.tiles)
		}
	}
	slices.SortFunc(out, func(a, b string) int {
		rank := func(s string) int {
			switch s {
			case tilesPlain:
				return 0
			case tilesStrips:
				return 1
			}
			return 2
		}
		if ra, rb := rank(a), rank(b); ra != rb {
			return ra - rb
		}
		return strings.Compare(a, b)
	})
	return out
}

func renderTiledHeadline(w io.Writer, res *results, pkg string, size int, workers []int) {
	var lines []string
	for _, op := range res.ops[pkg] {
		k := key{pkg, op, size, "off", "scalar", 1, tilesPlain}
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
		line := fmt.Sprintf("%-10s %9s %9s %12s", op, fmtRateOpt(scalar, okS), fmtRateOpt(simd, okV), speedup)
		for _, n := range workers {
			v, ok := res.median(key{pkg, op, size, "off", "simd", n, tilesStrips}, "Mcells/s")
			line += fmt.Sprintf(" %19s", fmtRateOpt(v, ok))
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%d × %d raster, no mask, M cells/sec (workers run the strips shape):\n\n```text\n", size, size)
	head := fmt.Sprintf("%-10s %9s %9s %12s", "", "scalar", "SIMD", "SIMD/scalar")
	for _, n := range workers {
		head += fmt.Sprintf(" %19s", fmt.Sprintf("SIMD + %d workers", n))
	}
	fmt.Fprintln(w, head)
	for _, l := range lines {
		fmt.Fprintln(w, l)
	}
	fmt.Fprint(w, "```\n")
}

func renderTiledOp(w io.Writer, res *results, pkg, op string) {
	var sizes []int
	for k := range res.entries {
		if k.pkg == pkg && k.op == op && !slices.Contains(sizes, k.size) {
			sizes = append(sizes, k.size)
		}
	}
	slices.Sort(sizes)
	workers := res.workerCounts(pkg, op)
	shapes := res.tileShapes(pkg, op)

	fmt.Fprintf(w, "\n### %s\n\n", op)
	fmt.Fprint(w, "| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain |")
	for _, n := range workers {
		fmt.Fprintf(w, " SIMD + %d workers M cells/s |", n)
	}
	fmt.Fprint(w, " scaling | SIMD GB/s, most workers | allocs/op |\n|---|---|---|---:|---:|---:|")
	for range workers {
		fmt.Fprint(w, "---:|")
	}
	fmt.Fprint(w, "---:|---:|---:|\n")

	var spreads []float64
	for _, mask := range []string{"off", "on"} {
		for _, size := range sizes {
			plain, okPlain := res.median(key{pkg, op, size, mask, "simd", 1, tilesPlain}, "Mcells/s")
			for _, tiles := range shapes {
				ks := key{pkg, op, size, mask, "scalar", 1, tiles}
				kv := key{pkg, op, size, mask, "simd", 1, tiles}
				scalar, okS := res.median(ks, "Mcells/s")
				simd, okV := res.median(kv, "Mcells/s")
				if !okS && !okV {
					continue
				}
				vsPlain := "–"
				if tiles != tilesPlain && okV && okPlain {
					vsPlain = fmt.Sprintf("%+.0f%%", 100*(simd/plain-1))
				}
				fmt.Fprintf(w, "| %d × %d | %s | %s | %s | %s | %s |", size, size, mask, tiles,
					fmtRateOpt(scalar, okS), fmtRateOpt(simd, okV), vsPlain)
				cases := []key{ks, kv}
				best, bestN := simd, 1
				gb, okGB := res.median(kv, "GB/s")
				for _, n := range workers {
					kn := key{pkg, op, size, mask, "simd", n, tiles}
					v, ok := res.median(kn, "Mcells/s")
					fmt.Fprintf(w, " %s |", fmtRateOpt(v, ok))
					if ok {
						cases = append(cases, kn)
						if v > best {
							best, bestN = v, n
						}
						gb, okGB = res.median(kn, "GB/s")
					}
				}
				scaling := "–"
				if okV && bestN > 1 {
					scaling = fmt.Sprintf("%s at %d", fmtSpeedup(best/simd), bestN)
				}
				allocs := 0.0
				for _, k := range cases {
					for _, v := range res.entries[k]["allocs/op"] {
						allocs = max(allocs, v)
					}
					if v := res.entries[k]["Mcells/s"]; len(v) > 1 {
						spreads = append(spreads, (slices.Max(v)-slices.Min(v))/median(v))
					}
				}
				fmt.Fprintf(w, " %s | %s | %g |\n", scaling, fmtGBOpt(gb, okGB), allocs)
			}
		}
	}
	fmt.Fprintln(w)
	if len(spreads) > 0 {
		fmt.Fprintf(w, "Run-to-run spread of M cells/s, (max − min)/median: median %.0f%%, worst %.0f%%.\n\n",
			100*median(spreads), 100*slices.Max(spreads))
	}
	for _, mask := range []string{"off", "on"} {
		if note := classify(res, pkg, op, mask, tilesPlain, sizes); note != "" {
			fmt.Fprintf(w, "- mask=%s, one worker: %s\n", mask, note)
		}
		if note := scalingNote(res, pkg, op, mask, sizes, workers); note != "" {
			fmt.Fprintf(w, "- mask=%s, workers: %s\n", mask, note)
		}
	}
}

// scalingNote summarises SIMD worker scaling of the strips shape: the
// speedup over one worker for each worker count and size, and the memory
// traffic at the largest count, where scaling flattens if memory
// bandwidth is the limit.
func scalingNote(res *results, pkg, op, mask string, sizes, workers []int) string {
	if len(workers) == 0 {
		return ""
	}
	var parts, gbs []string
	top := workers[len(workers)-1]
	for _, size := range sizes {
		one, ok := res.median(key{pkg, op, size, mask, "simd", 1, tilesStrips}, "Mcells/s")
		if !ok {
			continue
		}
		var sp []string
		for _, n := range workers {
			if v, ok := res.median(key{pkg, op, size, mask, "simd", n, tilesStrips}, "Mcells/s"); ok {
				sp = append(sp, fmt.Sprintf("%s with %d", fmtSpeedup(v/one), n))
			}
		}
		if len(sp) == 0 {
			continue
		}
		parts = append(parts, fmt.Sprintf("%d² %s", size, strings.Join(sp, ", ")))
		if gb, ok := res.median(key{pkg, op, size, mask, "simd", top, tilesStrips}, "GB/s"); ok {
			gbs = append(gbs, fmt.Sprintf("%s GB/s at %d²", fmtGB(gb), size))
		}
	}
	if len(parts) == 0 {
		return ""
	}
	note := "strips over one worker, SIMD: " + strings.Join(parts, "; ")
	if len(gbs) > 0 {
		note += fmt.Sprintf("; %d workers move %s", top, strings.Join(gbs, ", "))
	}
	return note + "."
}
