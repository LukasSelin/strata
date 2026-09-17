package main

import (
	"fmt"
	"io"
	"slices"
)

const mib = 1 << 20

// caseKey identifies the runs of one case.
type caseKey struct {
	kind, op string
	size     int
	tiles    string
	workers  int
	backend  string
}

// render prints one table per DEM size: every operation's reference and
// chunked cases, with medians over their runs, and the peak memory
// against the §27 bound.
func render(w io.Writer, results []result) {
	var keys []caseKey
	runs := map[caseKey][]result{}
	for _, r := range results {
		k := caseKey{r.Kind, r.Op, r.Size, r.Tiles, r.Workers, r.Backend}
		if _, ok := runs[k]; !ok {
			keys = append(keys, k)
		}
		runs[k] = append(runs[k], r)
	}
	var sizes []int
	for _, k := range keys {
		if !slices.Contains(sizes, k.size) {
			sizes = append(sizes, k.size)
		}
	}
	for _, size := range sizes {
		cells := float64(size) * float64(size)
		fmt.Fprintf(w, "### %d × %d DEM, raw float32 file of %.2f GiB\n\n", size, size, 4*cells/(1<<30))
		fmt.Fprint(w, "| op | tiles | backend | workers | M cells/s | GB/s | vs SIMD 1 worker | "+
			"peak private MiB | §27 bound MiB | peak − base MiB | flush s | identical |\n")
		fmt.Fprint(w, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
		for _, k := range keys {
			if k.size != size {
				continue
			}
			rs := runs[k]
			sec := median(field(rs, func(r result) float64 { return r.Seconds }))
			rate := cells / sec
			vs := "–"
			if k.kind == "run" && k.backend == "simd" {
				one := caseKey{"run", k.op, k.size, k.tiles, 1, "simd"}
				if r1, ok := runs[one]; ok {
					vs = fmt.Sprintf("%.2f×", median(field(r1, func(r result) float64 { return r.Seconds }))/sec)
				}
			}
			peak := slices.Max(field(rs, func(r result) float64 { return float64(r.PeakPrivate) }))
			base := median(field(rs, func(r result) float64 { return float64(r.BasePrivate) }))
			tiles, bound, identical := k.tiles, "–", "reference"
			if k.kind == "reference" {
				tiles = "whole raster in memory"
			} else {
				tiles = fmt.Sprintf("%s (%d×%d)", k.tiles, rs[0].TileW, rs[0].TileH)
				bound = fmt.Sprintf("%.0f", float64(rs[0].Bound)/mib)
				same := 0
				for _, r := range rs {
					if r.Identical != nil && *r.Identical {
						same++
					}
				}
				identical = fmt.Sprintf("%d/%d", same, len(rs))
			}
			workers := fmt.Sprint(k.workers)
			if len(rs) > 0 && rs[0].Used != k.workers {
				workers = fmt.Sprintf("%d (%d used)", k.workers, rs[0].Used)
			}
			flush := median(field(rs, func(r result) float64 { return r.SyncSec }))
			fmt.Fprintf(w, "| %s | %s | %s | %s | %.0f | %.2f | %s | %.0f | %s | %.0f | %.2f | %s |\n",
				k.op, tiles, k.backend, workers, rate/1e6, 8*rate/1e9, vs,
				peak/mib, bound, (peak-base)/mib, flush, identical)
		}
		fmt.Fprintln(w)
	}
	n := 0
	for _, rs := range runs {
		n = max(n, len(rs))
	}
	fmt.Fprintf(w, "Medians of up to %d runs per case; peak private bytes are the maximum. GB/s counts the "+
		"8 bytes per cell of the input and output files. \"identical\" counts the runs whose output file "+
		"equals the reference cell for cell.\n", n)
}

func field(rs []result, f func(result) float64) []float64 {
	out := make([]float64, len(rs))
	for i, r := range rs {
		out[i] = f(r)
	}
	return out
}

func median(v []float64) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}
