package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

const sample = `goversion: go1.27.0
goexperiment: simd
gomaxprocs: 1
usablecpus: 1
logicalcpus: 24
physicalcores: 12
kernels-vec: avx2
goos: windows
goarch: amd64
pkg: strata/benchmarks/algebra
cpu: Test CPU
BenchmarkAdd/size=256/mask=off/backend=scalar/workers=1-24  100  100 ns/op  12.0 GB/s  1000 Mcells/s  1.000 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=256/mask=off/backend=scalar/workers=1-24  100  100 ns/op  12.0 GB/s  1100 Mcells/s  0.909 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=256/mask=off/backend=scalar/workers=1-24  100  100 ns/op  12.0 GB/s   900 Mcells/s  1.111 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=256/mask=off/backend=simd/workers=1  100  100 ns/op  48.0 GB/s  4000 Mcells/s  0.250 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=4096/mask=off/backend=scalar/workers=1  100  100 ns/op  9.6 GB/s  800 Mcells/s  1.25 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=4096/mask=off/backend=simd/workers=1  100  100 ns/op  12.0 GB/s  1000 Mcells/s  1.00 ns/cell  0 B/op  0 allocs/op
BenchmarkAdd/size=4096/mask=off/backend=simd/workers=12  100  100 ns/op  96.0 GB/s  8000 Mcells/s  0.125 ns/cell  0 B/op  0 allocs/op
--- SKIP: BenchmarkMul/size=256/mask=off/backend=simd
BenchmarkMul/size=256/mask=off/backend=scalar/workers=1  100  100 ns/op  12.0 GB/s  500 Mcells/s  2.0 ns/cell  0 B/op  1 allocs/op
BenchmarkMul/size=1024/mask=off/backend=scalar/workers=1  100  100 ns/op  12.0 GB/s  450 Mcells/s  2.2 ns/cell  0 B/op  0 allocs/op
BenchmarkNodata/size=1024/nodata=0pct/mask-vec  100  100 ns/op
PASS
`

func TestParseAndRender(t *testing.T) {
	res, err := parse(strings.NewReader(sample))
	if err != nil {
		t.Fatal(err)
	}
	if got := res.ops["algebra"]; len(got) != 2 || got[0] != "Add" || got[1] != "Mul" {
		t.Fatalf("ops = %v, want [Add Mul]", got)
	}
	k := key{"algebra", "Add", 256, "off", "scalar", 1, ""}
	if v, _ := res.median(k, "Mcells/s"); v != 1000 {
		t.Errorf("median Mcells/s = %v, want 1000", v)
	}
	var out bytes.Buffer
	render(&out, res)
	got := out.String()
	for _, want := range []string{
		"| CPU | Test CPU |",
		"| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |",
		"| Kernels | vec: avx2 |",
		"| Runs | 3 per benchmark, medians shown |",
		"Add            800      1000        1.25×",
		"| 256 × 256 | off | 1000 | 4000 | 4.00× |",
		"SIMD + 12 workers M cells/s",
		"| 4096 × 4096 | off | 800 | 1000 | 1.25× | 8000 | 1.00 | 9.60 | 12.0 | 0 |",
		"memory-bandwidth-bound from 4096²: SIMD throughput falls to 25%",
		"| 256 × 256 | off | 500 | – | – | – | 12.0 | – | 1 |",
		"compute-bound: scalar throughput stays within 20% of 256²'s up to 1024²",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output lacks %q\n%s", want, got)
		}
	}
}

const tiledSample = `goversion: go1.27.0
physicalcores: 12
logicalcpus: 24
pkg: strata/benchmarks/engine
BenchmarkSlope/size=4096/mask=off/backend=scalar/workers=1/tiles=plain-24  1  1 ns/op  2.00 GB/s  250 Mcells/s  4.0 ns/cell  0 B/op  0 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=simd/workers=1/tiles=plain-24  1  1 ns/op  8.00 GB/s  1000 Mcells/s  1.0 ns/cell  0 B/op  9 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=scalar/workers=1/tiles=strips-24  1  1 ns/op  2.00 GB/s  250 Mcells/s  4.0 ns/cell  0 B/op  9 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=simd/workers=1/tiles=strips-24  1  1 ns/op  8.00 GB/s  980 Mcells/s  1.0 ns/cell  0 B/op  9 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=simd/workers=12/tiles=strips-24  1  1 ns/op  80.0 GB/s  9800 Mcells/s  0.1 ns/cell  0 B/op  20 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=simd/workers=24/tiles=strips-24  1  1 ns/op  96.0 GB/s  11760 Mcells/s  0.1 ns/cell  0 B/op  32 allocs/op
BenchmarkSlope/size=4096/mask=off/backend=simd/workers=1/tiles=256x256-24  1  1 ns/op  8.00 GB/s  900 Mcells/s  1.1 ns/cell  0 B/op  9 allocs/op
--- SKIP: BenchmarkSlope/size=4096/mask=off/backend=scalar/workers=12/tiles=strips
PASS
`

func TestParseAndRenderTiled(t *testing.T) {
	res, err := parse(strings.NewReader(tiledSample))
	if err != nil {
		t.Fatal(err)
	}
	if !res.tiled["engine"] {
		t.Fatalf("engine not detected as tiled")
	}
	var out bytes.Buffer
	render(&out, res)
	got := out.String()
	for _, want := range []string{
		"Slope            250      1000        4.00%s                9800               11760",
		"| 4096 × 4096 | off | plain | 250 | 1000 | – | – | – | – | 8.00 | 9 |",
		"| 4096 × 4096 | off | strips | 250 | 980 | -2% | 9800 | 11760 | 12.00× at 24 | 96.0 | 32 |",
		"| 4096 × 4096 | off | 256x256 | – | 900 | -10% | – | – | – | 8.00 | 9 |",
		"- mask=off, workers: strips over one worker, SIMD: 4096² 10.00× with 12, 12.00× with 24; 24 workers move 96.0 GB/s at 4096².",
	} {
		want = strings.ReplaceAll(want, "%s", "×")
		if !strings.Contains(got, want) {
			t.Errorf("output lacks %q\n%s", want, got)
		}
	}
}

// TestResultsMatch checks that the tables in each category's RESULTS.md
// are this command's output for the raw run committed next to it.
func TestResultsMatch(t *testing.T) {
	for _, category := range []string{"algebra", "engine", "chunked", "terrain", "resample"} {
		t.Run(category, func(t *testing.T) { testResultsMatch(t, category) })
	}
}

func testResultsMatch(t *testing.T, category string) {
	raw, err := os.Open("../../" + category + "/testdata/bench.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	res, err := parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	render(&out, res)

	doc, err := os.ReadFile("../../" + category + "/RESULTS.md")
	if err != nil {
		t.Fatal(err)
	}
	const begin, end = "<!-- stratabench output begin -->\n", "<!-- stratabench output end -->"
	text := strings.ReplaceAll(string(doc), "\r\n", "\n")
	_, rest, ok := strings.Cut(text, begin)
	section, _, ok2 := strings.Cut(rest, end)
	if !ok || !ok2 {
		t.Fatalf("RESULTS.md lacks the %q ... %q markers", strings.TrimSpace(begin), end)
	}
	if section != out.String() {
		t.Errorf("RESULTS.md differs from `go run ./benchmarks/cmd/stratabench < benchmarks/%s/testdata/bench.txt`;"+
			" regenerate the section between the markers.\ngot:\n%s", category, out.String())
	}
}
