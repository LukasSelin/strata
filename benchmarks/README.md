# Benchmarks

This directory is the project benchmark suite (DESIGN.md §27, §31) and the
spikes that informed the design.

| path | what |
|---|---|
| `algebra/` | suite: `strata/algebra` Add, Sub, Mul, Min, Max, Clamp. Numbers in [`algebra/RESULTS.md`](algebra/RESULTS.md) |
| `internal/suite/` | the shared harness: sizes, backend switching, metrics, machine configuration |
| `cmd/stratabench/` | turns `go test -bench` output into the §31 summary |
| `nodata/` | STRATA-3 spike: NoData representations ([`RESULTS.md`](nodata/RESULTS.md)). Not part of the suite |

Package-level micro-benchmarks, such as `algebra/bench_test.go` (whole
raster vs. per row vs. strided) and `internal/stencil/bench_test.go`, stay
next to their code. They compare internal paths. The suite measures the
public API at the §27 sizes and backends.

## Running

Benchmarks never run without `-bench`, so `go test ./...` stays fast. The
suite packages have one cheap test (0 allocs/op on a 64² raster).

```bash
go test ./benchmarks/algebra -run '^$' -bench . -short
```

```bash
GOEXPERIMENT=simd go test ./benchmarks/algebra -run '^$' -bench . -count 5 -timeout 2h > bench.txt
```

```bash
go run ./benchmarks/cmd/stratabench < bench.txt
```

- **Default build vs. `GOEXPERIMENT=simd`.** SIMD kernels only exist in
  `GOEXPERIMENT=simd` builds on amd64 with AVX2 (docs/adr/0001-simd-backend.md).
  In a default build every `backend=simd` case is skipped with that reason,
  and only `backend=scalar` runs. A SIMD build runs both, because the
  suite switches kernels at run time (`vec.UseScalar`), so publish numbers
  from a SIMD build.
- **`-count`.** Use at least 5. `stratabench` and `benchstat` both report
  the median. With `-count`, the testing package reruns each leaf
  benchmark consecutively, so a size's fixture is built once.
- **`-short`** skips 16384². That size needs about 4 GB: three float32
  operands of 1 GiB each plus masks. The whole algebra matrix takes about
  10 minutes with `-count 5` on the RESULTS.md machine; `-short` saves the
  16384² share of that and the 4 GB.
- **`-strata.sizes 256,1024`** runs other sizes. It is a flag of the test
  binary, so put it after the package.
- **`-bench 'Add/size=4096/mask=off'`** selects part of the matrix; each
  `/` level is matched separately.
- **Stable numbers.** Published runs pin the test binary to one logical
  CPU, with `GOMAXPROCS=1` and high priority, on an otherwise quiet
  machine. `RESULTS.md` records the exact commands. Memory-bandwidth-bound
  results are sensitive to anything else moving memory, including other
  builds.
- **benchstat** works on the output as is, since sub-benchmark names are
  `key=value`: `benchstat -col /backend -filter '.unit:Mcells/s' bench.txt`.

## Benchmark names

```text
Benchmark<Op>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=<W>
```

| key | values |
|---|---|
| `size` | square raster side: 256, 1024, 4096, 16384 (§27) |
| `mask` | `off`: no operand has a validity mask. `on`: every input has an independent mask with about 10% of cells invalid, and dst has one |
| `backend` | `scalar` or `simd`, switched in one binary |
| `workers` | only `1` for now. Multi-worker tiling needs the engine (STRATA-8/9) |

Before the results, `suite.Main` prints `key: value` configuration lines
(`goversion`, `goexperiment`, `goamd64`, `gomaxprocs`, `usablecpus`,
`physicalcores`, `logicalcpus`, `kernels-<pkg>`). `go test` adds `goos`,
`goarch`, `pkg` and `cpu`. `usablecpus` is `runtime.NumCPU`, which follows
the process's CPU affinity. `physicalcores` and `logicalcpus` describe the
machine.

## Metrics

Every leaf reports:

| metric | meaning |
|---|---|
| `ns/op` | one operation over the whole raster |
| `Mcells/s` | million cells per second (§31's "M cells/sec") |
| `ns/cell` | nanoseconds per cell, `1000 / Mcells/s` |
| `GB/s` | memory the operation touches per second, in 10⁹ bytes: cells/s × bytes per cell over **all operands including dst**, 4 bytes per float32 raster plus ⅛ byte per validity mask. Add with masks touches 3 × 4.125 = 12.375 bytes per cell, Clamp without masks 2 × 4 = 8 |
| `B/op`, `allocs/op` | from `b.ReportAllocs`. Must be 0 |

`stratabench` adds:

- **SIMD/scalar**: median SIMD M cells/s over median scalar M cells/s.
- **Run-to-run spread**: (max − min)/median of M cells/s over the `-count`
  runs of each case.
- **§19 class** for each operation and mask setting, from the measured
  throughput. *Compute-bound*: SIMD throughput stays within 20% of the
  smallest size's at every size. Otherwise the first size below 80% is
  where it starts to be *memory-bandwidth-bound*, if the SIMD/scalar
  speedup at the largest size is under two thirds of the speedup at the
  smallest, or *cache-bound* if SIMD keeps its lead. A build without SIMD
  can only say *working-set-bound*. The thresholds are in
  `cmd/stratabench/main.go`.

## Adding a category

A category is a package `benchmarks/<name>` (terrain, convolution,
resampling, nd, per §27) with a `doc.go` and a `bench_test.go` that uses
`internal/suite`:

```go
var stencilKernels = suite.Kernels{Name: "stencil", Backend: stencil.Backend, UseScalar: stencil.UseScalar}

func TestMain(m *testing.M) { suite.Main(m, stencilKernels) }

func BenchmarkSlope(b *testing.B) {
	suite.Run(b, suite.Matrix[*demFixture]{
		Kernels:  stencilKernels,
		Fixture:  newDEMFixture, // func(size int) *demFixture, pages touched
		Workload: func(f *demFixture, c suite.Case) suite.Workload {
			dst, dem := f.rasters(c.Masked)
			opts := terrain.SlopeOptions{CellSize: 10}
			return suite.Workload{
				Run:          func() { terrain.Slope(dst, dem, opts) },
				BytesPerCell: f.bytesPerCell(c.Masked), // operands incl. dst
			}
		},
	})
}
```

Rules:

- Call the **public API** of the package being measured. Internal paths
  belong in package micro-benchmarks.
- A kernel package needs a `Backend() string` / `UseScalar(bool)` pair
  (`internal/vec` and `internal/stencil` have one) so that both backends run
  in one binary. A workload that uses several kernel packages passes a
  `Kernels` whose functions switch all of them.
- Build fixtures once per size, touch every page before timing, and keep
  `Run` allocation-free. Add a cheap `TestZeroAllocs` like
  `algebra/bench_test.go`'s.
- Commit a `RESULTS.md` with the machine, the exact commands, the
  `stratabench` output between the `<!-- stratabench output begin -->`
  and `<!-- stratabench output end -->` markers, and the raw run in
  `testdata/bench.txt`. Then extend `TestResultsMatch` in
  `cmd/stratabench` to cover it.

### Workers

When the engine lands (STRATA-8/9), `suite.Workers` returns 1, the
physical core count and the logical CPU count (§27: SIMD + 1 worker, SIMD +
physical cores, SIMD + logical cores). Workloads then read `c.Workers` and
run through the engine. The benchmark names already carry `workers=N`, and
`stratabench` adds a "SIMD + N workers" column for every N > 1 it finds.
Do not add ad-hoc goroutine tiling here before then. Scaling runs must not
be pinned to one CPU.
