# Benchmarks

This directory is the project benchmark suite (DESIGN.md §38, §42) and the
spikes that informed the design.

| path | what |
|---|---|
| `algebra/` | suite: `strata/algebra` Add, Sub, Mul, Min, Max, Clamp. Numbers in [`algebra/RESULTS.md`](algebra/RESULTS.md) |
| `engine/` | suite: Slope, Hillshade and Clamp, plain and through the engine, by worker count and tile shape. Numbers in [`engine/RESULTS.md`](engine/RESULTS.md) |
| `chunked/` | suite: Slope, Hillshade and Clamp with bounded memory from a raw float32 file to another, by worker count and tile shape, and the §43 demo. Numbers in [`chunked/RESULTS.md`](chunked/RESULTS.md) |
| `fusion/` | suite: a six-factor product as five chained `algebra.MulTiled` calls, as one tile-level `Pipeline` (§52) and as a hand-written register-fused kernel (§29), by worker count and tile shape. Measures what register-level fusion would add before a generator is built. Numbers in [`fusion/RESULTS.md`](fusion/RESULTS.md) |
| `terrain/` | suite: the four terrain operations (Gradient, Slope, Aspect, Hillshade) through their plain public API, at every size and backend. Numbers in [`terrain/RESULTS.md`](terrain/RESULTS.md) |
| `gdal/` | against another program: the same three terrain operations timed against `gdaldem`, both in one container. Numbers in [`gdal/RESULTS.md`](gdal/RESULTS.md). Not a Go benchmark, so not part of the suite |
| `internal/suite/` | the shared harness: sizes, backend switching, metrics, machine configuration |
| `cmd/stratabench/` | turns `go test -bench` output into the §42 summary |
| `cmd/stratademo/` | the §43 validation target: a 20000² raw DEM with bounded memory, checked against the whole-raster result, with measured peak memory |
| `reduce/` | suite: `strata/reduce` Count, MinMax, Sum and Stats at every size, mask, backend and worker count, after the §49 accumulator decision (exact binned sums against float64 and Neumaier, benchmarked in `internal/accum`). Both in [`reduce/RESULTS.md`](reduce/RESULTS.md) |
| `nodata/` | STRATA-3 spike: NoData representations ([`RESULTS.md`](nodata/RESULTS.md)). Not part of the suite |

Package-level micro-benchmarks, such as `algebra/bench_test.go` (whole
raster vs. per row vs. strided), `internal/stencil/bench_test.go` and
`internal/vec/bench_test.go` (per-call cost by row width) and
`internal/exec/bench_test.go` (tile shapes on one worker), stay next to
their code. They compare internal paths. The suite measures the
public API at the §38 sizes and backends.

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
- **`-short`** skips 16384², where each float32 operand is 1 GiB.
  Measured peak private memory for the algebra suite at 16384² is
  3.15 GiB for the two-input operations (three operands plus masks) and
  2.12 GiB for Clamp (two). Fixtures are freed between sizes and between
  operations, so a full run peaks at 3.15 GiB too. The whole algebra
  matrix takes about 10 minutes with `-count 5` on the RESULTS.md
  machine; `-short` saves the 16384² share of that and the memory.
- **`-strata.sizes 256,1024`** runs other sizes. It is a flag of the test
  binary, so put it after the package.
- **`-bench 'Add/size=4096/mask=off'`** selects part of the matrix; each
  `/` level is matched separately.
- **Stable numbers.** Published one-worker runs pin the test binary to
  one logical CPU, with `GOMAXPROCS=1` and high priority, on an otherwise
  quiet machine. Runs with `workers` above 1 (the engine category) must not
  be pinned: they run at high priority with the default `GOMAXPROCS`.
  `RESULTS.md` records the exact commands. Memory-bandwidth-bound results
  are sensitive to anything else moving memory, including other builds.
- **benchstat** works on the output as is, since sub-benchmark names are
  `key=value`: `benchstat -col /backend -filter '.unit:Mcells/s' bench.txt`.

## Benchmark names

```text
Benchmark<Op>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=<W>[/tiles=<T>]
```

| key | values |
|---|---|
| `size` | square raster side: 256, 1024, 4096, 16384 (§38) |
| `mask` | `off`: no operand has a validity mask. `on`: every input has an independent mask with about 10% of cells invalid, and dst has one |
| `backend` | `scalar` or `simd`, switched in one binary |
| `workers` | `1` for categories that call plain functions. Engine categories run `suite.Workers()`: 1, one per physical core, one per logical CPU |
| `tiles` | engine categories only: the tile shape, e.g. `plain` (the plain function), `strips` (default `engine.Options`), `strips256` (full-width tiles of 256 rows), `256x256`. Scaling is reported for `strips`, or else the first `strips<rows>` shape |

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
| `Mcells/s` | million cells per second (§42's "M cells/sec") |
| `ns/cell` | nanoseconds per cell, `1000 / Mcells/s` |
| `GB/s` | memory the operation touches per second, in 10⁹ bytes: cells/s × bytes per cell over **all operands including dst**, 4 bytes per float32 raster plus ⅛ byte per validity mask. Add with masks touches 3 × 4.125 = 12.375 bytes per cell, Clamp without masks 2 × 4 = 8 |
| `B/op`, `allocs/op` | from `b.ReportAllocs`. Must be 0 for plain functions; engine calls allocate a few slices and their goroutines per call, never per tile |

`stratabench` adds (for categories with a `tiles` level it prints one row
per size, mask and tile shape with a column per worker count, the one-worker
SIMD cost against `plain`, the best scaling, and a note per mask with the
speedup of `strips` over one worker at each size):

- **SIMD/scalar**: median SIMD M cells/s over median scalar M cells/s.
- **Run-to-run spread**: (max − min)/median of M cells/s over the `-count`
  runs of each case.
- **§28 class** for each operation and mask setting, from the measured
  throughput. *Compute-bound*: SIMD throughput stays within 20% of the
  smallest size's at every size. Otherwise the first size below 80% is
  where it starts to be *memory-bandwidth-bound*, if the SIMD/scalar
  speedup at the largest size is under two thirds of the speedup at the
  smallest, or *cache-bound* if SIMD keeps its lead. A build without SIMD
  can only say *working-set-bound*. The thresholds are in
  `cmd/stratabench/main.go`.

## Adding a category

A category is a package `benchmarks/<name>` (convolution,
remote_sensing, pointcloud, nd, per §38) with a `doc.go` and a
`bench_test.go` that uses `internal/suite`. `terrain/bench_test.go` is the
smallest complete example; the sketch below is its Slope case:

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
- Build fixtures once per size, allocate only the operands the workload
  reads, touch every page before timing, and keep `Run` allocation-free.
  Document the measured 16384² peak in the category's `doc.go`. Add a cheap `TestZeroAllocs` like
  `algebra/bench_test.go`'s.
- Commit a `RESULTS.md` with the machine, the exact commands, the
  `stratabench` output between the `<!-- stratabench output begin -->`
  and `<!-- stratabench output end -->` markers, and the raw run in
  `testdata/bench.txt`. Then extend `TestResultsMatch` in
  `cmd/stratabench` to cover it.

### Workers and tiles

A category that runs through the engine sets `Matrix.Workers` to
`suite.Workers()` (1, the physical core count and the logical CPU count,
without duplicates) and `Matrix.Tiles` to its tile shapes, and reads
`c.Workers` and `c.Tiles` in its workload. A workload returns a `Workload`
with a nil `Run` for combinations that do not apply, such as a plain
function with more than one worker; the leaf is skipped. `stratabench`
recognises such a category by its `tiles` level. See `engine/bench_test.go`.
Scaling runs must not be pinned to one CPU.

### Files and peak memory

A category whose fixture writes files sets `Matrix.Release`, which runs
once a size is done, to close and remove them (`chunked/bench_test.go`).
`suite.ProcessMemory` reports the process's current and peak private
bytes and working set (Windows; resident memory only on Linux). A peak
covers the whole process, so a measurement that must be its own, such as
the §43 demo's, runs in a child process: `cmd/stratademo` starts one per
run.
