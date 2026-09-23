# focal benchmark suite, results

The focal category of the project benchmark suite (DESIGN.md §38, §53):
the operations of `strata/focal` through their plain public API —
Correlate with full (2r+1)² weights, CorrelateSeparable with Gaussian
taps (named `Gaussian` here), Mean and Min at radii 1, 2, 3 and 5, and
Max at 3 — at 256² to 16384², with and without validity masks, on the
scalar and SIMD kernels of `internal/focalrow`, on one worker. Metrics
and names are defined in [`../README.md`](../README.md).

It answers §28's open prediction: that convolution, like the terrain
kernels and unlike the algebra, is compute-bound rather than capped by
memory bandwidth. Every operation moves 8 bytes per cell whatever its
radius (the other rows of a neighbourhood come from cache), so its work
per byte grows with the radius.

The published numbers are AVX2 on the Zen 2 desktop every other category
uses. The Apple M4 NEON run that came first is kept below, under
[arm64 (NEON)](#arm64-neon).

## Headline

- **Compute-bound at every radius and size.** All 34 operation × mask
  rows hold their SIMD throughput from 256² to 16384². Until the change
  recorded under [16384² and the row stride](#16384-and-the-row-stride),
  six did not: Gaussian, Mean and Min at r = 5 ran at 42–56% of their
  4096² speed at 16384², because rows 64 KiB apart collided in one set
  of Zen 2's L2. The column pass now folds such rows in groups, and those
  rows run at 86–93% of their 4096² speed, without a bit of output
  changing.
- **Correlate costs what its products cost.** At 4096² without masks,
  AVX2 Correlate takes 0.96, 1.92, 3.44 and 7.98 ns per cell at r = 1, 2,
  3 and 5: 0.066–0.077 ns per product from r = 2 on (0.11 at r = 1, where
  the per-row fixed cost still shows), so its throughput falls as
  (2r+1)². That is about 14 billion products and sums a second on one
  core, without FMA (§15).
- **The separable forms grow linearly in r.** Gaussian takes 0.75 ns per
  cell at r = 1 and 1.65 at r = 5; Mean, 0.71 and 1.58; Min, 0.82 and
  2.45. At r = 5 Gaussian is 4.8× faster than Correlate with the same
  footprint, and at r = 1 still 1.3×.
- **Memory is not the limit.** The highest demand in the suite is 15.1
  GB/s (Mean at r = 1, 1024², in cache), under a core's 22 GB/s, and from
  r = 2 the demand falls with the radius, to 1.0 GB/s for Correlate at
  r = 5.
- **AVX2 is worth 2.9–7.5× over scalar** at 4096², rising with the
  radius, but read the upper half with care: the scalar Correlate and
  Mean kernels run 20–43% slower in this binary than in the previous run
  (`testdata/bench-prefold.txt`) for a reason that has nothing to do with
  them (see [Machine and method](#machine-and-method)), which inflates
  their ratios. Gaussian and Min, whose scalar speed did not move, give
  2.9–5.5×. As on NEON, this is lanes plus register blocking, not lanes
  alone: the scalar kernels, which are the canonical order and must stay
  bounds-check free (§39), accumulate through `dst` a term at a time,
  while the SIMD ones hold several accumulators per block in registers.
  The M4's faster core is 1.2–2.6× ahead of AVX2 in absolute terms:
  least on Correlate at large radii, most on Min.

| 4096², no mask | products/cell | scalar M cells/s | AVX2 M cells/s | AVX2/scalar | AVX2 ns/cell | AVX2 GB/s | NEON M cells/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CorrelateR1 | 9 | 222 | 1040 | 4.68× | 0.962 | 8.32 | 1724 |
| CorrelateR2 | 25 | 80.3 | 520 | 6.48× | 1.92 | 4.16 | 741 |
| CorrelateR3 | 49 | 41.2 | 291 | 7.06× | 3.44 | 2.33 | 386 |
| CorrelateR5 | 121 | 16.7 | 125 | 7.52× | 7.98 | 1.00 | 148 |
| GaussianR1 | 6 | 381 | 1343 | 3.52× | 0.745 | 10.8 | 2501 |
| GaussianR3 | 14 | 171 | 904 | 5.28× | 1.11 | 7.23 | 1329 |
| GaussianR5 | 22 | 110 | 605 | 5.49× | 1.65 | 4.84 | 892 |
| MeanR1 | — | 314 | 1412 | 4.50× | 0.708 | 11.3 | 2900 |
| MeanR5 | — | 90.5 | 632 | 6.99× | 1.58 | 5.06 | 1037 |
| MinR1 | — | 426 | 1218 | 2.86× | 0.821 | 9.74 | 2946 |
| MinR5 | — | 96.9 | 408 | 4.21× | 2.45 | 3.27 | 1039 |

## 16384² and the row stride

At 16384² a row of float32 is exactly 64 KiB. The SIMD separable
operations used to slow down there and nowhere else; they now run close
to their speed at the neighbouring sizes. The same pinning, unmasked,
AVX2, M cells/s, 5 samples each; the "before" column is the previous
kernels, measured on 2026-09-22 alongside `testdata/bench-prefold.txt`:

| operation | 16320² | **16384² before** | **16384² after** | 16448² |
|---|---:|---:|---:|---:|
| GaussianR5 | 597–609 | 251–258 | 540–552 | 577–609 |
| MeanR5 | 599–627 | 266–269 | 590–597 | 619–631 |
| MinR5 | 385–402 | 221–224 | 367–377 | 393–402 |
| CorrelateR5 | 122–126 | 92–94 | 124–126 | 122–126 |
| MeanR3 | 946–963 | 762–781 | 747–796 | 919–993 |

Before the change, 8192² and 12288² ran at full speed too (Gaussian r = 5:
569–578 and 556–558), and scalar and NEON did not show the effect.

### The cause: L2 sets, from the stride alone

Zen 2's L2 is 512 KiB and 8-way, so its set index repeats every 64 KiB.
A SIMD column pass reads a block of 32 cells from each of its 2r+1 rows
in lockstep, holding one accumulator per vector in registers; rows
64 KiB apart put all those lines in one set. No performance counters
back this: AMD uProf is not installed on the desktop, and Windows' own
PMU sources (`wpr -pmcsources`) have no L2 event and need an elevated
shell. The evidence is timing that separates the candidate causes, from
`BenchmarkColumnStride` in `internal/focalrow`: the column pass over 64
output rows of a fixed 16320 cells, only the stride varied, ns per cell,
fastest of 5, with the column pass as it was before the change
(ColumnSum; ColumnCorrelate behaves the same):

| rows (2r+1) | 16320 | **16384** | 16400 | 16416 | 16448 | 20480 | 24576 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.63 | **1.99** | 0.99 | 0.70 | 0.65 | 0.75 | 0.78 |
| 11 | 0.83 | **2.88** | 1.34 | 0.91 | 0.79 | 0.84 | 0.89 |
| 17 | 1.34 | **4.91** | 2.78 | 1.57 | 1.39 | 1.44 | 1.84 |

- **It is the stride, not the width or the size.** At a width of 16320
  with a stride of 16384, focal's Gaussian at r = 5 ran at 246–252 M
  cells/s; at a width of 16384 with a stride of 16448, at 431–512. A
  4096-cell window of a 16384-wide raster, as a tile of the Tiled
  functions is, was as slow (247–256).
- **Its period is 64 KiB, the L2's.** Strides of 32768 and 49152 cells
  are as slow as 16384 (11 rows: about 3.0 and 6.0 ns per cell), while
  20480 and 24576 (80 and 96 KiB) are not. Those two are multiples of
  4 KiB, as 16384 is, so the L1's sets (which repeat every 4 KiB) and 4K
  aliasing between loads and stores are ruled out.
- **Its threshold is the 8 ways.** 7 rows at stride 16384 lose 28%
  (0.47 → 0.60); 9 rows lose 3.2×. At 24576, where the rows alternate
  between two sets, only 17 rows (9 per set) slow down.
- **Near misses count.** Rows one cache line apart modulo 64 KiB (16400)
  lose 1.5–2.2×, two lines apart (16416) up to 24%, four (16448) nothing:
  the prefetcher's lines ahead of each row land in the neighbouring sets.

### The change

When more than seven of a column pass's rows lie within 1 KiB of one
another modulo 64 KiB (and are not simply adjacent in memory), the AVX2
kernels now fold them a group of at most seven rows at a time over chunks
of 4096 cells, the first group writing the destination row and the
others adding to it (`foldColumn` in `internal/focalrow/simd_amd64.go`).
Each cell still folds its rows in order into a float32, which is what
the scalar kernels do, so no output bit changes; `TestAliasedStride` in
package focal checks every operation at a 16384-cell stride against the
per-cell definition, and fails if the fold is broken. Correlate's 2-D
pass folds the same way. At other strides the kernels keep their single
pass: folding there costs 4–19%. The 1 KiB rule folds wherever
folding was faster in the microbenchmark and nowhere else, with one
exception: at 16416 cells and 9 rows it folds, and that costs 11–12%.

What is left:

- **r = 5 is 3–10% short of 16320².** Gaussian 540–552 against 597–609,
  Min 367–377 against 385–402, Mean within 3%, Correlate none. In the
  microbenchmark a folded pass is 4–19% slower than a single pass at a
  harmless stride, for its extra loads and stores of the destination
  row, and 6–15% slower again at 16384 than at 16320, so some conflict
  remains within each group.
- **r = 3 still loses about 20%** (Mean 747–796 against 946–963), as it
  did before. Seven rows are under the fold's threshold, and folding
  them in groups of three and four did not help in the microbenchmark
  (0.60 → 0.65 ns per cell), so that loss has a cause this change does
  not reach, and it stays open (DESIGN.md §53).
- Beyond the suite, Gaussian at r = 8 (17 rows) at 16384² went from
  148–152 to 268–342 M cells/s, against 341–367 at 16320².

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/focal`, then `focal.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 8h`, pinned to one logical CPU from start (`start /affinity 10 /high`), `GOMAXPROCS=1`, High priority. About 32 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) |
| Stats | median of 5 runs, each ≥1 s (`b.Loop`) |

This run was taken on the morning of 2026-09-23, after the column-pass
change, after a short pinned probe showed the core quiet (10 samples
within 6%). Two runs before it, taken while the desktop was in use, were
discarded: their worst per-operation spreads were 66% and
56% against 15% here. The previous published run, from the night of
2026-09-22 and before the change, is kept as
[`testdata/bench-prefold.txt`](testdata/bench-prefold.txt), and the busy
run that preceded it as [`testdata/bench-busy.txt`](testdata/bench-busy.txt).

**The scalar Correlate and Mean rows moved for a reason outside the focal
code.** Against `bench-prefold.txt`, scalar Correlate runs 11–29%
slower and scalar Mean 25–43% slower, at every size and steadily (their
spreads are a few percent), while scalar Gaussian, Min and Max are
within 6%, and the SIMD rows other than r ≥ 3 at 16384² within 12%.
The scalar kernels' source is unchanged, and the column-pass
change is not the cause: a binary of the commit just before it measures
the same as this one, pinned and interleaved (scalar Mean r = 5 at 1024²:
87–91 against 89–92 M cells/s), while a binary of the commit the previous
run measured, `a81fb65`, runs it at 147–160. In between, master's
unrelated commits moved every function of `internal/focalrow` by 32 bytes
(`scalarColumnSum` from 0 to 32 mod 64, `scalarRowMean` from 32 to 0),
and Go aligns functions only to 32 bytes. Scalar Mean and Correlate are
that sensitive to where their loops fall; exactly which boundary their
inner loops cross is not established here.

## Reproducing

```
GOEXPERIMENT=simd go test -c -o focal.test.exe ./benchmarks/focal
cd benchmarks/focal
focal.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 8h > testdata/bench.txt
cd ../..
go run ./benchmarks/cmd/stratabench < benchmarks/focal/testdata/bench.txt
```

Published runs pin the test binary to one logical CPU with
`GOMAXPROCS=1`, as for [`../terrain/RESULTS.md`](../terrain/RESULTS.md).
On Windows, pin from the start so that `runtime.NumCPU` sees one CPU:
`start "" /b /wait /high /affinity 10 focal.test.exe ...` from `cmd`.
The stride table above adds `-test.bench '(GaussianR5|MeanR5|MinR5|MeanR3|CorrelateR5)/size=.*/mask=off/backend=simd' -strata.sizes 16320,16384,16448`;
the column-pass table is `GOEXPERIMENT=simd go test -c ./internal/focalrow`
run pinned the same way with `-test.bench ColumnStride -test.count 5`, on
the commit before the change for the "before" figures.
`go test ./benchmarks/cmd/stratabench` checks that the section between
the markers is what stratabench renders from `testdata/bench.txt`.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | focalrow: avx2 |
| Runs | 5 per benchmark, medians shown |

## focal

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
CorrelateR1       222      1040        4.68×   not measured yet (tile engine, STRATA-8/9)
CorrelateR2      80.3       520        6.48×   not measured yet (tile engine, STRATA-8/9)
CorrelateR3      41.2       291        7.06×   not measured yet (tile engine, STRATA-8/9)
CorrelateR5      16.7       125        7.52×   not measured yet (tile engine, STRATA-8/9)
GaussianR1       381      1343        3.52×   not measured yet (tile engine, STRATA-8/9)
GaussianR2       239      1105        4.62×   not measured yet (tile engine, STRATA-8/9)
GaussianR3       171       904        5.28×   not measured yet (tile engine, STRATA-8/9)
GaussianR5       110       605        5.49×   not measured yet (tile engine, STRATA-8/9)
MeanR1         314      1412        4.50×   not measured yet (tile engine, STRATA-8/9)
MeanR2         195      1198        6.16×   not measured yet (tile engine, STRATA-8/9)
MeanR3         140      1000        7.16×   not measured yet (tile engine, STRATA-8/9)
MeanR5        90.5       632        6.99×   not measured yet (tile engine, STRATA-8/9)
MinR1          426      1218        2.86×   not measured yet (tile engine, STRATA-8/9)
MinR2          230       831        3.61×   not measured yet (tile engine, STRATA-8/9)
MinR3          159       614        3.86×   not measured yet (tile engine, STRATA-8/9)
MinR5         96.9       408        4.21×   not measured yet (tile engine, STRATA-8/9)
MaxR3          102       500        4.90×   not measured yet (tile engine, STRATA-8/9)
```

### CorrelateR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 214 | 776 | 3.62× | 1.29 | 1.72 | 6.21 | 9 |
| 1024 × 1024 | off | 229 | 1047 | 4.58× | 0.956 | 1.83 | 8.37 | 9 |
| 4096 × 4096 | off | 222 | 1040 | 4.68× | 0.962 | 1.78 | 8.32 | 9 |
| 16384 × 16384 | off | 207 | 1002 | 4.84× | 0.998 | 1.66 | 8.02 | 9 |
| 256 × 256 | on | 212 | 685 | 3.23× | 1.46 | 1.75 | 5.65 | 12 |
| 1024 × 1024 | on | 224 | 984 | 4.38× | 1.02 | 1.85 | 8.12 | 12 |
| 4096 × 4096 | on | 222 | 972 | 4.37× | 1.03 | 1.83 | 8.02 | 12 |
| 16384 × 16384 | on | 204 | 947 | 4.64× | 1.06 | 1.68 | 7.82 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (776 → 1002 M cells/s), SIMD/scalar 3.62× → 4.84×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (685 → 947 M cells/s), SIMD/scalar 3.23× → 4.64×.

### CorrelateR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 78.6 | 379 | 4.83× | 2.64 | 0.63 | 3.03 | 9 |
| 1024 × 1024 | off | 80.3 | 490 | 6.11× | 2.04 | 0.64 | 3.92 | 9 |
| 4096 × 4096 | off | 80.3 | 520 | 6.48× | 1.92 | 0.64 | 4.16 | 9 |
| 16384 × 16384 | off | 76.0 | 514 | 6.76× | 1.95 | 0.61 | 4.11 | 9 |
| 256 × 256 | on | 77.5 | 346 | 4.46× | 2.89 | 0.64 | 2.85 | 12 |
| 1024 × 1024 | on | 79.5 | 471 | 5.92× | 2.12 | 0.66 | 3.88 | 12 |
| 4096 × 4096 | on | 79.8 | 505 | 6.33× | 1.98 | 0.66 | 4.17 | 12 |
| 16384 × 16384 | on | 75.6 | 498 | 6.59× | 2.01 | 0.62 | 4.11 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 10%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (379 → 514 M cells/s), SIMD/scalar 4.83× → 6.76×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (346 → 498 M cells/s), SIMD/scalar 4.46× → 6.59×.

### CorrelateR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 40.7 | 229 | 5.63× | 4.37 | 0.33 | 1.83 | 9 |
| 1024 × 1024 | off | 41.0 | 278 | 6.77× | 3.60 | 0.33 | 2.22 | 9 |
| 4096 × 4096 | off | 41.2 | 291 | 7.06× | 3.44 | 0.33 | 2.33 | 9 |
| 16384 × 16384 | off | 39.6 | 278 | 7.02× | 3.59 | 0.32 | 2.23 | 9 |
| 256 × 256 | on | 40.1 | 213 | 5.30× | 4.70 | 0.33 | 1.76 | 12 |
| 1024 × 1024 | on | 40.8 | 267 | 6.55× | 3.74 | 0.34 | 2.21 | 12 |
| 4096 × 4096 | on | 41.0 | 287 | 7.00× | 3.49 | 0.34 | 2.37 | 12 |
| 16384 × 16384 | on | 39.3 | 270 | 6.89× | 3.70 | 0.32 | 2.23 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (229 → 278 M cells/s), SIMD/scalar 5.63× → 7.02×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (213 → 270 M cells/s), SIMD/scalar 5.30× → 6.89×.

### CorrelateR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 16.9 | 97.6 | 5.78× | 10.2 | 0.14 | 0.78 | 9 |
| 1024 × 1024 | off | 16.7 | 118 | 7.06× | 8.47 | 0.13 | 0.94 | 9 |
| 4096 × 4096 | off | 16.7 | 125 | 7.52× | 7.98 | 0.13 | 1.00 | 9 |
| 16384 × 16384 | off | 16.3 | 120 | 7.38× | 8.30 | 0.13 | 0.96 | 9 |
| 256 × 256 | on | 16.8 | 93.4 | 5.56× | 10.7 | 0.14 | 0.77 | 12 |
| 1024 × 1024 | on | 16.8 | 115 | 6.85× | 8.66 | 0.14 | 0.95 | 12 |
| 4096 × 4096 | on | 16.7 | 124 | 7.44× | 8.06 | 0.14 | 1.02 | 12 |
| 16384 × 16384 | on | 16.3 | 119 | 7.28× | 8.43 | 0.13 | 0.98 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (97.6 → 120 M cells/s), SIMD/scalar 5.78× → 7.38×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (93.4 → 119 M cells/s), SIMD/scalar 5.56× → 7.28×.

### GaussianR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 375 | 1352 | 3.60× | 0.740 | 3.00 | 10.8 | 10 |
| 1024 × 1024 | off | 400 | 1722 | 4.30× | 0.581 | 3.20 | 13.8 | 10 |
| 4096 × 4096 | off | 381 | 1343 | 3.52× | 0.745 | 3.05 | 10.8 | 10 |
| 16384 × 16384 | off | 365 | 1289 | 3.54× | 0.776 | 2.92 | 10.3 | 12 |
| 256 × 256 | on | 353 | 1092 | 3.09× | 0.915 | 2.91 | 9.01 | 13 |
| 1024 × 1024 | on | 395 | 1492 | 3.78× | 0.670 | 3.26 | 12.3 | 13 |
| 4096 × 4096 | on | 372 | 1152 | 3.09× | 0.868 | 3.07 | 9.51 | 13 |
| 16384 × 16384 | on | 374 | 1224 | 3.27× | 0.817 | 3.09 | 10.1 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 13%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1352 → 1289 M cells/s), SIMD/scalar 3.60× → 3.54×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1092 → 1224 M cells/s), SIMD/scalar 3.09× → 3.27×.

### GaussianR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 233 | 998 | 4.28× | 1.00 | 1.87 | 7.99 | 10 |
| 1024 × 1024 | off | 247 | 1215 | 4.93× | 0.823 | 1.97 | 9.72 | 10 |
| 4096 × 4096 | off | 239 | 1105 | 4.62× | 0.905 | 1.91 | 8.84 | 10 |
| 16384 × 16384 | off | 236 | 1079 | 4.57× | 0.926 | 1.89 | 8.64 | 14 |
| 256 × 256 | on | 219 | 804 | 3.67× | 1.24 | 1.81 | 6.63 | 13 |
| 1024 × 1024 | on | 240 | 1062 | 4.42× | 0.942 | 1.98 | 8.76 | 13 |
| 4096 × 4096 | on | 234 | 1024 | 4.37× | 0.977 | 1.94 | 8.45 | 13 |
| 16384 × 16384 | on | 234 | 989 | 4.22× | 1.01 | 1.93 | 8.16 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (998 → 1079 M cells/s), SIMD/scalar 4.28× → 4.57×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (804 → 989 M cells/s), SIMD/scalar 3.67× → 4.22×.

### GaussianR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 172 | 806 | 4.68× | 1.24 | 1.38 | 6.45 | 10 |
| 1024 × 1024 | off | 175 | 947 | 5.42× | 1.06 | 1.40 | 7.57 | 10 |
| 4096 × 4096 | off | 171 | 904 | 5.28× | 1.11 | 1.37 | 7.23 | 10 |
| 16384 × 16384 | off | 172 | 764 | 4.44× | 1.31 | 1.38 | 6.11 | 14 |
| 256 × 256 | on | 161 | 644 | 3.99× | 1.55 | 1.33 | 5.32 | 13 |
| 1024 × 1024 | on | 169 | 820 | 4.86× | 1.22 | 1.39 | 6.76 | 13 |
| 4096 × 4096 | on | 170 | 814 | 4.79× | 1.23 | 1.40 | 6.72 | 13 |
| 16384 × 16384 | on | 169 | 701 | 4.14× | 1.43 | 1.40 | 5.78 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (806 → 764 M cells/s), SIMD/scalar 4.68× → 4.44×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (644 → 701 M cells/s), SIMD/scalar 3.99× → 4.14×.

### GaussianR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 112 | 538 | 4.80× | 1.86 | 0.90 | 4.30 | 10 |
| 1024 × 1024 | off | 112 | 601 | 5.35× | 1.66 | 0.90 | 4.81 | 10 |
| 4096 × 4096 | off | 110 | 605 | 5.49× | 1.65 | 0.88 | 4.84 | 10 |
| 16384 × 16384 | off | 109 | 520 | 4.78× | 1.92 | 0.87 | 4.16 | 14 |
| 256 × 256 | on | 107 | 433 | 4.03× | 2.31 | 0.89 | 3.57 | 13 |
| 1024 × 1024 | on | 109 | 522 | 4.78× | 1.91 | 0.90 | 4.31 | 13 |
| 4096 × 4096 | on | 108 | 534 | 4.94× | 1.87 | 0.89 | 4.41 | 13 |
| 16384 × 16384 | on | 109 | 490 | 4.51× | 2.04 | 0.90 | 4.04 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (538 → 520 M cells/s), SIMD/scalar 4.80× → 4.78×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (433 → 490 M cells/s), SIMD/scalar 4.03× → 4.51×.

### MeanR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 313 | 1453 | 4.65× | 0.688 | 2.50 | 11.6 | 8 |
| 1024 × 1024 | off | 333 | 1885 | 5.67× | 0.530 | 2.66 | 15.1 | 8 |
| 4096 × 4096 | off | 314 | 1412 | 4.50× | 0.708 | 2.51 | 11.3 | 8 |
| 16384 × 16384 | off | 310 | 1378 | 4.45× | 0.726 | 2.48 | 11.0 | 10 |
| 256 × 256 | on | 295 | 1161 | 3.94× | 0.861 | 2.43 | 9.58 | 11 |
| 1024 × 1024 | on | 316 | 1664 | 5.27× | 0.601 | 2.60 | 13.7 | 11 |
| 4096 × 4096 | on | 309 | 1299 | 4.21× | 0.770 | 2.55 | 10.7 | 11 |
| 16384 × 16384 | on | 305 | 1283 | 4.20× | 0.779 | 2.52 | 10.6 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1453 → 1378 M cells/s), SIMD/scalar 4.65× → 4.45×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1161 → 1283 M cells/s), SIMD/scalar 3.94× → 4.20×.

### MeanR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 193 | 1120 | 5.79× | 0.893 | 1.55 | 8.96 | 8 |
| 1024 × 1024 | off | 198 | 1389 | 7.01× | 0.720 | 1.58 | 11.1 | 8 |
| 4096 × 4096 | off | 195 | 1198 | 6.16× | 0.835 | 1.56 | 9.58 | 8 |
| 16384 × 16384 | off | 192 | 1171 | 6.09× | 0.854 | 1.54 | 9.37 | 12 |
| 256 × 256 | on | 185 | 885 | 4.79× | 1.13 | 1.52 | 7.30 | 11 |
| 1024 × 1024 | on | 195 | 1205 | 6.19× | 0.830 | 1.61 | 9.94 | 11 |
| 4096 × 4096 | on | 191 | 1087 | 5.70× | 0.920 | 1.57 | 8.97 | 11 |
| 16384 × 16384 | on | 188 | 1056 | 5.62× | 0.947 | 1.55 | 8.71 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1120 → 1171 M cells/s), SIMD/scalar 5.79× → 6.09×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (885 → 1056 M cells/s), SIMD/scalar 4.79× → 5.62×.

### MeanR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 141 | 875 | 6.22× | 1.14 | 1.13 | 7.00 | 8 |
| 1024 × 1024 | off | 142 | 1086 | 7.62× | 0.921 | 1.14 | 8.69 | 8 |
| 4096 × 4096 | off | 140 | 1000 | 7.16× | 1.00 | 1.12 | 8.00 | 8 |
| 16384 × 16384 | off | 139 | 792 | 5.69× | 1.26 | 1.11 | 6.33 | 12 |
| 256 × 256 | on | 135 | 706 | 5.22× | 1.42 | 1.12 | 5.83 | 11 |
| 1024 × 1024 | on | 136 | 933 | 6.85× | 1.07 | 1.12 | 7.70 | 11 |
| 4096 × 4096 | on | 137 | 903 | 6.60× | 1.11 | 1.13 | 7.45 | 11 |
| 16384 × 16384 | on | 137 | 733 | 5.36× | 1.36 | 1.13 | 6.05 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (875 → 792 M cells/s), SIMD/scalar 6.22× → 5.69×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (706 → 733 M cells/s), SIMD/scalar 5.22× → 5.36×.

### MeanR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 91.4 | 553 | 6.05× | 1.81 | 0.73 | 4.43 | 8 |
| 1024 × 1024 | off | 91.1 | 680 | 7.47× | 1.47 | 0.73 | 5.44 | 8 |
| 4096 × 4096 | off | 90.5 | 632 | 6.99× | 1.58 | 0.72 | 5.06 | 8 |
| 16384 × 16384 | off | 89.7 | 558 | 6.23× | 1.79 | 0.72 | 4.47 | 12 |
| 256 × 256 | on | 88.0 | 481 | 5.46× | 2.08 | 0.73 | 3.97 | 11 |
| 1024 × 1024 | on | 88.8 | 578 | 6.50× | 1.73 | 0.73 | 4.76 | 11 |
| 4096 × 4096 | on | 88.9 | 560 | 6.30× | 1.79 | 0.73 | 4.62 | 11 |
| 16384 × 16384 | on | 87.5 | 517 | 5.90× | 1.94 | 0.72 | 4.26 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (553 → 558 M cells/s), SIMD/scalar 6.05× → 6.23×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (481 → 517 M cells/s), SIMD/scalar 5.46× → 5.90×.

### MinR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 420 | 1122 | 2.67× | 0.891 | 3.36 | 8.98 | 8 |
| 1024 × 1024 | off | 462 | 1369 | 2.96× | 0.731 | 3.69 | 10.9 | 8 |
| 4096 × 4096 | off | 426 | 1218 | 2.86× | 0.821 | 3.41 | 9.74 | 8 |
| 16384 × 16384 | off | 419 | 1215 | 2.90× | 0.823 | 3.35 | 9.72 | 10 |
| 256 × 256 | on | 392 | 939 | 2.40× | 1.06 | 3.23 | 7.74 | 11 |
| 1024 × 1024 | on | 440 | 1236 | 2.81× | 0.809 | 3.63 | 10.2 | 11 |
| 4096 × 4096 | on | 418 | 1131 | 2.71× | 0.884 | 3.45 | 9.33 | 11 |
| 16384 × 16384 | on | 410 | 1138 | 2.77× | 0.879 | 3.38 | 9.39 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1122 → 1215 M cells/s), SIMD/scalar 2.67× → 2.90×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (939 → 1138 M cells/s), SIMD/scalar 2.40× → 2.77×.

### MinR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 226 | 749 | 3.31× | 1.34 | 1.81 | 5.99 | 8 |
| 1024 × 1024 | off | 239 | 852 | 3.57× | 1.17 | 1.91 | 6.82 | 8 |
| 4096 × 4096 | off | 230 | 831 | 3.61× | 1.20 | 1.84 | 6.65 | 8 |
| 16384 × 16384 | off | 228 | 830 | 3.64× | 1.21 | 1.82 | 6.64 | 12 |
| 256 × 256 | on | 215 | 626 | 2.91× | 1.60 | 1.77 | 5.16 | 11 |
| 1024 × 1024 | on | 232 | 765 | 3.30× | 1.31 | 1.92 | 6.31 | 11 |
| 4096 × 4096 | on | 225 | 769 | 3.43× | 1.30 | 1.85 | 6.35 | 11 |
| 16384 × 16384 | on | 225 | 769 | 3.42× | 1.30 | 1.85 | 6.35 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (749 → 830 M cells/s), SIMD/scalar 3.31× → 3.64×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (626 → 769 M cells/s), SIMD/scalar 2.91× → 3.42×.

### MinR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 156 | 552 | 3.55× | 1.81 | 1.24 | 4.42 | 8 |
| 1024 × 1024 | off | 160 | 631 | 3.94× | 1.58 | 1.28 | 5.05 | 8 |
| 4096 × 4096 | off | 159 | 614 | 3.86× | 1.63 | 1.27 | 4.91 | 8 |
| 16384 × 16384 | off | 157 | 558 | 3.55× | 1.79 | 1.26 | 4.46 | 12 |
| 256 × 256 | on | 150 | 474 | 3.15× | 2.11 | 1.24 | 3.91 | 11 |
| 1024 × 1024 | on | 157 | 571 | 3.63× | 1.75 | 1.30 | 4.71 | 11 |
| 4096 × 4096 | on | 155 | 569 | 3.67× | 1.76 | 1.28 | 4.69 | 11 |
| 16384 × 16384 | on | 154 | 533 | 3.47× | 1.88 | 1.27 | 4.40 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (552 → 558 M cells/s), SIMD/scalar 3.55× → 3.55×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (474 → 533 M cells/s), SIMD/scalar 3.15× → 3.47×.

### MinR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 98.2 | 361 | 3.67× | 2.77 | 0.79 | 2.89 | 8 |
| 1024 × 1024 | off | 98.5 | 406 | 4.12× | 2.46 | 0.79 | 3.25 | 8 |
| 4096 × 4096 | off | 96.9 | 408 | 4.21× | 2.45 | 0.77 | 3.27 | 8 |
| 16384 × 16384 | off | 96.6 | 379 | 3.92× | 2.64 | 0.77 | 3.03 | 12 |
| 256 × 256 | on | 93.7 | 306 | 3.27× | 3.27 | 0.77 | 2.53 | 11 |
| 1024 × 1024 | on | 96.3 | 366 | 3.79× | 2.74 | 0.79 | 3.02 | 11 |
| 4096 × 4096 | on | 95.2 | 378 | 3.97× | 2.64 | 0.79 | 3.12 | 11 |
| 16384 × 16384 | on | 95.1 | 351 | 3.69× | 2.85 | 0.78 | 2.89 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (361 → 379 M cells/s), SIMD/scalar 3.67× → 3.92×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (306 → 351 M cells/s), SIMD/scalar 3.27× → 3.69×.

### MaxR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 105 | 459 | 4.36× | 2.18 | 0.84 | 3.67 | 8 |
| 1024 × 1024 | off | 102 | 502 | 4.91× | 1.99 | 0.82 | 4.01 | 8 |
| 4096 × 4096 | off | 102 | 500 | 4.90× | 2.00 | 0.82 | 4.00 | 8 |
| 16384 × 16384 | off | 102 | 465 | 4.56× | 2.15 | 0.82 | 3.72 | 12 |
| 256 × 256 | on | 102 | 402 | 3.96× | 2.49 | 0.84 | 3.32 | 11 |
| 1024 × 1024 | on | 101 | 466 | 4.62× | 2.15 | 0.83 | 3.85 | 11 |
| 4096 × 4096 | on | 101 | 468 | 4.62× | 2.13 | 0.84 | 3.87 | 11 |
| 16384 × 16384 | on | 101 | 449 | 4.45× | 2.23 | 0.83 | 3.71 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (459 → 465 M cells/s), SIMD/scalar 4.36× → 4.56×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (402 → 449 M cells/s), SIMD/scalar 3.96× → 4.45×.
<!-- stratabench output end -->

## arm64 (NEON)

The same suite on the NEON kernels of `internal/focalrow`, the run this
file published before the AVX2 one. Its headline, as first recorded:

- **Every operation is compute-bound at every radius and size** (§28).
  Throughput holds from 256² to 16384² in all 34 operation × mask rows.
  The one row `stratabench` labels otherwise, CorrelateR2 unmasked at
  16384², is scheduler noise: its three SIMD samples spread 45% (687,
  510, 458 M cells/s), the masked case at the same size — more work —
  ran at 669, and five further samples of the same case had a median of
  666 M cells/s, back within 20% of the 256² figure (below).
- **Correlate costs what its products cost.** At 4096² without masks,
  SIMD Correlate takes 0.58, 1.35, 2.59 and 6.77 ns per cell at r = 1, 2,
  3 and 5: 0.053–0.064 ns per product, flat across the radius, so 9, 25,
  49 and 121 products per cell cost in proportion. That is about 18
  billion products and sums a second on one core, without FMA (§15).
- **The separable forms grow linearly in r.** Gaussian (2(2r+1) products
  per cell) takes 0.40 ns at r = 1 and 1.12 at r = 5; Mean, 0.35 and 0.97;
  Min, 0.34 and 0.96. At r = 5 Gaussian is 6× faster than Correlate with
  the same footprint, and at r = 1 still 1.45×.
- **Memory is not the limit.** The highest demand in the suite is 28
  GB/s (Min and Mean at r = 1, 16384²), and even there throughput rises
  from 256² to 16384² (Min 2946 → 3523 M cells/s at r = 1) rather than
  falling off at a bandwidth ceiling. From r = 2 the demand falls with
  the radius, to 1.2 GB/s for Correlate at r = 5.
- **SIMD is worth 3.3–6.0× on four lanes** at 4096². More than four
  because the two backends differ in more than width: the scalar
  kernels, which are the canonical order and must stay bounds-check free
  (§39), accumulate through `dst` a term at a time, while the SIMD ones
  hold four accumulators per block in registers. The ratio is therefore
  lanes plus register blocking, not lanes alone, and should not be read
  as NEON's width advantage.

| 4096², no mask | products/cell | scalar M cells/s | NEON M cells/s | NEON/scalar | NEON ns/cell | NEON GB/s |
|---|---:|---:|---:|---:|---:|---:|
| CorrelateR1 | 9 | 425 | 1724 | 4.06× | 0.580 | 13.8 |
| CorrelateR2 | 25 | 150 | 741 | 4.94× | 1.35 | 5.92 |
| CorrelateR3 | 49 | 80.8 | 386 | 4.78× | 2.59 | 3.09 |
| CorrelateR5 | 121 | 32.9 | 148 | 4.49× | 6.77 | 1.18 |
| GaussianR1 | 6 | 636 | 2501 | 3.93× | 0.400 | 20.0 |
| GaussianR3 | 14 | 279 | 1329 | 4.76× | 0.752 | 10.6 |
| GaussianR5 | 22 | 179 | 892 | 4.97× | 1.12 | 7.14 |
| MeanR1 | — | 728 | 2900 | 3.99× | 0.345 | 23.2 |
| MeanR5 | — | 182 | 1037 | 5.71× | 0.965 | 8.29 |
| MinR1 | — | 890 | 2946 | 3.31× | 0.339 | 23.6 |
| MinR5 | — | 195 | 1039 | 5.33× | 0.963 | 8.31 |

| | |
|---|---|
| CPU | Apple M4 (4 performance + 6 efficiency cores), NEON 128-bit |
| OS | macOS (darwin/arm64), a laptop on mains power |
| Go | go1.27.0 darwin/arm64, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/focal`, then `focal.test -test.run '^$' -test.bench . -test.count 3 -test.timeout 6h`, default GOMAXPROCS (the plain functions run on one worker whatever it is). macOS cannot pin a thread to a core, so the scheduler chooses, and occasionally an efficiency core shows in a sample; the spread lines say how much |
| Raw output | [`testdata/bench-arm64.txt`](testdata/bench-arm64.txt) |
| Stats | median of 3 runs, rendered by `go run ./benchmarks/cmd/stratabench < testdata/bench-arm64.txt` |

A second look at the two noisiest 16384² cases, five more samples each
from the same binary, M cells/s: CorrelateR2 unmasked 505, 666, 720,
618, 718 (median 666, against 510 in the table); MinR5 masked 869, 719,
904, 694, 803 (median 803, against 667). Both are within 20% of their
256² figures, so compute-bound like the rest.

Working sets (the input and one output, 4 bytes per cell each, plus 1/8
byte per mask when masked): 0.5 MiB at 256², 8 MiB at 1024², 128 MiB at
4096² and 2 GiB at 16384².

| | |
|---|---|
| CPU | Apple M4 |
| Cores | 10 physical, 10 logical; 10 usable by the process, GOMAXPROCS 10 |
| Go | go1.27.0-X:simd darwin/arm64, GOEXPERIMENT=simd |
| Kernels | focalrow: neon |
| Runs | 3 per benchmark, medians shown |

### focal

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
CorrelateR1       425      1724        4.06×   not measured yet (tile engine, STRATA-8/9)
CorrelateR2       150       741        4.94×   not measured yet (tile engine, STRATA-8/9)
CorrelateR3      80.8       386        4.78×   not measured yet (tile engine, STRATA-8/9)
CorrelateR5      32.9       148        4.49×   not measured yet (tile engine, STRATA-8/9)
GaussianR1       636      2501        3.93×   not measured yet (tile engine, STRATA-8/9)
GaussianR2       392      1765        4.50×   not measured yet (tile engine, STRATA-8/9)
GaussianR3       279      1329        4.76×   not measured yet (tile engine, STRATA-8/9)
GaussianR5       179       892        4.97×   not measured yet (tile engine, STRATA-8/9)
MeanR1         728      2900        3.99×   not measured yet (tile engine, STRATA-8/9)
MeanR2         410      2148        5.23×   not measured yet (tile engine, STRATA-8/9)
MeanR3         288      1722        5.98×   not measured yet (tile engine, STRATA-8/9)
MeanR5         182      1037        5.71×   not measured yet (tile engine, STRATA-8/9)
MinR1          890      2946        3.31×   not measured yet (tile engine, STRATA-8/9)
MinR2          463      2117        4.57×   not measured yet (tile engine, STRATA-8/9)
MinR3          318      1623        5.11×   not measured yet (tile engine, STRATA-8/9)
MinR5          195      1039        5.33×   not measured yet (tile engine, STRATA-8/9)
MaxR3          311      1652        5.32×   not measured yet (tile engine, STRATA-8/9)
```

#### CorrelateR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 374 | 1427 | 3.82× | 0.701 | 2.99 | 11.4 | 9 |
| 1024 × 1024 | off | 413 | 1697 | 4.11× | 0.589 | 3.31 | 13.6 | 9 |
| 4096 × 4096 | off | 425 | 1724 | 4.06× | 0.580 | 3.40 | 13.8 | 9 |
| 16384 × 16384 | off | 393 | 1811 | 4.61× | 0.552 | 3.15 | 14.5 | 9 |
| 256 × 256 | on | 364 | 1258 | 3.46× | 0.795 | 3.00 | 10.4 | 12 |
| 1024 × 1024 | on | 406 | 1612 | 3.97× | 0.621 | 3.35 | 13.3 | 12 |
| 4096 × 4096 | on | 420 | 1632 | 3.89× | 0.613 | 3.46 | 13.5 | 12 |
| 16384 × 16384 | on | 416 | 1705 | 4.10× | 0.587 | 3.43 | 14.1 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 17%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1427 → 1811 M cells/s), SIMD/scalar 3.82× → 4.61×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1258 → 1705 M cells/s), SIMD/scalar 3.46× → 4.10×.

#### CorrelateR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 139 | 640 | 4.61× | 1.56 | 1.11 | 5.12 | 9 |
| 1024 × 1024 | off | 153 | 714 | 4.67× | 1.40 | 1.22 | 5.71 | 9 |
| 4096 × 4096 | off | 150 | 741 | 4.94× | 1.35 | 1.20 | 5.92 | 9 |
| 16384 × 16384 | off | 140 | 510 | 3.64× | 1.96 | 1.12 | 4.08 | 9 |
| 256 × 256 | on | 136 | 583 | 4.29× | 1.72 | 1.12 | 4.81 | 12 |
| 1024 × 1024 | on | 137 | 637 | 4.64× | 1.57 | 1.13 | 5.25 | 12 |
| 4096 × 4096 | on | 151 | 700 | 4.64× | 1.43 | 1.24 | 5.77 | 12 |
| 16384 × 16384 | on | 146 | 669 | 4.58× | 1.50 | 1.20 | 5.52 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 5%, worst 45%.

- mask=off: cache-bound from 16384²: SIMD throughput falls to 80% of 256²'s by 16384² (640 → 510 M cells/s), but SIMD keeps its lead (SIMD/scalar 4.61× → 3.64×).
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (583 → 669 M cells/s), SIMD/scalar 4.29× → 4.58×.

#### CorrelateR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 72.9 | 318 | 4.36× | 3.15 | 0.58 | 2.54 | 9 |
| 1024 × 1024 | off | 70.0 | 316 | 4.52× | 3.16 | 0.56 | 2.53 | 9 |
| 4096 × 4096 | off | 80.8 | 386 | 4.78× | 2.59 | 0.65 | 3.09 | 9 |
| 16384 × 16384 | off | 62.7 | 360 | 5.75× | 2.77 | 0.50 | 2.88 | 9 |
| 256 × 256 | on | 62.4 | 295 | 4.72× | 3.39 | 0.52 | 2.43 | 12 |
| 1024 × 1024 | on | 71.9 | 363 | 5.05× | 2.75 | 0.59 | 3.00 | 12 |
| 4096 × 4096 | on | 79.8 | 374 | 4.69× | 2.67 | 0.66 | 3.09 | 12 |
| 16384 × 16384 | on | 75.0 | 376 | 5.02× | 2.66 | 0.62 | 3.10 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 9%, worst 43%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (318 → 360 M cells/s), SIMD/scalar 4.36× → 5.75×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (295 → 376 M cells/s), SIMD/scalar 4.72× → 5.02×.

#### CorrelateR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 30.9 | 151 | 4.88× | 6.63 | 0.25 | 1.21 | 9 |
| 1024 × 1024 | off | 24.9 | 162 | 6.51× | 6.17 | 0.20 | 1.30 | 9 |
| 4096 × 4096 | off | 32.9 | 148 | 4.49× | 6.77 | 0.26 | 1.18 | 9 |
| 16384 × 16384 | off | 32.5 | 145 | 4.45× | 6.92 | 0.26 | 1.16 | 9 |
| 256 × 256 | on | 30.9 | 153 | 4.96× | 6.53 | 0.25 | 1.26 | 12 |
| 1024 × 1024 | on | 25.1 | 162 | 6.44× | 6.19 | 0.21 | 1.33 | 12 |
| 4096 × 4096 | on | 33.0 | 145 | 4.40× | 6.89 | 0.27 | 1.20 | 12 |
| 16384 × 16384 | on | 32.4 | 142 | 4.39× | 7.03 | 0.27 | 1.17 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (151 → 145 M cells/s), SIMD/scalar 4.88× → 4.45×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (153 → 142 M cells/s), SIMD/scalar 4.96× → 4.39×.

#### GaussianR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 538 | 2315 | 4.30× | 0.432 | 4.31 | 18.5 | 10 |
| 1024 × 1024 | off | 616 | 2625 | 4.26× | 0.381 | 4.93 | 21.0 | 10 |
| 4096 × 4096 | off | 636 | 2501 | 3.93× | 0.400 | 5.09 | 20.0 | 10 |
| 16384 × 16384 | off | 612 | 2893 | 4.72× | 0.346 | 4.90 | 23.1 | 11 |
| 256 × 256 | on | 513 | 1988 | 3.87× | 0.503 | 4.23 | 16.4 | 13 |
| 1024 × 1024 | on | 602 | 2489 | 4.13× | 0.402 | 4.97 | 20.5 | 13 |
| 4096 × 4096 | on | 626 | 2318 | 3.70× | 0.431 | 5.17 | 19.1 | 13 |
| 16384 × 16384 | on | 583 | 2624 | 4.50× | 0.381 | 4.81 | 21.6 | 14 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 37%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2315 → 2893 M cells/s), SIMD/scalar 4.30× → 4.72×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1988 → 2624 M cells/s), SIMD/scalar 3.87× → 4.50×.

#### GaussianR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 314 | 1619 | 5.16× | 0.618 | 2.51 | 12.9 | 10 |
| 1024 × 1024 | off | 374 | 1824 | 4.88× | 0.548 | 2.99 | 14.6 | 10 |
| 4096 × 4096 | off | 392 | 1765 | 4.50× | 0.567 | 3.14 | 14.1 | 10 |
| 16384 × 16384 | off | 383 | 1926 | 5.03× | 0.519 | 3.07 | 15.4 | 12 |
| 256 × 256 | on | 318 | 1370 | 4.30× | 0.730 | 2.63 | 11.3 | 13 |
| 1024 × 1024 | on | 369 | 1673 | 4.53× | 0.598 | 3.05 | 13.8 | 13 |
| 4096 × 4096 | on | 379 | 1632 | 4.31× | 0.613 | 3.12 | 13.5 | 13 |
| 16384 × 16384 | on | 388 | 1784 | 4.59× | 0.561 | 3.20 | 14.7 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 12%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1619 → 1926 M cells/s), SIMD/scalar 5.16× → 5.03×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1370 → 1784 M cells/s), SIMD/scalar 4.30× → 4.59×.

#### GaussianR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 241 | 1192 | 4.95× | 0.839 | 1.93 | 9.54 | 10 |
| 1024 × 1024 | off | 273 | 1201 | 4.39× | 0.833 | 2.19 | 9.61 | 10 |
| 4096 × 4096 | off | 279 | 1329 | 4.76× | 0.752 | 2.23 | 10.6 | 10 |
| 16384 × 16384 | off | 264 | 1349 | 5.11× | 0.741 | 2.11 | 10.8 | 12 |
| 256 × 256 | on | 239 | 1006 | 4.21× | 0.994 | 1.97 | 8.30 | 13 |
| 1024 × 1024 | on | 268 | 1212 | 4.52× | 0.825 | 2.21 | 9.99 | 13 |
| 4096 × 4096 | on | 275 | 1218 | 4.43× | 0.821 | 2.27 | 10.1 | 13 |
| 16384 × 16384 | on | 267 | 1259 | 4.71× | 0.794 | 2.20 | 10.4 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 28%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1192 → 1349 M cells/s), SIMD/scalar 4.95× → 5.11×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1006 → 1259 M cells/s), SIMD/scalar 4.21× → 4.71×.

#### GaussianR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 164 | 833 | 5.08× | 1.20 | 1.31 | 6.66 | 10 |
| 1024 × 1024 | off | 169 | 819 | 4.84× | 1.22 | 1.35 | 6.55 | 10 |
| 4096 × 4096 | off | 179 | 892 | 4.97× | 1.12 | 1.44 | 7.14 | 10 |
| 16384 × 16384 | off | 179 | 886 | 4.95× | 1.13 | 1.43 | 7.09 | 14 |
| 256 × 256 | on | 158 | 703 | 4.44× | 1.42 | 1.31 | 5.80 | 13 |
| 1024 × 1024 | on | 170 | 816 | 4.81× | 1.23 | 1.40 | 6.73 | 13 |
| 4096 × 4096 | on | 178 | 821 | 4.62× | 1.22 | 1.47 | 6.77 | 13 |
| 16384 × 16384 | on | 178 | 810 | 4.55× | 1.23 | 1.47 | 6.68 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (833 → 886 M cells/s), SIMD/scalar 5.08× → 4.95×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (703 → 810 M cells/s), SIMD/scalar 4.44× → 4.55×.

#### MeanR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 622 | 2795 | 4.49× | 0.358 | 4.98 | 22.4 | 8 |
| 1024 × 1024 | off | 713 | 3238 | 4.54× | 0.309 | 5.70 | 25.9 | 8 |
| 4096 × 4096 | off | 728 | 2900 | 3.99× | 0.345 | 5.82 | 23.2 | 8 |
| 16384 × 16384 | off | 669 | 3385 | 5.06× | 0.295 | 5.35 | 27.1 | 9 |
| 256 × 256 | on | 593 | 2354 | 3.97× | 0.425 | 4.89 | 19.4 | 11 |
| 1024 × 1024 | on | 690 | 2986 | 4.33× | 0.335 | 5.69 | 24.6 | 11 |
| 4096 × 4096 | on | 713 | 2527 | 3.54× | 0.396 | 5.88 | 20.9 | 11 |
| 16384 × 16384 | on | 670 | 3073 | 4.59× | 0.325 | 5.52 | 25.4 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 18%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2795 → 3385 M cells/s), SIMD/scalar 4.49× → 5.06×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2354 → 3073 M cells/s), SIMD/scalar 3.97× → 4.59×.

#### MeanR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 373 | 2044 | 5.48× | 0.489 | 2.98 | 16.4 | 8 |
| 1024 × 1024 | off | 412 | 2319 | 5.63× | 0.431 | 3.30 | 18.6 | 8 |
| 4096 × 4096 | off | 410 | 2148 | 5.23× | 0.466 | 3.28 | 17.2 | 8 |
| 16384 × 16384 | off | 419 | 2430 | 5.80× | 0.411 | 3.35 | 19.4 | 10 |
| 256 × 256 | on | 357 | 1654 | 4.63× | 0.605 | 2.95 | 13.7 | 11 |
| 1024 × 1024 | on | 400 | 2073 | 5.18× | 0.482 | 3.30 | 17.1 | 11 |
| 4096 × 4096 | on | 404 | 1955 | 4.84× | 0.511 | 3.33 | 16.1 | 11 |
| 16384 × 16384 | on | 413 | 2191 | 5.31× | 0.457 | 3.40 | 18.1 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 17%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2044 → 2430 M cells/s), SIMD/scalar 5.48× → 5.80×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1654 → 2191 M cells/s), SIMD/scalar 4.63× → 5.31×.

#### MeanR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 271 | 1536 | 5.67× | 0.651 | 2.17 | 12.3 | 8 |
| 1024 × 1024 | off | 292 | 1812 | 6.21× | 0.552 | 2.33 | 14.5 | 8 |
| 4096 × 4096 | off | 288 | 1722 | 5.98× | 0.581 | 2.30 | 13.8 | 8 |
| 16384 × 16384 | off | 293 | 1769 | 6.03× | 0.565 | 2.35 | 14.2 | 10 |
| 256 × 256 | on | 260 | 1243 | 4.77× | 0.805 | 2.15 | 10.2 | 11 |
| 1024 × 1024 | on | 284 | 1585 | 5.58× | 0.631 | 2.34 | 13.1 | 11 |
| 4096 × 4096 | on | 283 | 1557 | 5.50× | 0.642 | 2.34 | 12.8 | 11 |
| 16384 × 16384 | on | 281 | 1589 | 5.65× | 0.629 | 2.32 | 13.1 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1536 → 1769 M cells/s), SIMD/scalar 5.67× → 6.03×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1243 → 1589 M cells/s), SIMD/scalar 4.77× → 5.65×.

#### MeanR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 174 | 975 | 5.62× | 1.03 | 1.39 | 7.80 | 8 |
| 1024 × 1024 | off | 184 | 1097 | 5.97× | 0.911 | 1.47 | 8.78 | 8 |
| 4096 × 4096 | off | 182 | 1037 | 5.71× | 0.965 | 1.45 | 8.29 | 8 |
| 16384 × 16384 | off | 186 | 1035 | 5.56× | 0.967 | 1.49 | 8.28 | 12 |
| 256 × 256 | on | 167 | 793 | 4.74× | 1.26 | 1.38 | 6.54 | 11 |
| 1024 × 1024 | on | 179 | 944 | 5.27× | 1.06 | 1.48 | 7.79 | 11 |
| 4096 × 4096 | on | 182 | 938 | 5.15× | 1.07 | 1.50 | 7.74 | 11 |
| 16384 × 16384 | on | 183 | 906 | 4.96× | 1.10 | 1.51 | 7.48 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 32%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (975 → 1035 M cells/s), SIMD/scalar 5.62× → 5.56×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (793 → 906 M cells/s), SIMD/scalar 4.74× → 4.96×.

#### MinR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 769 | 2597 | 3.38× | 0.385 | 6.15 | 20.8 | 8 |
| 1024 × 1024 | off | 882 | 3084 | 3.50× | 0.324 | 7.05 | 24.7 | 8 |
| 4096 × 4096 | off | 890 | 2946 | 3.31× | 0.339 | 7.12 | 23.6 | 8 |
| 16384 × 16384 | off | 862 | 3523 | 4.09× | 0.284 | 6.89 | 28.2 | 9 |
| 256 × 256 | on | 734 | 2247 | 3.06× | 0.445 | 6.06 | 18.5 | 11 |
| 1024 × 1024 | on | 852 | 2762 | 3.24× | 0.362 | 7.03 | 22.8 | 11 |
| 4096 × 4096 | on | 868 | 2597 | 2.99× | 0.385 | 7.16 | 21.4 | 11 |
| 16384 × 16384 | on | 852 | 3238 | 3.80× | 0.309 | 7.03 | 26.7 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 11%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2597 → 3523 M cells/s), SIMD/scalar 3.38× → 4.09×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2247 → 3238 M cells/s), SIMD/scalar 3.06× → 3.80×.

#### MinR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 420 | 1953 | 4.65× | 0.512 | 3.36 | 15.6 | 8 |
| 1024 × 1024 | off | 462 | 2249 | 4.87× | 0.445 | 3.69 | 18.0 | 8 |
| 4096 × 4096 | off | 463 | 2117 | 4.57× | 0.472 | 3.71 | 16.9 | 8 |
| 16384 × 16384 | off | 455 | 2224 | 4.89× | 0.450 | 3.64 | 17.8 | 10 |
| 256 × 256 | on | 402 | 1638 | 4.07× | 0.611 | 3.32 | 13.5 | 11 |
| 1024 × 1024 | on | 451 | 1993 | 4.42× | 0.502 | 3.72 | 16.4 | 11 |
| 4096 × 4096 | on | 454 | 1932 | 4.25× | 0.517 | 3.75 | 15.9 | 11 |
| 16384 × 16384 | on | 450 | 2122 | 4.72× | 0.471 | 3.71 | 17.5 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 9%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1953 → 2224 M cells/s), SIMD/scalar 4.65× → 4.89×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1638 → 2122 M cells/s), SIMD/scalar 4.07× → 4.72×.

#### MinR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 294 | 1450 | 4.93× | 0.690 | 2.35 | 11.6 | 8 |
| 1024 × 1024 | off | 315 | 1531 | 4.86× | 0.653 | 2.52 | 12.2 | 8 |
| 4096 × 4096 | off | 318 | 1623 | 5.11× | 0.616 | 2.54 | 13.0 | 8 |
| 16384 × 16384 | off | 315 | 1547 | 4.91× | 0.647 | 2.52 | 12.4 | 10 |
| 256 × 256 | on | 283 | 1189 | 4.21× | 0.841 | 2.33 | 9.81 | 11 |
| 1024 × 1024 | on | 307 | 1473 | 4.80× | 0.679 | 2.53 | 12.2 | 11 |
| 4096 × 4096 | on | 312 | 1429 | 4.57× | 0.700 | 2.58 | 11.8 | 11 |
| 16384 × 16384 | on | 299 | 1516 | 5.06× | 0.660 | 2.47 | 12.5 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 32%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1450 → 1547 M cells/s), SIMD/scalar 4.93× → 4.91×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1189 → 1516 M cells/s), SIMD/scalar 4.21× → 5.06×.

#### MinR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 183 | 900 | 4.93× | 1.11 | 1.46 | 7.20 | 8 |
| 1024 × 1024 | off | 192 | 934 | 4.86× | 1.07 | 1.54 | 7.47 | 8 |
| 4096 × 4096 | off | 195 | 1039 | 5.33× | 0.963 | 1.56 | 8.31 | 8 |
| 16384 × 16384 | off | 191 | 1055 | 5.53× | 0.948 | 1.52 | 8.44 | 12 |
| 256 × 256 | on | 174 | 821 | 4.70× | 1.22 | 1.44 | 6.77 | 11 |
| 1024 × 1024 | on | 188 | 988 | 5.26× | 1.01 | 1.55 | 8.15 | 11 |
| 4096 × 4096 | on | 187 | 944 | 5.04× | 1.06 | 1.54 | 7.79 | 11 |
| 16384 × 16384 | on | 187 | 667 | 3.57× | 1.50 | 1.54 | 5.50 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 54%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (900 → 1055 M cells/s), SIMD/scalar 4.93× → 5.53×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (821 → 667 M cells/s), SIMD/scalar 4.70× → 3.57×.

#### MaxR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 288 | 1458 | 5.06× | 0.686 | 2.31 | 11.7 | 8 |
| 1024 × 1024 | off | 313 | 1743 | 5.58× | 0.574 | 2.50 | 13.9 | 8 |
| 4096 × 4096 | off | 311 | 1652 | 5.32× | 0.605 | 2.49 | 13.2 | 8 |
| 16384 × 16384 | off | 320 | 1734 | 5.42× | 0.577 | 2.56 | 13.9 | 10 |
| 256 × 256 | on | 281 | 1202 | 4.27× | 0.832 | 2.32 | 9.92 | 11 |
| 1024 × 1024 | on | 306 | 1531 | 5.00× | 0.653 | 2.53 | 12.6 | 11 |
| 4096 × 4096 | on | 312 | 1479 | 4.74× | 0.676 | 2.57 | 12.2 | 11 |
| 16384 × 16384 | on | 312 | 1561 | 5.01× | 0.640 | 2.57 | 12.9 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 12%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1458 → 1734 M cells/s), SIMD/scalar 5.06× → 5.42×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1202 → 1561 M cells/s), SIMD/scalar 4.27× → 5.01×.

The tables are `stratabench`'s: M cells/s and GB/s are medians, and
`allocs/op` is the per-call cost of the engine's one-worker path, 8
to 13 allocations depending on the operation and the mask (the separable
kernels also borrow a pooled scratch row), never per tile or per cell.
