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

- **Compute-bound, with one exception that is not bandwidth.** 28 of the
  34 operation × mask rows hold their SIMD throughput from 256² to
  16384². The other six are Gaussian, Mean and Min at r = 5 at 16384²,
  which run at 42–56% of their 4096² speed. `stratabench` labels them
  memory-bandwidth-bound by its rule, but they move 1.8–2.2 GB/s, a tenth
  of what one core pulls, and the same operations at 8192², 12288²,
  16320² and 16448² run at full speed. It is the power-of-two row stride,
  not the size: see [16384² and the row stride](#16384-and-the-row-stride).
- **Correlate costs what its products cost.** At 4096² without masks,
  AVX2 Correlate takes 0.93, 1.88, 3.36 and 7.95 ns per cell at r = 1, 2,
  3 and 5: 0.066–0.075 ns per product from r = 2 on (0.10 at r = 1, where
  the per-row fixed cost still shows), so its throughput falls as
  (2r+1)². That is about 14 billion products and sums a second on one
  core, without FMA (§15).
- **The separable forms grow linearly in r.** Gaussian takes 0.71 ns per
  cell at r = 1 and 1.66 at r = 5; Mean, 0.70 and 1.59; Min, 0.80 and
  2.38. At r = 5 Gaussian is 4.8× faster than Correlate with the same
  footprint, and at r = 1 still 1.3×.
- **Memory is not the limit** (outside the stride effect). The highest
  demand in the suite is 14.9 GB/s (Mean and Gaussian at r = 1, 1024²,
  in cache), under a core's 22 GB/s, and from r = 2 the demand falls
  with the radius, to 1.0 GB/s for Correlate at r = 5.
- **AVX2 is worth 3.0–5.4× over scalar** at 4096², rising with the
  radius. As on NEON, this is lanes plus register blocking, not lanes
  alone: the scalar kernels, which are the canonical order and must stay
  bounds-check free (§39), accumulate through `dst` a term at a time,
  while the SIMD ones hold several accumulators per block in registers.
  Eight lanes give about what four gave on the M4 (3.3–6.0×), and the M4's
  faster core is 1.2–2.5× ahead in absolute terms: least on Correlate at
  large radii, most on Min.

| 4096², no mask | products/cell | scalar M cells/s | AVX2 M cells/s | AVX2/scalar | AVX2 ns/cell | AVX2 GB/s | NEON M cells/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CorrelateR1 | 9 | 284 | 1074 | 3.78× | 0.931 | 8.59 | 1724 |
| CorrelateR2 | 25 | 109 | 532 | 4.89× | 1.88 | 4.25 | 741 |
| CorrelateR3 | 49 | 57.2 | 297 | 5.20× | 3.36 | 2.38 | 386 |
| CorrelateR5 | 121 | 23.5 | 126 | 5.36× | 7.95 | 1.01 | 148 |
| GaussianR1 | 6 | 394 | 1412 | 3.59× | 0.708 | 11.3 | 2501 |
| GaussianR3 | 14 | 174 | 897 | 5.16× | 1.11 | 7.17 | 1329 |
| GaussianR5 | 22 | 111 | 602 | 5.42× | 1.66 | 4.82 | 892 |
| MeanR1 | — | 434 | 1437 | 3.31× | 0.696 | 11.5 | 2900 |
| MeanR5 | — | 148 | 628 | 4.24× | 1.59 | 5.03 | 1037 |
| MinR1 | — | 424 | 1256 | 2.96× | 0.796 | 10.1 | 2946 |
| MinR5 | — | 96.8 | 419 | 4.33× | 2.38 | 3.35 | 1039 |

## 16384² and the row stride

At 16384² a row of float32 is exactly 64 KiB. The six rows above, and
milder cases, slow down there and nowhere else. The same binary, pinned
the same way, unmasked, AVX2, M cells/s (5 samples each unless marked):

| operation | 8192² (3 samples) | 12288² (3 samples) | 16320² | **16384²** | 16448² |
|---|---:|---:|---:|---:|---:|
| GaussianR5 | 569–578 | 556–558 | 580–596 | **251–258** | 584–599 |
| MeanR5 | 550–557 | 556–572 | 581–593 | **266–269** | 592–617 |
| MinR5 | | | 396–410 | **221–224** | 387–405 |
| CorrelateR5 | | | 122–125 | **92–94** | 119–123 |
| MeanR3 | | | 859–887 | **762–781** | 833–914 |

- The loss grows with the rows a column pass holds (in the suite, 4096²
  against 16384², unmasked): at most 7% at r ≤ 2 (5 rows); at r = 3 (7
  rows) 18% for Gaussian and Mean (897 → 739, 916 → 749) and 6–8% for
  Min, Max and Correlate; at r = 5 (11 rows) 2.3–2.4× for Gaussian and
  Mean, 1.9× for Min and 1.4× for Correlate.
- Only the SIMD path loses. Scalar Gaussian r = 5 runs at 111 M cells/s
  at 4096² and 112 at 16384²: slow enough that the extra misses hide.
- It is not the NEON run's: the M4 ran Gaussian r = 5 at 886 M cells/s at
  16384² against 892 at 4096².

This is consistent with set conflicts in Zen 2's L2 (512 KiB, 8-way,
whose set index repeats every 64 KiB): rows 64 KiB apart all map to the
same set, so a column pass over 11 of them, plus its destination,
exceeds the 8 ways, while at 16 KiB (4096²) or 32 KiB (8192²) the same
rows spread over four or two sets. That explanation made one prediction
before it was tested, that 8192² would be unaffected, and it held; it has
not been confirmed with performance counters. Any raster whose row is a
multiple of 64 KiB (16384, 32768, … float32 columns) is exposed, so this
is worth a kernel-side fix — blocking the column pass across the width,
or offsetting the rows it holds — rather than a note (DESIGN.md §53).

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

This is the second of two identical runs on the night of 2026-09-22. The
first, started at 23:24 while the desktop was still in use, is kept as
[`testdata/bench-busy.txt`](testdata/bench-busy.txt) and was not
published: its worst per-operation spread was 97% against 10% here, and
its case medians were a median 3.7% slower, with 28 of 272 cases 15–54%
slower, all in the same direction, as a busy sibling hyperthread on the
pinned core would make them. It showed the same six 16384² rows. The
machine's state was checked the same night against a committed category:
Slope and Hillshade at 4096² from `benchmarks/terrain`, rerun pinned,
came within 1–5% of their committed medians.

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
The stride table above adds `-test.bench '(GaussianR5|MeanR5|MinR5|MeanR3|CorrelateR5)/size=.*/mask=off/backend=simd' -strata.sizes 16320,16384,16448`.
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
CorrelateR1       284      1074        3.78×   not measured yet (tile engine, STRATA-8/9)
CorrelateR2       109       532        4.89×   not measured yet (tile engine, STRATA-8/9)
CorrelateR3      57.2       297        5.20×   not measured yet (tile engine, STRATA-8/9)
CorrelateR5      23.5       126        5.36×   not measured yet (tile engine, STRATA-8/9)
GaussianR1       394      1412        3.59×   not measured yet (tile engine, STRATA-8/9)
GaussianR2       236      1139        4.82×   not measured yet (tile engine, STRATA-8/9)
GaussianR3       174       897        5.16×   not measured yet (tile engine, STRATA-8/9)
GaussianR5       111       602        5.42×   not measured yet (tile engine, STRATA-8/9)
MeanR1         434      1437        3.31×   not measured yet (tile engine, STRATA-8/9)
MeanR2         294      1204        4.09×   not measured yet (tile engine, STRATA-8/9)
MeanR3         221       916        4.15×   not measured yet (tile engine, STRATA-8/9)
MeanR5         148       628        4.24×   not measured yet (tile engine, STRATA-8/9)
MinR1          424      1256        2.96×   not measured yet (tile engine, STRATA-8/9)
MinR2          229       839        3.66×   not measured yet (tile engine, STRATA-8/9)
MinR3          156       630        4.03×   not measured yet (tile engine, STRATA-8/9)
MinR5         96.8       419        4.33×   not measured yet (tile engine, STRATA-8/9)
MaxR3          103       508        4.93×   not measured yet (tile engine, STRATA-8/9)
```

### CorrelateR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 260 | 835 | 3.21× | 1.20 | 2.08 | 6.68 | 9 |
| 1024 × 1024 | off | 285 | 1102 | 3.87× | 0.908 | 2.28 | 8.81 | 9 |
| 4096 × 4096 | off | 284 | 1074 | 3.78× | 0.931 | 2.27 | 8.59 | 9 |
| 16384 × 16384 | off | 233 | 1037 | 4.46× | 0.964 | 1.86 | 8.30 | 9 |
| 256 × 256 | on | 249 | 718 | 2.88× | 1.39 | 2.06 | 5.92 | 12 |
| 1024 × 1024 | on | 275 | 1038 | 3.78× | 0.964 | 2.27 | 8.56 | 12 |
| 4096 × 4096 | on | 275 | 1015 | 3.69× | 0.985 | 2.27 | 8.38 | 12 |
| 16384 × 16384 | on | 228 | 969 | 4.24× | 1.03 | 1.88 | 7.99 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (835 → 1037 M cells/s), SIMD/scalar 3.21× → 4.46×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (718 → 969 M cells/s), SIMD/scalar 2.88× → 4.24×.

### CorrelateR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 104 | 388 | 3.73× | 2.58 | 0.83 | 3.11 | 9 |
| 1024 × 1024 | off | 106 | 504 | 4.74× | 1.99 | 0.85 | 4.03 | 9 |
| 4096 × 4096 | off | 109 | 532 | 4.89× | 1.88 | 0.87 | 4.25 | 9 |
| 16384 × 16384 | off | 96.3 | 526 | 5.47× | 1.90 | 0.77 | 4.21 | 9 |
| 256 × 256 | on | 100 | 357 | 3.56× | 2.80 | 0.83 | 2.95 | 12 |
| 1024 × 1024 | on | 106 | 480 | 4.52× | 2.08 | 0.88 | 3.96 | 12 |
| 4096 × 4096 | on | 109 | 513 | 4.72× | 1.95 | 0.90 | 4.23 | 12 |
| 16384 × 16384 | on | 96.1 | 495 | 5.15× | 2.02 | 0.79 | 4.08 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (388 → 526 M cells/s), SIMD/scalar 3.73× → 5.47×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (357 → 495 M cells/s), SIMD/scalar 3.56× → 5.15×.

### CorrelateR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 54.8 | 233 | 4.25× | 4.30 | 0.44 | 1.86 | 9 |
| 1024 × 1024 | off | 56.8 | 282 | 4.97× | 3.54 | 0.45 | 2.26 | 9 |
| 4096 × 4096 | off | 57.2 | 297 | 5.20× | 3.36 | 0.46 | 2.38 | 9 |
| 16384 × 16384 | off | 52.0 | 280 | 5.37× | 3.58 | 0.42 | 2.24 | 9 |
| 256 × 256 | on | 54.5 | 216 | 3.97× | 4.63 | 0.45 | 1.78 | 12 |
| 1024 × 1024 | on | 56.5 | 271 | 4.80× | 3.69 | 0.47 | 2.24 | 12 |
| 4096 × 4096 | on | 56.2 | 287 | 5.10× | 3.48 | 0.46 | 2.37 | 12 |
| 16384 × 16384 | on | 51.2 | 263 | 5.13× | 3.81 | 0.42 | 2.17 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (233 → 280 M cells/s), SIMD/scalar 4.25× → 5.37×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (216 → 263 M cells/s), SIMD/scalar 3.97× → 5.13×.

### CorrelateR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 22.5 | 104 | 4.60× | 9.65 | 0.18 | 0.83 | 9 |
| 1024 × 1024 | off | 23.6 | 119 | 5.07× | 8.38 | 0.19 | 0.96 | 9 |
| 4096 × 4096 | off | 23.5 | 126 | 5.36× | 7.95 | 0.19 | 1.01 | 9 |
| 16384 × 16384 | off | 22.3 | 89.9 | 4.03× | 11.1 | 0.18 | 0.72 | 9 |
| 256 × 256 | on | 23.0 | 100 | 4.35× | 9.99 | 0.19 | 0.83 | 12 |
| 1024 × 1024 | on | 23.4 | 117 | 4.99× | 8.56 | 0.19 | 0.96 | 12 |
| 4096 × 4096 | on | 23.2 | 123 | 5.29× | 8.12 | 0.19 | 1.02 | 12 |
| 16384 × 16384 | on | 22.2 | 90.8 | 4.08× | 11.0 | 0.18 | 0.75 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (104 → 89.9 M cells/s), SIMD/scalar 4.60× → 4.03×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (100 → 90.8 M cells/s), SIMD/scalar 4.35× → 4.08×.

### GaussianR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 374 | 1517 | 4.06× | 0.659 | 2.99 | 12.1 | 10 |
| 1024 × 1024 | off | 401 | 1849 | 4.62× | 0.541 | 3.21 | 14.8 | 10 |
| 4096 × 4096 | off | 394 | 1412 | 3.59× | 0.708 | 3.15 | 11.3 | 10 |
| 16384 × 16384 | off | 388 | 1338 | 3.45× | 0.747 | 3.10 | 10.7 | 12 |
| 256 × 256 | on | 352 | 1192 | 3.38× | 0.839 | 2.91 | 9.83 | 13 |
| 1024 × 1024 | on | 392 | 1689 | 4.31× | 0.592 | 3.23 | 13.9 | 13 |
| 4096 × 4096 | on | 379 | 1268 | 3.34× | 0.788 | 3.13 | 10.5 | 13 |
| 16384 × 16384 | on | 383 | 1278 | 3.33× | 0.782 | 3.16 | 10.6 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1517 → 1338 M cells/s), SIMD/scalar 4.06× → 3.45×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1192 → 1278 M cells/s), SIMD/scalar 3.38× → 3.33×.

### GaussianR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 231 | 1078 | 4.67× | 0.928 | 1.84 | 8.62 | 10 |
| 1024 × 1024 | off | 245 | 1234 | 5.03× | 0.810 | 1.96 | 9.88 | 10 |
| 4096 × 4096 | off | 236 | 1139 | 4.82× | 0.878 | 1.89 | 9.11 | 10 |
| 16384 × 16384 | off | 240 | 1061 | 4.42× | 0.943 | 1.92 | 8.49 | 14 |
| 256 × 256 | on | 219 | 841 | 3.84× | 1.19 | 1.81 | 6.94 | 13 |
| 1024 × 1024 | on | 238 | 1117 | 4.70× | 0.895 | 1.96 | 9.22 | 13 |
| 4096 × 4096 | on | 235 | 1028 | 4.37× | 0.973 | 1.94 | 8.48 | 13 |
| 16384 × 16384 | on | 236 | 973 | 4.13× | 1.03 | 1.95 | 8.03 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1078 → 1061 M cells/s), SIMD/scalar 4.67× → 4.42×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (841 → 973 M cells/s), SIMD/scalar 3.84× → 4.13×.

### GaussianR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 168 | 839 | 4.98× | 1.19 | 1.35 | 6.71 | 10 |
| 1024 × 1024 | off | 175 | 956 | 5.45× | 1.05 | 1.40 | 7.64 | 10 |
| 4096 × 4096 | off | 174 | 897 | 5.16× | 1.11 | 1.39 | 7.17 | 10 |
| 16384 × 16384 | off | 173 | 739 | 4.28× | 1.35 | 1.38 | 5.91 | 14 |
| 256 × 256 | on | 159 | 670 | 4.21× | 1.49 | 1.31 | 5.53 | 13 |
| 1024 × 1024 | on | 171 | 836 | 4.90× | 1.20 | 1.41 | 6.89 | 13 |
| 4096 × 4096 | on | 170 | 819 | 4.83× | 1.22 | 1.40 | 6.76 | 13 |
| 16384 × 16384 | on | 170 | 680 | 3.99× | 1.47 | 1.41 | 5.61 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (839 → 739 M cells/s), SIMD/scalar 4.98× → 4.28×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (670 → 680 M cells/s), SIMD/scalar 4.21× → 3.99×.

### GaussianR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 111 | 589 | 5.33× | 1.70 | 0.88 | 4.71 | 10 |
| 1024 × 1024 | off | 112 | 604 | 5.39× | 1.66 | 0.90 | 4.83 | 10 |
| 4096 × 4096 | off | 111 | 602 | 5.42× | 1.66 | 0.89 | 4.82 | 10 |
| 16384 × 16384 | off | 112 | 252 | 2.26× | 3.96 | 0.89 | 2.02 | 14 |
| 256 × 256 | on | 106 | 453 | 4.29× | 2.21 | 0.87 | 3.73 | 13 |
| 1024 × 1024 | on | 110 | 529 | 4.82× | 1.89 | 0.91 | 4.37 | 13 |
| 4096 × 4096 | on | 109 | 539 | 4.95× | 1.85 | 0.90 | 4.45 | 13 |
| 16384 × 16384 | on | 109 | 241 | 2.22× | 4.15 | 0.90 | 1.99 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: memory-bandwidth-bound from 16384²: SIMD throughput falls to 43% of 256²'s by 16384² (589 → 252 M cells/s) and the SIMD/scalar speedup flattens (5.33× → 2.26×); SIMD moves 2.02 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 16384²: SIMD throughput falls to 53% of 256²'s by 16384² (453 → 241 M cells/s) and the SIMD/scalar speedup flattens (4.29× → 2.22×); SIMD moves 1.99 GB/s at 16384².

### MeanR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 453 | 1542 | 3.41× | 0.648 | 3.62 | 12.3 | 8 |
| 1024 × 1024 | off | 488 | 1868 | 3.83× | 0.535 | 3.91 | 14.9 | 8 |
| 4096 × 4096 | off | 434 | 1437 | 3.31× | 0.696 | 3.47 | 11.5 | 8 |
| 16384 × 16384 | off | 417 | 1400 | 3.36× | 0.714 | 3.33 | 11.2 | 10 |
| 256 × 256 | on | 418 | 1230 | 2.94× | 0.813 | 3.45 | 10.2 | 11 |
| 1024 × 1024 | on | 468 | 1709 | 3.65× | 0.585 | 3.86 | 14.1 | 11 |
| 4096 × 4096 | on | 423 | 1314 | 3.10× | 0.761 | 3.49 | 10.8 | 11 |
| 16384 × 16384 | on | 408 | 1300 | 3.18× | 0.769 | 3.37 | 10.7 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1542 → 1400 M cells/s), SIMD/scalar 3.41× → 3.36×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1230 → 1300 M cells/s), SIMD/scalar 2.94× → 3.18×.

### MeanR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 304 | 1136 | 3.73× | 0.880 | 2.44 | 9.09 | 8 |
| 1024 × 1024 | off | 321 | 1382 | 4.30× | 0.724 | 2.57 | 11.1 | 8 |
| 4096 × 4096 | off | 294 | 1204 | 4.09× | 0.830 | 2.35 | 9.63 | 8 |
| 16384 × 16384 | off | 287 | 1139 | 3.97× | 0.878 | 2.30 | 9.11 | 10 |
| 256 × 256 | on | 286 | 912 | 3.19× | 1.10 | 2.36 | 7.52 | 11 |
| 1024 × 1024 | on | 306 | 1231 | 4.02× | 0.812 | 2.52 | 10.2 | 11 |
| 4096 × 4096 | on | 285 | 1097 | 3.85× | 0.911 | 2.35 | 9.05 | 11 |
| 16384 × 16384 | on | 278 | 1019 | 3.66× | 0.981 | 2.30 | 8.41 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 10%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1136 → 1139 M cells/s), SIMD/scalar 3.73× → 3.97×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (912 → 1019 M cells/s), SIMD/scalar 3.19× → 3.66×.

### MeanR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 232 | 878 | 3.78× | 1.14 | 1.86 | 7.03 | 8 |
| 1024 × 1024 | off | 240 | 1080 | 4.50× | 0.925 | 1.92 | 8.64 | 8 |
| 4096 × 4096 | off | 221 | 916 | 4.15× | 1.09 | 1.76 | 7.33 | 8 |
| 16384 × 16384 | off | 216 | 749 | 3.47× | 1.33 | 1.73 | 6.00 | 12 |
| 256 × 256 | on | 216 | 730 | 3.37× | 1.37 | 1.79 | 6.02 | 11 |
| 1024 × 1024 | on | 229 | 918 | 4.01× | 1.09 | 1.89 | 7.57 | 11 |
| 4096 × 4096 | on | 214 | 871 | 4.08× | 1.15 | 1.76 | 7.18 | 11 |
| 16384 × 16384 | on | 212 | 701 | 3.31× | 1.43 | 1.75 | 5.78 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 9%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (878 → 749 M cells/s), SIMD/scalar 3.78× → 3.47×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (730 → 701 M cells/s), SIMD/scalar 3.37× → 3.31×.

### MeanR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 160 | 619 | 3.87× | 1.61 | 1.28 | 4.95 | 8 |
| 1024 × 1024 | off | 159 | 661 | 4.15× | 1.51 | 1.27 | 5.29 | 8 |
| 4096 × 4096 | off | 148 | 628 | 4.24× | 1.59 | 1.18 | 5.03 | 8 |
| 16384 × 16384 | off | 147 | 270 | 1.84× | 3.71 | 1.17 | 2.16 | 12 |
| 256 × 256 | on | 147 | 498 | 3.38× | 2.01 | 1.22 | 4.11 | 11 |
| 1024 × 1024 | on | 154 | 573 | 3.72× | 1.75 | 1.27 | 4.73 | 11 |
| 4096 × 4096 | on | 145 | 561 | 3.86× | 1.78 | 1.20 | 4.63 | 11 |
| 16384 × 16384 | on | 143 | 257 | 1.80× | 3.90 | 1.18 | 2.12 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 8%.

- mask=off: memory-bandwidth-bound from 16384²: SIMD throughput falls to 44% of 256²'s by 16384² (619 → 270 M cells/s) and the SIMD/scalar speedup flattens (3.87× → 1.84×); SIMD moves 2.16 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 16384²: SIMD throughput falls to 52% of 256²'s by 16384² (498 → 257 M cells/s) and the SIMD/scalar speedup flattens (3.38× → 1.80×); SIMD moves 2.12 GB/s at 16384².

### MinR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 424 | 1224 | 2.88× | 0.817 | 3.40 | 9.79 | 8 |
| 1024 × 1024 | off | 463 | 1435 | 3.10× | 0.697 | 3.70 | 11.5 | 8 |
| 4096 × 4096 | off | 424 | 1256 | 2.96× | 0.796 | 3.39 | 10.1 | 8 |
| 16384 × 16384 | off | 419 | 1241 | 2.96× | 0.805 | 3.35 | 9.93 | 10 |
| 256 × 256 | on | 399 | 1008 | 2.53× | 0.993 | 3.29 | 8.31 | 11 |
| 1024 × 1024 | on | 445 | 1295 | 2.91× | 0.772 | 3.67 | 10.7 | 11 |
| 4096 × 4096 | on | 410 | 1170 | 2.86× | 0.855 | 3.38 | 9.65 | 11 |
| 16384 × 16384 | on | 408 | 1172 | 2.88× | 0.853 | 3.36 | 9.67 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1224 → 1241 M cells/s), SIMD/scalar 2.88× → 2.96×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1008 → 1172 M cells/s), SIMD/scalar 2.53× → 2.88×.

### MinR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 230 | 783 | 3.41× | 1.28 | 1.84 | 6.27 | 8 |
| 1024 × 1024 | off | 237 | 888 | 3.75× | 1.13 | 1.89 | 7.11 | 8 |
| 4096 × 4096 | off | 229 | 839 | 3.66× | 1.19 | 1.83 | 6.71 | 8 |
| 16384 × 16384 | off | 226 | 841 | 3.72× | 1.19 | 1.81 | 6.73 | 12 |
| 256 × 256 | on | 219 | 660 | 3.02× | 1.51 | 1.80 | 5.45 | 11 |
| 1024 × 1024 | on | 229 | 808 | 3.53× | 1.24 | 1.89 | 6.67 | 11 |
| 4096 × 4096 | on | 222 | 789 | 3.55× | 1.27 | 1.84 | 6.51 | 11 |
| 16384 × 16384 | on | 223 | 791 | 3.55× | 1.26 | 1.84 | 6.52 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (783 → 841 M cells/s), SIMD/scalar 3.41× → 3.72×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (660 → 791 M cells/s), SIMD/scalar 3.02× → 3.55×.

### MinR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 160 | 567 | 3.55× | 1.76 | 1.28 | 4.54 | 8 |
| 1024 × 1024 | off | 161 | 649 | 4.03× | 1.54 | 1.29 | 5.19 | 8 |
| 4096 × 4096 | off | 156 | 630 | 4.03× | 1.59 | 1.25 | 5.04 | 8 |
| 16384 × 16384 | off | 156 | 577 | 3.70× | 1.73 | 1.25 | 4.62 | 12 |
| 256 × 256 | on | 150 | 488 | 3.25× | 2.05 | 1.24 | 4.03 | 11 |
| 1024 × 1024 | on | 156 | 586 | 3.76× | 1.71 | 1.28 | 4.83 | 11 |
| 4096 × 4096 | on | 154 | 581 | 3.76× | 1.72 | 1.27 | 4.79 | 11 |
| 16384 × 16384 | on | 153 | 537 | 3.51× | 1.86 | 1.26 | 4.43 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (567 → 577 M cells/s), SIMD/scalar 3.55× → 3.70×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (488 → 537 M cells/s), SIMD/scalar 3.25× → 3.51×.

### MinR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 98.5 | 386 | 3.91× | 2.59 | 0.79 | 3.08 | 8 |
| 1024 × 1024 | off | 97.7 | 422 | 4.32× | 2.37 | 0.78 | 3.38 | 8 |
| 4096 × 4096 | off | 96.8 | 419 | 4.33× | 2.38 | 0.77 | 3.35 | 8 |
| 16384 × 16384 | off | 96.0 | 226 | 2.35× | 4.43 | 0.77 | 1.80 | 12 |
| 256 × 256 | on | 95.0 | 328 | 3.45× | 3.05 | 0.78 | 2.70 | 11 |
| 1024 × 1024 | on | 95.6 | 381 | 3.98× | 2.63 | 0.79 | 3.14 | 11 |
| 4096 × 4096 | on | 95.3 | 385 | 4.04× | 2.60 | 0.79 | 3.17 | 11 |
| 16384 × 16384 | on | 94.7 | 216 | 2.28× | 4.63 | 0.78 | 1.78 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: memory-bandwidth-bound from 16384²: SIMD throughput falls to 58% of 256²'s by 16384² (386 → 226 M cells/s) and the SIMD/scalar speedup flattens (3.91× → 2.35×); SIMD moves 1.80 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 16384²: SIMD throughput falls to 66% of 256²'s by 16384² (328 → 216 M cells/s) and the SIMD/scalar speedup flattens (3.45× → 2.28×); SIMD moves 1.78 GB/s at 16384².

### MaxR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 105 | 491 | 4.69× | 2.04 | 0.84 | 3.93 | 8 |
| 1024 × 1024 | off | 104 | 524 | 5.05× | 1.91 | 0.83 | 4.19 | 8 |
| 4096 × 4096 | off | 103 | 508 | 4.93× | 1.97 | 0.82 | 4.06 | 8 |
| 16384 × 16384 | off | 102 | 477 | 4.68× | 2.09 | 0.82 | 3.82 | 12 |
| 256 × 256 | on | 101 | 427 | 4.23× | 2.34 | 0.83 | 3.52 | 11 |
| 1024 × 1024 | on | 101 | 482 | 4.77× | 2.07 | 0.83 | 3.98 | 11 |
| 4096 × 4096 | on | 102 | 482 | 4.73× | 2.07 | 0.84 | 3.98 | 11 |
| 16384 × 16384 | on | 101 | 453 | 4.50× | 2.21 | 0.83 | 3.73 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (491 → 477 M cells/s), SIMD/scalar 4.69× → 4.68×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (427 → 453 M cells/s), SIMD/scalar 4.23× → 4.50×.
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
