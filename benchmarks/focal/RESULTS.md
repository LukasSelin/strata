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

## Headline (arm64, NEON; the AVX2 run is pending)

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

## AVX2 (Zen 2): pending

The suite's published numbers come from the Ryzen 9 3900X used by every
other category (§38). To produce them, in the repository root on that
machine:

```
GOEXPERIMENT=simd go test -c -o focal.test.exe ./benchmarks/focal
cd benchmarks/focal
# pinned to one logical CPU, High priority, as for benchmarks/terrain
../../focal.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 8h > testdata/bench.txt
cd ../..
go run ./benchmarks/cmd/stratabench < benchmarks/focal/testdata/bench.txt
```

and paste the output between markers here, as `benchmarks/terrain`'s
RESULTS.md has it, then add `"focal"` to `TestResultsMatch` in
`benchmarks/cmd/stratabench/main_test.go`. Correlate at r = 5 is the
long pole: at 16384² a scalar call is about 8 s.

## arm64 (NEON)

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
