# terrain benchmark suite, results

The terrain category of the project benchmark suite (DESIGN.md §38, §42):
the four v0.1 operations of `strata/terrain` (Gradient, Slope, Aspect,
Hillshade) through their plain public API, at 256² to 16384², with and
without validity masks, on the scalar and AVX2 kernels of
`internal/stencil`, on one worker. Metrics and names are defined in
[`../README.md`](../README.md). The engine category
([`../engine/RESULTS.md`](../engine/RESULTS.md)) measures what tiles and
workers add to two of these operations; this one measures the kernels
every path runs.

## Headline

- **Every operation is compute-bound at every size** (§28), unlike
  algebra's, which are memory-bandwidth-bound from 4096² on
  ([`../algebra/RESULTS.md`](../algebra/RESULTS.md)). Throughput is flat
  from 256² to 16384²: Aspect 489 → 543 M cells/s, Slope 597 → 706,
  Hillshade 1102 → 1248. A 3×3 stencil with an arctangent or a square
  root per cell touches only 8 bytes of memory, so SIMD Slope and Aspect
  ask for 3.9–5.7 GB/s and Hillshade for 8.8–10.9, where one core can
  pull about 22. The kernels, not the memory, are the limit — which is
  why workers scale them (engine: Slope 3.7× on 12 workers at 4096²).
- **SIMD pays most where the scalar code does transcendental maths.**
  At 4096² without masks: Aspect 8.7× (61.6 → 536 M cells/s), Hillshade
  5.7× (210 → 1202), Slope 4.1× (168 → 688), Gradient 2.6× (490 → 1282).
  Scalar Aspect, at about 62 M cells/s, is the slowest kernel the suite
  measures; its `atan2` is what the lanes speed up eightfold.
- **Gradient is the one operation near the memory ceiling.** It writes
  two outputs, so it moves 12 bytes per cell, and at about 1.3 G cells/s
  that is 15–18 GB/s — close enough to a single core's limit that other
  activity on the machine shows: its per-case medians differ by up to 64%
  between the two committed runs (median 15%), where Hillshade's differ
  by at most 3%.
- **Masks cost little for the single-output operations**: 2–10% for Slope
  and Aspect at every size. They cost most where mask traffic is highest
  and work per cell lowest: Gradient 28% at 256², falling to 4% at
  16384², and Hillshade 16% falling to 6%.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/terrain`, then `terrain.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 4h`, pinned to one logical CPU (affinity `0x10`), `GOMAXPROCS=1`, High priority. About 7 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt): **two** such runs of the same binary, one after the other, so every figure below is the median of 10 |
| Peak memory | 3.15 GiB private bytes over a whole run, sampled every 0.5 s: the 16384² Gradient fixture's three 1 GiB float32 operands and 96 MiB of masks |
| Stats | median of 10 runs, each ≥1 s (`b.Loop`). `benchstat -col /backend` gives the same medians |

To reproduce the tables below:

```
GOEXPERIMENT=simd go test ./benchmarks/terrain -run '^$' -bench . -count 5 -timeout 4h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
```

`go test ./benchmarks/cmd/stratabench` checks that the section between the
markers below is exactly the command's output for `testdata/bench.txt`.

Working sets of the operands (4 bytes per cell each, plus 1/8 byte per
mask when masked): the DEM and one output are 0.5 MiB at 256², 8 MiB at
1024², 128 MiB at 4096² and 2 GiB at 16384²; Gradient's second output
adds half as much again.

**Why ten runs.** Two runs of `-count 5` are committed because one is not
enough for Gradient. In the first run its 4096² masked case varied 86%
between samples; between the two runs its per-case medians differ by up
to 64%; and in the second run its 16384² unmasked median landed low
enough to flip that row's §28 class from compute-bound to cache-bound.
Over ten samples the classification is stable and the same for both mask
settings. The other operations are steadier: between the two runs
Hillshade's medians differ by at most 3%, Aspect's by 7% and Slope's by
25%. This was a desktop with other applications open. The spread lines
below are computed over all ten samples, so they include the drift
between the two runs as well as the variation within them.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | stencil: avx2 |
| Runs | 10 per benchmark, medians shown |

## terrain

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
Gradient       490      1282        2.62×   not measured yet (tile engine, STRATA-8/9)
Slope          168       688        4.09×   not measured yet (tile engine, STRATA-8/9)
Aspect        61.6       536        8.71×   not measured yet (tile engine, STRATA-8/9)
Hillshade       210      1202        5.71×   not measured yet (tile engine, STRATA-8/9)
```

### Gradient

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 461 | 1522 | 3.30× | 0.657 | 5.53 | 18.3 | 8 |
| 1024 × 1024 | off | 446 | 1542 | 3.46× | 0.649 | 5.35 | 18.5 | 8 |
| 4096 × 4096 | off | 490 | 1282 | 2.62× | 0.780 | 5.88 | 15.4 | 8 |
| 16384 × 16384 | off | 489 | 1268 | 2.59× | 0.789 | 5.86 | 15.2 | 8 |
| 256 × 256 | on | 427 | 1092 | 2.56× | 0.915 | 5.28 | 13.5 | 11 |
| 1024 × 1024 | on | 450 | 1300 | 2.89× | 0.769 | 5.56 | 16.1 | 11 |
| 4096 × 4096 | on | 466 | 1126 | 2.42× | 0.888 | 5.77 | 13.9 | 11 |
| 16384 × 16384 | on | 466 | 1212 | 2.60× | 0.825 | 5.76 | 15.0 | 11 |

Run-to-run spread of M cells/s, (max − min)/median: median 24%, worst 59%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1522 → 1268 M cells/s), SIMD/scalar 3.30× → 2.59×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1092 → 1212 M cells/s), SIMD/scalar 2.56× → 2.60×.

### Slope

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 170 | 597 | 3.50× | 1.67 | 1.36 | 4.78 | 8 |
| 1024 × 1024 | off | 164 | 694 | 4.23× | 1.44 | 1.31 | 5.55 | 8 |
| 4096 × 4096 | off | 168 | 688 | 4.09× | 1.45 | 1.34 | 5.50 | 8 |
| 16384 × 16384 | off | 169 | 706 | 4.17× | 1.42 | 1.36 | 5.65 | 8 |
| 256 × 256 | on | 162 | 540 | 3.34× | 1.85 | 1.33 | 4.46 | 11 |
| 1024 × 1024 | on | 168 | 646 | 3.86× | 1.55 | 1.38 | 5.33 | 11 |
| 4096 × 4096 | on | 167 | 675 | 4.05× | 1.48 | 1.38 | 5.57 | 11 |
| 16384 × 16384 | on | 146 | 644 | 4.42× | 1.55 | 1.20 | 5.32 | 11 |

Run-to-run spread of M cells/s, (max − min)/median: median 15%, worst 47%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (597 → 706 M cells/s), SIMD/scalar 3.50× → 4.17×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (540 → 644 M cells/s), SIMD/scalar 3.34× → 4.42×.

### Aspect

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 59.3 | 489 | 8.24× | 2.05 | 0.47 | 3.91 | 8 |
| 1024 × 1024 | off | 62.2 | 536 | 8.61× | 1.87 | 0.50 | 4.28 | 8 |
| 4096 × 4096 | off | 61.6 | 536 | 8.71× | 1.86 | 0.49 | 4.29 | 8 |
| 16384 × 16384 | off | 61.7 | 543 | 8.80× | 1.84 | 0.49 | 4.34 | 8 |
| 256 × 256 | on | 58.8 | 456 | 7.75× | 2.19 | 0.49 | 3.76 | 11 |
| 1024 × 1024 | on | 61.8 | 523 | 8.46× | 1.91 | 0.51 | 4.31 | 11 |
| 4096 × 4096 | on | 62.0 | 525 | 8.45× | 1.91 | 0.51 | 4.33 | 11 |
| 16384 × 16384 | on | 61.4 | 530 | 8.62× | 1.89 | 0.51 | 4.37 | 11 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 22%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (489 → 543 M cells/s), SIMD/scalar 8.24× → 8.80×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (456 → 530 M cells/s), SIMD/scalar 7.75× → 8.62×.

### Hillshade

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 214 | 1102 | 5.14× | 0.908 | 1.72 | 8.81 | 8 |
| 1024 × 1024 | off | 211 | 1358 | 6.44× | 0.737 | 1.69 | 10.9 | 8 |
| 4096 × 4096 | off | 210 | 1202 | 5.71× | 0.832 | 1.68 | 9.61 | 8 |
| 16384 × 16384 | off | 213 | 1248 | 5.86× | 0.801 | 1.70 | 9.98 | 8 |
| 256 × 256 | on | 206 | 928 | 4.51× | 1.08 | 1.70 | 7.65 | 11 |
| 1024 × 1024 | on | 207 | 1232 | 5.94× | 0.811 | 1.71 | 10.2 | 11 |
| 4096 × 4096 | on | 210 | 1152 | 5.50× | 0.868 | 1.73 | 9.50 | 11 |
| 16384 × 16384 | on | 211 | 1176 | 5.59× | 0.850 | 1.74 | 9.71 | 11 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 10%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1102 → 1248 M cells/s), SIMD/scalar 5.14× → 5.86×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (928 → 1176 M cells/s), SIMD/scalar 4.51× → 5.59×.
<!-- stratabench output end -->

The tables are `stratabench`'s: M cells/s and GB/s are the medians, and
each operation's `allocs/op` is the per-call cost of the engine's
one-worker path (8 allocations, 11 with masks), not per tile or per cell.
