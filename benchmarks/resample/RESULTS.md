# resample benchmark suite, results

The resampling category of the project benchmark suite (DESIGN.md §38,
§54): `resample.Resample`, the plain function on one goroutine, for every
method at 2× and 4× upsampling and ½ and 1/1.37 downsampling, with and
without masks for Bilinear and Lanczos, on the scalar and NEON kernels of
`internal/resamp`, and the direct 2-D evaluation (`resamp.Direct2D`) that
the separable passes replace. Metrics and names are defined in
[`../README.md`](../README.md) and in this package's `doc.go`: sizes are
the **output** side, throughput counts output cells, and GB/s counts each
source cell once.

These numbers are **arm64 (NEON) only**, from an Apple M4. The AVX2
numbers on the Zen 2 machine the other categories use are still to be
run (see *Reproducing* below); until then `testdata/bench.txt` holds the
NEON run, and it moves to `testdata/bench-arm64.txt` when they land.

## Headline

- **Every method at every scale is compute-bound** (§28), as §28
  predicted and had not measured: SIMD throughput holds from 256² to the
  largest size (16384² upsampling, 4096² downsampling, where the source is
  already 8192² or 5611²), for example Lanczos ×2 482 → 614 M cells/s and
  Lanczos ½ 131 → 144. The interpolating methods ask for 2–8 GB/s of
  source and output traffic, far below what a core can pull; what limits
  them is the taps, 2 to 12 per axis per cell. (Nearest's downsampling
  GB/s, up to 20, counts source cells it skips: it reads one per output
  cell.)
- **The separable passes beat direct 2-D evaluation by 3.3–6.5× on the
  same scalar code, and by 11–15× once they vectorise.** At 4096²:
  Cubic ×2 335 vs 73.5 M cells/s scalar (4.6×), 842 with NEON (11.5×);
  Lanczos ×2 265 vs 40.6 (6.5×), 596 NEON (14.7×); Cubic ½ 93.9 vs 28.8
  (3.3×), 197 NEON (6.8×); Lanczos ½ 72.8 vs 13.2 (5.5×), 144 NEON (10.9×).
  The direct evaluation recomputes every tap row's horizontal sum for
  every output cell (taps² products per cell instead of 2·taps), and it
  cannot vectorise without a gather per tap, which `simd/archsimd` lacks.
  Both give the same bits (`TestSeparableIsDirect`), so separability is
  the right design with nothing traded for it.
- **NEON is worth 2.0–3.3× on the interpolating methods** without masks
  (Average ×2 3.3×, Bilinear ×2 2.7×, Cubic ½ 2.1×, Lanczos ½ 2.0×), and
  nothing to Nearest, a gather copy with no arithmetic at about 1 G
  cells/s either way. Four lanes bound what NEON can give; AVX2's eight
  should do better, which the Zen 2 run will say.
- **Masks cost 5–9×** (at the largest size: Lanczos ×2 614 → 86 M cells/s, Bilinear ×2 1293 → 145), because the suite's mask (10% of source cells
  invalid, at random) puts an invalid cell in every footprint, so every
  chunk takes the masked path: two more horizontal passes, the tap counts
  and a per-cell finish with unpredictable branches. It is the worst
  case. Real NoData is clustered, and a 256-column chunk whose footprint
  is all valid takes the unmasked path, with the same bits.
- Cubic upsampling rises with size (488 → 882 M cells/s at ×2) because
  its gdalwarp bilinear fallback, recomputed per cell in scalar code, is
  a band of a few cells along the edges, a larger share of a small raster.

## Machine and method

| | |
|---|---|
| CPU | Apple M4, 10 cores (4 performance, 6 efficiency) |
| OS | macOS (darwin/arm64), a laptop on mains power with other applications and other build sessions running |
| Go | go1.27.0 darwin/arm64, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/resample`, then `resample.test -test.run '^$' -test.bench . -test.count 3 -test.timeout 4h`, not pinned, about 45 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt). The four `Lanczos<Scale>` benchmarks are from a second run, after Lanczos took gdalwarp's half-valid rule at every scale (DESIGN.md §54), which costs masked Lanczos 10–20%; the rest is unchanged by that change |
| Stats | median of 3 runs, each ≥1 s (`b.Loop`) |

The machine was not idle, so read single numbers to about ±10%; the
run-to-run spread is reported under every table.

## Reproducing

```
GOEXPERIMENT=simd go test ./benchmarks/resample -run '^$' -bench . -count 5 -timeout 4h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
```

For the AVX2 numbers, on the Zen 2 machine and as for
[`../terrain/RESULTS.md`](../terrain/RESULTS.md): build with
`GOEXPERIMENT=simd`, pin to one logical CPU with `GOMAXPROCS=1`, run with
`-count 5`, commit the output as `testdata/bench.txt` (moving this run to
`testdata/bench-arm64.txt`), and regenerate the section below with
`go run ./benchmarks/cmd/stratabench < testdata/bench.txt`.
`go test ./benchmarks/cmd/stratabench` checks that the section between
the markers is what stratabench renders from `testdata/bench.txt`.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | Apple M4 |
| Cores | 10 physical, 10 logical; 10 usable by the process, GOMAXPROCS 10 |
| Go | go1.27.0-X:simd darwin/arm64, GOEXPERIMENT=simd |
| Kernels | resamp: neon |
| Runs | 3 per benchmark, medians shown |

## resample

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
NearestUp2      1051      1060        1.01×   not measured yet (tile engine, STRATA-8/9)
NearestUp4      1067      1064        1.00×   not measured yet (tile engine, STRATA-8/9)
NearestDown2      1006      1009        1.00×   not measured yet (tile engine, STRATA-8/9)
NearestDown1p37      1054      1074        1.02×   not measured yet (tile engine, STRATA-8/9)
BilinearUp2       468      1279        2.73×   not measured yet (tile engine, STRATA-8/9)
BilinearUp4       724      1739        2.40×   not measured yet (tile engine, STRATA-8/9)
BilinearDown2       125       294        2.35×   not measured yet (tile engine, STRATA-8/9)
BilinearDown1p37       184       494        2.68×   not measured yet (tile engine, STRATA-8/9)
CubicUp2       335       842        2.51×   not measured yet (tile engine, STRATA-8/9)
CubicUp4       460      1060        2.31×   not measured yet (tile engine, STRATA-8/9)
CubicDown2      93.9       197        2.10×   not measured yet (tile engine, STRATA-8/9)
CubicDown1p37       145       339        2.34×   not measured yet (tile engine, STRATA-8/9)
LanczosUp2       265       596        2.25×   not measured yet (tile engine, STRATA-8/9)
LanczosUp4       351       748        2.13×   not measured yet (tile engine, STRATA-8/9)
LanczosDown2      72.8       144        1.98×   not measured yet (tile engine, STRATA-8/9)
LanczosDown1p37       115       250        2.18×   not measured yet (tile engine, STRATA-8/9)
AverageUp2       571      1900        3.33×   not measured yet (tile engine, STRATA-8/9)
AverageUp4       975      2907        2.98×   not measured yet (tile engine, STRATA-8/9)
AverageDown2       150       387        2.59×   not measured yet (tile engine, STRATA-8/9)
AverageDown1p37       200       545        2.72×   not measured yet (tile engine, STRATA-8/9)
CubicDirectUp2      73.5         –            –   not measured yet (tile engine, STRATA-8/9)
CubicDirectDown2      28.8         –            –   not measured yet (tile engine, STRATA-8/9)
LanczosDirectUp2      40.6         –            –   not measured yet (tile engine, STRATA-8/9)
LanczosDirectDown2      13.2         –            –   not measured yet (tile engine, STRATA-8/9)
```

### NearestUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 990 | 999 | 1.01× | 1.00 | 4.95 | 4.99 | 20 |
| 1024 × 1024 | off | 1036 | 1059 | 1.02× | 0.944 | 5.18 | 5.30 | 20 |
| 4096 × 4096 | off | 1051 | 1060 | 1.01× | 0.943 | 5.26 | 5.30 | 20 |
| 16384 × 16384 | off | 987 | 1008 | 1.02× | 0.992 | 4.93 | 5.04 | 21 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 13%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (999 → 1008 M cells/s), SIMD/scalar 1.01× → 1.02×.

### NearestUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 964 | 950 | 0.99× | 1.05 | 4.10 | 4.04 | 20 |
| 1024 × 1024 | off | 987 | 1043 | 1.06× | 0.959 | 4.19 | 4.43 | 20 |
| 4096 × 4096 | off | 1067 | 1064 | 1.00× | 0.940 | 4.54 | 4.52 | 20 |
| 16384 × 16384 | off | 1049 | 997 | 0.95× | 1.00 | 4.46 | 4.24 | 21 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 10%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (950 → 997 M cells/s), SIMD/scalar 0.99× → 0.95×.

### NearestDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 981 | 932 | 0.95× | 1.07 | 19.6 | 18.6 | 20 |
| 1024 × 1024 | off | 1012 | 998 | 0.99× | 1.00 | 20.2 | 20.0 | 20 |
| 4096 × 4096 | off | 1006 | 1009 | 1.00× | 0.991 | 20.1 | 20.2 | 20 |

Run-to-run spread of M cells/s, (max − min)/median: median 6%, worst 14%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (932 → 1009 M cells/s), SIMD/scalar 0.95× → 1.00×.

### NearestDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 1021 | 998 | 0.98× | 1.00 | 11.7 | 11.5 | 20 |
| 1024 × 1024 | off | 1115 | 1111 | 1.00× | 0.900 | 12.8 | 12.8 | 20 |
| 4096 × 4096 | off | 1054 | 1074 | 1.02× | 0.931 | 12.1 | 12.3 | 20 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (998 → 1074 M cells/s), SIMD/scalar 0.98× → 1.02×.

### BilinearUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 435 | 1186 | 2.72× | 0.843 | 2.18 | 5.93 | 24 |
| 1024 × 1024 | off | 471 | 1345 | 2.86× | 0.743 | 2.35 | 6.73 | 24 |
| 4096 × 4096 | off | 468 | 1279 | 2.73× | 0.782 | 2.34 | 6.39 | 24 |
| 16384 × 16384 | off | 469 | 1293 | 2.76× | 0.774 | 2.34 | 6.46 | 27 |
| 256 × 256 | on | 93.5 | 166 | 1.78× | 6.01 | 0.48 | 0.86 | 24 |
| 1024 × 1024 | on | 89.3 | 152 | 1.71× | 6.55 | 0.46 | 0.79 | 24 |
| 4096 × 4096 | on | 87.2 | 145 | 1.66× | 6.90 | 0.45 | 0.75 | 28 |
| 16384 × 16384 | on | 85.0 | 145 | 1.70× | 6.91 | 0.44 | 0.75 | 52 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1186 → 1293 M cells/s), SIMD/scalar 2.72× → 2.76×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (166 → 145 M cells/s), SIMD/scalar 1.78× → 1.70×.

### BilinearUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 653 | 1566 | 2.40× | 0.639 | 2.78 | 6.65 | 24 |
| 1024 × 1024 | off | 731 | 1845 | 2.52× | 0.542 | 3.11 | 7.84 | 24 |
| 4096 × 4096 | off | 724 | 1739 | 2.40× | 0.575 | 3.08 | 7.39 | 24 |
| 16384 × 16384 | off | 732 | 1761 | 2.41× | 0.568 | 3.11 | 7.49 | 26 |
| 256 × 256 | on | 128 | 206 | 1.61× | 4.84 | 0.56 | 0.90 | 24 |
| 1024 × 1024 | on | 127 | 205 | 1.61× | 4.88 | 0.56 | 0.90 | 24 |
| 4096 × 4096 | on | 123 | 193 | 1.56× | 5.19 | 0.54 | 0.84 | 27 |
| 16384 × 16384 | on | 123 | 192 | 1.56× | 5.20 | 0.54 | 0.84 | 52 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1566 → 1761 M cells/s), SIMD/scalar 2.40× → 2.41×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (206 → 192 M cells/s), SIMD/scalar 1.61× → 1.56×.

### BilinearDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 122 | 286 | 2.34× | 3.49 | 2.44 | 5.72 | 26 |
| 1024 × 1024 | off | 127 | 293 | 2.30× | 3.41 | 2.54 | 5.86 | 26 |
| 4096 × 4096 | off | 125 | 294 | 2.35× | 3.41 | 2.50 | 5.87 | 26 |
| 256 × 256 | on | 28.6 | 49.2 | 1.72× | 20.3 | 0.59 | 1.01 | 26 |
| 1024 × 1024 | on | 27.5 | 45.2 | 1.64× | 22.1 | 0.57 | 0.93 | 26 |
| 4096 × 4096 | on | 27.1 | 44.4 | 1.64× | 22.5 | 0.56 | 0.92 | 40 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (286 → 294 M cells/s), SIMD/scalar 2.34× → 2.35×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (49.2 → 44.4 M cells/s), SIMD/scalar 1.72× → 1.64×.

### BilinearDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 177 | 459 | 2.60× | 2.18 | 2.03 | 5.28 | 26 |
| 1024 × 1024 | off | 185 | 506 | 2.74× | 1.98 | 2.13 | 5.83 | 26 |
| 4096 × 4096 | off | 184 | 494 | 2.68× | 2.03 | 2.12 | 5.68 | 26 |
| 256 × 256 | on | 40.7 | 74.5 | 1.83× | 13.4 | 0.48 | 0.88 | 26 |
| 1024 × 1024 | on | 38.9 | 67.5 | 1.74× | 14.8 | 0.46 | 0.80 | 26 |
| 4096 × 4096 | on | 38.2 | 65.2 | 1.71× | 15.3 | 0.45 | 0.77 | 35 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (459 → 494 M cells/s), SIMD/scalar 2.60× → 2.68×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (74.5 → 65.2 M cells/s), SIMD/scalar 1.83× → 1.71×.

### CubicUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 262 | 488 | 1.86× | 2.05 | 1.31 | 2.44 | 49 |
| 1024 × 1024 | off | 328 | 749 | 2.28× | 1.33 | 1.64 | 3.75 | 49 |
| 4096 × 4096 | off | 335 | 842 | 2.51× | 1.19 | 1.67 | 4.21 | 49 |
| 16384 × 16384 | off | 340 | 882 | 2.60× | 1.13 | 1.70 | 4.41 | 52 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (488 → 882 M cells/s), SIMD/scalar 1.86× → 2.60×.

### CubicUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 287 | 436 | 1.52× | 2.29 | 1.22 | 1.85 | 50 |
| 1024 × 1024 | off | 426 | 833 | 1.96× | 1.20 | 1.81 | 3.54 | 50 |
| 4096 × 4096 | off | 460 | 1060 | 2.31× | 0.944 | 1.95 | 4.50 | 50 |
| 16384 × 16384 | off | 478 | 1173 | 2.45× | 0.853 | 2.03 | 4.99 | 53 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (436 → 1173 M cells/s), SIMD/scalar 1.52× → 2.45×.

### CubicDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 91.0 | 188 | 2.07× | 5.31 | 1.82 | 3.77 | 28 |
| 1024 × 1024 | off | 95.0 | 196 | 2.07× | 5.09 | 1.90 | 3.93 | 28 |
| 4096 × 4096 | off | 93.9 | 197 | 2.10× | 5.07 | 1.88 | 3.95 | 29 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (188 → 197 M cells/s), SIMD/scalar 2.07× → 2.10×.

### CubicDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 140 | 313 | 2.24× | 3.20 | 1.61 | 3.60 | 28 |
| 1024 × 1024 | off | 146 | 339 | 2.31× | 2.95 | 1.69 | 3.90 | 28 |
| 4096 × 4096 | off | 145 | 339 | 2.34× | 2.95 | 1.67 | 3.90 | 28 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (313 → 339 M cells/s), SIMD/scalar 2.24× → 2.34×.

### LanczosUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 241 | 482 | 2.00× | 2.07 | 1.21 | 2.41 | 30 |
| 1024 × 1024 | off | 271 | 586 | 2.16× | 1.71 | 1.36 | 2.93 | 30 |
| 4096 × 4096 | off | 265 | 596 | 2.25× | 1.68 | 1.32 | 2.98 | 30 |
| 16384 × 16384 | off | 268 | 614 | 2.30× | 1.63 | 1.34 | 3.07 | 37 |
| 256 × 256 | on | 53.2 | 92.0 | 1.73× | 10.9 | 0.27 | 0.47 | 30 |
| 1024 × 1024 | on | 51.7 | 90.5 | 1.75× | 11.1 | 0.27 | 0.47 | 31 |
| 4096 × 4096 | on | 49.8 | 86.0 | 1.73× | 11.6 | 0.26 | 0.44 | 40 |
| 16384 × 16384 | on | 49.8 | 86.1 | 1.73× | 11.6 | 0.26 | 0.44 | 61 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 19%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (482 → 614 M cells/s), SIMD/scalar 2.00× → 2.30×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (92.0 → 86.1 M cells/s), SIMD/scalar 1.73× → 1.73×.

### LanczosUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 306 | 562 | 1.84× | 1.78 | 1.30 | 2.39 | 30 |
| 1024 × 1024 | off | 360 | 726 | 2.02× | 1.38 | 1.53 | 3.08 | 30 |
| 4096 × 4096 | off | 351 | 748 | 2.13× | 1.34 | 1.49 | 3.18 | 30 |
| 16384 × 16384 | off | 358 | 775 | 2.17× | 1.29 | 1.52 | 3.29 | 33 |
| 256 × 256 | on | 64.8 | 109 | 1.69× | 9.15 | 0.28 | 0.48 | 30 |
| 1024 × 1024 | on | 63.9 | 110 | 1.72× | 9.11 | 0.28 | 0.48 | 31 |
| 4096 × 4096 | on | 61.0 | 104 | 1.71× | 9.61 | 0.27 | 0.46 | 37 |
| 16384 × 16384 | on | 61.0 | 104 | 1.70× | 9.65 | 0.27 | 0.45 | 61 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (562 → 775 M cells/s), SIMD/scalar 1.84× → 2.17×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (109 → 104 M cells/s), SIMD/scalar 1.69× → 1.70×.

### LanczosDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 70.0 | 131 | 1.87× | 7.66 | 1.40 | 2.61 | 32 |
| 1024 × 1024 | off | 75.0 | 144 | 1.92× | 6.96 | 1.50 | 2.87 | 32 |
| 4096 × 4096 | off | 72.8 | 144 | 1.98× | 6.95 | 1.46 | 2.88 | 33 |
| 256 × 256 | on | 16.9 | 28.5 | 1.68× | 35.1 | 0.35 | 0.59 | 32 |
| 1024 × 1024 | on | 16.8 | 27.8 | 1.66× | 36.0 | 0.35 | 0.57 | 33 |
| 4096 × 4096 | on | 16.3 | 26.9 | 1.65× | 37.2 | 0.34 | 0.55 | 63 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (131 → 144 M cells/s), SIMD/scalar 1.87× → 1.98×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (28.5 → 26.9 M cells/s), SIMD/scalar 1.68× → 1.65×.

### LanczosDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 107 | 216 | 2.01× | 4.63 | 1.24 | 2.48 | 32 |
| 1024 × 1024 | off | 116 | 250 | 2.15× | 4.01 | 1.34 | 2.87 | 32 |
| 4096 × 4096 | off | 115 | 250 | 2.18× | 4.00 | 1.32 | 2.88 | 33 |
| 256 × 256 | on | 26.4 | 46.7 | 1.77× | 21.4 | 0.31 | 0.55 | 32 |
| 1024 × 1024 | on | 25.5 | 44.7 | 1.76× | 22.4 | 0.30 | 0.53 | 33 |
| 4096 × 4096 | on | 24.6 | 42.9 | 1.74× | 23.3 | 0.29 | 0.51 | 47 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (216 → 250 M cells/s), SIMD/scalar 2.01× → 2.18×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (46.7 → 42.9 M cells/s), SIMD/scalar 1.77× → 1.74×.

### AverageUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 521 | 1460 | 2.80× | 0.685 | 2.60 | 7.30 | 22 |
| 1024 × 1024 | off | 576 | 1952 | 3.39× | 0.512 | 2.88 | 9.76 | 22 |
| 4096 × 4096 | off | 571 | 1900 | 3.33× | 0.526 | 2.86 | 9.50 | 22 |
| 16384 × 16384 | off | 580 | 1999 | 3.45× | 0.500 | 2.90 | 9.99 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1460 → 1999 M cells/s), SIMD/scalar 2.80× → 3.45×.

### AverageUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 831 | 2062 | 2.48× | 0.485 | 3.53 | 8.77 | 22 |
| 1024 × 1024 | off | 973 | 2964 | 3.05× | 0.337 | 4.13 | 12.6 | 22 |
| 4096 × 4096 | off | 975 | 2907 | 2.98× | 0.344 | 4.14 | 12.3 | 22 |
| 16384 × 16384 | off | 1008 | 3172 | 3.15× | 0.315 | 4.28 | 13.5 | 23 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2062 → 3172 M cells/s), SIMD/scalar 2.48× → 3.15×.

### AverageDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 144 | 372 | 2.59× | 2.69 | 2.87 | 7.44 | 24 |
| 1024 × 1024 | off | 150 | 401 | 2.67× | 2.50 | 3.00 | 8.02 | 24 |
| 4096 × 4096 | off | 150 | 387 | 2.59× | 2.58 | 2.99 | 7.75 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (372 → 387 M cells/s), SIMD/scalar 2.59× → 2.59×.

### AverageDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 188 | 488 | 2.60× | 2.05 | 2.16 | 5.62 | 26 |
| 1024 × 1024 | off | 200 | 560 | 2.80× | 1.79 | 2.30 | 6.44 | 26 |
| 4096 × 4096 | off | 200 | 545 | 2.72× | 1.83 | 2.31 | 6.27 | 26 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (488 → 545 M cells/s), SIMD/scalar 2.60× → 2.72×.

### CubicDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 72.8 | – | – | – | 0.36 | – | 46 |
| 1024 × 1024 | off | 74.2 | – | – | – | 0.37 | – | 46 |
| 4096 × 4096 | off | 73.5 | – | – | – | 0.37 | – | 46 |
| 16384 × 16384 | off | 73.4 | – | – | – | 0.37 | – | 46 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (72.8 → 73.4 M cells/s).

### CubicDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 31.2 | – | – | – | 0.62 | – | 25 |
| 1024 × 1024 | off | 30.7 | – | – | – | 0.61 | – | 25 |
| 4096 × 4096 | off | 28.8 | – | – | – | 0.57 | – | 25 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (31.2 → 28.8 M cells/s).

### LanczosDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 41.3 | – | – | – | 0.21 | – | 25 |
| 1024 × 1024 | off | 41.3 | – | – | – | 0.21 | – | 25 |
| 4096 × 4096 | off | 40.6 | – | – | – | 0.20 | – | 25 |
| 16384 × 16384 | off | 44.5 | – | – | – | 0.22 | – | 25 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 10%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (41.3 → 44.5 M cells/s).

### LanczosDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 14.4 | – | – | – | 0.29 | – | 27 |
| 1024 × 1024 | off | 14.0 | – | – | – | 0.28 | – | 27 |
| 4096 × 4096 | off | 13.2 | – | – | – | 0.26 | – | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 2%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (14.4 → 13.2 M cells/s).
<!-- stratabench output end -->
