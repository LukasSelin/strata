# resample benchmark suite, results

The resampling category of the project benchmark suite (DESIGN.md §38,
§54): `resample.Resample`, the plain function on one goroutine, for every
method at 2× and 4× upsampling and ½ and 1/1.37 downsampling, with and
without masks for Bilinear and Lanczos, on the scalar and SIMD kernels of
`internal/resamp`, and the direct 2-D evaluation (`resamp.Direct2D`) that
the separable passes replace. Metrics and names are defined in
[`../README.md`](../README.md) and in this package's `doc.go`: sizes are
the **output** side, throughput counts output cells, and GB/s counts each
source cell once.

The published numbers are AVX2 on the Zen 2 desktop every other category
uses. The Apple M4 NEON run that came first is kept below, under
[arm64 (NEON)](#arm64-neon).

## Headline

- **Every method at every scale is compute-bound** (§28), on AVX2 as on
  NEON: `stratabench` gives every one of the suite's operation × mask
  rows that class, with SIMD throughput holding from 256² to the largest
  size (16384² upsampling, 4096² downsampling), for example Lanczos ×2
  272 → 350 M cells/s and Lanczos ½ 82.8 → 100. The interpolating
  methods ask for at most 7.7 GB/s of source and output traffic, a third of
  what one Zen 2 core pulls (about 22 GB/s, benchmarks/algebra); the taps
  are the limit.
- **The separable passes beat direct 2-D evaluation by 3.3–6.4× on the
  same scalar code, and by 8.7–17× once they vectorise.** At 4096²:
  Cubic ×2 184 vs 36.2 M cells/s scalar (5.1×), 478 with AVX2 (13.2×);
  Lanczos ×2 138 vs 21.5 (6.4×), 362 AVX2 (16.8×); Cubic ½ 46.2 vs 14.2
  (3.3×), 123 AVX2 (8.7×); Lanczos ½ 35.8 vs 7.1 (5.0×), 100 AVX2 (14.1×).
  The NEON run gave the same shape (3.3–6.5×, 11–15×), so the design
  choice does not depend on the machine.
- **AVX2 is worth 2.3–3.2× on the interpolating methods** without masks
  at 4096² (Average ×2 3.2×, Bilinear 1/1.37 3.0×, Cubic ×2 2.6×, Lanczos
  ×4 2.3×), and nothing to Nearest. That is no more than NEON's 2.0–3.3×
  on half the lanes, which DESIGN.md §54 had expected to grow: the passes
  are not limited by lane width alone. Per core, the M4 is about 1.6–1.7×
  faster in absolute terms (Lanczos ×2 596 against 362 M cells/s,
  Bilinear ×2 1279 against 811, Average ×2 1900 against 1105).
- **Nearest runs at about 400 M cells/s**, 2.5 ns per cell, whatever the
  scale, and the same on both backends: a gather copy with no arithmetic
  and no SIMD path. On the M4 it is about 1 G cells/s.
- **Masks cost 5–9×** (at the largest size: Lanczos ×2 350 → 47.3 M
  cells/s, Bilinear ×2 741 → 81.5, Lanczos ½ 100 → 17.5), because the
  suite's mask (10% of source cells invalid, at random) puts an invalid
  cell in every footprint, so every chunk takes the masked path: two more
  horizontal passes, the tap counts and a per-cell finish with
  unpredictable branches. It is the worst case. Real NoData is clustered,
  and a 256-column chunk whose footprint is all valid takes the unmasked
  path, with the same bits.
- Cubic upsampling rises with size (283 → 509 M cells/s at ×2) because
  its gdalwarp bilinear fallback, recomputed per cell in scalar code, is
  a band of a few cells along the edges, a larger share of a small
  raster.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/resample`, then `resample.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 4h`, pinned to one logical CPU from start (`start /affinity 10 /high`), `GOMAXPROCS=1`, High priority. About 26 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) |
| Stats | median of 5 runs, each ≥1 s (`b.Loop`) |

This is the second of two identical runs on the night of 2026-09-22. The
first, started at 22:56 while the desktop was still in use, is kept as
[`testdata/bench-busy.txt`](testdata/bench-busy.txt) and was not
published: its worst per-operation spread was 63% against 11% here, and
its case medians were a median 4.5% slower, with 22 of 210 cases 17–46%
slower, all in the same direction, as a busy sibling hyperthread on the
pinned core would make them. Its slow patches got two rows labelled
cache-bound (Bilinear 1/1.37 and masked Lanczos ×4); everything else in
the headline holds for both runs. As a check
that the machine was in its usual state, Slope and Hillshade at 4096²
from `benchmarks/terrain` were rerun pinned the same night and came
within 1–5% of that category's committed medians (Slope SIMD 720 against
692, Hillshade SIMD 1237 against 1202 M cells/s).

## Reproducing

```
GOEXPERIMENT=simd go test ./benchmarks/resample -run '^$' -bench . -count 5 -timeout 4h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
```

As for [`../terrain/RESULTS.md`](../terrain/RESULTS.md), published runs pin
the test binary to one logical CPU with `GOMAXPROCS=1`. On Windows, pin
from the start so that `runtime.NumCPU` sees one CPU:
`start "" /b /wait /high /affinity 10 resample.test.exe ...` from `cmd`.
`go test ./benchmarks/cmd/stratabench` checks that the section between
the markers is what stratabench renders from `testdata/bench.txt`.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | resamp: avx2 |
| Runs | 5 per benchmark, medians shown |

## resample

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
NearestUp2       405       394        0.97×   not measured yet (tile engine, STRATA-8/9)
NearestUp4       410       404        0.99×   not measured yet (tile engine, STRATA-8/9)
NearestDown2       377       375        0.99×   not measured yet (tile engine, STRATA-8/9)
NearestDown1p37       367       373        1.02×   not measured yet (tile engine, STRATA-8/9)
BilinearUp2       273       811        2.98×   not measured yet (tile engine, STRATA-8/9)
BilinearUp4       426      1032        2.42×   not measured yet (tile engine, STRATA-8/9)
BilinearDown2      69.9       182        2.60×   not measured yet (tile engine, STRATA-8/9)
BilinearDown1p37       106       322        3.03×   not measured yet (tile engine, STRATA-8/9)
CubicUp2       184       478        2.60×   not measured yet (tile engine, STRATA-8/9)
CubicUp4       251       603        2.40×   not measured yet (tile engine, STRATA-8/9)
CubicDown2      46.2       123        2.66×   not measured yet (tile engine, STRATA-8/9)
CubicDown1p37      76.7       224        2.92×   not measured yet (tile engine, STRATA-8/9)
LanczosUp2       138       362        2.62×   not measured yet (tile engine, STRATA-8/9)
LanczosUp4       185       432        2.33×   not measured yet (tile engine, STRATA-8/9)
LanczosDown2      35.8       100        2.79×   not measured yet (tile engine, STRATA-8/9)
LanczosDown1p37      58.4       164        2.81×   not measured yet (tile engine, STRATA-8/9)
AverageUp2       341      1105        3.24×   not measured yet (tile engine, STRATA-8/9)
AverageUp4       586      1588        2.71×   not measured yet (tile engine, STRATA-8/9)
AverageDown2      87.7       242        2.77×   not measured yet (tile engine, STRATA-8/9)
AverageDown1p37       112       348        3.11×   not measured yet (tile engine, STRATA-8/9)
CubicDirectUp2      36.2         –            –   not measured yet (tile engine, STRATA-8/9)
CubicDirectDown2      14.2         –            –   not measured yet (tile engine, STRATA-8/9)
LanczosDirectUp2      21.5         –            –   not measured yet (tile engine, STRATA-8/9)
LanczosDirectDown2       7.1         –            –   not measured yet (tile engine, STRATA-8/9)
```

### NearestUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 380 | 386 | 1.01× | 2.59 | 1.90 | 1.93 | 22 |
| 1024 × 1024 | off | 408 | 400 | 0.98× | 2.50 | 2.04 | 2.00 | 22 |
| 4096 × 4096 | off | 405 | 394 | 0.97× | 2.54 | 2.02 | 1.97 | 22 |
| 16384 × 16384 | off | 403 | 405 | 1.01× | 2.47 | 2.02 | 2.03 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (386 → 405 M cells/s), SIMD/scalar 1.01× → 1.01×.

### NearestUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 385 | 389 | 1.01× | 2.57 | 1.64 | 1.65 | 22 |
| 1024 × 1024 | off | 414 | 411 | 0.99× | 2.43 | 1.76 | 1.75 | 22 |
| 4096 × 4096 | off | 410 | 404 | 0.99× | 2.47 | 1.74 | 1.72 | 22 |
| 16384 × 16384 | off | 405 | 407 | 1.01× | 2.46 | 1.72 | 1.73 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (389 → 407 M cells/s), SIMD/scalar 1.01× → 1.01×.

### NearestDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 376 | 382 | 1.02× | 2.62 | 7.53 | 7.65 | 22 |
| 1024 × 1024 | off | 370 | 366 | 0.99× | 2.73 | 7.40 | 7.32 | 22 |
| 4096 × 4096 | off | 377 | 375 | 0.99× | 2.67 | 7.54 | 7.50 | 22 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (382 → 375 M cells/s), SIMD/scalar 1.02× → 0.99×.

### NearestDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 379 | 377 | 0.99× | 2.66 | 4.37 | 4.33 | 22 |
| 1024 × 1024 | off | 393 | 400 | 1.02× | 2.50 | 4.52 | 4.60 | 22 |
| 4096 × 4096 | off | 367 | 373 | 1.02× | 2.68 | 4.22 | 4.29 | 22 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 11%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (377 → 373 M cells/s), SIMD/scalar 0.99× → 1.02×.

### BilinearUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 258 | 718 | 2.79× | 1.39 | 1.29 | 3.59 | 26 |
| 1024 × 1024 | off | 282 | 880 | 3.12× | 1.14 | 1.41 | 4.40 | 26 |
| 4096 × 4096 | off | 273 | 811 | 2.98× | 1.23 | 1.36 | 4.05 | 26 |
| 16384 × 16384 | off | 271 | 741 | 2.73× | 1.35 | 1.36 | 3.71 | 33 |
| 256 × 256 | on | 50.8 | 85.5 | 1.68× | 11.7 | 0.26 | 0.44 | 26 |
| 1024 × 1024 | on | 49.1 | 84.3 | 1.72× | 11.9 | 0.25 | 0.43 | 26 |
| 4096 × 4096 | on | 49.5 | 81.5 | 1.65× | 12.3 | 0.26 | 0.42 | 35 |
| 16384 × 16384 | on | 49.3 | 81.5 | 1.65× | 12.3 | 0.25 | 0.42 | 55 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (718 → 741 M cells/s), SIMD/scalar 2.79× → 2.73×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (85.5 → 81.5 M cells/s), SIMD/scalar 1.68× → 1.65×.

### BilinearUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 392 | 982 | 2.51× | 1.02 | 1.66 | 4.17 | 26 |
| 1024 × 1024 | off | 444 | 1259 | 2.84× | 0.794 | 1.89 | 5.35 | 26 |
| 4096 × 4096 | off | 426 | 1032 | 2.42× | 0.969 | 1.81 | 4.38 | 26 |
| 16384 × 16384 | off | 425 | 1047 | 2.46× | 0.955 | 1.81 | 4.45 | 29 |
| 256 × 256 | on | 70.1 | 108 | 1.55× | 9.21 | 0.31 | 0.48 | 26 |
| 1024 × 1024 | on | 69.9 | 108 | 1.55× | 9.23 | 0.31 | 0.47 | 26 |
| 4096 × 4096 | on | 68.2 | 105 | 1.54× | 9.54 | 0.30 | 0.46 | 33 |
| 16384 × 16384 | on | 67.8 | 102 | 1.51× | 9.77 | 0.30 | 0.45 | 55 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (982 → 1047 M cells/s), SIMD/scalar 2.51× → 2.46×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (108 → 102 M cells/s), SIMD/scalar 1.55× → 1.51×.

### BilinearDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 67.0 | 179 | 2.67× | 5.58 | 1.34 | 3.58 | 28 |
| 1024 × 1024 | off | 69.1 | 181 | 2.62× | 5.52 | 1.38 | 3.62 | 28 |
| 4096 × 4096 | off | 69.9 | 182 | 2.60× | 5.50 | 1.40 | 3.63 | 29 |
| 256 × 256 | on | 15.8 | 28.4 | 1.80× | 35.2 | 0.33 | 0.59 | 28 |
| 1024 × 1024 | on | 15.4 | 27.4 | 1.78× | 36.5 | 0.32 | 0.56 | 29 |
| 4096 × 4096 | on | 15.5 | 27.4 | 1.77× | 36.5 | 0.32 | 0.56 | 57 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (179 → 182 M cells/s), SIMD/scalar 2.67× → 2.60×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (28.4 → 27.4 M cells/s), SIMD/scalar 1.80× → 1.77×.

### BilinearDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 105 | 285 | 2.72× | 3.50 | 1.21 | 3.28 | 28 |
| 1024 × 1024 | off | 106 | 325 | 3.05× | 3.08 | 1.23 | 3.74 | 28 |
| 4096 × 4096 | off | 106 | 322 | 3.03× | 3.10 | 1.22 | 3.71 | 29 |
| 256 × 256 | on | 22.7 | 40.5 | 1.79× | 24.7 | 0.27 | 0.48 | 28 |
| 1024 × 1024 | on | 22.5 | 40.9 | 1.82× | 24.4 | 0.27 | 0.49 | 29 |
| 4096 × 4096 | on | 22.4 | 40.8 | 1.82× | 24.5 | 0.27 | 0.48 | 42 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (285 → 322 M cells/s), SIMD/scalar 2.72× → 3.03×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (40.5 → 40.8 M cells/s), SIMD/scalar 1.79× → 1.82×.

### CubicUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 143 | 283 | 1.98× | 3.54 | 0.72 | 1.42 | 53 |
| 1024 × 1024 | off | 181 | 480 | 2.64× | 2.08 | 0.91 | 2.40 | 53 |
| 4096 × 4096 | off | 184 | 478 | 2.60× | 2.09 | 0.92 | 2.39 | 53 |
| 16384 × 16384 | off | 183 | 509 | 2.78× | 1.97 | 0.92 | 2.54 | 60 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (283 → 509 M cells/s), SIMD/scalar 1.98× → 2.78×.

### CubicUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 155 | 248 | 1.60× | 4.04 | 0.66 | 1.05 | 54 |
| 1024 × 1024 | off | 235 | 520 | 2.21× | 1.92 | 1.00 | 2.21 | 54 |
| 4096 × 4096 | off | 251 | 603 | 2.40× | 1.66 | 1.07 | 2.56 | 54 |
| 16384 × 16384 | off | 258 | 598 | 2.32× | 1.67 | 1.10 | 2.54 | 61 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (248 → 598 M cells/s), SIMD/scalar 1.60× → 2.32×.

### CubicDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 44.2 | 129 | 2.92× | 7.75 | 0.88 | 2.58 | 30 |
| 1024 × 1024 | off | 46.5 | 130 | 2.80× | 7.67 | 0.93 | 2.61 | 30 |
| 4096 × 4096 | off | 46.2 | 123 | 2.66× | 8.13 | 0.93 | 2.46 | 32 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (129 → 123 M cells/s), SIMD/scalar 2.92× → 2.66×.

### CubicDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 75.8 | 198 | 2.61× | 5.06 | 0.87 | 2.27 | 30 |
| 1024 × 1024 | off | 78.2 | 231 | 2.96× | 4.32 | 0.90 | 2.66 | 30 |
| 4096 × 4096 | off | 76.7 | 224 | 2.92× | 4.47 | 0.88 | 2.57 | 31 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (198 → 224 M cells/s), SIMD/scalar 2.61× → 2.92×.

### LanczosUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 121 | 272 | 2.25× | 3.68 | 0.60 | 1.36 | 30 |
| 1024 × 1024 | off | 140 | 376 | 2.69× | 2.66 | 0.70 | 1.88 | 30 |
| 4096 × 4096 | off | 138 | 362 | 2.62× | 2.76 | 0.69 | 1.81 | 30 |
| 16384 × 16384 | off | 138 | 350 | 2.53× | 2.86 | 0.69 | 1.75 | 37 |
| 256 × 256 | on | 26.6 | 50.1 | 1.88× | 20.0 | 0.14 | 0.26 | 30 |
| 1024 × 1024 | on | 26.4 | 49.9 | 1.89× | 20.0 | 0.14 | 0.26 | 31 |
| 4096 × 4096 | on | 25.4 | 47.6 | 1.87× | 21.0 | 0.13 | 0.25 | 45 |
| 16384 × 16384 | on | 25.6 | 47.3 | 1.85× | 21.2 | 0.13 | 0.24 | 61 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (272 → 350 M cells/s), SIMD/scalar 2.25× → 2.53×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (50.1 → 47.3 M cells/s), SIMD/scalar 1.88× → 1.85×.

### LanczosUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 153 | 320 | 2.09× | 3.12 | 0.65 | 1.36 | 30 |
| 1024 × 1024 | off | 187 | 474 | 2.54× | 2.11 | 0.79 | 2.02 | 30 |
| 4096 × 4096 | off | 185 | 432 | 2.33× | 2.32 | 0.79 | 1.83 | 30 |
| 16384 × 16384 | off | 186 | 403 | 2.17× | 2.48 | 0.79 | 1.71 | 37 |
| 256 × 256 | on | 32.7 | 60.1 | 1.84× | 16.6 | 0.14 | 0.26 | 30 |
| 1024 × 1024 | on | 32.5 | 59.5 | 1.83× | 16.8 | 0.14 | 0.26 | 30 |
| 4096 × 4096 | on | 31.1 | 56.2 | 1.81× | 17.8 | 0.14 | 0.25 | 45 |
| 16384 × 16384 | on | 31.2 | 55.3 | 1.77× | 18.1 | 0.14 | 0.24 | 61 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (320 → 403 M cells/s), SIMD/scalar 2.09× → 2.17×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (60.1 → 55.3 M cells/s), SIMD/scalar 1.84× → 1.77×.

### LanczosDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 33.8 | 82.8 | 2.45× | 12.1 | 0.68 | 1.66 | 32 |
| 1024 × 1024 | off | 36.0 | 96.8 | 2.69× | 10.3 | 0.72 | 1.94 | 32 |
| 4096 × 4096 | off | 35.8 | 100 | 2.79× | 9.99 | 0.72 | 2.00 | 34 |
| 256 × 256 | on | 8.6 | 18.0 | 2.10× | 55.6 | 0.18 | 0.37 | 32 |
| 1024 × 1024 | on | 8.5 | 17.8 | 2.10× | 56.2 | 0.17 | 0.37 | 35 |
| 4096 × 4096 | on | 8.3 | 17.5 | 2.11× | 57.0 | 0.17 | 0.36 | 63 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (82.8 → 100 M cells/s), SIMD/scalar 2.45× → 2.79×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (18.0 → 17.5 M cells/s), SIMD/scalar 2.10× → 2.11×.

### LanczosDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 53.0 | 130 | 2.46× | 7.67 | 0.61 | 1.50 | 32 |
| 1024 × 1024 | off | 58.9 | 164 | 2.78× | 6.11 | 0.68 | 1.88 | 32 |
| 4096 × 4096 | off | 58.4 | 164 | 2.81× | 6.09 | 0.67 | 1.89 | 33 |
| 256 × 256 | on | 13.0 | 26.9 | 2.07× | 37.1 | 0.15 | 0.32 | 32 |
| 1024 × 1024 | on | 12.8 | 27.1 | 2.11× | 37.0 | 0.15 | 0.32 | 34 |
| 4096 × 4096 | on | 12.7 | 25.9 | 2.03× | 38.6 | 0.15 | 0.31 | 63 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (130 → 164 M cells/s), SIMD/scalar 2.46× → 2.81×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (26.9 → 25.9 M cells/s), SIMD/scalar 2.07× → 2.03×.

### AverageUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 312 | 877 | 2.81× | 1.14 | 1.56 | 4.38 | 24 |
| 1024 × 1024 | off | 352 | 1201 | 3.41× | 0.832 | 1.76 | 6.01 | 24 |
| 4096 × 4096 | off | 341 | 1105 | 3.24× | 0.905 | 1.71 | 5.52 | 24 |
| 16384 × 16384 | off | 343 | 1111 | 3.24× | 0.900 | 1.72 | 5.55 | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (877 → 1111 M cells/s), SIMD/scalar 2.81× → 3.24×.

### AverageUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 502 | 1119 | 2.23× | 0.893 | 2.13 | 4.76 | 24 |
| 1024 × 1024 | off | 618 | 1800 | 2.91× | 0.555 | 2.63 | 7.65 | 24 |
| 4096 × 4096 | off | 586 | 1588 | 2.71× | 0.630 | 2.49 | 6.75 | 24 |
| 16384 × 16384 | off | 591 | 1523 | 2.58× | 0.657 | 2.51 | 6.47 | 26 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1119 → 1523 M cells/s), SIMD/scalar 2.23× → 2.58×.

### AverageDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 85.2 | 232 | 2.72× | 4.31 | 1.70 | 4.64 | 26 |
| 1024 × 1024 | off | 88.0 | 238 | 2.70× | 4.21 | 1.76 | 4.76 | 26 |
| 4096 × 4096 | off | 87.7 | 242 | 2.77× | 4.12 | 1.75 | 4.85 | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (232 → 242 M cells/s), SIMD/scalar 2.72× → 2.77×.

### AverageDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 108 | 299 | 2.78× | 3.34 | 1.24 | 3.44 | 28 |
| 1024 × 1024 | off | 111 | 342 | 3.08× | 2.92 | 1.28 | 3.94 | 28 |
| 4096 × 4096 | off | 112 | 348 | 3.11× | 2.88 | 1.29 | 4.00 | 29 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 7%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (299 → 348 M cells/s), SIMD/scalar 2.78× → 3.11×.

### CubicDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 35.9 | – | – | – | 0.18 | – | 50 |
| 1024 × 1024 | off | 35.8 | – | – | – | 0.18 | – | 50 |
| 4096 × 4096 | off | 36.2 | – | – | – | 0.18 | – | 50 |
| 16384 × 16384 | off | 36.2 | – | – | – | 0.18 | – | 50 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (35.9 → 36.2 M cells/s).

### CubicDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 14.3 | – | – | – | 0.29 | – | 27 |
| 1024 × 1024 | off | 14.3 | – | – | – | 0.29 | – | 27 |
| 4096 × 4096 | off | 14.2 | – | – | – | 0.28 | – | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (14.3 → 14.2 M cells/s).

### LanczosDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 21.1 | – | – | – | 0.11 | – | 27 |
| 1024 × 1024 | off | 21.6 | – | – | – | 0.11 | – | 27 |
| 4096 × 4096 | off | 21.5 | – | – | – | 0.11 | – | 27 |
| 16384 × 16384 | off | 21.5 | – | – | – | 0.11 | – | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 2%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (21.1 → 21.5 M cells/s).

### LanczosDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 7.6 | – | – | – | 0.15 | – | 29 |
| 1024 × 1024 | off | 7.7 | – | – | – | 0.15 | – | 29 |
| 4096 × 4096 | off | 7.1 | – | – | – | 0.14 | – | 29 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 5%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (7.6 → 7.1 M cells/s).
<!-- stratabench output end -->

## arm64 (NEON)

The same suite on the NEON kernels of `internal/resamp`, the run this file
published before the AVX2 one. Four lanes gave 2.0–3.3× over scalar on
the interpolating methods, eight give 2.3–3.2× (above); the M4's faster
core puts its absolute throughput 1.6–1.7× above the Zen 2's.

| | |
|---|---|
| CPU | Apple M4, 10 cores (4 performance, 6 efficiency) |
| OS | macOS (darwin/arm64), a laptop on mains power with other applications and other build sessions running |
| Go | go1.27.0 darwin/arm64, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/resample`, then `resample.test -test.run '^$' -test.bench . -test.count 3 -test.timeout 4h`, not pinned, about 45 minutes |
| Raw output | [`testdata/bench-arm64.txt`](testdata/bench-arm64.txt). The four `Lanczos<Scale>` benchmarks are from a second run, after Lanczos took gdalwarp's half-valid rule at every scale (DESIGN.md §54), which costs masked Lanczos 10–20%; the rest is unchanged by that change |
| Stats | median of 3 runs, each ≥1 s (`b.Loop`) |

The machine was not idle, so read single numbers to about ±10%; the
run-to-run spread is reported under every table.

Rendered by `go run ./benchmarks/cmd/stratabench < testdata/bench-arm64.txt`:

| | |
|---|---|
| CPU | Apple M4 |
| Cores | 10 physical, 10 logical; 10 usable by the process, GOMAXPROCS 10 |
| Go | go1.27.0-X:simd darwin/arm64, GOEXPERIMENT=simd |
| Kernels | resamp: neon |
| Runs | 3 per benchmark, medians shown |

### resample

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

#### NearestUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 990 | 999 | 1.01× | 1.00 | 4.95 | 4.99 | 20 |
| 1024 × 1024 | off | 1036 | 1059 | 1.02× | 0.944 | 5.18 | 5.30 | 20 |
| 4096 × 4096 | off | 1051 | 1060 | 1.01× | 0.943 | 5.26 | 5.30 | 20 |
| 16384 × 16384 | off | 987 | 1008 | 1.02× | 0.992 | 4.93 | 5.04 | 21 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 13%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (999 → 1008 M cells/s), SIMD/scalar 1.01× → 1.02×.

#### NearestUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 964 | 950 | 0.99× | 1.05 | 4.10 | 4.04 | 20 |
| 1024 × 1024 | off | 987 | 1043 | 1.06× | 0.959 | 4.19 | 4.43 | 20 |
| 4096 × 4096 | off | 1067 | 1064 | 1.00× | 0.940 | 4.54 | 4.52 | 20 |
| 16384 × 16384 | off | 1049 | 997 | 0.95× | 1.00 | 4.46 | 4.24 | 21 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 10%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (950 → 997 M cells/s), SIMD/scalar 0.99× → 0.95×.

#### NearestDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 981 | 932 | 0.95× | 1.07 | 19.6 | 18.6 | 20 |
| 1024 × 1024 | off | 1012 | 998 | 0.99× | 1.00 | 20.2 | 20.0 | 20 |
| 4096 × 4096 | off | 1006 | 1009 | 1.00× | 0.991 | 20.1 | 20.2 | 20 |

Run-to-run spread of M cells/s, (max − min)/median: median 6%, worst 14%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (932 → 1009 M cells/s), SIMD/scalar 0.95× → 1.00×.

#### NearestDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 1021 | 998 | 0.98× | 1.00 | 11.7 | 11.5 | 20 |
| 1024 × 1024 | off | 1115 | 1111 | 1.00× | 0.900 | 12.8 | 12.8 | 20 |
| 4096 × 4096 | off | 1054 | 1074 | 1.02× | 0.931 | 12.1 | 12.3 | 20 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (998 → 1074 M cells/s), SIMD/scalar 0.98× → 1.02×.

#### BilinearUp2

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

#### BilinearUp4

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

#### BilinearDown2

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

#### BilinearDown1p37

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

#### CubicUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 262 | 488 | 1.86× | 2.05 | 1.31 | 2.44 | 49 |
| 1024 × 1024 | off | 328 | 749 | 2.28× | 1.33 | 1.64 | 3.75 | 49 |
| 4096 × 4096 | off | 335 | 842 | 2.51× | 1.19 | 1.67 | 4.21 | 49 |
| 16384 × 16384 | off | 340 | 882 | 2.60× | 1.13 | 1.70 | 4.41 | 52 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (488 → 882 M cells/s), SIMD/scalar 1.86× → 2.60×.

#### CubicUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 287 | 436 | 1.52× | 2.29 | 1.22 | 1.85 | 50 |
| 1024 × 1024 | off | 426 | 833 | 1.96× | 1.20 | 1.81 | 3.54 | 50 |
| 4096 × 4096 | off | 460 | 1060 | 2.31× | 0.944 | 1.95 | 4.50 | 50 |
| 16384 × 16384 | off | 478 | 1173 | 2.45× | 0.853 | 2.03 | 4.99 | 53 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (436 → 1173 M cells/s), SIMD/scalar 1.52× → 2.45×.

#### CubicDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 91.0 | 188 | 2.07× | 5.31 | 1.82 | 3.77 | 28 |
| 1024 × 1024 | off | 95.0 | 196 | 2.07× | 5.09 | 1.90 | 3.93 | 28 |
| 4096 × 4096 | off | 93.9 | 197 | 2.10× | 5.07 | 1.88 | 3.95 | 29 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (188 → 197 M cells/s), SIMD/scalar 2.07× → 2.10×.

#### CubicDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 140 | 313 | 2.24× | 3.20 | 1.61 | 3.60 | 28 |
| 1024 × 1024 | off | 146 | 339 | 2.31× | 2.95 | 1.69 | 3.90 | 28 |
| 4096 × 4096 | off | 145 | 339 | 2.34× | 2.95 | 1.67 | 3.90 | 28 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (313 → 339 M cells/s), SIMD/scalar 2.24× → 2.34×.

#### LanczosUp2

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

#### LanczosUp4

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

#### LanczosDown2

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

#### LanczosDown1p37

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

#### AverageUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 521 | 1460 | 2.80× | 0.685 | 2.60 | 7.30 | 22 |
| 1024 × 1024 | off | 576 | 1952 | 3.39× | 0.512 | 2.88 | 9.76 | 22 |
| 4096 × 4096 | off | 571 | 1900 | 3.33× | 0.526 | 2.86 | 9.50 | 22 |
| 16384 × 16384 | off | 580 | 1999 | 3.45× | 0.500 | 2.90 | 9.99 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1460 → 1999 M cells/s), SIMD/scalar 2.80× → 3.45×.

#### AverageUp4

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 831 | 2062 | 2.48× | 0.485 | 3.53 | 8.77 | 22 |
| 1024 × 1024 | off | 973 | 2964 | 3.05× | 0.337 | 4.13 | 12.6 | 22 |
| 4096 × 4096 | off | 975 | 2907 | 2.98× | 0.344 | 4.14 | 12.3 | 22 |
| 16384 × 16384 | off | 1008 | 3172 | 3.15× | 0.315 | 4.28 | 13.5 | 23 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (2062 → 3172 M cells/s), SIMD/scalar 2.48× → 3.15×.

#### AverageDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 144 | 372 | 2.59× | 2.69 | 2.87 | 7.44 | 24 |
| 1024 × 1024 | off | 150 | 401 | 2.67× | 2.50 | 3.00 | 8.02 | 24 |
| 4096 × 4096 | off | 150 | 387 | 2.59× | 2.58 | 2.99 | 7.75 | 24 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (372 → 387 M cells/s), SIMD/scalar 2.59× → 2.59×.

#### AverageDown1p37

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 188 | 488 | 2.60× | 2.05 | 2.16 | 5.62 | 26 |
| 1024 × 1024 | off | 200 | 560 | 2.80× | 1.79 | 2.30 | 6.44 | 26 |
| 4096 × 4096 | off | 200 | 545 | 2.72× | 1.83 | 2.31 | 6.27 | 26 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (488 → 545 M cells/s), SIMD/scalar 2.60× → 2.72×.

#### CubicDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 72.8 | – | – | – | 0.36 | – | 46 |
| 1024 × 1024 | off | 74.2 | – | – | – | 0.37 | – | 46 |
| 4096 × 4096 | off | 73.5 | – | – | – | 0.37 | – | 46 |
| 16384 × 16384 | off | 73.4 | – | – | – | 0.37 | – | 46 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (72.8 → 73.4 M cells/s).

#### CubicDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 31.2 | – | – | – | 0.62 | – | 25 |
| 1024 × 1024 | off | 30.7 | – | – | – | 0.61 | – | 25 |
| 4096 × 4096 | off | 28.8 | – | – | – | 0.57 | – | 25 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (31.2 → 28.8 M cells/s).

#### LanczosDirectUp2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 41.3 | – | – | – | 0.21 | – | 25 |
| 1024 × 1024 | off | 41.3 | – | – | – | 0.21 | – | 25 |
| 4096 × 4096 | off | 40.6 | – | – | – | 0.20 | – | 25 |
| 16384 × 16384 | off | 44.5 | – | – | – | 0.22 | – | 25 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 10%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 16384² (41.3 → 44.5 M cells/s).

#### LanczosDirectDown2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 14.4 | – | – | – | 0.29 | – | 27 |
| 1024 × 1024 | off | 14.0 | – | – | – | 0.28 | – | 27 |
| 4096 × 4096 | off | 13.2 | – | – | – | 0.26 | – | 27 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 2%.

- mask=off: compute-bound: scalar throughput stays within 20% of 256²'s up to 4096² (14.4 → 13.2 M cells/s).
