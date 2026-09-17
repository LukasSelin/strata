# STRATA-9: engine benchmark suite, results

The first multi-worker run of the project benchmark suite (DESIGN.md §26,
§28, §38): terrain.Slope, terrain.Hillshade and algebra.Clamp, plain and
through the engine with 1, 12 and 24 workers, in full-width strips and in
256×256 tiles, at 1024², 4096² and 16384². It also records the per-row
cost investigation behind the default tile shape. Metrics and names are
defined in [`../README.md`](../README.md).

## Headline

- **Full-width strips are the right default.** On one worker they are
  within ±3% of the plain function at every size. 256×256 tiles cost
  15–21% for Slope, 21–40% for Hillshade and 23–35% for Clamp at
  1024²–4096², and 36–52% at 16384². With 12 or 24 workers strips are
  still the fastest shape at every size.
- **The SIMD row kernels' fixed cost was an SSE/AVX transition**, not the
  n%8 scalar tail: about 65 ns per legacy-SSE instruction run while the
  upper YMM bits were dirty, paid once per row. Removing it cut Slope in
  degrees at 254 cells per row from 2.45 to 1.54 ns/cell and Aspect from
  3.33 to 2.04. On one pinned worker, 256×256 tiles went from +93% to +29%
  over the whole-raster call for Slope at 4096², and the plain Slope and
  Hillshade got 4–14% faster.
- **Scaling flattens at about 19–23 GB/s.** In cache (1024²) Slope runs
  5.5× faster on 12 workers than on one, Hillshade 4.4×, Clamp 1.9×. From
  4096² every operation lands at 2.4–2.9 billion cells/s whatever its
  compute cost: Slope 3.7×, Hillshade 2.1×, Clamp 1.2–1.7×. 24 workers
  are within −6% and +5% of 12.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Suite run | `GOEXPERIMENT=simd go test -c ./benchmarks/engine`, then `engine.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 4h`, **not pinned**, `GOMAXPROCS=24`, High priority |
| Micro-benchmarks | `internal/stencil` `BenchmarkRowWidth`, `internal/vec` `BenchmarkWidth` and the `internal/exec` benchmarks, pinned to one logical CPU (affinity `0x10`), `GOMAXPROCS=1`, High priority, the parent commit (5582a31) and this change alternating: 6 runs for row widths, 5 for the engine |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) (suite only) |
| Peak memory | 2.14 GiB private bytes over the whole suite run, sampled every 0.5 s: the 16384² fixture's two 1 GiB float32 operands and two masks |
| Stats | median of 5 runs, each ≥1 s (`b.Loop`) |

To reproduce the tables between the markers:

```
GOEXPERIMENT=simd go test ./benchmarks/engine -run '^$' -bench . -count 5 -timeout 4h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
```

`go test ./benchmarks/cmd/stratabench` checks that the section between the
markers is exactly the command's output for `testdata/bench.txt`.

The scalar backend runs with one worker only. The desktop had other
applications open; most cases vary 1–6% between runs, the worst 19%
(multi-worker Clamp, whose limit is memory traffic that anything else on
the machine competes for).

## Per-row cost of the row kernels

STRATA-8 found 256² tiles 60–85% slower than a whole-raster call for Slope
in the SIMD build and traced it to a fixed cost of about 270 ns per call of
the Horn row kernel. `BenchmarkRowWidth` times one call at widths from 8
to 4094 cells. Before this change (pinned, ns/cell, median of 6):

| op | 8 | 16 | 64 | 254 | 256 | 1022 | 4094 |
|---|---:|---:|---:|---:|---:|---:|---:|
| slope-percent | 3.21 | 1.82 | 0.774 | 0.585 | 0.503 | 0.465 | 0.453 |
| slope-degrees | 32.3 | 17.1 | 5.23 | 2.45 | 2.29 | 1.58 | 1.42 |
| aspect | 34.3 | 18.1 | 6.00 | 3.33 | 3.03 | 2.34 | 2.12 |
| hillshade | 11.6 | 6.48 | 1.99 | 1.09 | 0.997 | 0.777 | 0.686 |
| gradient | 3.00 | 1.73 | 0.762 | 0.539 | 0.501 | 0.450 | 0.424 |

Slope in degrees cost 259 ns at width 8, which has no tail, against 26 ns
in percent, so the n%8 scalar tail (including the scalar arctangent) was
not the fixed cost. Probes showed `hornSlopeAtanLanes` costing 229 ns over
zero cells, and one `BroadcastFloat32x8` in a function that returned
without `VZEROUPPER` costing 66 ns instead of 2. The assembly explained it:
`newAtanConsts` returned 13 vectors in Y0–Y12 and the caller copied them
into a struct with legacy (non-VEX) `MOVUPS` while the upper YMM bits were
dirty, and `hornHillshadeLanes` zeroed a vector with `XORPS` after five
broadcasts. Each such instruction costs about 65 ns on this Zen 2
(golang/go#80835). Gradient and slope in percent had none, which is why
they had no fixed cost. Broadcasts of parameters, `ClearAVXUpperBits`,
slice re-bounding and dispatch through function variables together cost a
few nanoseconds per call.

The fix builds the constant vectors once in `init`, so the lane functions
broadcast only their float32 parameters. A padded SIMD last lane instead of
the scalar tail was tried for every kernel. It paid only for Aspect (−15%
at 254 cells, −33% at 12), whose scalar tail runs Atan2F32, and made
gradient, percent slope and hillshade 21–43% slower at 12 cells, so only
Aspect keeps it. After (same method):

| op | 8 | 16 | 64 | 254 | 256 | 1022 | 4094 |
|---|---:|---:|---:|---:|---:|---:|---:|
| slope-percent | 3.25 | 1.85 | 0.760 | 0.581 | 0.509 | 0.464 | 0.457 |
| slope-degrees | 5.18 | 3.04 | 1.75 | 1.54 | 1.43 | 1.40 | 1.34 |
| aspect | 5.03 | 3.22 | 2.20 | 2.04 | 1.86 | 1.82 | 1.79 |
| hillshade | 3.49 | 2.04 | 0.991 | 0.875 | 0.738 | 0.694 | 0.675 |
| gradient | 2.95 | 1.76 | 0.734 | 0.527 | 0.499 | 0.446 | 0.422 |

Every change in the slope-degrees, aspect and hillshade rows up to 1022
cells is significant (p = 0.002); gradient and slope-percent did not
change. What remains is 5–30 ns per call: the call chain, the scalar tail
and cache effects. The `internal/vec` kernels behind Clamp had no
transition (their legacy SSE instructions are all in the scalar tail,
after `VZEROUPPER`) and cost 4–6 ns per call: Clamp is 0.259 ns/cell at
254 cells and 0.225 at 4096, unchanged by this work.

## Tile overhead on one worker

`internal/exec`'s benchmarks, pinned, median of 5, against the Tiled
function on the whole raster with one worker, in the same build:

| op | raster | 256×256 tiles, before | 256×256 tiles, after | strips of 256 rows, after |
|---|---|---:|---:|---:|
| Slope | 1024² | +63% | +21% | −1% |
| Slope | 4096² | +93% | +29% | +1% |
| Hillshade | 1024² | +65% | +30% | +0% |
| Hillshade | 4096² | +90% | +60% | +2% |
| Clamp | 1024² | +28% | +26% | −2% |
| Clamp | 4096² | +22% | +42% | −2% |

Nothing on Clamp's path changed; its before and after differ by noise. At
256 cells the row kernels' per-call cost is now 6–9% of a row, so most of
what is left is not the kernel, and it grows with the raster: each short
row of a tile starts memory streams the prefetcher has not seen, while
full-width rows stream through memory sequentially, and pointwise kernels
lose their one-vector-call-per-band path because tile views are not
compact. So the default stays one full-width tile, and a small TileWidth
is honoured as given: it is slower, and still bit-identical.

## Band size

Bands (whole rows of about 2¹⁶ cells) are the unit of scheduling across
workers. A probe over bands of 2¹², 2¹⁴, 2¹⁶, 2¹⁸ and 2²⁰ cells (Slope and
Clamp, 4096² and 16384², 1, 12 and 24 workers, 3 unpinned runs) found no
difference on one worker or for Clamp, and 3–10% more for Slope on 12–24
workers with 2¹⁸-cell bands, within the spread of unpinned runs. Larger
bands would leave a 1024² raster only 4 bands, too few for its
compute-bound scaling, so 2¹⁶ stays.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 24 usable by the process, GOMAXPROCS 24 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | stencil: avx2, vec: avx2 |
| Runs | 5 per benchmark, medians shown |

## engine

4096 × 4096 raster, no mask, M cells/sec (workers run the strips shape):

```text
              scalar      SIMD  SIMD/scalar   SIMD + 12 workers   SIMD + 24 workers
Slope            164       686        4.19×                2644                2592
Hillshade        209      1247        5.97×                2654                2642
Clamp            924      2303        2.49×                2812                2733
```

### Slope

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 163 | 709 | – | – | – | – | 5.67 | 7 |
| 1024 × 1024 | off | strips | 165 | 705 | -1% | 3877 | 3634 | 5.50× at 12 | 29.1 | 24 |
| 1024 × 1024 | off | 256x256 | 154 | 596 | -16% | 2367 | 2461 | 4.13× at 24 | 19.7 | 24 |
| 4096 × 4096 | off | plain | 164 | 686 | – | – | – | – | 5.49 | 7 |
| 4096 × 4096 | off | strips | 165 | 701 | +2% | 2644 | 2592 | 3.77× at 12 | 20.7 | 32 |
| 4096 × 4096 | off | 256x256 | 154 | 583 | -15% | 2182 | 2196 | 3.76× at 24 | 17.6 | 32 |
| 16384 × 16384 | off | plain | 147 | 668 | – | – | – | – | 5.34 | 7 |
| 16384 × 16384 | off | strips | 150 | 680 | +2% | 2456 | 2547 | 3.75× at 24 | 20.4 | 32 |
| 16384 × 16384 | off | 256x256 | 133 | 426 | -36% | 2053 | 2280 | 5.36× at 24 | 18.2 | 32 |
| 1024 × 1024 | on | plain | 162 | 665 | – | – | – | – | 5.49 | 10 |
| 1024 × 1024 | on | strips | 162 | 657 | -1% | 3534 | 3346 | 5.38× at 12 | 27.6 | 27 |
| 1024 × 1024 | on | 256x256 | 149 | 531 | -20% | 1939 | 1952 | 3.68× at 24 | 16.1 | 27 |
| 4096 × 4096 | on | plain | 163 | 668 | – | – | – | – | 5.51 | 10 |
| 4096 × 4096 | on | strips | 162 | 670 | +0% | 2514 | 2465 | 3.75× at 12 | 20.3 | 35 |
| 4096 × 4096 | on | 256x256 | 152 | 527 | -21% | 1836 | 1843 | 3.50× at 24 | 15.2 | 36 |
| 16384 × 16384 | on | plain | 148 | 660 | – | – | – | – | 5.44 | 10 |
| 16384 × 16384 | on | strips | 148 | 648 | -2% | 2375 | 2292 | 3.66× at 12 | 18.9 | 42 |
| 16384 × 16384 | on | 256x256 | 128 | 389 | -41% | 1751 | 1791 | 4.61× at 24 | 14.8 | 62 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (709 → 668 M cells/s), SIMD/scalar 4.34× → 4.53×.
- mask=off, workers: strips over one worker, SIMD: 1024² 5.50× with 12, 5.15× with 24; 4096² 3.77× with 12, 3.70× with 24; 16384² 3.61× with 12, 3.75× with 24; 24 workers move 29.1 GB/s at 1024², 20.7 GB/s at 4096², 20.4 GB/s at 16384².
- mask=on, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (665 → 660 M cells/s), SIMD/scalar 4.12× → 4.46×.
- mask=on, workers: strips over one worker, SIMD: 1024² 5.38× with 12, 5.09× with 24; 4096² 3.75× with 12, 3.68× with 24; 16384² 3.66× with 12, 3.54× with 24; 24 workers move 27.6 GB/s at 1024², 20.3 GB/s at 4096², 18.9 GB/s at 16384².

### Hillshade

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 213 | 1350 | – | – | – | – | 10.8 | 7 |
| 1024 × 1024 | off | strips | 209 | 1371 | +2% | 5976 | 6074 | 4.43× at 24 | 48.6 | 24 |
| 1024 × 1024 | off | 256x256 | 202 | 1060 | -21% | 3409 | 3366 | 3.22× at 12 | 26.9 | 24 |
| 4096 × 4096 | off | plain | 209 | 1247 | – | – | – | – | 9.98 | 7 |
| 4096 × 4096 | off | strips | 211 | 1255 | +1% | 2654 | 2642 | 2.11× at 12 | 21.1 | 32 |
| 4096 × 4096 | off | 256x256 | 202 | 818 | -34% | 2375 | 2326 | 2.90× at 12 | 18.6 | 32 |
| 16384 × 16384 | off | plain | 168 | 1250 | – | – | – | – | 10.00 | 7 |
| 16384 × 16384 | off | strips | 167 | 1254 | +0% | 2437 | 2557 | 2.04× at 24 | 20.5 | 32 |
| 16384 × 16384 | off | 256x256 | 160 | 634 | -49% | 2205 | 2335 | 3.68× at 24 | 18.7 | 32 |
| 1024 × 1024 | on | plain | 208 | 1220 | – | – | – | – | 10.1 | 10 |
| 1024 × 1024 | on | strips | 207 | 1213 | -1% | 4812 | 4749 | 3.97× at 12 | 39.2 | 27 |
| 1024 × 1024 | on | 256x256 | 196 | 893 | -27% | 2291 | 2228 | 2.57× at 12 | 18.4 | 27 |
| 4096 × 4096 | on | plain | 209 | 1174 | – | – | – | – | 9.68 | 10 |
| 4096 × 4096 | on | strips | 209 | 1179 | +0% | 2592 | 2541 | 2.20× at 12 | 21.0 | 35 |
| 4096 × 4096 | on | 256x256 | 193 | 703 | -40% | 2050 | 2009 | 2.92× at 12 | 16.6 | 36 |
| 16384 × 16384 | on | plain | 166 | 1167 | – | – | – | – | 9.63 | 10 |
| 16384 × 16384 | on | strips | 167 | 1167 | +0% | 2329 | 2310 | 2.00× at 12 | 19.1 | 43 |
| 16384 × 16384 | on | 256x256 | 153 | 560 | -52% | 1889 | 1878 | 3.37× at 12 | 15.5 | 58 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 15%.

- mask=off, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (1350 → 1250 M cells/s), SIMD/scalar 6.34× → 7.44×.
- mask=off, workers: strips over one worker, SIMD: 1024² 4.36× with 12, 4.43× with 24; 4096² 2.11× with 12, 2.11× with 24; 16384² 1.94× with 12, 2.04× with 24; 24 workers move 48.6 GB/s at 1024², 21.1 GB/s at 4096², 20.5 GB/s at 16384².
- mask=on, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (1220 → 1167 M cells/s), SIMD/scalar 5.85× → 7.02×.
- mask=on, workers: strips over one worker, SIMD: 1024² 3.97× with 12, 3.92× with 24; 4096² 2.20× with 12, 2.16× with 24; 16384² 2.00× with 12, 1.98× with 24; 24 workers move 39.2 GB/s at 1024², 21.0 GB/s at 4096², 19.1 GB/s at 16384².

### Clamp

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 1017 | 4438 | – | – | – | – | 35.5 | 0 |
| 1024 × 1024 | off | strips | 1013 | 4429 | -0% | 8407 | 8224 | 1.90× at 12 | 65.8 | 24 |
| 1024 × 1024 | off | 256x256 | 935 | 3407 | -23% | 4627 | 4516 | 1.36× at 12 | 36.1 | 24 |
| 4096 × 4096 | off | plain | 924 | 2303 | – | – | – | – | 18.4 | 0 |
| 4096 × 4096 | off | strips | 930 | 2269 | -1% | 2812 | 2733 | 1.24× at 12 | 21.9 | 32 |
| 4096 × 4096 | off | 256x256 | 794 | 1661 | -28% | 2515 | 2398 | 1.51× at 12 | 19.2 | 32 |
| 16384 × 16384 | off | plain | 882 | 1666 | – | – | – | – | 13.3 | 0 |
| 16384 × 16384 | off | strips | 890 | 1657 | -1% | 2764 | 2879 | 1.74× at 24 | 23.0 | 32 |
| 16384 × 16384 | off | 256x256 | 538 | 881 | -47% | 2607 | 2707 | 3.07× at 24 | 21.7 | 32 |
| 1024 × 1024 | on | plain | 1012 | 4338 | – | – | – | – | 35.8 | 0 |
| 1024 × 1024 | on | strips | 1010 | 4341 | +0% | 8166 | 8032 | 1.88× at 12 | 66.3 | 26 |
| 1024 × 1024 | on | 256x256 | 884 | 2926 | -33% | 3691 | 3663 | 1.26× at 12 | 30.2 | 26 |
| 4096 × 4096 | on | plain | 919 | 2255 | – | – | – | – | 18.6 | 0 |
| 4096 × 4096 | on | strips | 921 | 2179 | -3% | 2698 | 2611 | 1.24× at 12 | 21.5 | 34 |
| 4096 × 4096 | on | 256x256 | 743 | 1460 | -35% | 2287 | 2236 | 1.57× at 12 | 18.4 | 34 |
| 16384 × 16384 | on | plain | 870 | 1631 | – | – | – | – | 13.5 | 0 |
| 16384 × 16384 | on | strips | 859 | 1620 | -1% | 2669 | 2617 | 1.65× at 12 | 21.6 | 39 |
| 16384 × 16384 | on | 256x256 | 497 | 820 | -50% | 2368 | 2311 | 2.89× at 12 | 19.1 | 46 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 19%.

- mask=off, one worker: memory-bandwidth-bound from 4096²: SIMD throughput falls to 38% of 1024²'s by 16384² (4438 → 1666 M cells/s) and the SIMD/scalar speedup flattens (4.36× → 1.89×); SIMD moves 18.4 GB/s at 4096², 13.3 GB/s at 16384².
- mask=off, workers: strips over one worker, SIMD: 1024² 1.90× with 12, 1.86× with 24; 4096² 1.24× with 12, 1.20× with 24; 16384² 1.67× with 12, 1.74× with 24; 24 workers move 65.8 GB/s at 1024², 21.9 GB/s at 4096², 23.0 GB/s at 16384².
- mask=on, one worker: memory-bandwidth-bound from 4096²: SIMD throughput falls to 38% of 1024²'s by 16384² (4338 → 1631 M cells/s) and the SIMD/scalar speedup flattens (4.29× → 1.88×); SIMD moves 18.6 GB/s at 4096², 13.5 GB/s at 16384².
- mask=on, workers: strips over one worker, SIMD: 1024² 1.88× with 12, 1.85× with 24; 4096² 1.24× with 12, 1.20× with 24; 16384² 1.65× with 12, 1.62× with 24; 24 workers move 66.3 GB/s at 1024², 21.5 GB/s at 4096², 21.6 GB/s at 16384².
<!-- stratabench output end -->

## What the numbers say (§28)

- **On one worker the terrain kernels are compute-bound** at every size
  (Slope 709 → 668 M cells/s from 1024² to 16384²): they do 0.8–1.4 ns of
  arithmetic per cell for 8 bytes of traffic. Clamp is
  memory-bandwidth-bound from 4096², as in benchmarks/algebra.
- **Workers scale while the working set is in cache.** At 1024² (8 MiB of
  operands, half of one CCX's L3) Slope gains 5.5× on 12 workers and
  Hillshade 4.4×, moving 29 and 49 GB/s. Clamp gains only 1.9× even there:
  one worker already moves 36 GB/s.
- **From 4096² scaling flattens at 19–23 GB/s for every operation.** Slope,
  Hillshade and Clamp end at 2.4–2.9 billion cells/s although one-worker
  Clamp is 3.4× faster than Slope at 4096²: the limit is the two DDR4-3200
  channels, not the cores. How much a kernel gains depends on how far
  below that ceiling one worker is: Slope 3.7×, Hillshade 2.1×, Clamp
  1.2–1.7×. Clamp gains more at 16384² than at 4096² because its one
  worker is slower there (TLB and prefetcher misses).
- **24 workers are no faster than 12** (−6% to +5%): SMT siblings share
  the memory stream as well as the core. `Workers: 0` (GOMAXPROCS) and one
  worker per physical core give the same throughput on this machine.
- **Masks do not limit scaling.** mask=on scales within 0–10% of mask=off
  (the largest gap is Hillshade at 1024²), so the lock that serialises
  validity words costs little next to the Data work.
- **Tiles do not help these kernels' cache behaviour.** 256×256 tiles
  scale better relative to their own one-worker run only because that run
  is slower; in absolute throughput strips win at every size and worker
  count.

For the engine demo (§43) this means about 2.5 billion cells/s for Slope
on this class of machine from 4096² up with 12 workers. Beyond that,
throughput needs fewer bytes moved per cell, which is what operation
fusion (§29) offers.
