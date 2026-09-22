# STRATA-10: algebra benchmark suite, results

The first run of the project benchmark suite (DESIGN.md §38, §42): the six
v0.1 operations of `strata/algebra` through the public API, at 256² to
16384², with and without validity masks, on the scalar and AVX2 kernels of
`internal/vec`, with one worker. Metrics and names are defined in
[`../README.md`](../README.md).

## Headline

At 4096 × 4096 the operations run at 1.8–2.3 billion cells/s on one core
with SIMD, but SIMD adds only 1.1–1.2× to Add, Sub and Mul, 1.5–1.8× to Min
and Max, and 2.9× to Clamp. From 4096² on, every operation is
memory-bandwidth-bound (§28). The CPU could compute 4–6 billion cells/s, as
it does while the operands fit in cache (256², 1024²), but single-threaded
access can only move about 22 GB/s at 4096² and 14 GB/s at 16384², and
throughput is capped there. Workers (STRATA-8/9) and operation fusion (§29),
not wider lanes, are what can raise large-raster throughput for these
kernels.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/algebra`, then `algebra.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 3h`, pinned to one logical CPU (affinity `0x10`), `GOMAXPROCS=1`, High priority. About 10 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) |
| Stats | median of 5 runs, each ≥1 s (`b.Loop`). `benchstat -col /backend` gives the same medians |

To reproduce the tables below:

```
GOEXPERIMENT=simd go test ./benchmarks/algebra -run '^$' -bench . -count 5 -timeout 2h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
```

`go test ./benchmarks/cmd/stratabench` checks that the section between the
markers below is exactly the command's output for `testdata/bench.txt`.

Working sets of the three operands (two inputs and dst, 4 bytes per cell,
plus 1/8 byte per mask when masked): 256² 0.79 MiB (larger than L2, fits in
L3), 1024² 12.6 MiB (fits in one CCX's L3), 4096² 201 MiB, 16384² 3.2 GiB.
Clamp has two operands, so two thirds of that.

This was a desktop with other applications open. Most cases vary 1–6%
between runs. The worst outlier (35%) is the first run of the first
benchmark (Add, 256², scalar), which warms up the machine. Medians hide it.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | vec: avx2 |
| Runs | 5 per benchmark, medians shown |

## algebra

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
Add           1657      1836        1.11×   not measured yet (tile engine, STRATA-8/9)
Sub           1581      1832        1.16×   not measured yet (tile engine, STRATA-8/9)
Mul           1700      1848        1.09×   not measured yet (tile engine, STRATA-8/9)
Min           1212      1865        1.54×   not measured yet (tile engine, STRATA-8/9)
Max           1043      1873        1.80×   not measured yet (tile engine, STRATA-8/9)
Clamp          802      2336        2.91×   not measured yet (tile engine, STRATA-8/9)
```

### Add

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 2737 | 5552 | 2.03× | 0.180 | 32.8 | 66.6 | 0 |
| 1024 × 1024 | off | 2543 | 5454 | 2.14× | 0.183 | 30.5 | 65.4 | 0 |
| 4096 × 4096 | off | 1657 | 1836 | 1.11× | 0.545 | 19.9 | 22.0 | 0 |
| 16384 × 16384 | off | 1049 | 1159 | 1.10× | 0.863 | 12.6 | 13.9 | 0 |
| 256 × 256 | on | 2668 | 5375 | 2.01× | 0.186 | 33.0 | 66.5 | 0 |
| 1024 × 1024 | on | 2522 | 5349 | 2.12× | 0.187 | 31.2 | 66.2 | 0 |
| 4096 × 4096 | on | 1646 | 1806 | 1.10× | 0.554 | 20.4 | 22.4 | 0 |
| 16384 × 16384 | on | 1011 | 1131 | 1.12× | 0.884 | 12.5 | 14.0 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 35%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 21% of 256²'s by 16384² (5552 → 1159 M cells/s) and the SIMD/scalar speedup flattens (2.03× → 1.10×); SIMD moves 22.0 GB/s at 4096², 13.9 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 21% of 256²'s by 16384² (5375 → 1131 M cells/s) and the SIMD/scalar speedup flattens (2.01× → 1.12×); SIMD moves 22.4 GB/s at 4096², 14.0 GB/s at 16384².

### Sub

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 2085 | 6007 | 2.88× | 0.167 | 25.0 | 72.1 | 0 |
| 1024 × 1024 | off | 2028 | 5632 | 2.78× | 0.177 | 24.3 | 67.6 | 0 |
| 4096 × 4096 | off | 1581 | 1832 | 1.16× | 0.546 | 19.0 | 22.0 | 0 |
| 16384 × 16384 | off | 1018 | 1161 | 1.14× | 0.861 | 12.2 | 13.9 | 0 |
| 256 × 256 | on | 2058 | 5716 | 2.78× | 0.175 | 25.5 | 70.7 | 0 |
| 1024 × 1024 | on | 1990 | 5177 | 2.60× | 0.193 | 24.6 | 64.1 | 0 |
| 4096 × 4096 | on | 1535 | 1812 | 1.18× | 0.552 | 19.0 | 22.4 | 0 |
| 16384 × 16384 | on | 1002 | 1148 | 1.15× | 0.871 | 12.4 | 14.2 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 3%, worst 6%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 19% of 256²'s by 16384² (6007 → 1161 M cells/s) and the SIMD/scalar speedup flattens (2.88× → 1.14×); SIMD moves 22.0 GB/s at 4096², 13.9 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 20% of 256²'s by 16384² (5716 → 1148 M cells/s) and the SIMD/scalar speedup flattens (2.78× → 1.15×); SIMD moves 22.4 GB/s at 4096², 14.2 GB/s at 16384².

### Mul

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 2774 | 5872 | 2.12× | 0.170 | 33.3 | 70.5 | 0 |
| 1024 × 1024 | off | 2672 | 5701 | 2.13× | 0.175 | 32.1 | 68.4 | 0 |
| 4096 × 4096 | off | 1700 | 1848 | 1.09× | 0.541 | 20.4 | 22.2 | 0 |
| 16384 × 16384 | off | 1035 | 1163 | 1.12× | 0.860 | 12.4 | 14.0 | 0 |
| 256 × 256 | on | 2714 | 5610 | 2.07× | 0.178 | 33.6 | 69.4 | 0 |
| 1024 × 1024 | on | 2614 | 5267 | 2.01× | 0.190 | 32.4 | 65.2 | 0 |
| 4096 × 4096 | on | 1660 | 1789 | 1.08× | 0.559 | 20.5 | 22.1 | 0 |
| 16384 × 16384 | on | 1011 | 1144 | 1.13× | 0.874 | 12.5 | 14.2 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 10%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 20% of 256²'s by 16384² (5872 → 1163 M cells/s) and the SIMD/scalar speedup flattens (2.12× → 1.12×); SIMD moves 22.2 GB/s at 4096², 14.0 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 20% of 256²'s by 16384² (5610 → 1144 M cells/s) and the SIMD/scalar speedup flattens (2.07× → 1.13×); SIMD moves 22.1 GB/s at 4096², 14.2 GB/s at 16384².

### Min

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 1397 | 4147 | 2.97× | 0.241 | 16.8 | 49.8 | 0 |
| 1024 × 1024 | off | 1368 | 4016 | 2.94× | 0.249 | 16.4 | 48.2 | 0 |
| 4096 × 4096 | off | 1212 | 1865 | 1.54× | 0.536 | 14.5 | 22.4 | 0 |
| 16384 × 16384 | off | 903 | 1225 | 1.36× | 0.817 | 10.8 | 14.7 | 0 |
| 256 × 256 | on | 1388 | 4013 | 2.89× | 0.249 | 17.2 | 49.7 | 0 |
| 1024 × 1024 | on | 1339 | 3857 | 2.88× | 0.259 | 16.6 | 47.7 | 0 |
| 4096 × 4096 | on | 1187 | 1793 | 1.51× | 0.558 | 14.7 | 22.2 | 0 |
| 16384 × 16384 | on | 894 | 1205 | 1.35× | 0.830 | 11.1 | 14.9 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 12%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 30% of 256²'s by 16384² (4147 → 1225 M cells/s) and the SIMD/scalar speedup flattens (2.97× → 1.36×); SIMD moves 22.4 GB/s at 4096², 14.7 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 30% of 256²'s by 16384² (4013 → 1205 M cells/s) and the SIMD/scalar speedup flattens (2.89× → 1.35×); SIMD moves 22.2 GB/s at 4096², 14.9 GB/s at 16384².

### Max

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 1193 | 4148 | 3.48× | 0.241 | 14.3 | 49.8 | 0 |
| 1024 × 1024 | off | 1164 | 3975 | 3.41× | 0.252 | 14.0 | 47.7 | 0 |
| 4096 × 4096 | off | 1043 | 1873 | 1.80× | 0.534 | 12.5 | 22.5 | 0 |
| 16384 × 16384 | off | 769 | 1231 | 1.60× | 0.813 | 9.23 | 14.8 | 0 |
| 256 × 256 | on | 1182 | 4003 | 3.39× | 0.250 | 14.6 | 49.5 | 0 |
| 1024 × 1024 | on | 1139 | 3851 | 3.38× | 0.260 | 14.1 | 47.7 | 0 |
| 4096 × 4096 | on | 1014 | 1808 | 1.78× | 0.553 | 12.6 | 22.4 | 0 |
| 16384 × 16384 | on | 761 | 1204 | 1.58× | 0.831 | 9.42 | 14.9 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 6%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 30% of 256²'s by 16384² (4148 → 1231 M cells/s) and the SIMD/scalar speedup flattens (3.48× → 1.60×); SIMD moves 22.5 GB/s at 4096², 14.8 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 30% of 256²'s by 16384² (4003 → 1204 M cells/s) and the SIMD/scalar speedup flattens (3.39× → 1.58×); SIMD moves 22.4 GB/s at 4096², 14.9 GB/s at 16384².

### Clamp

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 850 | 4605 | 5.42× | 0.217 | 6.80 | 36.8 | 0 |
| 1024 × 1024 | off | 843 | 4558 | 5.41× | 0.219 | 6.74 | 36.5 | 0 |
| 4096 × 4096 | off | 802 | 2336 | 2.91× | 0.428 | 6.41 | 18.7 | 0 |
| 16384 × 16384 | off | 718 | 1777 | 2.47× | 0.563 | 5.75 | 14.2 | 0 |
| 256 × 256 | on | 848 | 4549 | 5.36× | 0.220 | 7.00 | 37.5 | 0 |
| 1024 × 1024 | on | 839 | 4507 | 5.37× | 0.222 | 6.92 | 37.2 | 0 |
| 4096 × 4096 | on | 795 | 2283 | 2.87× | 0.438 | 6.56 | 18.8 | 0 |
| 16384 × 16384 | on | 690 | 1722 | 2.49× | 0.581 | 5.70 | 14.2 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: memory-bandwidth-bound from 4096²: SIMD throughput falls to 39% of 256²'s by 16384² (4605 → 1777 M cells/s) and the SIMD/scalar speedup flattens (5.42× → 2.47×); SIMD moves 18.7 GB/s at 4096², 14.2 GB/s at 16384².
- mask=on: memory-bandwidth-bound from 4096²: SIMD throughput falls to 38% of 256²'s by 16384² (4549 → 1722 M cells/s) and the SIMD/scalar speedup flattens (5.36× → 2.49×); SIMD moves 18.8 GB/s at 4096², 14.2 GB/s at 16384².
<!-- stratabench output end -->

## What the numbers say

- **Cache-resident sizes are compute-bound.** At 256² and 1024² throughput
  is within 10% for every operation, both backends and both mask
  settings, even though 256² already overflows L2. The
  hardware prefetcher keeps up with three sequential streams. SIMD gives
  2.0–2.1× on Add and Mul, 2.6–2.9× on Sub, 2.9–3.5× on Min and Max, and
  5.4× on Clamp.
- **From 4096² they are memory-bandwidth-bound.** SIMD throughput drops to
  about a third (Add, Sub, Mul: 5.5–6.0 → 1.8 billion cells/s), and all
  five binary operations land on the same ~22 GB/s whatever their compute
  cost: SIMD Mul and Min differ by 1.4× at 1024² and by 1% at 4096².
  The scalar kernels are slower than the memory for Min, Max and Clamp,
  so they lose less, and that is where SIMD keeps a lead (1.5–2.9×).
  For Add, Sub and Mul, scalar code already comes close to what memory
  delivers, and SIMD adds 8–18%.
- **16384² is slower again, and GB/s does not flatten.** SIMD falls from
  about 22 to 14 GB/s between 4096² and 16384², and scalar about as much.
  The benchmark cannot tell why. The likely cause is address translation:
  each 1 GiB operand is 262 144 4 KiB pages, so the loops leave the TLB's
  reach, and the prefetcher stops at every page boundary. That was not
  measured (for example with large pages). Either way it favours the
  tiled, chunked execution of §25 and §27 over whole-raster passes, and
  the tile size is worth benchmarking when the engine lands.
- **Masks cost 0–8% on these operations.** The word-level mask pass (64
  cells per AND) is small next to the float data: 1/32 of the bytes. The
  largest gaps are SIMD Sub and Mul at 1024² (8%), which were not
  investigated further. At 4096² and above the masked and unmasked cases move the same
  GB/s, so the mask bytes cost bandwidth in proportion to their size.
- **0 allocs/op everywhere**, as the suite's `TestZeroAllocs` also checks
  on a small raster.
- **Scalar oddities.** Scalar Sub is about 25% slower than scalar Add and
  Mul at cache-resident sizes, and scalar Max about 15% slower than Min.
  All four are one-line loops in `internal/vec/scalar.go`. The cause
  (code alignment or instruction selection) was not investigated. It
  does not affect the SIMD path or the large sizes.

## Caveats

- One machine: Zen 2, dual-channel DDR4, Windows, AVX2 only. Memory
  bandwidth sets the large-raster numbers, so machines with more channels
  (servers) or faster memory will show different large-size figures and
  larger SIMD gains there. Laptops will show smaller ones.
- Single worker, pinned to one logical CPU. "SIMD + workers" waits for the
  tile engine (STRATA-8/9). Bandwidth-bound kernels are exactly the ones
  where worker scaling will be limited by the memory channels rather than
  by cores, which is what that measurement should show.
- The §28 classification is the command's heuristic (thresholds in
  `cmd/stratabench`), applied to five runs on one machine. The
  classification does not depend on noise here: every SIMD throughput drop
  at 4096² is about 2× or more, and every speedup loss more than a third.
- Operands are uniform random values, dst is written in place of a
  previous result, and pages are touched before timing. Rasters read from
  disk or freshly allocated would add page-fault costs that the suite
  deliberately leaves out.

## arm64 (NEON), STRATA-11

The same suite on the NEON kernels of `internal/vec`, at 256² to 4096².
This is a different machine from the one above, so compare the
SIMD/scalar ratios, not the absolute numbers, across the two.

**Headline.** NEON adds 1.4–1.6× to Add, Sub, Mul, Min and Max, and 1.8×
to Clamp, at every size measured. Unlike on the Zen 2 desktop, nothing
here is memory-bandwidth-bound by 4096²: one M4 core moves about 70 GB/s
through the NEON kernels there, three times the desktop's 22, so the
201 MiB working set is still compute-bound. The ratios are lower than
AVX2's in-cache 2–3× because the lanes are half as wide and the scalar
loop is relatively faster: arm64 scalar Min and Max are single `FMIN`/`FMAX`
instructions, which is also why NEON's Min and Max gain no more than Add.

| | |
|---|---|
| CPU | Apple M4 (4 performance + 6 efficiency cores), NEON 128-bit |
| OS | macOS (darwin/arm64), a laptop on mains power with other applications open |
| Go | go1.27.0 darwin/arm64, **`GOEXPERIMENT=simd`** |
| Run | `GOMAXPROCS=1 GOEXPERIMENT=simd go test ./benchmarks/algebra -run '^$' -bench '/size=(256\|1024\|4096)/' -count 3`. macOS cannot pin a thread to a core, so the scheduler chooses; the spreads below say how much that cost |
| Raw output | [`testdata/bench-arm64.txt`](testdata/bench-arm64.txt) |
| Stats | median of 3 runs, rendered by `go run ./benchmarks/cmd/stratabench < testdata/bench-arm64.txt` |

16384² was left out: it takes most of the run time, and the AVX2 results
above already show what that size adds.

| | |
|---|---|
| CPU | Apple M4 |
| Cores | 10 physical, 10 logical; 10 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd darwin/arm64, GOEXPERIMENT=simd |
| Kernels | vec: neon |
| Runs | 3 per benchmark, medians shown |

### algebra

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
Add           3920      5899        1.50×   not measured yet (tile engine, STRATA-8/9)
Sub           3755      5829        1.55×   not measured yet (tile engine, STRATA-8/9)
Mul           3936      5814        1.48×   not measured yet (tile engine, STRATA-8/9)
Min           3970      5652        1.42×   not measured yet (tile engine, STRATA-8/9)
Max           3968      5633        1.42×   not measured yet (tile engine, STRATA-8/9)
Clamp         4010      7194        1.79×   not measured yet (tile engine, STRATA-8/9)
```

#### Add

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 4048 | 5741 | 1.42× | 0.174 | 48.6 | 68.9 | 0 |
| 1024 × 1024 | off | 3897 | 5949 | 1.53× | 0.168 | 46.8 | 71.4 | 0 |
| 4096 × 4096 | off | 3920 | 5899 | 1.50× | 0.170 | 47.0 | 70.8 | 0 |
| 256 × 256 | on | 3954 | 5760 | 1.46× | 0.174 | 48.9 | 71.3 | 0 |
| 1024 × 1024 | on | 3851 | 5735 | 1.49× | 0.174 | 47.6 | 71.0 | 0 |
| 4096 × 4096 | on | 3831 | 5664 | 1.48× | 0.176 | 47.4 | 70.1 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5741 → 5899 M cells/s), SIMD/scalar 1.42× → 1.50×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5760 → 5664 M cells/s), SIMD/scalar 1.46× → 1.48×.

#### Sub

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 3590 | 5912 | 1.65× | 0.169 | 43.1 | 70.9 | 0 |
| 1024 × 1024 | off | 3934 | 5890 | 1.50× | 0.170 | 47.2 | 70.7 | 0 |
| 4096 × 4096 | off | 3755 | 5829 | 1.55× | 0.172 | 45.1 | 69.9 | 0 |
| 256 × 256 | on | 3851 | 5587 | 1.45× | 0.179 | 47.7 | 69.1 | 0 |
| 1024 × 1024 | on | 3722 | 5570 | 1.50× | 0.179 | 46.0 | 68.9 | 0 |
| 4096 × 4096 | on | 3838 | 5660 | 1.47× | 0.177 | 47.5 | 70.0 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 4%, worst 14%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5912 → 5829 M cells/s), SIMD/scalar 1.65× → 1.55×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5587 → 5660 M cells/s), SIMD/scalar 1.45× → 1.47×.

#### Mul

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 4012 | 5901 | 1.47× | 0.170 | 48.1 | 70.8 | 0 |
| 1024 × 1024 | off | 3984 | 5983 | 1.50× | 0.167 | 47.8 | 71.8 | 0 |
| 4096 × 4096 | off | 3936 | 5814 | 1.48× | 0.172 | 47.2 | 69.8 | 0 |
| 256 × 256 | on | 3901 | 5743 | 1.47× | 0.174 | 48.3 | 71.1 | 0 |
| 1024 × 1024 | on | 3890 | 5600 | 1.44× | 0.179 | 48.1 | 69.3 | 0 |
| 4096 × 4096 | on | 3874 | 5707 | 1.47× | 0.175 | 47.9 | 70.6 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 20%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5901 → 5814 M cells/s), SIMD/scalar 1.47× → 1.48×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5743 → 5707 M cells/s), SIMD/scalar 1.47× → 1.47×.

#### Min

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 4013 | 5664 | 1.41× | 0.176 | 48.1 | 68.0 | 0 |
| 1024 × 1024 | off | 3985 | 5692 | 1.43× | 0.176 | 47.8 | 68.3 | 0 |
| 4096 × 4096 | off | 3970 | 5652 | 1.42× | 0.177 | 47.6 | 67.8 | 0 |
| 256 × 256 | on | 3923 | 5505 | 1.40× | 0.182 | 48.5 | 68.1 | 0 |
| 1024 × 1024 | on | 3896 | 5499 | 1.41× | 0.182 | 48.2 | 68.0 | 0 |
| 4096 × 4096 | on | 3872 | 5478 | 1.41× | 0.183 | 47.9 | 67.8 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5664 → 5652 M cells/s), SIMD/scalar 1.41× → 1.42×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5505 → 5478 M cells/s), SIMD/scalar 1.40× → 1.41×.

#### Max

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 4011 | 5681 | 1.42× | 0.176 | 48.1 | 68.2 | 0 |
| 1024 × 1024 | off | 3972 | 5696 | 1.43× | 0.176 | 47.7 | 68.3 | 0 |
| 4096 × 4096 | off | 3968 | 5633 | 1.42× | 0.177 | 47.6 | 67.6 | 0 |
| 256 × 256 | on | 3919 | 5474 | 1.40× | 0.183 | 48.5 | 67.7 | 0 |
| 1024 × 1024 | on | 3896 | 5507 | 1.41× | 0.182 | 48.2 | 68.2 | 0 |
| 4096 × 4096 | on | 3867 | 5406 | 1.40× | 0.185 | 47.9 | 66.9 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5681 → 5633 M cells/s), SIMD/scalar 1.42× → 1.42×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (5474 → 5406 M cells/s), SIMD/scalar 1.40× → 1.40×.

#### Clamp

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 4015 | 7276 | 1.81× | 0.137 | 32.1 | 58.2 | 0 |
| 1024 × 1024 | off | 4028 | 7317 | 1.82× | 0.137 | 32.2 | 58.5 | 0 |
| 4096 × 4096 | off | 4010 | 7194 | 1.79× | 0.139 | 32.1 | 57.6 | 0 |
| 256 × 256 | on | 3971 | 7157 | 1.80× | 0.140 | 32.8 | 59.0 | 0 |
| 1024 × 1024 | on | 3996 | 7199 | 1.80× | 0.139 | 33.0 | 59.4 | 0 |
| 4096 × 4096 | on | 3995 | 7211 | 1.81× | 0.139 | 33.0 | 59.5 | 0 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 3%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (7276 → 7194 M cells/s), SIMD/scalar 1.81× → 1.79×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 4096² (7157 → 7211 M cells/s), SIMD/scalar 1.80× → 1.81×.
