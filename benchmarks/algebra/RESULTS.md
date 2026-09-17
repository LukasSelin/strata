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
