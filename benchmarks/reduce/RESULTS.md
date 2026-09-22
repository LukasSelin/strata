# Reduction accumulator: results

This is the accumulator decision DESIGN.md §49 asks for before `reduce.Sum`
and `reduce.Stats`. The benchmarks are
`internal/accum/bench_test.go` (`BenchmarkAccumulators`,
`BenchmarkWorkers`). The `reduce` suite at the §38 sizes lands with
`reduce.Sum`.

## Recommendation

Use the **exponent-binned exact accumulator** in `internal/accum`, with its
AVX2 backend. Keep §49's guarantee as written:

> A reduction returns the correctly rounded result of the exact
> arithmetic over the valid cells, for every tiling, worker count and
> backend.

- **Speed.** In a SIMD build the exact sum is as fast as a plain float64
  loop. Exact sum and sum of squares together is as fast as Neumaier,
  which is neither exact nor independent of order.
- **Scaling.** With the engine's workers, the exact sum reaches 85% of
  the machine's read bandwidth, and the version with squares 68%.
- **No weakening needed.** Nothing here makes a case for dropping to a
  documented evaluation order.

## How it works

- **Scalar backend.** Each finite float32 is a signed 24-bit significand
  times a power of two. The accumulator keeps one `int64` bin per
  exponent field and adds each significand to its bin. Integer addition
  is exact, associative and commutative, so the bins do not depend on
  order, split or combine order. They are normalised (carried) before any
  bin could overflow. Each result is rounded once, through `math/big`,
  when it is read.
- **Squares.** A float32 squared is exact in 48 bits, so it is added as
  two 24-bit halves.
- **AVX2 backend.** It takes 64-cell blocks.
  - A first pass finds the block's largest and smallest non-zero exponent
    fields.
  - If they are within 32 fields of each other (16 with squares), a second
    pass shifts every significand onto the block's base exponent and adds
    it in 64-bit lanes (`VPSLLVQ`). Squares use `VPMULUDQ`. Each block sum
    stays below 2^62.
  - The block sum goes into the bins in 24-bit pieces. That is the same
    integer the scalar loop adds cell by cell, so the backends agree on
    the exact bins by construction. `TestBackendsAgree` compares the
    integers, not only the rounded results.
  - A block holding a NaN or an infinity, or a spread too wide for the
    window, goes to the scalar loop whole.

## Numbers

These are medians of 5 runs on one core. The machine is a Ryzen 9 3900X
running Windows 11 with Go 1.27.0, `GOEXPERIMENT=simd`. The distribution
is `dem` (uniform 300–2300); `normal` (σ = 100) gives the same figures
within 1%. The max fold reads every cell and does almost nothing else,
so it marks the bandwidth bound.

| accumulator | exact | order-free | 1M cells (in cache) GB/s | ns/cell | 64M cells (memory) GB/s |
|---|---|---|---|---|---|
| max fold (bandwidth) | — | — | 23.6 | 0.17 | 15.3 |
| float64 | no | no | 5.65 | 0.71 | 5.01 |
| Neumaier | nearly | no | 3.54 | 1.13 | 3.33 |
| binned, 1 bin set, scalar | yes | yes | 1.36 | 2.94 | 1.29 |
| binned, 4 bin sets, scalar (default build) | yes | yes | 1.82 | 2.20 | 1.80 |
| **binned, AVX2** | yes | yes | **5.24** | **0.76** | **4.90** |
| moments (sum + squares), scalar | yes | yes | 1.16 | 3.45 | 1.15 |
| **moments, AVX2** | yes | yes | **3.32** | **1.20** | **3.24** |

Scaling from memory: 64M cells split across goroutines, medians of 3
runs, in GB/s.

| workers | max fold | binned, AVX2 | moments, AVX2 |
|---|---|---|---|
| 1 | 15.1 | 4.9 | 3.2 |
| 2 | 27.9 | 9.6 | 6.4 |
| 4 | 37.8 | 18.5 | 12.2 |
| 8 | 39.9 | 32.9 | 21.3 |
| 12 | 39.3 | 33.0 | 26.7 |

Reading a result (`Mean` and `StdDev` together) takes about 10 µs. It
happens once per call.

## What was tried and why it lost

- **More bin sets (8 and 16).** The guess was that runs of equal
  exponents, which DEMs are full of, serialise on the same bin's
  store-to-load chain. They do not.
  - Values spread over all 254 exponents ran as fast as DEM values
    (1.58 vs 1.57 ns/cell with the special-value test removed).
  - 8 unrolled sets were no faster than 4 (1.56 ns/cell).
  - 8 or 16 sets in a rolled loop were slower (2.6–2.9 ns/cell).
  - The scalar loop is limited by its instruction count, about 6 cycles
    per cell. The NaN/Inf test cost another 0.6 ns/cell until it became
    one max and one compare per four cells.
- **One register sum per exponent run, flushed when the exponent
  changes:** 3.2–4.0 ns/cell, because the branch mispredicts.
- **Scalar block window** (the AVX2 algorithm without vectors):
  3.5 ns/cell. Each cell's variable shift feeds one dependent add chain.
  It only pays with 4 or 8 lanes.

## Known costs

- **Data that defeats the window.** Blocks whose non-zero values span
  more than 32 exponent fields (16 with squares) pay for the first pass
  and then the scalar loop. On random bit patterns over the whole float32
  range, that is 4.0 ns/cell against 2.2 for the scalar loop alone.
  - A factor-of-2^32 spread (2^16 with squares) inside 64 neighbouring
    cells is rare in elevation and in its derivatives. Exact zeros never
    widen a block.
  - Curvature near flat ground is the case to watch. If `benchmarks/reduce`
    shows it at the §38 sizes, the fix is a narrower window per block or
    splitting a block in two, not a different accumulator.
- **Memory.** A `Moments` partial is about 28 KiB (4 bin sets of sums and
  squares). The engine keeps one per worker, so this is negligible.
- **Count limit.** An accumulator takes up to 2^38 values (about a
  terabyte of float32) and panics beyond that.
- **StdDev rounding.** StdDev is the 256-bit square root of the exact
  variance, rounded to float64. It is within one ulp of the true value
  and depends only on the values. Sum, Mean and Variance are correctly
  rounded.
