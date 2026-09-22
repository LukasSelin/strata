# Register-level fusion spike: results

What register-level operation fusion (DESIGN.md §29) adds on top of the
tile-level `Pipeline` (§52), measured before building a generator for it.
The workload is §52's six-factor product
dst = ((((a0·a1)·a2)·a3)·a4)·a5, computed three ways: five chained
`algebra.MulTiled` calls, one `exec.Pipeline`, and the hand-written
`Fused` kernel (one loop, six loads, five multiplies, one store).
All three are bit-identical (`TestFormsAgree`). Metrics and names are
defined in [`../README.md`](../README.md) and [`doc.go`](doc.go).

## Headline

- **Out of cache, register-level fusion is worth about 5%.** At 4096²
  and 8192² in default strips, Fused runs 0.96–1.07× the Pipeline on one
  SIMD worker and 1.04–1.11× on 12. Unmasked, both reach 896–953
  Mcells/s with 12 workers, which at 28 B/cell is 25–27 GB/s: the two DDR4 channels, as
  in benchmarks/engine. The scratch traffic the engine's counter does not
  see was being absorbed by cache.
- **Tile-level fusion is the win.** Chained to Pipeline is 1.6–2.6× in
  strips from 4096² up, 2.3–2.6× unmasked on 12 SIMD workers (344–396 →
  896–908 Mcells/s), as §52 predicted from 60 against 28 B/cell.
- **256×256 tiles show a larger gap, 1.03–1.95×, because the Pipeline
  does worse in them**, not because Fused does better: unmasked, Fused
  in 256×256 tiles (742–818 Mcells/s with 12 workers) is still below the
  Pipeline in strips (896–908).
- **The 1024² gap, up to 5.5×, is not fusion.** The Pipeline allocates
  its scratch per worker on every call: 1.05 MB/op on one worker and
  12.6 MB/op on 12, against 28 MB of operands. That is why it gets slower
  from 1 to 12 workers at 1024² (498 → 384 Mcells/s in strips) while
  Fused scales from 809 to 1935. It is an engine cost with a fix of its
  own, and until it is fixed the in-cache comparison cannot separate the
  two effects.

So a fusion generator for pointwise chains is not worth building on this
evidence. Reasons to measure again: stencil or longer chains, whose
intermediates would not fit in cache, and the in-cache sizes once the
per-call scratch allocation is gone.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631 |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test ./benchmarks/fusion -run '^$' -bench . -count 3 -timeout 2h`, `GOMAXPROCS=24`, not pinned, normal priority |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) |

## Caveats

- **One run, `-count 3`, not on a quiet machine.** The README asks for
  `-count 5`, High priority and nothing else moving memory for published
  numbers. A few percent is inside this run's noise, so "about 5%" means
  "small", not a measured 5%. Re-run under the README's conditions before
  quoting a figure.
- **One chain.** Five multiplies are the cheapest possible stages. A chain
  with more arithmetic per stage would be even more compute-bound per
  byte and gain less; a longer chain, more scratch slots, or stencil
  stages whose windows grow would spill the Pipeline's intermediates out
  of L2 and gain more. Neither was measured.
- **The in-cache sizes measure the scratch allocation**, as above.
- **Tiled only.** The chunked path was not run; its 2.00× source and sink
  copy (§52) is the same for Pipeline and Fused.

## Medians

Mcells/s, the median of three runs. Scalar runs one worker only, as in
the engine category.

### 1024²

| mask | backend | workers | tiles | Chained | Pipeline | Fused | Fused ÷ Pipeline |
|---|---|---:|---|---:|---:|---:|---:|
| off | scalar | 1 | 256x256 | 206 | 249 | 467 | 1.88× |
| off | scalar | 1 | strips | 318 | 430 | 774 | 1.80× |
| off | simd | 1 | 256x256 | 241 | 294 | 475 | 1.62× |
| off | simd | 1 | strips | 359 | 498 | 809 | 1.63× |
| off | simd | 12 | 256x256 | 328 | 328 | 1316 | 4.01× |
| off | simd | 12 | strips | 487 | 384 | 1935 | 5.04× |
| off | simd | 24 | 256x256 | 445 | 272 | 1387 | 5.09× |
| off | simd | 24 | strips | 554 | 327 | 1804 | 5.52× |
| on | scalar | 1 | 256x256 | 197 | 226 | 391 | 1.73× |
| on | scalar | 1 | strips | 308 | 423 | 718 | 1.70× |
| on | simd | 1 | 256x256 | 219 | 265 | 407 | 1.54× |
| on | simd | 1 | strips | 349 | 489 | 767 | 1.57× |
| on | simd | 12 | 256x256 | 417 | 329 | 1052 | 3.20× |
| on | simd | 12 | strips | 535 | 380 | 1767 | 4.64× |
| on | simd | 24 | 256x256 | 341 | 248 | 1028 | 4.14× |
| on | simd | 24 | strips | 516 | 308 | 1691 | 5.48× |

### 4096²

| mask | backend | workers | tiles | Chained | Pipeline | Fused | Fused ÷ Pipeline |
|---|---|---:|---|---:|---:|---:|---:|
| off | scalar | 1 | 256x256 | 198 | 163 | 250 | 1.53× |
| off | scalar | 1 | strips | 278 | 471 | 466 | 0.99× |
| off | simd | 1 | 256x256 | 198 | 182 | 239 | 1.32× |
| off | simd | 1 | strips | 273 | 550 | 588 | 1.07× |
| off | simd | 12 | 256x256 | 312 | 612 | 818 | 1.34× |
| off | simd | 12 | strips | 344 | 896 | 927 | 1.04× |
| off | simd | 24 | 256x256 | 323 | 483 | 733 | 1.52× |
| off | simd | 24 | strips | 355 | 712 | 916 | 1.29× |
| on | scalar | 1 | 256x256 | 184 | 151 | 198 | 1.31× |
| on | scalar | 1 | strips | 270 | 447 | 405 | 0.91× |
| on | simd | 1 | 256x256 | 202 | 172 | 202 | 1.18× |
| on | simd | 1 | strips | 296 | 531 | 509 | 0.96× |
| on | simd | 12 | 256x256 | 285 | 513 | 728 | 1.42× |
| on | simd | 12 | strips | 373 | 763 | 839 | 1.10× |
| on | simd | 24 | 256x256 | 304 | 378 | 737 | 1.95× |
| on | simd | 24 | strips | 361 | 578 | 909 | 1.57× |

### 8192²

| mask | backend | workers | tiles | Chained | Pipeline | Fused | Fused ÷ Pipeline |
|---|---|---:|---|---:|---:|---:|---:|
| off | scalar | 1 | 256x256 | 177 | 182 | 246 | 1.35× |
| off | scalar | 1 | strips | 246 | 400 | 475 | 1.19× |
| off | simd | 1 | 256x256 | 182 | 215 | 222 | 1.03× |
| off | simd | 1 | strips | 277 | 506 | 486 | 0.96× |
| off | simd | 12 | 256x256 | 354 | 616 | 742 | 1.20× |
| off | simd | 12 | strips | 396 | 908 | 953 | 1.05× |
| off | simd | 24 | 256x256 | 336 | 490 | 781 | 1.59× |
| off | simd | 24 | strips | 385 | 810 | 960 | 1.18× |
| on | scalar | 1 | 256x256 | 161 | 172 | 225 | 1.31× |
| on | scalar | 1 | strips | 219 | 400 | 458 | 1.14× |
| on | simd | 1 | 256x256 | 166 | 193 | 205 | 1.06× |
| on | simd | 1 | strips | 272 | 468 | 460 | 0.98× |
| on | simd | 12 | 256x256 | 321 | 501 | 677 | 1.35× |
| on | simd | 12 | strips | 347 | 840 | 931 | 1.11× |
| on | simd | 24 | 256x256 | 318 | 416 | 620 | 1.49× |
| on | simd | 24 | strips | 381 | 638 | 869 | 1.36× |

## arm64 (NEON)

The same benchmark on an Apple M4 with the NEON kernels, after the
`Fused` form gained its NEON loop and after the Pipeline's scratch moved
into a pool (so the per-call allocation behind the 1024² caveat above is
gone: 2.7 KB/op for the Pipeline on one worker in strips, against
1.05 MB/op before).

**Headline.** On this machine, register-level fusion is worth far more
than the ~5% measured on Zen 2. At 4096² on NEON, Fused runs 1.5–2.2×
the Pipeline: 1.82× on one worker in strips (1093 → 1986 Mcells/s) and
1.47× on ten (2168 → 3191). At 1024², with no allocation left to blame,
it is 1.7–2.2×. The M4 core has about three times the Zen 2 desktop's
single-core bandwidth (see `../algebra/RESULTS.md`); the likely reading,
not separately measured, is that the traffic the Pipeline's scratch adds
in cache is no longer hidden behind a DRAM limit.
The spike's conclusion — a fusion generator for pointwise chains is not
worth building — was drawn on a bandwidth-starved machine; this is one
of the reasons to measure again that it names, and it points the other
way. It is one laptop run of three repetitions, so treat it as a reason to re-run
under the README's conditions, not yet as a reversal.

| | |
|---|---|
| CPU | Apple M4, 10 cores (4 performance + 6 efficiency), NEON 128-bit |
| OS | macOS, darwin/arm64, a laptop with other applications open |
| Go | go1.27.0 darwin/arm64, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test ./benchmarks/fusion -run '^$' -bench . -count 3 -timeout 2h`, `GOMAXPROCS=10`, not pinned (macOS cannot pin threads), normal priority. Workers 1 and 10 |
| Raw output | [`testdata/bench-arm64.txt`](testdata/bench-arm64.txt) |

Caveats beyond those above:

- **8192² is not reported.** Its runs are not usable: 12 of its 36 cases
  spread by more than 25% between runs, the scalar Pipeline by up to
  268% (497, 149 and 96 Mcells/s for one case), against a median spread
  of 3% at 4096². The six inputs, output and two intermediates are about
  2.3 GiB at that size, and the laptop had little memory free, so memory
  pressure is the likely cause, but it was not confirmed. The raw lines
  are kept in the testdata file.
- **One 1024² row is odd and unexplained:** scalar Fused, unmasked, in
  strips runs at 476 Mcells/s, below the masked case's 1392 and the
  4096² case's 1132, consistently over its three runs (446–601).
- Otherwise the median run-to-run spread is 6% at 1024² and 3% at 4096².

### Medians, arm64

Mcells/s, the median of three runs. Scalar runs one worker only, as
above.

#### 1024²

| mask | backend | workers | tiles | Chained | Pipeline | Fused | Fused ÷ Pipeline |
|---|---|---:|---|---:|---:|---:|---:|
| off | scalar | 1 | 256x256 | 399 | 265 | 495 | 1.87× |
| off | scalar | 1 | strips | 746 | 776 | 476 | 0.61× |
| off | simd | 1 | 256x256 | 510 | 381 | 804 | 2.11× |
| off | simd | 1 | strips | 1127 | 1121 | 1993 | 1.78× |
| off | simd | 10 | 256x256 | 998 | 1323 | 2676 | 2.02× |
| off | simd | 10 | strips | 1525 | 2524 | 4908 | 1.94× |
| on | scalar | 1 | 256x256 | 368 | 272 | 518 | 1.91× |
| on | scalar | 1 | strips | 752 | 753 | 1392 | 1.85× |
| on | simd | 1 | 256x256 | 501 | 324 | 704 | 2.17× |
| on | simd | 1 | strips | 1108 | 1067 | 1910 | 1.79× |
| on | simd | 10 | 256x256 | 887 | 1044 | 1793 | 1.72× |
| on | simd | 10 | strips | 1724 | 2549 | 4735 | 1.86× |

#### 4096²

| mask | backend | workers | tiles | Chained | Pipeline | Fused | Fused ÷ Pipeline |
|---|---|---:|---|---:|---:|---:|---:|
| off | scalar | 1 | 256x256 | 116 | 191 | 314 | 1.64× |
| off | scalar | 1 | strips | 709 | 776 | 1132 | 1.46× |
| off | simd | 1 | 256x256 | 197 | 250 | 402 | 1.60× |
| off | simd | 1 | strips | 1125 | 1093 | 1986 | 1.82× |
| off | simd | 10 | 256x256 | 745 | 802 | 1756 | 2.19× |
| off | simd | 10 | strips | 1401 | 2168 | 3191 | 1.47× |
| on | scalar | 1 | 256x256 | 106 | 151 | 284 | 1.88× |
| on | scalar | 1 | strips | 720 | 668 | 1076 | 1.61× |
| on | simd | 1 | 256x256 | 183 | 199 | 366 | 1.83× |
| on | simd | 1 | strips | 1059 | 939 | 1884 | 2.01× |
| on | simd | 10 | 256x256 | 644 | 540 | 1164 | 2.16× |
| on | simd | 10 | strips | 1396 | 1854 | 3096 | 1.67× |
