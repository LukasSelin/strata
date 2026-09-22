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
