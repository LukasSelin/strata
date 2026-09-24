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

The published numbers are AVX2 on the Zen 2 desktop every other category
uses. The Apple M4 NEON run that came first is kept below, under
[arm64 (NEON)](#arm64-neon).

## Headline

- **Compute-bound at every radius and size.** All 34 operation × mask
  rows hold their SIMD throughput from 256² to 16384². Until the change
  recorded under [16384² and the row stride](#16384-and-the-row-stride),
  six did not: Gaussian, Mean and Min at r = 5 ran at 42–56% of their
  4096² speed at 16384², because rows 64 KiB apart collided in one set
  of Zen 2's L2. The column pass now folds such rows in groups, and those
  rows run at 92–97% of their 4096² speed, without a bit of output
  changing.
- **Correlate costs what its products cost.** At 4096² without masks,
  AVX2 Correlate takes 0.92, 1.83, 3.27 and 7.65 ns per cell at r = 1, 2,
  3 and 5: 0.063–0.073 ns per product from r = 2 on (0.10 at r = 1, where
  the per-row fixed cost still shows), so its throughput falls as
  (2r+1)². That is about 16 billion products and sums a second on one
  core, without FMA (§15).
- **The separable forms grow linearly in r.** Gaussian takes 0.69 ns per
  cell at r = 1 and 1.60 at r = 5; Mean, 0.67 and 1.51; Min, 0.78 and
  2.33. At r = 5 Gaussian is 4.8× faster than Correlate with the same
  footprint, and at r = 1 still 1.3×.
- **Memory is not the limit.** The highest demand in the suite is 15.4
  GB/s (Mean at r = 1, 1024², in cache), under a core's 22 GB/s, and from
  r = 2 the demand falls with the radius, to 1.0 GB/s for Correlate at
  r = 5.
- **AVX2 is worth 2.6–4.9× over scalar** at 4096², rising with the
  radius. Until the change under [Loop placement](#loop-placement), the
  scalar Correlate and Mean kernels ran at one of two speeds depending
  on where the linker put them, which put their ratios as high as 7.5×
  in the previous run (`testdata/bench-preloop.txt`); they are now 1.3–1.9×
  faster than there and no longer depend on placement. As on NEON, this
  is lanes plus register blocking, not lanes alone: the scalar kernels,
  which are the canonical order and must stay bounds-check free (§39),
  accumulate through `dst` a term at a time, while the SIMD ones hold
  several accumulators per block in registers. The M4's faster core is
  1.1–2.4× ahead of AVX2 in absolute terms: least on Correlate at large
  radii, most on Min.

| 4096², no mask | products/cell | scalar M cells/s | AVX2 M cells/s | AVX2/scalar | AVX2 ns/cell | AVX2 GB/s | NEON M cells/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CorrelateR1 | 9 | 296 | 1092 | 3.69× | 0.916 | 8.73 | 1724 |
| CorrelateR2 | 25 | 131 | 545 | 4.17× | 1.83 | 4.36 | 741 |
| CorrelateR3 | 49 | 69.5 | 306 | 4.41× | 3.27 | 2.45 | 386 |
| CorrelateR5 | 121 | 28.7 | 131 | 4.56× | 7.65 | 1.05 | 148 |
| GaussianR1 | 6 | 564 | 1454 | 2.58× | 0.688 | 11.6 | 2501 |
| GaussianR3 | 14 | 255 | 952 | 3.73× | 1.05 | 7.61 | 1329 |
| GaussianR5 | 22 | 164 | 624 | 3.80× | 1.60 | 4.99 | 892 |
| MeanR1 | — | 455 | 1489 | 3.28× | 0.671 | 11.9 | 2900 |
| MeanR5 | — | 158 | 663 | 4.21× | 1.51 | 5.30 | 1037 |
| MinR1 | — | 446 | 1277 | 2.86× | 0.783 | 10.2 | 2946 |
| MinR5 | — | 102 | 429 | 4.21× | 2.33 | 3.43 | 1039 |

## 16384² and the row stride

At 16384² a row of float32 is exactly 64 KiB. The SIMD separable
operations used to slow down there and nowhere else; they now run close
to their speed at the neighbouring sizes. The same pinning, unmasked,
AVX2, M cells/s, 5 samples each; the "before" column is the previous
kernels, measured on 2026-09-22 alongside `testdata/bench-prefold.txt`:

| operation | 16320² | **16384² before** | **16384² after** | 16448² |
|---|---:|---:|---:|---:|
| GaussianR5 | 597–609 | 251–258 | 540–552 | 577–609 |
| MeanR5 | 599–627 | 266–269 | 590–597 | 619–631 |
| MinR5 | 385–402 | 221–224 | 367–377 | 393–402 |
| CorrelateR5 | 122–126 | 92–94 | 124–126 | 122–126 |
| MeanR3 | 946–963 | 762–781 | 747–796 | 919–993 |

Before the change, 8192² and 12288² ran at full speed too (Gaussian r = 5:
569–578 and 556–558), and scalar and NEON did not show the effect.

### The cause: L2 sets, from the stride alone

Zen 2's L2 is 512 KiB and 8-way, so its set index repeats every 64 KiB.
A SIMD column pass reads a block of 32 cells from each of its 2r+1 rows
in lockstep, holding one accumulator per vector in registers; rows
64 KiB apart put all those lines in one set. No performance counters
back this: AMD uProf is not installed on the desktop, and Windows' own
PMU sources (`wpr -pmcsources`) have no L2 event and need an elevated
shell. The evidence is timing that separates the candidate causes, from
`BenchmarkColumnStride` in `internal/focalrow`: the column pass over 64
output rows of a fixed 16320 cells, only the stride varied, ns per cell,
fastest of 5, with the column pass as it was before the change
(ColumnSum; ColumnCorrelate behaves the same):

| rows (2r+1) | 16320 | **16384** | 16400 | 16416 | 16448 | 20480 | 24576 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.63 | **1.99** | 0.99 | 0.70 | 0.65 | 0.75 | 0.78 |
| 11 | 0.83 | **2.88** | 1.34 | 0.91 | 0.79 | 0.84 | 0.89 |
| 17 | 1.34 | **4.91** | 2.78 | 1.57 | 1.39 | 1.44 | 1.84 |

- **It is the stride, not the width or the size.** At a width of 16320
  with a stride of 16384, focal's Gaussian at r = 5 ran at 246–252 M
  cells/s; at a width of 16384 with a stride of 16448, at 431–512. A
  4096-cell window of a 16384-wide raster, as a tile of the Tiled
  functions is, was as slow (247–256).
- **Its period is 64 KiB, the L2's.** Strides of 32768 and 49152 cells
  are as slow as 16384 (11 rows: about 3.0 and 6.0 ns per cell), while
  20480 and 24576 (80 and 96 KiB) are not. Those two are multiples of
  4 KiB, as 16384 is, so the L1's sets (which repeat every 4 KiB) and 4K
  aliasing between loads and stores are ruled out.
- **Its threshold is the 8 ways.** 7 rows at stride 16384 lose 28%
  (0.47 → 0.60); 9 rows lose 3.2×. At 24576, where the rows alternate
  between two sets, only 17 rows (9 per set) slow down.
- **Near misses count.** Rows one cache line apart modulo 64 KiB (16400)
  lose 1.5–2.2×, two lines apart (16416) up to 24%, four (16448) nothing:
  the prefetcher's lines ahead of each row land in the neighbouring sets.

### The change

When more than seven of a column pass's rows lie within 1 KiB of one
another modulo 64 KiB (and are not simply adjacent in memory), the AVX2
kernels now fold them a group of at most seven rows at a time over chunks
of 4096 cells, the first group writing the destination row and the
others adding to it (`foldColumn` in `internal/focalrow/simd_amd64.go`).
Each cell still folds its rows in order into a float32, which is what
the scalar kernels do, so no output bit changes; `TestAliasedStride` in
package focal checks every operation at a 16384-cell stride against the
per-cell definition, and fails if the fold is broken. Correlate's 2-D
pass folds the same way. At other strides the kernels keep their single
pass: folding there costs 4–19%. The 1 KiB rule folds wherever
folding was faster in the microbenchmark and nowhere else, with one
exception: at 16416 cells and 9 rows it folds, and that costs 11–12%.

What is left:

- **r = 5 is 3–10% short of 16320².** Gaussian 540–552 against 597–609,
  Min 367–377 against 385–402, Mean within 3%, Correlate none. In the
  microbenchmark a folded pass is 4–19% slower than a single pass at a
  harmless stride, for its extra loads and stores of the destination
  row, and 6–15% slower again at 16384 than at 16320, so some conflict
  remains within each group.
- **r = 3 still loses about 20%** (Mean 747–796 against 946–963), as it
  did before. Seven rows are under the fold's threshold, and folding
  them in groups of three and four did not help in the microbenchmark
  (0.60 → 0.65 ns per cell), so that loss has a cause this change does
  not reach, and it stays open (DESIGN.md §53).
- Beyond the suite, Gaussian at r = 8 (17 rows) at 16384² went from
  148–152 to 268–342 M cells/s, against 341–367 at 16320².

## Loop placement

The scalar weighted-sum, sum and mean kernels used to run at one of two
speeds, depending on where the linker put them: scalar Mean at r = 5 ran
at 147–160 M cells/s in one binary and 87–92 in another built from the
same kernel source (see [Machine and method](#machine-and-method)). On
2026-09-23 their cell loops were changed to take eight cells a step,
without a bit of output changing. They now run at the same speed wherever
they land, and faster than before at either placement.

### The cause: a 25-byte loop across a 64-byte line

Each of those kernels spends its time in a one-cell loop of 23–35 bytes:
a load, an add (or a multiply and an add) and a store per iteration. Go
aligns functions to 32 bytes, so relative to a 64-byte line of code each
function has two possible positions, and unrelated code elsewhere in the
binary decides which. From `go tool objdump` of the two binaries, the
hot loops' first and last byte within their 64-byte line:

| loop | `a81fb65` (fast) | `5746c71` (slow) |
|---|---|---|
| ColumnSum, `dst[i] += v[i]` | 11–36, one line | 43–4, two lines |
| RowMean, the sum | 16–38, one line | 48–6, two lines |
| CorrelateRow, the later terms | 2–36, one line | 34–4, two lines |
| ColumnCorrelate, the later terms | 10–36, one line | 42–4, two lines |
| RowCorrelate, the later terms | 24–50, one line | 56–18, two lines |

Every hot loop that fits in one line in the fast binary spans two in the
slow one. Both binaries cross 32-byte boundaries alike. The Min and Max
loops (41–65 bytes) moved too, and their speed did not change.

Two experiments separate the cause from everything else that differs
between the two commits:

- **The loop alone.** ColumnSum's loop, byte for byte as the compiler
  emits it, in assembly at a chosen offset from a 64-byte boundary, over
  1024 cells in L1, took 0.28–0.38 ns per cell starting at offsets 2–38
  and 0.55–0.78 from 39 on, the first offset at which its last byte (the
  branch back) falls in the next line. Offsets 11–38 cross a 32-byte
  boundary and cost nothing extra.
- **The same source at both placements.** A 32-byte function added
  ahead of the kernels in a scratch build moves them all by 32 bytes and
  swaps fast and slow: scalar Mean r = 5 at 1024², 88 M cells/s at the
  current placement and 136 moved (medians of 6, interleaved).

That a loop spanning two lines costs a second cycle an iteration fits a
front end that fetches one 64-byte line of code a cycle; no performance
counter confirms the mechanism (none are available on this machine
without an elevated shell, see [the cause](#the-cause-l2-sets-from-the-stride-alone)
of the stride effect).

### The change

The cell loops of the five kernels moved into four helpers (`mulRow`,
`mulAddRow`, `addRow`, `divRow` in `internal/focalrow/focalrow.go`)
that take eight cells a step while more than eight remain, then the rest
one at a time. An iteration then makes eight stores, which take eight
cycles, more than fetching the three or four lines it spans. The same
experiment as above, with the plain sum's stepped loop written out in
assembly at every third start offset from 2 to 65, fastest of 6, ns per
cell (the one-cell loop, for comparison, took 0.28–0.38 at offsets 2–38
and 0.55–0.78 from 39):

| step | offsets 2–50 | offsets 53–62 |
|---|---:|---:|
| four cells | 0.257–0.276 | 0.313–0.360 |
| **eight cells** | 0.254–0.276 | 0.256–0.265 |

Four cells a step was not enough: the plain sum still lost up to 39%
when its loop started in the last 11 bytes of a line.

Every cell gets the same operations in the same order as before. The
focal and `internal/focalrow` tests, the fuzzers
(`FuzzFocalRows`, `FuzzFocal`, `FuzzFocalRelations`) and
`TestNoBoundsChecksInLoops` pass. A comparison against the previous
kernels (every length to 130, every odd neighbourhood to 11, 6.1 million
cells per setting, NaN, ±Inf, ±0 and subnormals among the inputs) found
every non-NaN output bit identical. Where two NaNs meet, the NaN's sign
or payload differed in 0.08% of those cells: x86 keeps the first
operand's NaN, and Go orders a float addition's operands as it likes, so
the previous kernels' NaN payloads were never fixed either; the package
promises any NaN for any NaN.

Both placements of the new kernels, built as above and measured
interleaved with the old at theirs, scalar at 1024², medians of 6, M
cells/s. The desktop was in use, and single samples spread 20–30%, so
read these to the nearest 10%:

| scalar, 1024² | old, slow placement | old, fast placement | new, placement A | new, placement B |
|---|---:|---:|---:|---:|
| MeanR5 | 88 | 136 | 151 | 155 |
| CorrelateR2 | 75 | 95 | 128 | 121 |
| GaussianR3 | 171 | 155 | 223 | 218 |
| MinR3 (unchanged) | 152 | 152 | 138 | 149 |

In the suite run (quiet, 5 samples a case), against
`bench-preloop.txt`, where their loops sat at the slow placement,
the scalar rows went, per operation across sizes and masks:
Correlate 1.31–1.88× faster (median 1.72×), Mean 1.42–1.92× (1.67×),
Gaussian 1.44–1.56× (1.48×); Min and Max, whose loops did not change,
1.01–1.07×. Against `bench-prefold.txt`, taken when the old loops sat at
the fast placement, Mean is 1.04–1.11× faster, so for the plain sum the
change mostly removes the loss rather than adding speed; Correlate is
1.04–1.46× and Gaussian 1.43–1.57× faster.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Run | `GOEXPERIMENT=simd go test -c ./benchmarks/focal`, then `focal.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 8h`, pinned to one logical CPU from start (`start /affinity 10 /high`), `GOMAXPROCS=1`, High priority. About 32 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) |
| Stats | median of 5 runs, each ≥1 s (`b.Loop`) |

This run was taken in the early hours of 2026-09-24, after the
loop-placement change, when a short pinned probe showed the core quiet
(10 samples within 1.4%); its worst per-operation spread is 12%. The
previous published run, from the morning of 2026-09-23 with the same
code apart from that change, is kept as
[`testdata/bench-preloop.txt`](testdata/bench-preloop.txt); the one
before it, from the night of 2026-09-22 and before the column-pass
change, as [`testdata/bench-prefold.txt`](testdata/bench-prefold.txt),
and the busy run that preceded that as
[`testdata/bench-busy.txt`](testdata/bench-busy.txt).

Against `bench-preloop.txt`, the rows this change does not touch moved
little: SIMD within −12% to +17% (median +3–4% per operation), scalar
Min and Max +1–7%, so the machine ran a few percent faster overall. The
scalar rows it does touch moved far more; see
[Loop placement](#loop-placement).

**Why `bench-preloop.txt`'s scalar Correlate and Mean rows were slow.**
Against `bench-prefold.txt`, its scalar Correlate ran 11–29% slower and
scalar Mean 25–43% slower, at every size and steadily, while scalar
Gaussian, Min and Max were within 6%. Their source had not changed, and
the column-pass change was not the cause: a binary of the commit just
before it measured the same (scalar Mean r = 5 at 1024²: 87–91 against
89–92 M cells/s), while a binary of the commit `bench-prefold.txt`
measured, `a81fb65`, ran it at 147–160. In between, master's unrelated
commits moved every function of `internal/focalrow` by 32 bytes
(`scalarColumnSum` from 0 to 32 mod 64, `scalarRowMean` from 32 to 0),
and Go aligns functions only to 32 bytes. The cause, a one-cell loop
spanning two 64-byte lines of code, and the change that removed it are
under [Loop placement](#loop-placement).

## Reproducing

```
GOEXPERIMENT=simd go test -c -o focal.test.exe ./benchmarks/focal
cd benchmarks/focal
focal.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 8h > testdata/bench.txt
cd ../..
go run ./benchmarks/cmd/stratabench < benchmarks/focal/testdata/bench.txt
```

Published runs pin the test binary to one logical CPU with
`GOMAXPROCS=1`, as for [`../terrain/RESULTS.md`](../terrain/RESULTS.md).
On Windows, pin from the start so that `runtime.NumCPU` sees one CPU:
`start "" /b /wait /high /affinity 10 focal.test.exe ...` from `cmd`.
The stride table above adds `-test.bench '(GaussianR5|MeanR5|MinR5|MeanR3|CorrelateR5)/size=.*/mask=off/backend=simd' -strata.sizes 16320,16384,16448`;
the column-pass table is `GOEXPERIMENT=simd go test -c ./internal/focalrow`
run pinned the same way with `-test.bench ColumnStride -test.count 5`, on
the commit before the change for the "before" figures.
`go test ./benchmarks/cmd/stratabench` checks that the section between
the markers is what stratabench renders from `testdata/bench.txt`.

## Results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 1 usable by the process, GOMAXPROCS 1 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | focalrow: avx2 |
| Runs | 5 per benchmark, medians shown |

## focal

4096 × 4096 raster, no mask, M cells/sec:

```text
            scalar      SIMD  SIMD/scalar   SIMD + workers
CorrelateR1       296      1092        3.69×   not measured yet (tile engine, STRATA-8/9)
CorrelateR2       131       545        4.17×   not measured yet (tile engine, STRATA-8/9)
CorrelateR3      69.5       306        4.41×   not measured yet (tile engine, STRATA-8/9)
CorrelateR5      28.7       131        4.56×   not measured yet (tile engine, STRATA-8/9)
GaussianR1       564      1454        2.58×   not measured yet (tile engine, STRATA-8/9)
GaussianR2       347      1170        3.37×   not measured yet (tile engine, STRATA-8/9)
GaussianR3       255       952        3.73×   not measured yet (tile engine, STRATA-8/9)
GaussianR5       164       624        3.80×   not measured yet (tile engine, STRATA-8/9)
MeanR1         455      1489        3.28×   not measured yet (tile engine, STRATA-8/9)
MeanR2         310      1290        4.16×   not measured yet (tile engine, STRATA-8/9)
MeanR3         233      1068        4.58×   not measured yet (tile engine, STRATA-8/9)
MeanR5         158       663        4.21×   not measured yet (tile engine, STRATA-8/9)
MinR1          446      1277        2.86×   not measured yet (tile engine, STRATA-8/9)
MinR2          240       871        3.62×   not measured yet (tile engine, STRATA-8/9)
MinR3          166       647        3.91×   not measured yet (tile engine, STRATA-8/9)
MinR5          102       429        4.21×   not measured yet (tile engine, STRATA-8/9)
MaxR3          105       518        4.93×   not measured yet (tile engine, STRATA-8/9)
```

### CorrelateR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 379 | 739 | 1.95× | 1.35 | 3.03 | 5.91 | 9 |
| 1024 × 1024 | off | 393 | 1089 | 2.77× | 0.918 | 3.15 | 8.71 | 9 |
| 4096 × 4096 | off | 296 | 1092 | 3.69× | 0.916 | 2.37 | 8.73 | 9 |
| 16384 × 16384 | off | 275 | 1069 | 3.88× | 0.935 | 2.20 | 8.55 | 9 |
| 256 × 256 | on | 357 | 661 | 1.85× | 1.51 | 2.95 | 5.45 | 12 |
| 1024 × 1024 | on | 381 | 1012 | 2.66× | 0.988 | 3.14 | 8.35 | 12 |
| 4096 × 4096 | on | 292 | 1035 | 3.54× | 0.966 | 2.41 | 8.54 | 12 |
| 16384 × 16384 | on | 272 | 1011 | 3.72× | 0.989 | 2.24 | 8.34 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (739 → 1069 M cells/s), SIMD/scalar 1.95× → 3.88×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (661 → 1011 M cells/s), SIMD/scalar 1.85× → 3.72×.

### CorrelateR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 143 | 361 | 2.52× | 2.77 | 1.14 | 2.88 | 9 |
| 1024 × 1024 | off | 143 | 501 | 3.50× | 1.99 | 1.15 | 4.01 | 9 |
| 4096 × 4096 | off | 131 | 545 | 4.17× | 1.83 | 1.04 | 4.36 | 9 |
| 16384 × 16384 | off | 120 | 542 | 4.53× | 1.84 | 0.96 | 4.33 | 9 |
| 256 × 256 | on | 138 | 332 | 2.40× | 3.01 | 1.14 | 2.74 | 12 |
| 1024 × 1024 | on | 141 | 474 | 3.35× | 2.11 | 1.17 | 3.91 | 12 |
| 4096 × 4096 | on | 128 | 523 | 4.07× | 1.91 | 1.06 | 4.32 | 12 |
| 16384 × 16384 | on | 118 | 519 | 4.40× | 1.93 | 0.97 | 4.28 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (361 → 542 M cells/s), SIMD/scalar 2.52× → 4.53×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (332 → 519 M cells/s), SIMD/scalar 2.40× → 4.40×.

### CorrelateR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 75.7 | 221 | 2.92× | 4.52 | 0.61 | 1.77 | 9 |
| 1024 × 1024 | off | 74.6 | 284 | 3.81× | 3.52 | 0.60 | 2.28 | 9 |
| 4096 × 4096 | off | 69.5 | 306 | 4.41× | 3.27 | 0.56 | 2.45 | 9 |
| 16384 × 16384 | off | 66.0 | 303 | 4.59× | 3.30 | 0.53 | 2.42 | 9 |
| 256 × 256 | on | 74.1 | 207 | 2.79× | 4.84 | 0.61 | 1.71 | 12 |
| 1024 × 1024 | on | 73.7 | 272 | 3.69× | 3.68 | 0.61 | 2.24 | 12 |
| 4096 × 4096 | on | 68.9 | 297 | 4.31× | 3.37 | 0.57 | 2.45 | 12 |
| 16384 × 16384 | on | 65.6 | 294 | 4.47× | 3.41 | 0.54 | 2.42 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (221 → 303 M cells/s), SIMD/scalar 2.92× → 4.59×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (207 → 294 M cells/s), SIMD/scalar 2.79× → 4.47×.

### CorrelateR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 31.7 | 89.6 | 2.83× | 11.2 | 0.25 | 0.72 | 9 |
| 1024 × 1024 | off | 30.5 | 119 | 3.90× | 8.40 | 0.24 | 0.95 | 9 |
| 4096 × 4096 | off | 28.7 | 131 | 4.56× | 7.65 | 0.23 | 1.05 | 9 |
| 16384 × 16384 | off | 28.7 | 132 | 4.61× | 7.55 | 0.23 | 1.06 | 9 |
| 256 × 256 | on | 31.2 | 85.7 | 2.74× | 11.7 | 0.26 | 0.71 | 12 |
| 1024 × 1024 | on | 30.4 | 116 | 3.83× | 8.58 | 0.25 | 0.96 | 12 |
| 4096 × 4096 | on | 28.5 | 128 | 4.47× | 7.84 | 0.24 | 1.05 | 12 |
| 16384 × 16384 | on | 28.6 | 130 | 4.55× | 7.71 | 0.24 | 1.07 | 12 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (89.6 → 132 M cells/s), SIMD/scalar 2.83× → 4.61×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (85.7 → 130 M cells/s), SIMD/scalar 2.74× → 4.55×.

### GaussianR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 586 | 1217 | 2.08× | 0.822 | 4.69 | 9.74 | 10 |
| 1024 × 1024 | off | 600 | 1738 | 2.89× | 0.575 | 4.80 | 13.9 | 10 |
| 4096 × 4096 | off | 564 | 1454 | 2.58× | 0.688 | 4.51 | 11.6 | 10 |
| 16384 × 16384 | off | 569 | 1419 | 2.49× | 0.705 | 4.55 | 11.3 | 11 |
| 256 × 256 | on | 532 | 1013 | 1.90× | 0.987 | 4.39 | 8.36 | 13 |
| 1024 × 1024 | on | 567 | 1536 | 2.71× | 0.651 | 4.68 | 12.7 | 13 |
| 4096 × 4096 | on | 545 | 1347 | 2.47× | 0.742 | 4.50 | 11.1 | 13 |
| 16384 × 16384 | on | 553 | 1322 | 2.39× | 0.756 | 4.57 | 10.9 | 14 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1217 → 1419 M cells/s), SIMD/scalar 2.08× → 2.49×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1013 → 1322 M cells/s), SIMD/scalar 1.90× → 2.39×.

### GaussianR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 355 | 909 | 2.56× | 1.10 | 2.84 | 7.27 | 10 |
| 1024 × 1024 | off | 363 | 1238 | 3.41× | 0.808 | 2.90 | 9.90 | 10 |
| 4096 × 4096 | off | 347 | 1170 | 3.37× | 0.855 | 2.78 | 9.36 | 10 |
| 16384 × 16384 | off | 355 | 1168 | 3.29× | 0.856 | 2.84 | 9.34 | 12 |
| 256 × 256 | on | 328 | 748 | 2.28× | 1.34 | 2.71 | 6.17 | 13 |
| 1024 × 1024 | on | 349 | 1082 | 3.10× | 0.924 | 2.88 | 8.93 | 13 |
| 4096 × 4096 | on | 339 | 1079 | 3.19× | 0.927 | 2.79 | 8.90 | 13 |
| 16384 × 16384 | on | 345 | 1060 | 3.07× | 0.943 | 2.85 | 8.74 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (909 → 1168 M cells/s), SIMD/scalar 2.56× → 3.29×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (748 → 1060 M cells/s), SIMD/scalar 2.28× → 3.07×.

### GaussianR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 259 | 736 | 2.85× | 1.36 | 2.07 | 5.89 | 10 |
| 1024 × 1024 | off | 260 | 948 | 3.64× | 1.05 | 2.08 | 7.58 | 10 |
| 4096 × 4096 | off | 255 | 952 | 3.73× | 1.05 | 2.04 | 7.61 | 10 |
| 16384 × 16384 | off | 257 | 803 | 3.12× | 1.25 | 2.06 | 6.43 | 14 |
| 256 × 256 | on | 240 | 603 | 2.51× | 1.66 | 1.98 | 4.97 | 13 |
| 1024 × 1024 | on | 250 | 823 | 3.29× | 1.22 | 2.07 | 6.79 | 13 |
| 4096 × 4096 | on | 247 | 850 | 3.45× | 1.18 | 2.04 | 7.02 | 13 |
| 16384 × 16384 | on | 250 | 718 | 2.87× | 1.39 | 2.06 | 5.92 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (736 → 803 M cells/s), SIMD/scalar 2.85× → 3.12×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (603 → 718 M cells/s), SIMD/scalar 2.51× → 2.87×.

### GaussianR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 169 | 492 | 2.91× | 2.03 | 1.35 | 3.94 | 10 |
| 1024 × 1024 | off | 167 | 607 | 3.64× | 1.65 | 1.33 | 4.86 | 10 |
| 4096 × 4096 | off | 164 | 624 | 3.80× | 1.60 | 1.31 | 4.99 | 10 |
| 16384 × 16384 | off | 166 | 577 | 3.48× | 1.73 | 1.33 | 4.62 | 14 |
| 256 × 256 | on | 158 | 401 | 2.55× | 2.49 | 1.30 | 3.31 | 13 |
| 1024 × 1024 | on | 160 | 526 | 3.28× | 1.90 | 1.32 | 4.34 | 13 |
| 4096 × 4096 | on | 159 | 562 | 3.53× | 1.78 | 1.31 | 4.64 | 13 |
| 16384 × 16384 | on | 161 | 524 | 3.25× | 1.91 | 1.33 | 4.32 | 17 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 6%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (492 → 577 M cells/s), SIMD/scalar 2.91× → 3.48×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (401 → 524 M cells/s), SIMD/scalar 2.55× → 3.25×.

### MeanR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 504 | 1301 | 2.58× | 0.769 | 4.03 | 10.4 | 8 |
| 1024 × 1024 | off | 515 | 1920 | 3.73× | 0.521 | 4.12 | 15.4 | 8 |
| 4096 × 4096 | off | 455 | 1489 | 3.28× | 0.671 | 3.64 | 11.9 | 8 |
| 16384 × 16384 | off | 443 | 1453 | 3.28× | 0.688 | 3.54 | 11.6 | 10 |
| 256 × 256 | on | 464 | 1072 | 2.31× | 0.933 | 3.83 | 8.85 | 11 |
| 1024 × 1024 | on | 495 | 1671 | 3.37× | 0.598 | 4.09 | 13.8 | 11 |
| 4096 × 4096 | on | 444 | 1385 | 3.12× | 0.722 | 3.66 | 11.4 | 11 |
| 16384 × 16384 | on | 434 | 1354 | 3.12× | 0.738 | 3.58 | 11.2 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1301 → 1453 M cells/s), SIMD/scalar 2.58× → 3.28×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1072 → 1354 M cells/s), SIMD/scalar 2.31× → 3.12×.

### MeanR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 334 | 981 | 2.93× | 1.02 | 2.68 | 7.85 | 8 |
| 1024 × 1024 | off | 339 | 1398 | 4.13× | 0.715 | 2.71 | 11.2 | 8 |
| 4096 × 4096 | off | 310 | 1290 | 4.16× | 0.775 | 2.48 | 10.3 | 8 |
| 16384 × 16384 | off | 304 | 1212 | 3.99× | 0.825 | 2.43 | 9.70 | 10 |
| 256 × 256 | on | 310 | 800 | 2.58× | 1.25 | 2.56 | 6.60 | 11 |
| 1024 × 1024 | on | 326 | 1208 | 3.71× | 0.828 | 2.69 | 9.96 | 11 |
| 4096 × 4096 | on | 301 | 1154 | 3.83× | 0.867 | 2.49 | 9.52 | 11 |
| 16384 × 16384 | on | 297 | 1103 | 3.72× | 0.906 | 2.45 | 9.10 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (981 → 1212 M cells/s), SIMD/scalar 2.93× → 3.99×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (800 → 1103 M cells/s), SIMD/scalar 2.58× → 3.72×.

### MeanR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 256 | 802 | 3.14× | 1.25 | 2.04 | 6.41 | 8 |
| 1024 × 1024 | off | 254 | 1086 | 4.27× | 0.920 | 2.03 | 8.69 | 8 |
| 4096 × 4096 | off | 233 | 1068 | 4.58× | 0.936 | 1.86 | 8.54 | 8 |
| 16384 × 16384 | off | 231 | 849 | 3.68× | 1.18 | 1.85 | 6.79 | 12 |
| 256 × 256 | on | 237 | 644 | 2.71× | 1.55 | 1.96 | 5.31 | 11 |
| 1024 × 1024 | on | 244 | 915 | 3.75× | 1.09 | 2.01 | 7.54 | 11 |
| 4096 × 4096 | on | 227 | 942 | 4.15× | 1.06 | 1.87 | 7.77 | 11 |
| 16384 × 16384 | on | 224 | 763 | 3.41× | 1.31 | 1.85 | 6.30 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 8%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (802 → 849 M cells/s), SIMD/scalar 3.14× → 3.68×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (644 → 763 M cells/s), SIMD/scalar 2.71× → 3.41×.

### MeanR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 175 | 583 | 3.33× | 1.72 | 1.40 | 4.66 | 8 |
| 1024 × 1024 | off | 171 | 686 | 4.01× | 1.46 | 1.37 | 5.49 | 8 |
| 4096 × 4096 | off | 158 | 663 | 4.21× | 1.51 | 1.26 | 5.30 | 8 |
| 16384 × 16384 | off | 156 | 627 | 4.03× | 1.60 | 1.25 | 5.01 | 12 |
| 256 × 256 | on | 163 | 464 | 2.85× | 2.16 | 1.34 | 3.83 | 11 |
| 1024 × 1024 | on | 164 | 590 | 3.60× | 1.70 | 1.35 | 4.87 | 11 |
| 4096 × 4096 | on | 151 | 588 | 3.90× | 1.70 | 1.25 | 4.85 | 11 |
| 16384 × 16384 | on | 152 | 572 | 3.77× | 1.75 | 1.25 | 4.72 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (583 → 627 M cells/s), SIMD/scalar 3.33× → 4.03×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (464 → 572 M cells/s), SIMD/scalar 2.85× → 3.77×.

### MinR1

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 451 | 1192 | 2.64× | 0.839 | 3.61 | 9.54 | 8 |
| 1024 × 1024 | off | 487 | 1428 | 2.93× | 0.701 | 3.90 | 11.4 | 8 |
| 4096 × 4096 | off | 446 | 1277 | 2.86× | 0.783 | 3.57 | 10.2 | 8 |
| 16384 × 16384 | off | 439 | 1282 | 2.92× | 0.780 | 3.51 | 10.3 | 10 |
| 256 × 256 | on | 419 | 995 | 2.38× | 1.00 | 3.46 | 8.21 | 11 |
| 1024 × 1024 | on | 468 | 1297 | 2.77× | 0.771 | 3.86 | 10.7 | 11 |
| 4096 × 4096 | on | 435 | 1195 | 2.75× | 0.837 | 3.59 | 9.86 | 11 |
| 16384 × 16384 | on | 429 | 1204 | 2.81× | 0.830 | 3.54 | 9.94 | 13 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 2%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (1192 → 1282 M cells/s), SIMD/scalar 2.64× → 2.92×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (995 → 1204 M cells/s), SIMD/scalar 2.38× → 2.81×.

### MinR2

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 241 | 795 | 3.30× | 1.26 | 1.93 | 6.36 | 8 |
| 1024 × 1024 | off | 252 | 898 | 3.57× | 1.11 | 2.01 | 7.18 | 8 |
| 4096 × 4096 | off | 240 | 871 | 3.62× | 1.15 | 1.92 | 6.97 | 8 |
| 16384 × 16384 | off | 239 | 857 | 3.59× | 1.17 | 1.91 | 6.86 | 12 |
| 256 × 256 | on | 228 | 669 | 2.93× | 1.49 | 1.89 | 5.52 | 11 |
| 1024 × 1024 | on | 244 | 805 | 3.30× | 1.24 | 2.01 | 6.64 | 11 |
| 4096 × 4096 | on | 236 | 811 | 3.44× | 1.23 | 1.95 | 6.69 | 11 |
| 16384 × 16384 | on | 234 | 799 | 3.42× | 1.25 | 1.93 | 6.59 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 4%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (795 → 857 M cells/s), SIMD/scalar 3.30× → 3.59×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (669 → 799 M cells/s), SIMD/scalar 2.93× → 3.42×.

### MinR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 166 | 583 | 3.52× | 1.72 | 1.33 | 4.66 | 8 |
| 1024 × 1024 | off | 170 | 658 | 3.87× | 1.52 | 1.36 | 5.26 | 8 |
| 4096 × 4096 | off | 166 | 647 | 3.91× | 1.55 | 1.32 | 5.17 | 8 |
| 16384 × 16384 | off | 164 | 591 | 3.60× | 1.69 | 1.31 | 4.73 | 12 |
| 256 × 256 | on | 158 | 496 | 3.13× | 2.02 | 1.30 | 4.09 | 11 |
| 1024 × 1024 | on | 165 | 599 | 3.63× | 1.67 | 1.36 | 4.94 | 11 |
| 4096 × 4096 | on | 162 | 601 | 3.71× | 1.67 | 1.34 | 4.96 | 11 |
| 16384 × 16384 | on | 161 | 557 | 3.45× | 1.80 | 1.33 | 4.59 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (583 → 591 M cells/s), SIMD/scalar 3.52× → 3.60×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (496 → 557 M cells/s), SIMD/scalar 3.13× → 3.45×.

### MinR5

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 104 | 381 | 3.67× | 2.62 | 0.83 | 3.05 | 8 |
| 1024 × 1024 | off | 104 | 426 | 4.11× | 2.35 | 0.83 | 3.40 | 8 |
| 4096 × 4096 | off | 102 | 429 | 4.21× | 2.33 | 0.81 | 3.43 | 8 |
| 16384 × 16384 | off | 101 | 398 | 3.94× | 2.51 | 0.81 | 3.18 | 12 |
| 256 × 256 | on | 99.1 | 327 | 3.30× | 3.06 | 0.82 | 2.70 | 11 |
| 1024 × 1024 | on | 101 | 386 | 3.83× | 2.59 | 0.83 | 3.18 | 11 |
| 4096 × 4096 | on | 99.9 | 397 | 3.98× | 2.52 | 0.82 | 3.28 | 11 |
| 16384 × 16384 | on | 99.2 | 372 | 3.75× | 2.69 | 0.82 | 3.07 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 0%, worst 1%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (381 → 398 M cells/s), SIMD/scalar 3.67× → 3.94×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (327 → 372 M cells/s), SIMD/scalar 3.30× → 3.75×.

### MaxR3

| raster | mask | scalar M cells/s | SIMD M cells/s | SIMD/scalar | SIMD ns/cell | scalar GB/s | SIMD GB/s | allocs/op |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | off | 110 | 495 | 4.51× | 2.02 | 0.88 | 3.96 | 8 |
| 1024 × 1024 | off | 109 | 523 | 4.80× | 1.91 | 0.87 | 4.19 | 8 |
| 4096 × 4096 | off | 105 | 518 | 4.93× | 1.93 | 0.84 | 4.14 | 8 |
| 16384 × 16384 | off | 106 | 484 | 4.54× | 2.07 | 0.85 | 3.87 | 12 |
| 256 × 256 | on | 106 | 430 | 4.07× | 2.32 | 0.87 | 3.55 | 11 |
| 1024 × 1024 | on | 102 | 465 | 4.56× | 2.15 | 0.84 | 3.83 | 11 |
| 4096 × 4096 | on | 106 | 488 | 4.62× | 2.05 | 0.87 | 4.03 | 11 |
| 16384 × 16384 | on | 105 | 462 | 4.39× | 2.17 | 0.87 | 3.81 | 15 |

Run-to-run spread of M cells/s, (max − min)/median: median 1%, worst 12%.

- mask=off: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (495 → 484 M cells/s), SIMD/scalar 4.51× → 4.54×.
- mask=on: compute-bound: SIMD throughput stays within 20% of 256²'s up to 16384² (430 → 462 M cells/s), SIMD/scalar 4.07× → 4.39×.
<!-- stratabench output end -->

## arm64 (NEON)

The same suite on the NEON kernels of `internal/focalrow`, the run this
file published before the AVX2 one. Its headline, as first recorded:

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
