# Chunked execution: results

Bounded-memory execution over sources and sinks (DESIGN.md §24, §27) and
the first validation target (§43): terrain.SlopeChunked,
terrain.HillshadeChunked and algebra.ClampChunked from a raw float32 file
to another, by worker count and tile shape, and a 20000 × 20000 DEM
checked against the whole-raster result with measured peak memory.
Metrics and names are defined in [`../README.md`](../README.md).

## Headline

- **Memory is bounded by the tile size and the workers, not the raster.**
  Every chunked run's peak private bytes are its §27 bound, Workers ×
  (TileW+2r) × (TileH+2r) × 8 bytes, plus 14–22 MiB of process overhead.
  At 20000² (a 1.49 GiB file) 12 workers in full-width strips of 256 rows
  peak at 489 MiB against a 472 MiB bound; the whole raster in memory
  peaks at 3076 MiB. 1024 × 1024 tiles with 12 workers peak at 113 MiB on
  a 4096² DEM and 114 MiB on a 20000² one, against the same 96 MiB bound.
- **The output is the whole-raster result.** All 126 demo runs (three
  operations, scalar and SIMD, 1, 12 and 24 workers, two tile shapes, two
  DEM sizes, three runs each) wrote files equal to the plain function's
  output cell for cell.
- **12 workers process the 20000² DEM at about 830 M cells/s** for Slope,
  Hillshade and Clamp alike (Slope: 121 scalar, 276 SIMD, 832 SIMD with 12
  workers), half a second for the 400 million cells. That is 3.0× one SIMD worker
  for Slope, but a flat ceiling for every operation, like the memory
  sources' (about 1.1 billion cells/s at 16384²): the limit is copying
  every cell into the tile buffers and out again, not file IO or compute.
  The whole raster in memory computes Slope at 707 M cells/s on one
  worker and about 2.5 billion with 12 (benchmarks/engine).
- **Files need one handle per worker and few calls.** Calls on one
  `*os.File` queue on Windows; engine.RawFile's handle per worker made 12
  workers 1.6× faster. Reading and writing full-width rows in 1 MiB calls
  is 17–21% faster than a call per row. Narrow tiles take two calls per
  row: 1024 × 1024 tiles of a 20000-wide file run at about 200 M cells/s
  with 12 workers, 256 × 256 tiles at 65–85.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512). L2 512 KiB per core, L3 64 MiB (16 MiB per CCX) |
| Memory | 64 GB DDR4-3200, 4 × 16 GB, dual channel |
| Storage | local disk, NTFS; every file is in the OS file cache during the runs (64 GB of RAM) |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Suite run | `GOEXPERIMENT=simd go test -c ./benchmarks/chunked`, then `chunked.test.exe -test.run '^$' -test.bench . -test.count 5 -test.timeout 6h`, **not pinned**, `GOMAXPROCS=24`, High priority: 42 minutes |
| Demo run | `GOEXPERIMENT=simd go build ./benchmarks/cmd/stratademo`, then `stratademo.exe -dir <dir> -sizes 4096,20000 -count 3`, High priority, right after the suite's build, before the suite: 5 minutes |
| Raw output | [`testdata/bench.txt`](testdata/bench.txt) (suite), [`testdata/demo.txt`](testdata/demo.txt) (demo) |
| Peak memory, suite | 3.69 GiB private bytes for all Slope cases at 16384² (measured with `suite.ProcessMemory`): the fixture's two 1 GiB operands and masks, 24 workers' tile buffers and the garbage of earlier calls' buffers |
| Stats | suite: median of 5 runs, each ≥1 s (`b.Loop`); demo: median of 3 runs, peak memory the maximum |

To reproduce the tables between the markers:

```
GOEXPERIMENT=simd go test ./benchmarks/chunked -run '^$' -bench . -count 5 -timeout 6h > bench.txt
go run ./benchmarks/cmd/stratabench < bench.txt
GOEXPERIMENT=simd go run ./benchmarks/cmd/stratademo -dir <dir> -sizes 4096,20000 > demo.txt
go run ./benchmarks/cmd/stratademo -render demo.txt
```

`go test ./benchmarks/cmd/stratabench ./benchmarks/cmd/stratademo` checks
that the sections between the markers are exactly the commands' output for
the files in `testdata`.

The demo needs about 7.5 GiB of disk at 20000² (the DEM, a reference per
operation and the output) and 3 GiB of memory for the in-memory
references. The desktop had other applications open; the median suite
case varies 2% between runs, the worst 30–79% (Clamp's worst, see its
table).

## Validity from a fill value (2026-09-24)

The masked rows below use a DEM file whose invalid cells, 10% of them,
scattered, hold the fill value -9999. So every tile reads validity from
the fill (`RawSource`) and writes the fill back under invalid cells
(`RawSink`), and no tile is all valid. They were taken before the fill
test became a vector kernel (DESIGN.md §31, "Rules 4 and 5 at the file
boundary"). This is master (8ad43da) against that change, for the one
case that isolates it: 4096², one worker, strips of 256 rows. Both are
GOEXPERIMENT=simd test binaries, pinned to one core at High priority
with GOMAXPROCS=1, run interleaved 6 times, median shown. `scalar` is
the suite's scalar kernels, which include `vec.ValidBits`'s scalar form.

| op, mask on | kernels | master, M cells/s | this change | speedup | spread (master / change) |
|---|---|---:|---:|---:|---|
| Slope | SIMD | 145 | 242 | 1.67× | 15% / 9% |
| Slope | scalar | 68 | 82 | 1.21× | 10% / 5% |
| Hillshade | SIMD | 158 | 298 | 1.89× | 7% / 8% |
| Hillshade | scalar | 68 | 79 | 1.15× | 4% / 4% |
| Clamp | SIMD | 190 | 395 | 2.08× | 14% / 28% |
| Clamp | scalar | 163 | 239 | 1.46× | 10% / 11% |

Unmasked cases were not rerun quietly. The change does not touch
their path (`RawSource` without a fill value, and no masks for the
engine to check), and the one run that included them, too noisy to
publish (45–90% spreads, with the desktop in use), showed no
difference beyond its noise. Raw
output: [`testdata/validity-ab.txt`](testdata/validity-ab.txt). The
suite tables below are from before the change and have not been
regenerated.

## Shapes

The suite's tile shapes, over the benchmarks/engine DEM (a smooth surface
of 800 ± 300 with ±1 noise):

- `plain`: the plain function with both rasters in memory, one goroutine,
  no IO. The reference.
- `strips256`: the Chunked function from a raw file to another through
  engine.RawSource and engine.RawSink on engine.RawFile (one handle per
  logical CPU), in full-width tiles of 256 rows.
- `256x256`: the same in 256 × 256 tiles.
- `memstrips256`: the Chunked function over engine.MemorySource and
  engine.MemorySink on the in-memory rasters, in full-width tiles of 256
  rows: the cost of the tile buffers without files.

With `mask=on` the raw files carry a fill value (-9999) under the DEM's
invalid cells (10%) and the sink writes it under invalid output cells, so
the source and sink are Masked; the memory shapes use the masks directly.

The demo runs `stratademo`: for each operation the plain function in
memory (timed without reading and writing), then the Chunked function
from the DEM file to an output file in full-width strips of 256 rows
(scalar with one worker, SIMD with 1, 12 and 24) and in 1024 × 1024 tiles
(SIMD), each in a child process that reports its peak private bytes, and
compares each output file with the reference.

## File handles and call size

Before RawFile and grouped calls, the first 20000² demo ran Slope at 256 M
cells/s on one worker, 463 with 12 and 418 with 24, and 1024 × 1024 tiles
at 81–114 whatever the workers. A probe of SlopeChunked on the 20000² DEM
(median of 3, M cells/s, before grouped calls and compact buffers) against
memory sources and sinks over the same data:

| tiles | workers | one `*os.File` per file | one handle per worker | memory |
|---|---:|---:|---:|---:|
| 20000×256 | 1 | 211 | 209 | 356 |
| 20000×256 | 12 | 361 | 563 | 836 |
| 20000×256 | 24 | 396 | 592 | 736 |
| 1024×1024 | 1 | 82 | 83 | 422 |
| 1024×1024 | 12 | 122 | 199 | 861 |
| 1024×1024 | 24 | 119 | 197 | 844 |

- **Handles.** Go's `os.File.ReadAt` and `WriteAt` on Windows take the
  handle's read-write lock and seek twice around each positioned
  `ReadFile` or `WriteFile`, so workers sharing a handle take turns.
  Separate handles gave 1.56–1.63× with 12 workers. Without contention
  (one worker) they change nothing.
- **Calls.** One `ReadAt` of 4104 bytes (a 1026-cell row) from the cache
  costs 2.98 µs, and a positioned `ReadFile` without Go's lock and seeks
  2.31 µs, so the calls, not Go, are the cost: 1024 × 1024 tiles make
  about 820 000 calls per pass. Joining short rows into one read would
  copy the 18 974 cells between them, 12.9 µs per row at the measured
  rate, more than the call it saves. Only full-width windows are grouped.
- **Call size.** Sequential reads and writes of 1 GiB of the cached files
  on one handle:

  | bytes per call | read GB/s | write GB/s |
  |---:|---:|---:|
  | 80 008 (one row) | 5.47 | 3.01 |
  | 262 144 | 6.19 | 3.43 |
  | 1 048 576 | 6.63 | 3.53 |
  | 4 194 304 | 6.57 | 3.46 |
  | 16 777 216 | 4.88 | 3.11 |

  So raw sources and sinks move full-width windows in 1 MiB calls, into
  and out of compact buffers: the engine pads rows to a multiple of 64
  cells only in buffers with masks.

With both, the 20000² demo went from 463 to 832 M cells/s for Slope with
12 workers and from 106 to 192 with 1024 × 1024 tiles; the final run is
below.

## Suite results

<!-- stratabench output begin -->
| | |
|---|---|
| CPU | AMD Ryzen 9 3900X 12-Core Processor |
| Cores | 12 physical, 24 logical; 24 usable by the process, GOMAXPROCS 24 |
| Go | go1.27.0-X:simd windows/amd64, GOAMD64=v1, GOEXPERIMENT=simd |
| Kernels | stencil: avx2, vec: avx2 |
| Runs | 5 per benchmark, medians shown |

## chunked

4096 × 4096 raster, no mask, M cells/sec (workers run the strips256 shape):

```text
              scalar      SIMD  SIMD/scalar   SIMD + 12 workers   SIMD + 24 workers
Slope            168       722        4.29×                 586                 542
Hillshade        216      1339        6.20×                 588                 530
Clamp            790      2331        2.95×                 579                 514
```

### Slope

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 173 | 728 | – | – | – | – | 5.82 | 8 |
| 1024 × 1024 | off | strips256 | 135 | 352 | -52% | 527 | 516 | 1.50× at 12 | 4.13 | 34 |
| 1024 × 1024 | off | 256x256 | 34.6 | 41.0 | -94% | 73.5 | 69.0 | 1.79× at 12 | 0.55 | 114 |
| 1024 × 1024 | off | memstrips256 | 154 | 495 | -32% | 677 | 674 | 1.37× at 12 | 5.39 | 32 |
| 4096 × 4096 | off | plain | 168 | 722 | – | – | – | – | 5.77 | 8 |
| 4096 × 4096 | off | strips256 | 128 | 299 | -59% | 586 | 542 | 1.96× at 12 | 4.33 | 112 |
| 4096 × 4096 | off | 256x256 | 31.6 | 36.9 | -95% | 82.5 | 71.0 | 2.24× at 12 | 0.57 | 161 |
| 4096 × 4096 | off | memstrips256 | 156 | 515 | -29% | 834 | 735 | 1.62× at 12 | 5.88 | 104 |
| 16384 × 16384 | off | plain | 153 | 698 | – | – | – | – | 5.58 | 8 |
| 16384 × 16384 | off | strips256 | 125 | 285 | -59% | 790 | 710 | 2.77× at 12 | 5.68 | 197 |
| 16384 × 16384 | off | 256x256 | 28.4 | 32.2 | -95% | 67.7 | 61.0 | 2.10× at 12 | 0.49 | 165 |
| 16384 × 16384 | off | memstrips256 | 158 | 530 | -24% | 1126 | 996 | 2.12× at 12 | 7.97 | 152 |
| 1024 × 1024 | on | plain | 168 | 688 | – | – | – | – | 5.68 | 11 |
| 1024 × 1024 | on | strips256 | 71.5 | 156 | -77% | 360 | 356 | 2.32× at 12 | 2.94 | 47 |
| 1024 × 1024 | on | 256x256 | 27.4 | 35.1 | -95% | 70.6 | 66.7 | 2.01× at 12 | 0.55 | 164 |
| 1024 × 1024 | on | memstrips256 | 151 | 467 | -32% | 685 | 687 | 1.47× at 24 | 5.67 | 45 |
| 4096 × 4096 | on | plain | 166 | 694 | – | – | – | – | 5.73 | 11 |
| 4096 × 4096 | on | strips256 | 69.8 | 146 | -79% | 477 | 468 | 3.28× at 12 | 3.86 | 168 |
| 4096 × 4096 | on | 256x256 | 25.9 | 31.8 | -95% | 79.9 | 69.2 | 2.52× at 12 | 0.57 | 231 |
| 4096 × 4096 | on | memstrips256 | 153 | 500 | -28% | 793 | 688 | 1.58× at 12 | 5.68 | 153 |
| 16384 × 16384 | on | plain | 150 | 673 | – | – | – | – | 5.55 | 11 |
| 16384 × 16384 | on | strips256 | 69.9 | 147 | -78% | 637 | 588 | 4.32× at 12 | 4.86 | 270 |
| 16384 × 16384 | on | 256x256 | 23.6 | 28.5 | -96% | 66.3 | 59.8 | 2.33× at 12 | 0.49 | 243 |
| 16384 × 16384 | on | memstrips256 | 156 | 510 | -24% | 1039 | 946 | 2.04× at 12 | 7.81 | 225 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 30%.

- mask=off, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (728 → 698 M cells/s), SIMD/scalar 4.22× → 4.56×.
- mask=off, workers: strips256 over one worker, SIMD: 1024² 1.50× with 12, 1.47× with 24; 4096² 1.96× with 12, 1.81× with 24; 16384² 2.77× with 12, 2.49× with 24; 24 workers move 4.13 GB/s at 1024², 4.33 GB/s at 4096², 5.68 GB/s at 16384².
- mask=on, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (688 → 673 M cells/s), SIMD/scalar 4.09× → 4.49×.
- mask=on, workers: strips256 over one worker, SIMD: 1024² 2.32× with 12, 2.29× with 24; 4096² 3.28× with 12, 3.21× with 24; 16384² 4.32× with 12, 4.00× with 24; 24 workers move 2.94 GB/s at 1024², 3.86 GB/s at 4096², 4.86 GB/s at 16384².

### Hillshade

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 212 | 1406 | – | – | – | – | 11.2 | 8 |
| 1024 × 1024 | off | strips256 | 160 | 474 | -66% | 550 | 549 | 1.16× at 12 | 4.39 | 34 |
| 1024 × 1024 | off | 256x256 | 36.3 | 42.5 | -97% | 74.6 | 70.2 | 1.75× at 12 | 0.56 | 114 |
| 1024 × 1024 | off | memstrips256 | 189 | 702 | -50% | 746 | 741 | 1.06× at 12 | 5.92 | 32 |
| 4096 × 4096 | off | plain | 216 | 1339 | – | – | – | – | 10.7 | 8 |
| 4096 × 4096 | off | strips256 | 149 | 374 | -72% | 588 | 530 | 1.57× at 12 | 4.24 | 114 |
| 4096 × 4096 | off | 256x256 | 32.8 | 37.7 | -97% | 80.9 | 70.8 | 2.15× at 12 | 0.57 | 154 |
| 4096 × 4096 | off | memstrips256 | 192 | 768 | -43% | 812 | 726 | 1.06× at 12 | 5.81 | 104 |
| 16384 × 16384 | off | plain | 171 | 1284 | – | – | – | – | 10.3 | 8 |
| 16384 × 16384 | off | strips256 | 147 | 347 | -73% | 782 | 698 | 2.26× at 12 | 5.59 | 201 |
| 16384 × 16384 | off | 256x256 | 29.3 | 33.1 | -97% | 67.6 | 61.1 | 2.04× at 12 | 0.49 | 170 |
| 16384 × 16384 | off | memstrips256 | 195 | 789 | -39% | 1125 | 984 | 1.43× at 12 | 7.87 | 152 |
| 1024 × 1024 | on | plain | 213 | 1278 | – | – | – | – | 10.6 | 11 |
| 1024 × 1024 | on | strips256 | 71.9 | 171 | -87% | 374 | 374 | 2.18× at 12 | 3.08 | 47 |
| 1024 × 1024 | on | 256x256 | 28.3 | 36.0 | -97% | 73.9 | 69.2 | 2.05× at 12 | 0.57 | 167 |
| 1024 × 1024 | on | memstrips256 | 184 | 669 | -48% | 723 | 724 | 1.08× at 24 | 5.97 | 45 |
| 4096 × 4096 | on | plain | 212 | 1239 | – | – | – | – | 10.2 | 11 |
| 4096 × 4096 | on | strips256 | 69.9 | 165 | -87% | 479 | 474 | 2.91× at 12 | 3.91 | 164 |
| 4096 × 4096 | on | 256x256 | 25.9 | 32.3 | -97% | 77.4 | 69.3 | 2.40× at 12 | 0.57 | 229 |
| 4096 × 4096 | on | memstrips256 | 191 | 738 | -40% | 820 | 702 | 1.11× at 12 | 5.79 | 153 |
| 16384 × 16384 | on | plain | 168 | 1210 | – | – | – | – | 9.98 | 11 |
| 16384 × 16384 | on | strips256 | 70.4 | 163 | -87% | 657 | 592 | 4.04× at 12 | 4.88 | 266 |
| 16384 × 16384 | on | 256x256 | 23.7 | 29.2 | -98% | 65.9 | 59.9 | 2.26× at 12 | 0.49 | 236 |
| 16384 × 16384 | on | memstrips256 | 192 | 748 | -38% | 1096 | 945 | 1.47× at 12 | 7.80 | 226 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 32%.

- mask=off, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (1406 → 1284 M cells/s), SIMD/scalar 6.62× → 7.53×.
- mask=off, workers: strips256 over one worker, SIMD: 1024² 1.16× with 12, 1.16× with 24; 4096² 1.57× with 12, 1.42× with 24; 16384² 2.26× with 12, 2.01× with 24; 24 workers move 4.39 GB/s at 1024², 4.24 GB/s at 4096², 5.59 GB/s at 16384².
- mask=on, one worker: compute-bound: SIMD throughput stays within 20% of 1024²'s up to 16384² (1278 → 1210 M cells/s), SIMD/scalar 6.00× → 7.22×.
- mask=on, workers: strips256 over one worker, SIMD: 1024² 2.18× with 12, 2.18× with 24; 4096² 2.91× with 12, 2.88× with 24; 16384² 4.04× with 12, 3.64× with 24; 24 workers move 3.08 GB/s at 1024², 3.91 GB/s at 4096², 4.88 GB/s at 16384².

### Clamp

| raster | mask | tiles | scalar M cells/s | SIMD M cells/s | SIMD vs plain | SIMD + 12 workers M cells/s | SIMD + 24 workers M cells/s | scaling | SIMD GB/s, most workers | allocs/op |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 | off | plain | 825 | 4536 | – | – | – | – | 36.3 | 0 |
| 1024 × 1024 | off | strips256 | 382 | 632 | -86% | 577 | 554 | – | 4.43 | 34 |
| 1024 × 1024 | off | 256x256 | 43.3 | 45.5 | -99% | 77.0 | 67.8 | 1.69× at 12 | 0.54 | 114 |
| 1024 × 1024 | off | memstrips256 | 530 | 1023 | -77% | 725 | 620 | – | 4.96 | 32 |
| 4096 × 4096 | off | plain | 790 | 2331 | – | – | – | – | 18.6 | 0 |
| 4096 × 4096 | off | strips256 | 310 | 462 | -80% | 579 | 514 | 1.25× at 12 | 4.11 | 118 |
| 4096 × 4096 | off | 256x256 | 38.4 | 40.0 | -98% | 80.4 | 70.4 | 2.01× at 12 | 0.56 | 155 |
| 4096 × 4096 | off | memstrips256 | 552 | 1174 | -50% | 870 | 716 | – | 5.72 | 104 |
| 16384 × 16384 | off | plain | 714 | 1782 | – | – | – | – | 14.3 | 0 |
| 16384 × 16384 | off | strips256 | 292 | 392 | -78% | 771 | 713 | 1.97× at 12 | 5.70 | 194 |
| 16384 × 16384 | off | 256x256 | 33.4 | 34.7 | -98% | 68.0 | 62.5 | 1.96× at 12 | 0.50 | 166 |
| 16384 × 16384 | off | memstrips256 | 560 | 1102 | -38% | 1143 | 992 | 1.04× at 12 | 7.94 | 152 |
| 1024 × 1024 | on | plain | 796 | 4505 | – | – | – | – | 37.2 | 0 |
| 1024 × 1024 | on | strips256 | 179 | 219 | -95% | 426 | 408 | 1.94× at 12 | 3.36 | 43 |
| 1024 × 1024 | on | 256x256 | 38.0 | 39.6 | -99% | 76.7 | 72.1 | 1.94× at 12 | 0.59 | 147 |
| 1024 × 1024 | on | memstrips256 | 522 | 989 | -78% | 697 | 686 | – | 5.66 | 41 |
| 4096 × 4096 | on | plain | 782 | 2273 | – | – | – | – | 18.8 | 0 |
| 4096 × 4096 | on | strips256 | 165 | 197 | -91% | 489 | 465 | 2.48× at 12 | 3.84 | 151 |
| 4096 × 4096 | on | 256x256 | 34.0 | 35.5 | -98% | 80.6 | 70.2 | 2.27× at 12 | 0.58 | 209 |
| 4096 × 4096 | on | memstrips256 | 548 | 1115 | -51% | 803 | 680 | – | 5.61 | 137 |
| 16384 × 16384 | on | plain | 704 | 1730 | – | – | – | – | 14.3 | 0 |
| 16384 × 16384 | on | strips256 | 163 | 193 | -89% | 675 | 603 | 3.50× at 12 | 4.98 | 240 |
| 16384 × 16384 | on | 256x256 | 30.2 | 31.1 | -98% | 67.1 | 62.0 | 2.16× at 12 | 0.51 | 211 |
| 16384 × 16384 | on | memstrips256 | 553 | 1055 | -39% | 1084 | 958 | 1.03× at 12 | 7.91 | 202 |

Run-to-run spread of M cells/s, (max − min)/median: median 2%, worst 79%.

- mask=off, one worker: memory-bandwidth-bound from 4096²: SIMD throughput falls to 39% of 1024²'s by 16384² (4536 → 1782 M cells/s) and the SIMD/scalar speedup flattens (5.50× → 2.50×); SIMD moves 18.6 GB/s at 4096², 14.3 GB/s at 16384².
- mask=off, workers: strips256 over one worker, SIMD: 1024² 0.91× with 12, 0.88× with 24; 4096² 1.25× with 12, 1.11× with 24; 16384² 1.97× with 12, 1.82× with 24; 24 workers move 4.43 GB/s at 1024², 4.11 GB/s at 4096², 5.70 GB/s at 16384².
- mask=on, one worker: memory-bandwidth-bound from 4096²: SIMD throughput falls to 38% of 1024²'s by 16384² (4505 → 1730 M cells/s) and the SIMD/scalar speedup flattens (5.66× → 2.46×); SIMD moves 18.8 GB/s at 4096², 14.3 GB/s at 16384².
- mask=on, workers: strips256 over one worker, SIMD: 1024² 1.94× with 12, 1.86× with 24; 4096² 2.48× with 12, 2.36× with 24; 16384² 3.50× with 12, 3.13× with 24; 24 workers move 3.36 GB/s at 1024², 3.84 GB/s at 4096², 4.98 GB/s at 16384².
<!-- stratabench output end -->

## Demo results (§43)

<!-- stratademo output begin -->
### 4096 × 4096 DEM, raw float32 file of 0.06 GiB

| op | tiles | backend | workers | M cells/s | GB/s | vs SIMD 1 worker | peak private MiB | §27 bound MiB | peak − base MiB | flush s | identical |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| slope | whole raster in memory | simd | 1 | 706 | 5.65 | – | 143 | – | 0 | 0.03 | reference |
| slope | strips256 (4096×256) | scalar | 1 | 121 | 0.97 | – | 22 | 8 | 10 | 0.03 | 3/3 |
| slope | strips256 (4096×256) | simd | 1 | 277 | 2.22 | 1.00× | 22 | 8 | 10 | 0.03 | 3/3 |
| slope | strips256 (4096×256) | simd | 12 | 621 | 4.97 | 2.24× | 113 | 97 | 101 | 0.09 | 3/3 |
| slope | strips256 (4096×256) | simd | 24 (16 used) | 576 | 4.61 | 2.08× | 147 | 129 | 135 | 0.09 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 1 | 102 | 0.82 | 1.00× | 22 | 8 | 10 | 0.03 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 12 | 198 | 1.58 | 1.94× | 113 | 96 | 101 | 0.03 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 24 (16 used) | 202 | 1.61 | 1.97× | 147 | 129 | 134 | 0.03 | 3/3 |
| hillshade | whole raster in memory | simd | 1 | 1228 | 9.83 | – | 143 | – | 0 | 0.03 | reference |
| hillshade | strips256 (4096×256) | scalar | 1 | 143 | 1.14 | – | 22 | 8 | 10 | 0.11 | 3/3 |
| hillshade | strips256 (4096×256) | simd | 1 | 342 | 2.73 | 1.00× | 22 | 8 | 10 | 0.04 | 3/3 |
| hillshade | strips256 (4096×256) | simd | 12 | 625 | 5.00 | 1.83× | 113 | 97 | 101 | 0.04 | 3/3 |
| hillshade | strips256 (4096×256) | simd | 24 (16 used) | 610 | 4.88 | 1.78× | 147 | 129 | 135 | 0.05 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 1 | 111 | 0.89 | 1.00× | 22 | 8 | 10 | 0.05 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 12 | 187 | 1.50 | 1.69× | 113 | 96 | 101 | 0.05 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 24 (16 used) | 186 | 1.49 | 1.67× | 146 | 129 | 134 | 0.05 | 3/3 |
| clamp | whole raster in memory | simd | 1 | 2376 | 19.01 | – | 143 | – | 0 | 0.04 | reference |
| clamp | strips256 (4096×256) | scalar | 1 | 277 | 2.21 | – | 22 | 8 | 11 | 0.05 | 3/3 |
| clamp | strips256 (4096×256) | simd | 1 | 425 | 3.40 | 1.00× | 22 | 8 | 10 | 0.05 | 3/3 |
| clamp | strips256 (4096×256) | simd | 12 | 612 | 4.89 | 1.44× | 113 | 96 | 101 | 0.05 | 3/3 |
| clamp | strips256 (4096×256) | simd | 24 (16 used) | 635 | 5.08 | 1.49× | 146 | 128 | 134 | 0.05 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 1 | 125 | 1.00 | 1.00× | 22 | 8 | 10 | 0.05 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 12 | 207 | 1.66 | 1.65× | 113 | 96 | 101 | 0.05 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 24 (16 used) | 200 | 1.60 | 1.60× | 146 | 128 | 135 | 0.04 | 3/3 |

### 20000 × 20000 DEM, raw float32 file of 1.49 GiB

| op | tiles | backend | workers | M cells/s | GB/s | vs SIMD 1 worker | peak private MiB | §27 bound MiB | peak − base MiB | flush s | identical |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| slope | whole raster in memory | simd | 1 | 707 | 5.66 | – | 3076 | – | 1 | 0.73 | reference |
| slope | strips256 (20000×256) | scalar | 1 | 121 | 0.97 | – | 53 | 39 | 41 | 1.00 | 3/3 |
| slope | strips256 (20000×256) | simd | 1 | 276 | 2.21 | 1.00× | 54 | 39 | 41 | 0.81 | 3/3 |
| slope | strips256 (20000×256) | simd | 12 | 832 | 6.66 | 3.01× | 489 | 472 | 477 | 1.62 | 3/3 |
| slope | strips256 (20000×256) | simd | 24 | 820 | 6.56 | 2.97× | 964 | 945 | 952 | 1.71 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 1 | 85 | 0.68 | 1.00× | 22 | 8 | 10 | 0.70 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 12 | 192 | 1.54 | 2.25× | 114 | 96 | 102 | 0.98 | 3/3 |
| slope | 1024x1024 (1024×1024) | simd | 24 | 202 | 1.62 | 2.37× | 212 | 193 | 200 | 0.80 | 3/3 |
| hillshade | whole raster in memory | simd | 1 | 1307 | 10.46 | – | 3075 | – | 0 | 4.78 | reference |
| hillshade | strips256 (20000×256) | scalar | 1 | 145 | 1.16 | – | 54 | 39 | 42 | 0.79 | 3/3 |
| hillshade | strips256 (20000×256) | simd | 1 | 334 | 2.67 | 1.00× | 53 | 39 | 41 | 1.20 | 3/3 |
| hillshade | strips256 (20000×256) | simd | 12 | 829 | 6.63 | 2.48× | 489 | 472 | 477 | 0.90 | 3/3 |
| hillshade | strips256 (20000×256) | simd | 24 | 839 | 6.71 | 2.51× | 965 | 945 | 953 | 1.00 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 1 | 91 | 0.73 | 1.00× | 22 | 8 | 10 | 0.54 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 12 | 204 | 1.63 | 2.24× | 114 | 96 | 102 | 1.12 | 3/3 |
| hillshade | 1024x1024 (1024×1024) | simd | 24 | 207 | 1.65 | 2.27× | 212 | 193 | 200 | 0.68 | 3/3 |
| clamp | whole raster in memory | simd | 1 | 2240 | 17.92 | – | 3075 | – | 1 | 0.59 | reference |
| clamp | strips256 (20000×256) | scalar | 1 | 292 | 2.33 | – | 54 | 39 | 41 | 1.24 | 3/3 |
| clamp | strips256 (20000×256) | simd | 1 | 389 | 3.11 | 1.00× | 54 | 39 | 42 | 0.82 | 3/3 |
| clamp | strips256 (20000×256) | simd | 12 | 843 | 6.74 | 2.17× | 487 | 469 | 475 | 0.78 | 3/3 |
| clamp | strips256 (20000×256) | simd | 24 | 841 | 6.73 | 2.16× | 960 | 938 | 948 | 0.94 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 1 | 96 | 0.77 | 1.00× | 22 | 8 | 10 | 1.44 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 12 | 211 | 1.69 | 2.20× | 113 | 96 | 101 | 0.69 | 3/3 |
| clamp | 1024x1024 (1024×1024) | simd | 24 | 207 | 1.66 | 2.16× | 212 | 192 | 200 | 0.57 | 3/3 |

Medians of up to 3 runs per case; peak private bytes are the maximum. GB/s counts the 8 bytes per cell of the input and output files. "identical" counts the runs whose output file equals the reference cell for cell.
<!-- stratademo output end -->

## What the numbers say (§27, §28, §43)

- **The bound holds and is the whole cost.** Peak private bytes minus the
  bound are 14–22 MiB in every chunked case, at 4096² and at 20000², on 1
  worker or 24: the Go runtime, the binary, goroutine stacks and a little
  heap. The buffers' row padding in masked buffers and the input halo are
  inside the bound; outputs are TileW × TileH, below it. A 24-worker call
  with 256-row strips of a 20000-wide raster holds 945 MiB, so strip
  height and worker count are the knobs.
- **Chunked calls are copy-bound.** From 12 workers every operation lands
  at 770–845 M cells/s on the 16384² and 20000² files and at about 1.1
  billion on memory sources at 16384², whatever its compute cost: Clamp is 2.5×
  faster than Slope in memory and no faster chunked. Each cell's 8 bytes
  of input and output are copied from the file cache into a buffer, read
  and written by the kernel, and copied back: about four passes over
  memory per cell against the in-memory path's one, which is where
  benchmarks/engine's 19–23 GB/s ceiling of memory bandwidth goes. 24
  workers are no faster than 12.
- **One worker loses most on cheap kernels.** Against the plain function,
  strips256 on one worker costs 59% for Slope, 73% for Hillshade and 78%
  for Clamp at 16384²; the memory shape costs 24–39%. The kernels are the
  same; the copies are not free.
- **Workers help more than in memory.** Slope scales 2.8× with 12 workers
  on the 16384² file (3.0× on the 20000² demo), because the copies of
  different tiles run in parallel; in memory it scaled 3.7× from a much
  higher start.
- **Small rasters do not scale.** At 1024² a call has 4 strips, so at most
  4 workers run, and each call allocates and zeroes its buffers: Clamp is
  slower with 12 workers than with one. Chunked calls are for rasters
  larger than memory, where one call runs for seconds.
- **Masks cost twice in files.** With a fill value, strips256 on one
  worker drops from 285 to 147 M cells/s for Slope at 16384²: masked
  buffers keep their padded rows, so rows are read a call at a time, and
  the fill value is compared cell by cell on read and written under
  invalid cells on write. Workers recover most of it (637 with 12).
- **Narrow tiles are an IO trap.** 256 × 256 tiles run at 24–46 M cells/s
  on one worker and 65–85 with 12, whatever the operation: two calls per
  row of 256 cells.

For the §43 target this means: a 20000² DEM in a raw file is processed
with bounded memory (489 MiB for 12 workers), in tiles, on 12 workers, with
SIMD, at about 830 M cells/s, and matches the whole-raster result bit for
bit. Beyond that, throughput needs fewer copies per cell: memory-mapped
sources that the kernel reads in place, and operation fusion (§29) so a
pipeline copies each cell in and out once.
