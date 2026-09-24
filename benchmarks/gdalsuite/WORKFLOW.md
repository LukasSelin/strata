# A fused workflow against GDAL: slope + aspect + hillshade

This asks whether a fused workflow beats GDAL by more than a single
operation does. It is evidence for making a planner the main thing
strata offers: lazy graphs of operations, run as one pass on the
`internal/exec` Pipeline (DESIGN.md §52). The suite times one job, all
three terrain products of one DEM, in three ways:

- **(a) fused:** one `stratasuite -op surface` process. One
  `terrain.SurfaceChunked` call reads the DEM once, computes the Horn
  gradient once, and writes slope, aspect and hillshade to three raw
  files.
- **(b) separate:** three `stratasuite` processes (`-op slope`,
  `aspect`, `hillshade`) run one after another. Together they are timed
  as one case, wall and CPU. This is strata today, without a planner.
- **(c) GDAL:** three `gdaldem` runs, timed as one case. Each tier uses
  GDAL's fastest configuration under the suite's rules. That is always
  GeoTIFF output on one thread, because `GDAL_NUM_THREADS=12` makes
  gdaldem from the COG take 23.3 s instead of 10.9 s.

The job runs in the same container, on the same window as
[RESULTS.md](RESULTS.md): 11264 × 11264 of HGV_leaf.tif, a Deflate
predictor-3 COG with 512 blocks, and GDAL 3.14. Each case has one
warm-up run and 5 timed runs, and the tables give the median. `runsuite.sh`'s
`workflow()` does this. `summarize.py` prints the table, from
[testdata/workflow-timings.txt](testdata/workflow-timings.txt) (tmpfs) and
[testdata/workflow-timings-disk.txt](testdata/workflow-timings-disk.txt)
(`WORKVOL`, an ext4 volume in Docker Desktop's VM). Both runs are on
strata bf20dac:

```bash
OPS=surface REPEATS=5 ./benchmarks/gdalsuite/gdalsuite.sh HGV_leaf.tif 1024 320 11264 11264
WORKVOL=strata-gdalsuite-workflow OPS=surface REPEATS=5 ./benchmarks/gdalsuite/gdalsuite.sh ...
```

## The numbers

A ratio says how many times faster its denominator is. (c)/(a) is the
fused pass against GDAL. CPU is user + sys seconds, over all the
processes of the case.

**tmpfs** (the suite's default):

| tier | threads | (a) fused | (b) separate | (c) GDAL | (b)/(a) | (c)/(a) | (c)/(b) | CPU (a) | CPU (b) | CPU (c) | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compute | 1 | 559.7 ms | 536.4 ms | 7.59 s (mem, 1) | 1.0× | 13.6× | 14.1× | – | – | – | 3% |
| compute | 12 | 118.0 ms | 157.9 ms | 7.59 s (mem, 1) | 1.3× | 64.3× | 48.1× | – | – | – | 12% |
| file to file | 1 | 1.62 s | 1.85 s | 9.53 s (gtiff, 1) | 1.1× | 5.9× | 5.2× | 1.71 | 2.14 | 9.52 | 5% |
| file to file | 12 | 783.0 ms | 1.03 s | 9.53 s (gtiff, 1) | 1.3× | 12.2× | 9.2× | 4.49 | 6.78 | 9.52 | 15% |
| whole flow from a COG | 1 | 2.46 s | 4.36 s | 10.93 s (to-gtiff, 1) | 1.8× | 4.4× | 2.5× | 2.51 | 4.39 | 10.91 | 3% |
| whole flow from a COG | 12 | 879.0 ms | 1.45 s | 10.93 s (to-gtiff, 1) | 1.6× | 12.4× | 7.5× | 5.02 | 7.85 | 10.91 | 3% |

**Docker volume on a real disk** (`WORKVOL`; the harness runs `sync`
before each timed run):

| tier | threads | (a) fused | (b) separate | (c) GDAL | (b)/(a) | (c)/(a) | (c)/(b) | CPU (a) | CPU (b) | CPU (c) | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compute | 1 | 563.0 ms | 539.8 ms | 7.53 s (mem, 1) | 1.0× | 13.4× | 13.9× | – | – | – | 7% |
| compute | 12 | 120.3 ms | 156.3 ms | 7.53 s (mem, 1) | 1.3× | 62.6× | 48.2× | – | – | – | 10% |
| file to file | 1 | 1.11 s | 1.39 s | 9.44 s (gtiff, 1) | 1.2× | 8.5× | 6.8× | 1.15 | 1.56 | 9.44 | 13% |
| file to file | 12 | 897.0 ms | 1.18 s | 9.44 s (gtiff, 1) | 1.3× | 10.5× | 8.0× | 6.22 | 8.50 | 9.44 | 8% |
| whole flow from a COG | 1 | 1.94 s | 3.99 s | 10.94 s (to-gtiff, 1) | 2.1× | 5.6× | 2.7× | 1.95 | 4.00 | 10.92 | 6% |
| whole flow from a COG | 12 | 944.0 ms | 1.49 s | 10.94 s (to-gtiff, 1) | 1.6× | 11.6× | 7.3× | 6.54 | 9.11 | 10.92 | 7% |

No case spread more than 15%, so none is flagged ⚠. The machine was
also running an unrelated game-server container at about 1.7 cores
throughout.

Three things about the setup:

- **The disk run does not time writeback.** `sync` runs before each
  timed run, untimed, so no run pays for an earlier run's writeback.
  But a run's own dirty pages are still in the page cache when the
  process exits. So the disk run measures writing into the page cache
  of an ext4 file, not durable writes.
- **The disk run is faster on one worker.** It is 1.94 s against
  2.46 s on tmpfs. On tmpfs, dropping the previous run's outputs frees
  their pages and writing allocates them again. That is the tmpfs
  effect `ReuseRawFile` documents (benchmarks/rawio).
- **Every output is overwritten.** Each timed run truncates the
  previous run's outputs, as the rest of the suite does. Fused and
  separate pay this equally: both write three 508 MB files per run.

### Correctness

Every check passed in both runs:

| product | fused == separate (`cmp`) | fused from COG, 12 workers == fused from raw, 1 worker (`cmp`) | cells compared with gdaldem | validity differs | max abs diff | mean abs diff |
| --- | :---: | :---: | ---: | ---: | ---: | ---: |
| slope | yes | yes | 123,785,590 (97.6%) | 0 | 7.63e-06 | 1.26e-06 |
| aspect | yes | yes | 99,492,495 (78.4%) | 24,293,095 | 3.05e-05 | 4.62e-06 |
| hillshade | yes | yes | 123,785,590 (97.6%) | 0 | 1.5 | 0.614 |

- **fused == separate:** each fused output, from the raw file on one
  worker, is byte-identical to that product's own op.
- **COG == raw:** the fused whole flow from the COG on 12 workers is
  byte-identical to the fused raw path on one worker.
- **The gdaldem rows** are the numbers `agree.py` gives for each
  product alone in RESULTS.md. Two cells differ for known reasons:
  - Aspect's validity differs on flat cells, which are −1 in strata and
    NoData in gdaldem.
  - Hillshade differs by up to 1.5 because gdaldem rounds to a byte.

## Against the ~2 s estimate

The estimate was fused ≈ 0.85 s decode + 0.55 s compute + three writes,
about 2 s. It put separate strata at about 4.4 s and three gdaldem runs
at about 11.2 s, so the workflow would be about 5× or more against GDAL.

| | estimate | measured, tmpfs | measured, disk |
| --- | ---: | ---: | ---: |
| (b) three strata processes | ~4.4 s | 4.36 s | 3.99 s |
| (c) three gdaldem runs | ~11.2 s | 10.93 s | 10.94 s |
| (a) fused | ~2 s | **2.46 s** | **1.94 s** |
| (c)/(a), one thread | ≥ 5× | **4.4×** | **5.6×** |

The separate and GDAL estimates were right. The fused one was about
0.45 s too optimistic on tmpfs and about right on the disk volume. In
both cases the error is in the writes, which the estimate treated as
small. Writing three outputs costs about 1.0 s on tmpfs, as much as
decoding the COG (see below).

So on one thread, **fusion takes the workflow from 2.5× GDAL to 4.4×
(tmpfs) or 5.6× (disk)**. The 2.5× is (c)/(b), which is also this job's
per-operation lead: 10.93 s against 4.36 s. The suite-wide geometric
mean is 3.7×. The ≥5× claim holds on the disk volume and falls short on
tmpfs.

On 12 workers the fused lead is 12.4× / 11.6× against 7.5× / 7.3× for
separate. That is well above the suite-wide 8.0×. But fusion itself is
worth less there: (b)/(a) is only 1.6×, against 1.8–2.1× on one thread.

## Where the fused pass's time goes

Method:

- **CPU profile:** `stratasuite -cpuprofile`, on the tmpfs setup, 5
  in-process runs of the fused whole flow from the COG. Samples are
  bucketed by stack: `cog.(*Source)` is decode; `internal/stencil` and
  the Pipeline stages are compute; `RawSink`, `Behind` and
  `RawFile.WriteAt` are writes.
- **Wall-clock phases:** from stratasuite's `phases` line. `open`
  opens the COG and creates the three outputs, which frees the
  previous run's. `run` is the `SurfaceChunked` call. `close` unmaps.

**One worker, 2.46 s wall (tmpfs):**

| part | time | how measured |
| --- | ---: | --- |
| create the three outputs (freeing the previous run's 3 × 508 MB) | 0.24 s | `open` phase |
| decode the COG (inflate + floating-point predictor) | 0.79 s | profile |
| compute: gradient, then slope, aspect, hillshade | 0.54 s | profile (compute tier: 0.56 s) |
| write the three outputs (`pwrite` into tmpfs, 1.5 GB) | 0.79 s | profile |
| process start, exit, the rest | ~0.1 s | remainder |

**Twelve workers, 0.88 s wall (tmpfs):**

| part | time |
| --- | ---: |
| create outputs (one thread, serial) | 0.24 s wall |
| `SurfaceChunked` | 0.44 s wall |
| unmap | 0.07 s wall |

Of the 12-worker run's 4.7 CPU-seconds, 46% is copying into the mapped
outputs, most of it page faults in the kernel. Decode is 26% and
compute is 25%.

What this shows:

- **Once decode is shared, writing is the largest cost.**
  - On one thread, writing (creating plus writing) is about 1.0 s,
    decode 0.8 s and compute 0.55 s.
  - On twelve workers, writing is the biggest CPU consumer. The
    serial truncation alone is 27% of the wall time.
  - Three separate processes decode the COG three times. That is
    where (b) loses its 1.9 s on one thread.
  - They write the same 1.5 GB as the fused pass, which is why fusion
    gains less on 12 workers: decode parallelises well and writing
    does not.
- **Fusing the arithmetic gains almost nothing.** In memory, fused
  against three kernels is 1.0× on one thread and 1.3× on 12. The
  shared Horn gradient (about 12% of the profiled compute) is cheap
  next to the products' own work: aspect, mostly its `atan2`, is about
  40%. So the
  win from fusion here is almost entirely I/O. The DEM is read and
  decoded once instead of three times, and two process starts and
  two output-file setups are skipped.
- **Reusing the output pages saves most of the create cost.** In an
  ad hoc probe outside the suite, fused from raw on 12 workers with
  `stratasuite -reuse` (`engine.ReuseRawFile`) took 0.3 ms to open,
  against 240 ms with truncation, and 315 ms to run against 350 ms.
  That was not timed under the suite's rules.

## What this implies for a planner

- **The case for a planner is decode and I/O, not arithmetic.** A lazy
  graph that reads a COG once for several consumers turns a 2.5×
  workflow into a 4.4–5.6× one on one thread. It turns 7.5× into
  11.6–12.4× on twelve. Kernel-level fusion alone (compute tier: 1.0–1.3×)
  would not justify the work.
- **After fusion, the outputs are the bottleneck.** When every product
  is written to its own file, the three writes cost as much as decode
  on one thread, and more CPU on twelve.
  - A planner's biggest further gain is not writing values nobody
    asked to keep: intermediates stay in the Pipeline's scratch, and
    only the requested outputs go to files. In this job every product
    was requested, so there was nothing to drop.
  - For requested outputs, the levers are the write path:
    - reuse or preallocate pages instead of truncate-then-fault, as
      `ReuseRawFile` shows;
    - avoid the serial truncation on the critical path;
    - write compressed output, which trades write bytes for encode CPU.
- **Scaling will be bounded by writes, not decode.** The fused pass
  runs 2.8× faster on 12 workers than on one (2.46 s to 0.88 s). The
  compute tier runs 4.7× faster. A planner that fuses deeper graphs
  with more outputs will hit this first.
- **GDAL cannot do this at all from the command line.** Each gdaldem
  run decodes the COG again. Its threads only decode, and on this
  machine they make the three runs twice as slow (23.3 s). So the gap
  grows with every product added to the job, for the same reason
  (b)/(a) does.

## Problems met

- **The first run was invalid on 12 workers.** `separate_strata` set
  the shared `SARGS` array, so every fused case timed after a separate
  one ran hillshade alone. That was the 12-worker rows (0.33 s and
  0.47 s, which looked too good). The one-worker rows and the
  correctness checks were not affected. It was fixed in bf20dac by
  making `SARGS` local there, and both runs above were taken after the
  fix. It was found because the phases from a standalone profile
  (0.8 s) disagreed with the suite's number.
- **gdaldem's validity and rounding differ from strata's.** These are
  the known cases above: aspect's flat cells and hillshade's byte
  rounding.
