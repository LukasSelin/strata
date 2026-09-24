# Writing a chunked call's output: results

How a chunked call writes its raw float32 output (`engine.RawSink` over
`engine.RawFile`), and what changed. Before this, writing was the
bottleneck of multi-worker runs: 12 workers doing slope from one raw file
into another took 0.43 s, and a reduction over the same file, which
writes nothing, took 0.09 s. A CPU profile of 12-worker slope from a COG
had 53% of its CPU in system calls. Writing did not parallelise at all.

Two changes, both measured before they were chosen:

1. **`engine.CreateRawFile`** sizes the output file up front and, when it
   is opened for concurrent writers (more than one handle), maps it
   shared into memory on Linux and Windows. `WriteAt` becomes a copy into
   the mapping.
2. **Write-behind** in the chunked engine: every worker hands a finished
   tile to a writer goroutine of its own and computes the next tile into
   a second set of output buffers while the first is written.

Not one output bit changed: [`acceptance/`](../../acceptance/) passed all
714 terrain and 375 resampling checks (plain == tiled == chunked, bit for
bit, with its chunked runs writing through the mapped path), and both
gdalsuite runs below wrote byte-identical slope files from the COG on 12
workers and from the raw file on one.

## Headline

- **12 workers: raw slope 1.6–2.0× faster, slope from a COG 1.2–1.5×**
  (interleaved A/B against master 8ad43da, in-process time).
  - Raw file to raw file: 453 → 288 ms on tmpfs, 360 → 292 ms on ext4
    and 545 → 274 ms on NTFS. On NTFS the old path was also erratic
    (339–1316 ms); the new one stays within 9%.
  - From the Deflate COG: 546 → 404 ms on tmpfs, 437 → 374 ms on ext4,
    617 → 405 ms on NTFS.
- **One worker gains 17–34% when a second core is free.**
  - With GOMAXPROCS=2, raw slope goes 854 → 563 ms on tmpfs and
    732 → 495 ms on ext4; slope from the COG goes 1489 → 1207 ms and
    1401 → 1164 ms.
  - In benchmarks/chunked on Windows (default GOMAXPROCS), one worker is
    31–36% faster.
  - Under GOMAXPROCS=1, as gdalsuite runs one thread, the gain is 13–21%
    for raw input on Linux and nothing from a COG or on Windows (see
    [Why GOMAXPROCS=1 gains little](#why-gomaxprocs1-gains-little)).
- **The targets were not reached.** They were ~0.15 s for 12-worker raw
  slope and ~0.3 s from the COG. In gdalsuite's whole-process timing on
  tmpfs, the two now take 0.35 s and 0.49 s (0.39 s and 0.50 s on ext4),
  down from the 0.43 s and ~0.60 s the task started from. The 0.35 s
  splits as follows (medians):

  | part | ms |
  |---|---:|
  | Freeing the previous run's 508 MB output (`O_TRUNC`, which every suite run of either tool pays) | 82 |
  | The operation | 180 |
  | Unmapping on Close | 24 |
  | Process start and exit | ~60 |

  From the COG the operation takes 300 ms. What is left in the operation
  is mostly the kernel allocating the output's pages: page faults inside
  the copy were 44% of CPU.
- **A full disk is an error, not a crash.**
  - A mapped write into a full 100 MB tmpfs returns `ErrMappedFault`.
  - A 1-worker `WriteAt` returns `ENOSPC`.
  - On ext4, `fallocate` fails in `CreateRawFile`, before any work, and
    leaves an empty file ([`testdata/diskfull.txt`](testdata/diskfull.txt)).

## Why writing did not scale

The write probe ([`probe/`](probe/)) writes the 508 MB output in the
engine's pattern (strips of 256 full-width rows, 1 MiB `WriteAt` calls,
one handle per writer) with no computation. Medians of 5 runs into a new
file, in seconds, including unmap and close
([`testdata/probe-linux.txt`](testdata/probe-linux.txt),
[`testdata/probe-windows.txt`](testdata/probe-windows.txt)):

| | writers | WriteAt | fallocate + WriteAt | mapped | fallocate + mapped | overwrite, WriteAt | overwrite, mapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| tmpfs | 1 | 0.24 | 0.30 | 0.38 | 0.42 | 0.11 | 0.20 |
| tmpfs | 12 | 0.28 | 0.30 | **0.13** | 0.26 | 0.12 | 0.07 |
| ext4 | 1 | 0.17 | **0.14** | 0.19 | 0.16 | 0.09 | 0.05 |
| ext4 | 12 | 0.19 | **0.14** | 0.22 | 0.19 | 0.13 | 0.10 |
| NTFS | 1 | **0.19** | 0.20 | 0.27 | 0.25 | 0.19 | 0.18 |
| NTFS | 12 | 0.45 (±63%) | 0.57 | **0.17** | 0.20 | 0.10 | 0.12 |

- **Twelve writers into one file are no faster than one.** Linux takes
  the file's lock (`i_rwsem`) for every buffered write, and NTFS
  serialises writes that extend the valid data length. With the same
  bytes split into 12 separate files, `dd` took 0.09 s.
- **Most of the cost is the page cache, not the system call.**
  Overwriting a file whose pages exist is 2–3× cheaper than writing a
  new one.
- **A shared mapping makes the copies parallel,** at 0.13 s on tmpfs
  and 0.17 s on NTFS. But it costs a page fault per 4 KiB, which one
  writer pays serially (0.38 s against 0.24 s on tmpfs). Hence
  `CreateRawFile` maps only with more than one handle.
- **On ext4, `fallocate` is what helps.** It only marks blocks as
  unwritten, costs 1 ms, and speeds up later writes, mapped or not.
  Without it, every mapped page fault allocates a block. On tmpfs,
  `fallocate` zeroes all 508 MB on one core first (0.18 s), which
  undoes the gain. So `CreateRawFile` calls `fallocate` except on tmpfs,
  which it detects with `fstatfs`. There, a full tmpfs faults the
  mapped write instead, and `debug.SetPanicOnFault` turns that fault
  into an error.
- **In the probe, mapping does not beat fallocate + WriteAt on ext4**
  (0.19 against 0.14 s). In the slope below it does, 358 → 320 ms
  with write-behind. With `WriteAt`, a worker blocked on the file's
  lock does nothing; with the mapping, twelve workers fault pages
  concurrently alongside the compute.

## The design, measured in parts

Before settling, a prototype build switched each part with environment
variables: `STRATA_NOMAP`, `STRATA_WB` and `STRATA_FALLOC`. None of
them is in the final code. Slope, 11264² from HGV_leaf.tif, median of 5
interleaved runs, in-process ms. On tmpfs and NTFS the file was not
preallocated; on ext4, "+ fallocate" rows used it.

| | workers | WriteAt | + write-behind | mapped | mapped + write-behind | fallocate + WriteAt | fallocate + mapped + write-behind |
|---|---:|---:|---:|---:|---:|---:|---:|
| tmpfs, raw | 1 | 911 | **820** | 1061 | 1060 | | |
| tmpfs, raw | 12 | 448 | 481 | 308 | **307** | | |
| tmpfs, COG | 1 | **1511** | 1528 | 1703 | 1675 | | |
| tmpfs, COG | 12 | 582 | 529 | 460 | **419** | | |
| ext4, raw | 1 | 883 | **754** | 983 | 842 | 796 | 838 |
| ext4, raw | 12 | 398 | 465 | 413 | 411 | 358 | **320** |
| ext4, COG | 1 | 1582 | 1470 | 1654 | 1560 | **1460** | 1502 |
| ext4, COG | 12 | 511 | 467 | 523 | 489 | 503 | **423** |
| NTFS, raw | 1 | **758** | 758 | 803 | 791 | | |
| NTFS, raw | 12 | 545 | 401 | **274** | 280 | | |
| NTFS, COG | 1 | 1372 | **1365** | 1399 | 1377 | | |
| NTFS, COG | 12 | 511 | **392** | 397 | 401 | | |

Raw data: [`testdata/design-linux-run1.txt`](testdata/design-linux-run1.txt)
(tmpfs; its ext4 rows ran without the sync and are not used),
[`testdata/design-linux-run2.txt`](testdata/design-linux-run2.txt) (ext4,
sync before every run) and
[`testdata/design-windows.txt`](testdata/design-windows.txt). Each part's
weak spot is visible:

- **Mapping at one worker is slower everywhere.** That is why it is
  tied to more than one handle.
- **Write-behind alone does not fix 12 workers.** The writes are still
  serialised in the kernel, so the writer queues fill and the workers
  wait for them.
- **Write-behind helps least under GOMAXPROCS=1,** particularly from the
  COG.

Together, mapped for concurrent writers and write-behind always, the
combination is the best or within noise of it in every row.

A CPU profile of the mapped prototype, 12 workers, raw input on tmpfs:
44% of CPU in `memmove` into the mapping (page faults, charged to the
copy), 17% in read system calls, 10% in the input's fill-value scan
(`RawSource.rowValidity`) and 22% in the slope kernel.

## The final build against master

[`ab.sh`](ab.sh) builds gdalsuite's `stratasuite` at master (8ad43da) and
from the working tree, then runs the two alternately: 7 rounds of every
case, in-process time from opening the inputs to closing the output. It
syncs before every run on ext4 so no run pays for an earlier run's
writeback. The ext4 target is a Docker volume, which is ext4 on the
Docker Desktop VM's virtual disk on NVMe. NTFS runs are native Windows
([`abrun.sh`](abrun.sh) with `NOSYNC=1`). `open`, `run` and `close` are
the new build's phases: opening the inputs and creating the output, the
operation, and closing (unmapping) the output.

| disk | input | workers | GOMAXPROCS | old ms | new ms | change | spread old / new | open / run / close |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tmpfs | raw | 1 | 1 | 863 | 755 | −13% | 8% / 17% | 82 / 669 / 0 |
| tmpfs | raw | 1 | 2 | 854 | 563 | −34% | 21% / 17% | 83 / 476 / 0 |
| tmpfs | raw | 12 | 12 | 453 | **288** | −37% | 12% / 12% | 82 / 180 / 24 |
| tmpfs | COG | 1 | 1 | 1510 | 1503 | 0% | 6% / 6% | 83 / 1421 / 0 |
| tmpfs | COG | 1 | 2 | 1489 | 1207 | −19% | 5% / 3% | 82 / 1123 / 0 |
| tmpfs | COG | 12 | 12 | 546 | **404** | −26% | 15% / 6% | 80 / 300 / 24 |
| ext4 | raw | 1 | 1 | 803 | 636 | −21% | 20% / 8% | 28 / 607 / 0 |
| ext4 | raw | 1 | 2 | 732 | 495 | −32% | 36% / 9% | 29 / 467 / 0 |
| ext4 | raw | 12 | 12 | 360 | **292** | −19% | 33% / 20% | 28 / 237 / 27 |
| ext4 | COG | 1 | 1 | 1420 | 1391 | −2% | 13% / 19% | 30 / 1361 / 0 |
| ext4 | COG | 1 | 2 | 1401 | 1164 | −17% | 10% / 12% | 30 / 1131 / 0 |
| ext4 | COG | 12 | 12 | 437 | **374** | −14% | 20% / 37% | 31 / 320 / 26 |
| NTFS | raw | 1 | 1 | 776 | 751 | −3% | 11% / 47% | 42 / 708 / 0 |
| NTFS | raw | 12 | 12 | 545 | **274** | −50% | 399% / 9% | 40 / 196 / 37 |
| NTFS | COG | 1 | 1 | 1353 | 1338 | −1% | 2% / 3% | 39 / 1298 / 0 |
| NTFS | COG | 12 | 12 | 617 | **405** | −34% | 27% / 9% | 40 / 327 / 37 |

Raw data: [`testdata/ab-linux.txt`](testdata/ab-linux.txt),
[`testdata/ab-linux-gomaxprocs2.txt`](testdata/ab-linux-gomaxprocs2.txt)
and [`testdata/ab-windows.txt`](testdata/ab-windows.txt).

Some ext4 cases spread by 20–37%, above the 10–15% this repository
usually accepts. The virtual disk's writeback is part of that, since
sync empties the cache but not the host's. Treat the ext4 rows as
direction and size, not as three significant digits. The tmpfs and NTFS
rows for the new build are within 3–17%. The NTFS old 12-worker row
spread 399%: that path really is erratic, and the probe shows the same.

**benchmarks/chunked**, native Windows, SIMD, default GOMAXPROCS (24),
Slope with no mask in 256-row strips, 5 interleaved runs of each test
binary. This benchmark overwrites an existing output, the cheap case
([`testdata/chunked-windows.txt`](testdata/chunked-windows.txt)):

| size | workers | old ms | new ms | change |
|---|---:|---:|---:|---:|
| 4096² | 1 | 60.4 | 41.9 | −31% |
| 4096² | 12 | 29.7 | 28.5 | −4% |
| 4096² | 24 | 32.3 | 33.4 | +3% |
| 16384² | 1 | 991 | 638 | −36% |
| 16384² | 12 | 348 | 292 | −16% |
| 16384² | 24 | 378 | 357 | −6% |

**gdalsuite**, `OPS="slope stats" TIERS="raw cog"
BASELINE=testdata/timings-3ed48ad.txt ./gdalsuite.sh HGV_leaf.tif 1024 320 11264
11264`, whole-process medians of 5. It ran once on tmpfs and once with
`WORKVOL=` a Docker volume on ext4, which is new in `gdalsuite.sh` for
this ([`testdata/gdalsuite-tmpfs.txt`](testdata/gdalsuite-tmpfs.txt),
[`testdata/gdalsuite-ext4.txt`](testdata/gdalsuite-ext4.txt)). The
baseline, `testdata/timings-3ed48ad.txt`, is 3ed48ad, from before the cog
inflater work, so its "since" column mixes both changes. The A/B above
isolates this one.

| | strata 1 | strata 12 | GDAL, best | stats 12 (writes nothing) |
|---|---:|---:|---:|---:|
| tmpfs, raw → raw | 0.78 s | 0.35 s | 3.81 s | 0.10 s |
| tmpfs, COG → raw | 1.46 s | 0.49 s | 4.14 s | 0.27 s |
| ext4, raw → raw | 0.71 s | 0.39 s | 3.69 s | 0.10 s |
| ext4, COG → raw | 1.36 s | 0.50 s | 4.16 s | 0.26 s |

Whole-process time runs 55–90 ms above in-process time at 12 workers.
Part of that is the exit tearing down a larger heap: write-behind
doubles the output buffers, +140 MB at 12 workers in 256-row strips of
an 11264-wide raster.

## Why GOMAXPROCS=1 gains little

A write blocked in a system call does not hold a P, so the writer
goroutine's `pwrite` runs on another core while the worker computes.
But when the system call returns, the writer needs a P again before it
can issue the next one, and under GOMAXPROCS=1 the worker holds the
only P. The writer waits until the worker blocks, on a read of its own
or on a full writer, or until the scheduler preempts it after up to a
10 ms slice.

A tile is 11 calls of 1 MiB, so from a COG, where the worker computes
without system calls, the writer falls behind. The worker then waits for
it at the next tile and nothing is gained. From a raw file, the worker's
own reads hand the P over, and the overlap is partial (−13% on tmpfs,
−21% on ext4). With a spare P, the overlap is complete.

The engine documents that a chunked call with one worker keeps up to two
cores busy. The overlap cannot be forced from inside the library without
changing how Go schedules.

## Prototypes: reusing the output, and smaller tiles

These are two ways toward the targets, measured against the build above
(`pr`). They ran through the same interleaved loop, 7 rounds, in-process
ms, with variants given as `name:binary:flags`.

- **`reuse`** uses `engine.ReuseRawFile` (stratasuite `-reuse`). It opens
  an existing output without truncating it, then sets its size, so the
  pages and blocks of the last run's output are overwritten instead of
  freed and allocated again.
- **`t128`, `t64`** use tiles of 128 or 64 rows instead of 256. Smaller
  tiles mean smaller buffers to fault in and tear down: 12 workers hold
  about 420 MB at 256 rows, and a quarter of that at 64.

| disk | input | workers | pr | t128 | t64 | reuse | reuse, t64 |
|---|---|---:|---:|---:|---:|---:|---:|
| tmpfs | raw | 1 | 738 | 710 (−4%) | 708 (−4%) | 615 (−17%) | **565 (−24%)** |
| tmpfs | raw | 12 | 288 | 281 (−2%) | 278 (−3%) | 211 (−27%) | **175 (−39%)** |
| tmpfs | COG | 1 | 1459 | 1464 (0%) | 1433 (−2%) | 1328 (−9%) | **1213 (−17%)** |
| tmpfs | COG | 12 | 403 | 501 (+25%) | 625 (+55%) | **347 (−14%)** | 545 (+35%) |
| ext4 | raw | 1 | 717 | 629 (−12%) | 596 (−17%) | 558 (−22%) | **514 (−28%)** |
| ext4 | raw | 12 | 288 | 356 (+24%) | 369 (+28%) | 239 (−17%) | **203 (−30%)** |
| ext4 | COG | 1 | 1400 | 1282 (−8%) | 1281 (−9%) | 1217 (−13%) | **1177 (−16%)** |
| ext4 | COG | 12 | 372 | 503 (+35%) | 647 (+74%) | **345 (−7%)** | 550 (+48%) |
| NTFS | raw | 1 | 779 | 747 (−4%) | 737 (−5%) | 708 (−9%) | **689 (−12%)** |
| NTFS | raw | 12 | 277 | 261 (−6%) | 247 (−11%) | 266 (−4%, ±184%) | **233 (−16%, ±181%)** |
| NTFS | COG | 1 | 1338 | 1328 (−1%) | 1327 (−1%) | 1286 (−4%) | **1266 (−5%)** |
| NTFS | COG | 12 | **381** | 488 (+28%) | 593 (+55%) | 385 (+1%) | 578 (+52%) |

Raw data: [`testdata/proto-reuse-tile-linux.txt`](testdata/proto-reuse-tile-linux.txt)
and [`testdata/proto-reuse-tile-windows.txt`](testdata/proto-reuse-tile-windows.txt).

What the table shows:

- **Reusing the output removes the free-and-reallocate cost.** On
  tmpfs the open phase drops from 82 ms to 0 and the operation from 183
  to 157 ms. On ext4 the open phase drops by about 30 ms, and on NTFS by
  35 ms (40 to 6), so the gain is smallest on NTFS. Written over a longer file of random
  bytes, the output is identical to the `pr` build's, from raw and COG
  input, with 256- and 64-row tiles
  ([`testdata/proto-reuse-exact.txt`](testdata/proto-reuse-exact.txt)).
  The cost is semantic: after a failed or cancelled run, cells never
  reached keep the previous run's values instead of zeros. That is why
  it is a separate function and not CreateRawFile's behaviour.
- **Close costs more when reusing.** On tmpfs it is 52 ms instead of 24
  with 256-row tiles, but 25–26 with 64-row tiles. I have not explained
  this.
- **Smaller tiles help raw input and hurt COG input.**
  - With reuse on 12 workers, 64-row tiles take raw slope from 211 to
    175 ms on tmpfs.
  - From the Deflate COG (512-row blocks), 12 workers slow down by
    25–74% at 128 or 64 rows. 512-row tiles are 16% slower too, so 256
    is the best height here.
  - A bigger block cache does not help: 64-row tiles took 694 ms with
    a 2 GB cache against 623 with the default, and 986 with no cache
    ([`testdata/proto-cog-tile-cache.txt`](testdata/proto-cog-tile-cache.txt)).
    So the cost is not eviction.
  - Probable cause, not proven: 12 workers on 64-row tiles all read
    the same 22 blocks of one block row at once and wait for each
    other's single decode, so decoding loses its parallelism.
  - The best tile height therefore depends on the input. No single
    default is right for both.
- **Against the targets:**
  - 12-worker raw slope reaches 175 ms in-process on tmpfs with reuse
    and 64-row tiles (203 on ext4, 233 on NTFS). Whole-process time
    adds about 50 ms, so ~0.22 s in gdalsuite, against ~0.15.
  - 12-worker COG slope reaches 345–347 ms with reuse and 256-row tiles
    (whole process ~0.40 s), against ~0.3. Its operation phase is 293
    ms, at the decode floor: stats from the same COG, which writes
    nothing, takes 0.27 s in gdalsuite.

## What the numbers do not cover

- **One machine** (Ryzen 9 3900X, 64 GB, NVMe). The ext4 numbers come
  from Docker Desktop's VM, whose virtual disk sits on the NTFS host, not
  a bare-metal Linux disk.
- **Everything was in the page cache.** No run waited for the disk
  itself. On a slow disk, the writeback of dirty pages, mapped or not,
  would set the pace.
- **Mapping only on Linux and Windows.** macOS and the BSDs use the
  handles, since mapping there was not measured, and Sync's coverage of
  mapped pages differs.
- **Only raw files are mapped.** A future COG or other format writer
  gets write-behind but has to find its own parallelism.

## Reproducing

```bash
./benchmarks/rawio/ab.sh <HGV_leaf.tif> 8ad43da
GMP=2 NS=1 ./benchmarks/rawio/ab.sh <HGV_leaf.tif> 8ad43da
./benchmarks/rawio/probe.sh <probe binary> <dir>...
OPS="slope stats" TIERS="raw cog" BASELINE=testdata/timings-3ed48ad.txt ./benchmarks/gdalsuite/gdalsuite.sh <HGV_leaf.tif> 1024 320 11264 11264
WORKVOL=strata-suite OPS="slope stats" TIERS="raw cog" BASELINE=testdata/timings-3ed48ad.txt ./benchmarks/gdalsuite/gdalsuite.sh <HGV_leaf.tif> 1024 320 11264 11264
```

On Windows, run `abrun.sh` with `DIRS=. BINDIR=. EXE=.exe NOSYNC=1` from
a directory holding `dem.raw`, `dem-cog.tif`, `suite-old.exe` and
`suite-new.exe`. The disk-full check is [`diskfull.sh`](diskfull.sh).
Runs were taken 2026-09-23/24 on a quiet desktop: a probe of 10
one-worker in-memory slopes spread 3%, with one outlier at 7%. The
multi-worker runs are not pinned, as benchmarks/README.md prescribes.
