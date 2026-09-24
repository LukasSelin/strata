# Against GDAL, all of it: results

Every strata operation that has a GDAL counterpart, 25 of them, timed
against that counterpart in the same container, on the same machine,
over the same bytes, under the same timer. Each is timed at three
levels, so that the speed of the arithmetic and the speed of the job a
user runs can be told apart:

| tier | what is timed | strata | GDAL |
|---|---|---|---|
| **compute** | the call, with the input already in memory, in process | the plain function (Tiled on 12 workers) | its own algorithm through its Python API, MEM dataset to MEM dataset: `DEMProcessing`, `gdal raster neighbors`, `gdal raster calc`, `Warp`, `ComputeStatistics` |
| **file to file** | the whole process, from a float32 file to a file | the Chunked form, raw file to raw file | the command line: `gdaldem`, `gdal raster neighbors`, `gdal raster calc` and `gdal_calc.py`, `gdalinfo -stats`/`-mm`, `gdalwarp`, on ENVI and on GeoTIFF |
| **whole flow** | the whole process, from a Deflate COG (predictor 3, 512 blocks) to a file | the `cog` module's reader into the Chunked form | the same command lines, reading the COG, writing GeoTIFF or ENVI |

[`../gdal/`](../gdal/RESULTS.md) did this for three terrain operations
and [`../cog/`](../cog/RESULTS.md) for reading. This directory is both,
for everything, with the two tools' answers compared on every operation
(see [Do they agree](#do-they-agree)).

A fused workflow (slope + aspect + hillshade in one pass) against the same three as separate runs and as three gdaldem runs: [WORKFLOW.md](WORKFLOW.md).

## Headline

- **The arithmetic is 25× GDAL's on one core, and 57× on twelve.** That
  is the geometric mean over all 25 operations at the compute tier. It
  ranges from 4.7× (min/max, a scan GDAL also does well) to 170× (the
  11×11 focal mean, which `gdal raster neighbors` computes as a general
  121-tap kernel: 40.8 s against strata's 0.24 s). By family, on one
  core: focal 55×, algebra 49×, resample 22×, terrain 12×, statistics
  5.0×.
- **The whole flow a user runs is 3.7× GDAL's on one core, and 8.0× on
  twelve.** From a Deflate COG to a file, geometric mean. strata is now
  faster than GDAL on every operation in it, on one core too; the
  closest are min/max (1.2×), hillshade (1.5×), and TPI, statistics and
  nearest resampling (1.7×). In the first run of this suite (3ed48ad)
  it was 2.3× and 4.5×, and three operations were slower than GDAL on
  one core.
- **Reading the COG is still most of strata's flow.** strata's
  one-worker whole flow takes 0.97–1.56 s for every single-input
  operation, whatever it computes, from min/max to the 11×11 mean
  (2.2–2.3 s for the two-input algebra, which reads two COGs; 3.7 s for
  the 2× upsample, which writes four times the cells). Compute is 3–28%
  of it (36% for the 2× upsample), and the same operation from a raw
  file takes 0.8–0.95 s less: that difference is decoding the COG. For
  GDAL it is the other way round: compute is 42–98% of its flow. So
  the lead is smallest where GDAL's arithmetic is cheap. **A faster
  kernel would move the whole flow very little; a faster read still
  would move every operation.**
- **From a raw float32 file, where there is nothing to decode, the lead
  is 8.7× on one core and 11.2× on twelve.** strata's file-to-file time
  is flat (0.43–0.68 s for every terrain and focal operation and
  clamp), and GDAL's command-line tools pay 1.0–1.2 s just to copy this
  raster (`gdal_translate`, the floors table).
- **The SIMD kernels are about 3× of the 25×.** A default `go build`
  (no `GOEXPERIMENT=simd`) is still 8.6× GDAL at the compute tier and
  4.1× file to file, and faster than GDAL on all 25 operations at both
  (the closest: roughness 1.06× and hillshade 1.07×). But not evenly:
  terrain drops to 2.0× and statistics to 1.4× without SIMD, while
  algebra (37×) and focal (17×) keep most of their lead, because there
  the gap is GDAL's general-purpose machinery (`muparser` expressions
  per cell, a generic kernel filter), not lanes.
- **Twelve threads help strata more than GDAL.** Of GDAL's tools here
  only `gdalwarp -multi` puts more than one thread on the arithmetic. On
  the COG, `GDAL_NUM_THREADS=12` makes `gdaldem` *slower* (slope 4.4 s →
  8.5 s, and 2.8× the CPU), so the one-thread GDAL configuration is the
  baseline there. strata's twelve-worker whole flow is 0.46–0.48 s for
  every terrain and focal operation.
- **Every speed is a speed at the same answer.** Ruggedness, all six
  focal operations, all five algebra operations, min/max and nearest,
  bilinear, average and 2× cubic resampling are **bit-identical** to
  GDAL on every cell both compute (up to 495M cells). Slope differs by at most
  7.6e-6°, aspect 3.1e-5°, cubic and Lanczos ½ by 6e-5 and 1.5e-4,
  statistics by 2e-13 relative. Hillshade differs by up to 1.5 of 254
  because gdaldem rounds to a byte. The cells where *validity* differs
  are three documented conventions, below. And strata's whole flow from
  the COG on 12 workers wrote exactly the bytes of its raw path on one
  worker, for all 24 raster operations. None of this moved since the
  first run.

## How far it has come

The same window, machine, container and GDAL build have been timed
since 09-21, so this run has a history. strata's own times:

| | 2026-09-21, [`../gdal/`](../gdal/RESULTS.md) (d46d082) | 2026-09-23, [`../cog/`](../cog/RESULTS.md), reader as first proposed (9f658d3) | 2026-09-23, same, after its decoder fixes (b3c31c2) | 2026-09-23, this suite's first run (3ed48ad) | this run (10a5e33) |
|---|---:|---:|---:|---:|---:|
| slope, raw file to file, 1 worker | 0.863 s | – | 0.892 s | 0.883 s | **0.567 s** |
| aspect, raw file to file, 1 worker | 0.922 s | – | – | 0.899 s | **0.673 s** |
| hillshade, raw file to file, 1 worker | 0.789 s | – | – | 0.784 s | **0.504 s** |
| slope, raw file to file, 12 workers | 0.448 s | – | 0.463 s | 0.435 s | **0.336 s** |
| slope kernel alone, in memory | 190 ms | – | – | 190 ms | 189 ms |
| aspect kernel alone, in memory | 246 ms | – | – | 242 ms | 243 ms |
| hillshade kernel alone, in memory | 114 ms | – | – | 156 ms ⚠ | 117 ms |
| **slope, whole flow from the Deflate COG, 1 worker** | – | **4.55 s** | **2.28 s** | **2.25 s** | **1.52 s** |
| **… against gdaldem on the same COG** | – | **0.96×** | **1.9×** | **1.9×** | **2.9×** |
| slope, whole flow from the COG, 12 workers | – | 1.70 s | 0.93 s | 0.91 s | **0.47 s** |

The kernels have not moved since they were first measured. The
hillshade kernel's 156 ms in the first run was the machine: it is back
at 117 ms. What moved is everything around them. Since the first run,
over all 25 operations (geometric mean of now/then, strata one worker):

- **file to file: 0.62**, 38% less time (0.44 for min/max, 0.47 for
  the two-input algebra, 0.93 for the 2× upsample). The source's
  validity became a vector kernel, and a tile with no NoData carries no
  mask (#51, [below](#validity-from-a-fill-value-2026-09-24)); outputs
  are preallocated, mapped files written behind the computation (#53,
  [`../rawio/`](../rawio/RESULTS.md)).
- **whole flow from the COG: 0.64**, 36% less (0.55–0.68; 0.84 for the
  2× upsample). To the above add the decoder: the whole-block inflater
  and its tuning (#44, #49), reused block buffers (#42), and a fused
  SIMD floating-point predictor (#50) ([`../cog/`](../cog/RESULTS.md)).
  On 12 workers slope from the COG takes half the time it did.
- **compute: 0.99.** Unchanged, as it should be. The two rows that look
  slower are noise: min (1.26) is one of the noisier compute cases
  (14%), and TPI (1.09) is marked ⚠.

GDAL's own numbers are stable across all the runs (gdaldem slope on
GeoTIFF, file to file: 3.75, 3.96, 3.98 and 3.83 s), so the change is
strata's.

The committed timings are this run's. The first run's are
[`testdata/timings-3ed48ad.txt`](testdata/timings-3ed48ad.txt), and
[The numbers](#the-numbers) ends with the "since" table against it, per
operation and tier: rerun with `BASELINE=` pointing at committed
timings and `summarize.py` adds it.

## Validity from a fill value (2026-09-24)

A CPU profile of the one-worker raw flow put 26% of slope's time in
deriving the validity mask from the file's NoData value: a branchy test
per cell, then a range copy per 64 cells. That was more than the slope
kernel itself. The change (DESIGN.md §31, "Rules 4 and 5 at the file
boundary") has three parts: a vector kernel that writes mask words
straight from the cells, `vec.ValidBits`; a per-tile rule that drops
the mask of a tile whose cells are all valid; and a pooled mask buffer
in cog. The same window was timed on master (8ad43da) and on the change,
`OPS="slope stats minmax" TIERS="raw cog"`, 5 runs per case, one run
after the other.

| strata, one worker unless noted | committed run (3ed48ad) | master (8ad43da) | this change | against the committed run |
|---|---:|---:|---:|---:|
| slope, raw file to file | 0.883 s | 0.929 s | **0.635 s** | **0.72×** |
| stats, raw file | 0.508 s | 0.520 s | **0.307 s** | **0.60×** |
| minmax, raw file | 0.397 s | 0.520 s ⚠ | **0.179 s** | **0.45×** |
| slope, raw, scalar build | 1.739 s | 1.741 s | 1.646 s | 0.95× |
| stats, raw, scalar build | 1.121 s | 1.122 s | 1.077 s | 0.96× |
| minmax, raw, scalar build | 0.589 s | 0.606 s | 0.566 s | 0.96× |
| slope, raw, 12 workers | 0.435 s | 0.577 s ⚠ | 0.436 s | 1.00× |
| slope, whole flow from the COG | 2.247 s | 1.522 s ⚠ | 1.472 s | (0.97× master) |
| stats, whole flow from the COG | 1.895 s | 1.132 s | 1.180 s | (1.04× master) |
| minmax, whole flow from the COG | 1.756 s | 1.133 s ⚠ | 0.994 s | (0.88× master) |

- **The one-worker raw flow is 28–55% faster**: slope 28%, stats 40%,
  minmax 55%. The raw path did not change between the committed run and
  master, and master's own run was noisy (⚠: spreads of 49–77% on the
  marked cases, with the machine in use), so the committed run is the
  better baseline for the raw rows. In a profile of slope, the source's
  validity fell from 0.19 s a run to 4 ms (natively on Windows, window at (0, 0)). What remains is file IO
  (about 0.37 s) and the kernel.
- **Only the SIMD build gains much.** In the scalar build (a plain `go
  build`) the old per-cell branches predicted well on data that is
  mostly valid. An integer form of the test, eight cells at a time, is
  1.4–1.8× their speed in isolation, and 4–5% end to end.
- **Twelve workers do not move.** Raw slope on 12 workers is bound by
  the file and memory traffic of the tile copies (../chunked/), not by
  the test.
- **From the COG, little changes.** #49 already moved cog's NoData test
  into AVX2 row kernels, so what is left is the mask allocation per
  block, now pooled (about 10 MB a run less garbage), and the all-valid
  tiles. Those tiles pay mostly for the reductions: minmax is 12% faster
  than master. The COG columns against the committed run are #49's
  gain, not this change's.
- **The answers did not move.** slope from the COG on 12 workers was
  byte-identical to slope from the raw file on one, in both runs, and
  the two runs' agree lines against GDAL are equal. Separately, the
  change's slope output was byte-identical to master's for raw and COG
  input, on 1 and 12 workers (SHA-256 of the output file, window at
  (0, 0)).
- **The all-valid tiles matter less here than they could.** In the
  window at (0, 0), 30 of the 44 strips of 256 rows hold NoData, even
  with their one-row halos, and so do 50 of 484 COG blocks. NoData here
  is lakes, scattered through the raster. Measured by switching the
  per-tile check off, the check gives minmax 11% and stats 3% on the
  raw flow, and nothing measurable on slope. The vector test accounts
  for the rest.

Raw output: [`testdata/validity-master.txt`](testdata/validity-master.txt)
and [`testdata/validity-branch.txt`](testdata/validity-branch.txt).

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200; the Docker Desktop VM is given 24 CPUs and 31 GiB |
| Host OS | Windows 11 Home 10.0.22631, Docker Desktop 29.8.0 |
| Container | `ghcr.io/osgeo/gdal:ubuntu-small-latest`, Ubuntu 26.04, Linux 6.18 WSL2 |
| GDAL | 3.14.0dev-17759e56, released 2026/08/18, with libdeflate and numpy 2.3.5 |
| strata | 10a5e33, cross-compiled `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, go1.27.0: once with `GOEXPERIMENT=simd` (every "strata" number) and once without (the `scalar` rows) |
| Raster | `HGV_leaf.tif`, a 12.5 m Swedish canopy-height grid, UInt16, NoData 65535; the window at (1024, 320), 11264 × 11264 = 126.9M cells, promoted to Float32: the window of `../gdal/` and `../cog/`. 97.6% of it carries data. The two-input algebra adds the window at (0, 0) as B |
| COG | `gdal_translate -of COG`, Deflate, predictor 3, 512 × 512 blocks, no overviews, no stored statistics (117 MB) |
| Working files | a 20 GB tmpfs inside the container |
| Runs | per case one untimed warm-up, then **3** timed; the median is reported, never the minimum |
| Duration | 2 h 00 min for the whole suite |
| Raw output | [`testdata/timings.txt`](testdata/timings.txt). One run is missing from it: bash's `time` printed a garbled field (`real=6.:00`, the third one-thread run of GDAL's max from the COG to ENVI), so it was dropped rather than guessed, and that case has 2 runs. `runsuite.sh` now times such a run again |

```bash
./gdalsuite.sh /path/to/HGV_leaf.tif 1024 320 11264 11264   # REPEATS=3 for this run
python summarize.py testdata/timings.txt --baseline testdata/timings-3ed48ad.txt   # the tables below
```

[`gdalsuite.sh`](gdalsuite.sh) builds [`stratasuite`](main.go) twice and
starts the container, [`runsuite.sh`](runsuite.sh) times every case
there, [`gdalcompute.py`](gdalcompute.py) is GDAL's compute tier and
[`agree.py`](agree.py) compares the outputs. The operations and their
GDAL command lines are side by side in [`ops.go`](ops.go) and
`runsuite.sh`'s `gdal_cmd`. `OPS=` and `TIERS=` select part of the
matrix. Like `../gdal/` this directory has no `TestResultsMatch`: the
tables between the markers are exactly what `summarize.py` prints for
the committed timings, and rerunning that one line is the check.

### What is controlled, and what is not

The rules of [`../gdal/`](../gdal/RESULTS.md) apply: one container, one
shell timer, a tmpfs so Docker's NTFS mount is not measured, process
start measured in the floors rather than assumed away. Beyond those:

- **GDAL gets its fastest configuration, per operation and tier.** ENVI
  or GeoTIFF in, GeoTIFF or ENVI out, `gdal raster calc` or
  `gdal_calc.py` (the numpy one, faster here), `GDAL_CACHEMAX=4096`,
  `gdalwarp -wm 2048`, and where GDAL has threads (`gdalwarp -multi`,
  statistics, COG decoding) a 12-thread row. The one-thread speedup is
  against its fastest one-thread configuration; the 12-worker speedup
  against its fastest at any thread count.
- **The compute tier is as close to GDAL's arithmetic as its API
  reaches, and slightly favours strata.** GDAL's calls create their
  output dataset every call, and its API does not let a caller reuse
  one; strata writes into a buffer it was given. The floors table puts a
  number on that: a MEM-to-MEM copy of the raster costs GDAL 145 ms,
  under 5% of its compute time on 17 of the 25 operations and 6–28% on
  the eight cheapest (most for min/max, 518 ms, and nearest ½,
  831 ms). On those eight the compute speedup is overstated by up to
  that much. The algebra
  reads its inputs from uncompressed GeoTIFFs in `/vsimem/`, because
  `gdal raster calc` names inputs only by file.
- **3 timed runs, not 5, and the machine was not idle.** A game
  server's container was running for the whole suite. Cases with a
  spread over 15% are marked ⚠. Most of them are GDAL's own 12-thread
  `gdalwarp -multi` (up to 42%), which varies run to run by itself; the
  rest are strata's 12-worker cases, which take 0.08–0.47 s, so a few
  tens of milliseconds is 15%, and its TPI kernel (18%, 81–99 ms). The
  other one-thread cases are within 15%. The copy floors spread 7–19%.
- **strata reads and writes less format.** Its raw path reads a
  headerless file it is told the shape of; its outputs are raw float32.
  GDAL writes a real GeoTIFF or ENVI with a header and a geotransform.
  That is strata's design (it is not a format library), but it is part
  of the file-to-file gap, and the `gdal_translate` floors size it.
- **Hillshade is measured against strata**: gdaldem writes one byte per
  cell, strata a float32.
- **Throughput counts source cells**, valid or not; for resampling too,
  so the 2× upsample writes four times the cells its M cells/s counts.

## The numbers

<!-- summarize.py output begin -->

# strata: 10a5e33
# go: go1.27.0
# date: 2026-09-24T08:13Z
# gdal: GDAL 3.14.0dev-17759e56cbe7d693a39c3d5ccf339364de844ee7, released 2026/08/18
# nproc: 24
# work: tmpfs            20G   15M   20G   1% /work
# ops: slope aspect hillshade tri tpi roughness mean3 mean11 min3 max3 gauss5 conv5 add mul min max clamp stats minmax near-half bilinear-half cubic-half lanczos-half average-half cubic-double 
# tiers: compute raw cog  repeats: 3  tile: 256  N: 12
# Size is 11264, 11264
# Pixel Size = (12.500000000000000,-12.500000000000000)
#   NoData Value=65535
# dem-cog.tif: 116691772 bytes; stored statistics: 0
# cell: 12.5  nodata: 65535
# dropped, unreadable time: tier=cog op=max tool=gdal cfg=to-envi threads=1 run=3 real=6.:00 user=5.399 sys=1.596 ms=-

11264 × 11264 = 126.9M cells, 3 timed runs per case, median reported. N = 12.

### Scoreboard: how many times faster strata is than GDAL

`1` is one thread each; `12` is strata on 12 workers against GDAL's fastest configuration at any thread count. Above 1× strata is faster.

| op | compute, 1 | compute, 12 | file to file, 1 | file to file, 12 | whole flow from a COG, 1 | whole flow from a COG, 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 16.1× | 58.2× | 6.7× | 11.4× | 2.9× | 9.5× |
| aspect | 14.0× | 64.8× | 6.2× | 12.6× | 3.0× | 9.9× |
| hillshade | 10.0× | 22.2× | 3.3× | 5.1× | 1.5× | 4.4× |
| tri | 6.7× | 29.1× | 3.4× | 6.6× | 1.8× | 5.9× |
| tpi | 12.4× | 23.3× | 4.2× | 5.8× | 1.7× | 5.0× |
| roughness | 13.6× | 27.0× | 4.4× | 6.4× | 1.9× | 5.6× |
| **terrain** (geometric mean) | **11.7×** | **34.0×** | **4.5×** | **7.5×** | **2.1×** | **6.4×** |
| mean3 | 46.2× | 94.4× | 11.8× | 17.5× | 4.5× | 11.9× |
| mean11 | 170× | 497× | 66.1× | 123× | 27.3× | 86.3× |
| min3 | 36.8× | 83.4× | 10.1× | 15.2× | 4.1× | 10.9× |
| max3 | 33.0× | 83.4× | 10.1× | 15.5× | 3.9× | 10.4× |
| gauss5 | 75.0× | 162× | 21.5× | 31.8× | 7.5× | 22.4× |
| conv5 | 38.6× | 166× | 15.8× | 31.8× | 7.3× | 21.8× |
| **focal** (geometric mean) | **55.0×** | **144×** | **17.3×** | **28.3×** | **6.9×** | **19.6×** |
| add | 46.4× | 56.7× | 6.0× | 7.6× | 2.6× | 6.4× |
| mul | 46.0× | 58.5× | 5.8× | 7.1× | 2.5× | 6.3× |
| min | 38.9× | 65.6× | 6.2× | 7.4× | 2.7× | 7.3× |
| max | 53.8× | 69.8× | 5.8× | 7.6× | 2.7× | 7.3× |
| clamp | 64.2× | 89.4× | 5.5× | 7.5× | 3.9× | 9.7× |
| **algebra** (geometric mean) | **49.2×** | **67.1×** | **5.9×** | **7.4×** | **2.8×** | **7.3×** |
| stats | 5.4× | 50.0× | 4.7× | 15.4× | 1.7× | 6.8× |
| minmax | 4.7× | 38.6× | 4.4× | 9.4× | 1.2× | 4.7× |
| **reduce** (geometric mean) | **5.0×** | **44.0×** | **4.5×** | **12.1×** | **1.4×** | **5.6×** |
| near-half | 6.6× | 20.3× | 5.1× | 4.9× | 1.7× | 2.2× |
| bilinear-half | 27.0× | 65.1× | 16.9× | 15.8× | 6.0× | 7.7× |
| cubic-half | 35.8× | 39.8× | 23.6× | 16.6× | 9.6× | 9.4× |
| lanczos-half | 39.0× | 83.1× | 26.7× | 11.1× | 11.9× | 5.4× |
| average-half | 13.3× | 17.3× | 8.2× | 5.1× | 3.0× | 2.6× |
| cubic-double | 34.7× | 31.9× | 18.5× | 7.3× | 13.5× | 7.5× |
| **resample** (geometric mean) | **22.1×** | **36.6×** | **14.2×** | **9.0×** | **6.0×** | **5.1×** |
| **all 25 operations** | **24.6×** | **57.2×** | **8.7×** | **11.2×** | **3.7×** | **8.0×** |

### Where the time goes: compute against the whole flow, one thread

The compute tier times the call with the input already in memory; the whole flow times the process from a Deflate COG to a file. `compute share` is the first over the second: how much of what a user waits for is arithmetic. The last columns are the speedup at each end.

| op | strata compute | strata flow | compute share | GDAL compute | GDAL flow | compute share | × compute | × flow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 188.5 ms | 1.52 s | 12% | 3.04 s | 4.42 s | 69% | 16.1× | 2.9× |
| aspect | 243.2 ms | 1.56 s | 16% | 3.41 s | 4.71 s | 72% | 14.0× | 3.0× |
| hillshade | 117.4 ms | 1.35 s | 9% | 1.17 s | 2.07 s | 57% | 10.0× | 1.5× |
| tri | 227.7 ms | 1.53 s | 15% | 1.54 s | 2.79 s | 55% | 6.7× | 1.8× |
| tpi | 97.9 ms | 1.34 s | 7% | 1.21 s | 2.33 s | 52% | 12.4× | 1.7× |
| roughness | 104.0 ms | 1.35 s | 8% | 1.41 s | 2.60 s | 54% | 13.6× | 1.9× |
| mean3 | 105.5 ms | 1.36 s | 8% | 4.88 s | 6.19 s | 79% | 46.2× | 4.5× |
| mean11 | 240.6 ms | 1.52 s | 16% | 40.84 s | 41.47 s | 98% | 170× | 27.3× |
| min3 | 119.5 ms | 1.37 s | 9% | 4.40 s | 5.56 s | 79% | 36.8× | 4.1× |
| max3 | 129.0 ms | 1.41 s | 9% | 4.26 s | 5.51 s | 77% | 33.0× | 3.9× |
| gauss5 | 131.6 ms | 1.48 s | 9% | 9.87 s | 11.18 s | 88% | 75.0× | 7.5× |
| conv5 | 255.7 ms | 1.54 s | 17% | 9.86 s | 11.18 s | 88% | 38.6× | 7.3× |
| add | 79.8 ms | 2.23 s | 4% | 3.71 s | 5.71 s | 65% | 46.4× | 2.6× |
| mul | 80.9 ms | 2.23 s | 4% | 3.73 s | 5.61 s | 66% | 46.0× | 2.5× |
| min | 109.3 ms | 2.23 s | 5% | 4.25 s | 6.08 s | 70% | 38.9× | 2.7× |
| max | 80.0 ms | 2.31 s | 3% | 4.31 s | 6.15 s | 70% | 53.8× | 2.7× |
| clamp | 65.2 ms | 1.34 s | 5% | 4.18 s | 5.27 s | 79% | 64.2× | 3.9× |
| stats | 225.0 ms | 1.11 s | 20% | 1.21 s | 1.87 s | 65% | 5.4× | 1.7× |
| minmax | 111.3 ms | 974.0 ms | 11% | 518.4 ms | 1.19 s | 44% | 4.7× | 1.2× |
| near-half | 125.1 ms | 1.14 s | 11% | 830.5 ms | 1.99 s | 42% | 6.6× | 1.7× |
| bilinear-half | 238.7 ms | 1.27 s | 19% | 6.44 s | 7.62 s | 85% | 27.0× | 6.0× |
| cubic-half | 326.1 ms | 1.34 s | 24% | 11.67 s | 12.88 s | 91% | 35.8× | 9.6× |
| lanczos-half | 404.8 ms | 1.42 s | 28% | 15.80 s | 16.94 s | 93% | 39.0× | 11.9× |
| average-half | 173.1 ms | 1.18 s | 15% | 2.30 s | 3.52 s | 65% | 13.3× | 3.0× |
| cubic-double | 1.32 s | 3.69 s | 36% | 45.89 s | 49.90 s | 92% | 34.7× | 13.5× |

### compute: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal mem, 1 | gdal mem, 12 | strata simd, 1 | strata simd, 12 | strata scalar, 1 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 3.04 s | – | 188.5 ms | 52.3 ms | 1.13 s | 673 | 6.0× | 2% |
| aspect | 3.41 s | – | 243.2 ms | 52.6 ms | 3.01 s | 522 | 12.4× | 6% |
| hillshade | 1.17 s | – | 117.4 ms | 52.7 ms | 1.01 s | 1081 | 8.6× | 3% |
| tri | 1.54 s | – | 227.7 ms | 52.8 ms | 578.5 ms | 557 | 2.5× | 10% |
| tpi | 1.21 s | – | 97.9 ms | 52.0 ms | 198.0 ms | 1297 | 2.0× | 18% ⚠ |
| roughness | 1.41 s | – | 104.0 ms | 52.3 ms | 1.34 s | 1220 | 12.9× | 8% |
| mean3 | 4.88 s | – | 105.5 ms | 51.7 ms | 314.9 ms | 1202 | 3.0× | 4% |
| mean11 | 40.84 s | – | 240.6 ms | 82.1 ms | 910.4 ms | 527 | 3.8× | 4% |
| min3 | 4.40 s | – | 119.5 ms | 52.8 ms | 317.6 ms | 1061 | 2.7× | 6% |
| max3 | 4.26 s | – | 129.0 ms | 51.1 ms | 456.8 ms | 983 | 3.5× | 2% |
| gauss5 | 9.87 s | – | 131.6 ms | 60.9 ms | 397.9 ms | 964 | 3.0× | 3% |
| conv5 | 9.86 s | – | 255.7 ms | 59.5 ms | 977.3 ms | 496 | 3.8× | 4% |
| add | 3.71 s | – | 79.8 ms | 65.4 ms | 86.9 ms | 1590 | 1.1× | 3% |
| mul | 3.73 s | – | 80.9 ms | 63.7 ms | 87.7 ms | 1568 | 1.1× | 2% |
| min | 4.25 s | – | 109.3 ms | 64.9 ms | 101.1 ms | 1161 | 0.9× | 14% |
| max | 4.31 s | – | 80.0 ms | 61.7 ms | 136.1 ms | 1585 | 1.7× | 7% |
| clamp | 4.18 s | – | 65.2 ms | 46.8 ms | 148.4 ms | 1947 | 2.3× | 3% |
| stats | 1.21 s | 1.16 s | 225.0 ms | 23.1 ms | 887.3 ms | 564 | 3.9× | 10% |
| minmax | 518.4 ms | 513.5 ms | 111.3 ms | 13.3 ms | 356.1 ms | 1140 | 3.2× | 8% |
| near-half | 830.5 ms | 339.7 ms | 125.1 ms | 16.8 ms | 129.8 ms | 1014 | 1.0× | 8% |
| bilinear-half | 6.44 s | 2.17 s | 238.7 ms | 33.4 ms | 554.8 ms | 532 | 2.3× | 11% |
| cubic-half | 11.67 s | 1.56 s | 326.1 ms | 39.2 ms | 817.9 ms | 389 | 2.5× | 5% |
| lanczos-half | 15.80 s | 4.19 s | 404.8 ms | 50.5 ms | 1.03 s | 313 | 2.5× | 28% ⚠ |
| average-half | 2.30 s | 507.7 ms | 173.1 ms | 29.3 ms | 441.4 ms | 733 | 2.6× | 4% |
| cubic-double | 45.89 s | 5.80 s | 1.32 s | 182.2 ms | 3.08 s | 96 | 2.3× | 4% |

### file to file: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal envi, 1 | gdal gdal_calc-envi, 1 | gdal gdal_calc-gtiff, 1 | gdal gtiff, 1 | gdal envi, 12 | gdal gtiff, 12 | strata simd, 1 | strata simd, 12 | strata scalar, 1 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 4.36 s | – | – | 3.83 s | – | – | 567.0 ms | 336.0 ms | 1.67 s | 224 | 3.0× | 4% |
| aspect | 4.51 s | – | – | 4.18 s | – | – | 673.0 ms | 331.0 ms | 3.56 s | 189 | 5.3× | 7% |
| hillshade | 1.79 s | – | – | 1.65 s | – | – | 504.0 ms | 327.0 ms | 1.54 s | 252 | 3.1× | 7% |
| tri | 2.78 s | – | – | 2.25 s | – | – | 662.0 ms | 339.0 ms | 1.11 s | 192 | 1.7× | 6% |
| tpi | 2.23 s | – | – | 1.95 s | – | – | 463.0 ms | 334.0 ms | 720.0 ms | 274 | 1.6× | 9% |
| roughness | 2.46 s | – | – | 2.14 s | – | – | 485.0 ms | 333.0 ms | 1.86 s | 262 | 3.8× | 5% |
| mean3 | 6.10 s | – | – | 5.69 s | – | – | 483.0 ms | 326.0 ms | 831.0 ms | 263 | 1.7× | 5% |
| mean11 | 42.34 s | – | – | 40.86 s | – | – | 618.0 ms | 333.0 ms | 1.43 s | 205 | 2.3× | 5% |
| min3 | 5.44 s | – | – | 5.08 s | – | – | 504.0 ms | 334.0 ms | 857.0 ms | 252 | 1.7× | 5% |
| max3 | 5.38 s | – | – | 5.01 s | – | – | 498.0 ms | 323.0 ms | 1.00 s | 255 | 2.0× | 6% |
| gauss5 | 11.08 s | – | – | 10.68 s | – | – | 497.0 ms | 336.0 ms | 952.0 ms | 255 | 1.9× | 6% |
| conv5 | 11.11 s | – | – | 10.76 s | – | – | 682.0 ms | 339.0 ms | 1.52 s | 186 | 2.2× | 6% |
| add | 5.11 s | 3.71 s | 3.02 s | 4.68 s | – | – | 506.0 ms | 400.0 ms | 710.0 ms | 251 | 1.4× | 5% |
| mul | 5.15 s | 3.26 s | 2.88 s | 4.70 s | – | – | 501.0 ms | 403.0 ms | 703.0 ms | 253 | 1.4× | 6% |
| min | 5.65 s | 3.15 s | 3.05 s | 5.29 s | – | – | 492.0 ms | 412.0 ms | 753.0 ms | 258 | 1.5× | 10% |
| max | 5.74 s | 3.30 s | 3.03 s | 5.32 s | – | – | 518.0 ms | 397.0 ms | 827.0 ms | 245 | 1.6× | 10% |
| clamp | 5.21 s | 2.59 s | 2.39 s | 4.81 s | – | – | 431.0 ms | 319.0 ms | 624.0 ms | 294 | 1.4× | 5% |
| stats | 1.49 s | – | – | 1.51 s | 1.47 s | 1.48 s | 320.0 ms | 95.0 ms | 1.10 s | 396 | 3.5× | 21% ⚠ |
| minmax | 779.0 ms | – | – | 754.0 ms | 781.0 ms | 835.0 ms | 173.0 ms | 80.0 ms | 572.0 ms | 733 | 3.3× | 21% ⚠ |
| near-half | 1.63 s | – | – | 1.63 s | 1.18 s | 862.0 ms | 321.0 ms | 177.0 ms | 437.0 ms | 395 | 1.4× | 9% |
| bilinear-half | 7.23 s | – | – | 7.44 s | 5.44 s | 3.37 s | 427.0 ms | 213.0 ms | 881.0 ms | 297 | 2.1× | 19% ⚠ |
| cubic-half | 12.56 s | – | – | 12.52 s | 4.95 s | 3.62 s | 530.0 ms | 218.0 ms | 1.15 s | 239 | 2.2× | 11% |
| lanczos-half | 16.54 s | – | – | 16.56 s | 4.04 s | 2.69 s | 620.0 ms | 243.0 ms | 1.35 s | 205 | 2.2× | 22% ⚠ |
| average-half | 3.09 s | – | – | 3.08 s | 1.37 s | 1.03 s | 374.0 ms | 204.0 ms | 774.0 ms | 339 | 2.1× | 3% |
| cubic-double | 51.26 s | – | – | 49.30 s | 10.00 s | 9.40 s | 2.67 s | 1.29 s | 4.68 s | 48 | 1.8× | 10% |

### whole flow from a COG: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal to-envi, 1 | gdal to-gtiff, 1 | gdal to-envi, 12 | gdal to-gtiff, 12 | strata simd, 1 | strata simd, 12 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 5.37 s | 4.42 s | 9.14 s | 8.45 s | 1.52 s | 467.0 ms | 84 | – | 16% ⚠ |
| aspect | 5.49 s | 4.71 s | 9.46 s | 8.82 s | 1.56 s | 473.0 ms | 81 | – | 4% |
| hillshade | 2.23 s | 2.07 s | 6.39 s | 6.18 s | 1.35 s | 466.0 ms | 94 | – | 3% |
| tri | 3.38 s | 2.79 s | 7.57 s | 6.92 s | 1.53 s | 474.0 ms | 83 | – | 6% |
| tpi | 2.95 s | 2.33 s | 7.10 s | 6.65 s | 1.34 s | 463.0 ms | 95 | – | 5% |
| roughness | 3.23 s | 2.60 s | 7.44 s | 6.80 s | 1.35 s | 464.0 ms | 94 | – | 3% |
| mean3 | 6.85 s | 6.19 s | 6.16 s | 5.55 s | 1.36 s | 466.0 ms | 93 | – | 3% |
| mean11 | 42.38 s | 41.47 s | 41.35 s | 40.72 s | 1.52 s | 472.0 ms | 84 | – | 3% |
| min3 | 6.43 s | 5.56 s | 5.55 s | 4.99 s | 1.37 s | 458.0 ms | 93 | – | 4% |
| max3 | 6.44 s | 5.51 s | 5.39 s | 4.84 s | 1.41 s | 468.0 ms | 90 | – | 11% |
| gauss5 | 12.12 s | 11.18 s | 11.22 s | 10.62 s | 1.48 s | 474.0 ms | 86 | – | 5% |
| conv5 | 12.03 s | 11.18 s | 11.17 s | 10.49 s | 1.54 s | 480.0 ms | 82 | – | 9% |
| add | 6.60 s | 5.71 s | 4.76 s | 4.37 s | 2.23 s | 681.0 ms | 57 | – | 6% |
| mul | 6.49 s | 5.61 s | 4.72 s | 4.30 s | 2.23 s | 678.0 ms | 57 | – | 5% |
| min | 6.96 s | 6.08 s | 5.38 s | 4.91 s | 2.23 s | 677.0 ms | 57 | – | 5% |
| max | 6.97 s | 6.15 s | 5.48 s | 4.97 s | 2.31 s | 685.0 ms | 55 | – | 4% |
| clamp | 6.10 s | 5.27 s | 5.00 s | 4.61 s | 1.34 s | 476.0 ms | 94 | – | 5% |
| stats | 1.88 s | 1.87 s | 1.94 s | 1.84 s | 1.11 s | 271.0 ms | 114 | – | 6% |
| minmax | 1.21 s | 1.19 s | 1.20 s | 1.21 s | 974.0 ms | 253.0 ms | 130 | – | 17% ⚠ |
| near-half | 2.09 s | 1.99 s | 677.0 ms | 641.0 ms | 1.14 s | 292.0 ms | 111 | – | 4% |
| bilinear-half | 7.64 s | 7.62 s | 3.19 s | 3.65 s | 1.27 s | 412.0 ms | 100 | – | 27% ⚠ |
| cubic-half | 12.89 s | 12.88 s | 4.06 s | 4.03 s | 1.34 s | 430.0 ms | 95 | – | 42% ⚠ |
| lanczos-half | 16.94 s | 16.99 s | 3.07 s | 2.48 s | 1.42 s | 456.0 ms | 89 | – | 30% ⚠ |
| average-half | 3.52 s | 3.53 s | 873.0 ms | 806.0 ms | 1.18 s | 313.0 ms | 107 | – | 8% |
| cubic-double | 52.33 s | 49.90 s | 10.39 s | 9.10 s | 3.69 s | 1.21 s | 34 | – | 16% ⚠ |

### CPU spent: the whole flow from a COG, CPU-seconds (user + sys)

What each tool burns for the same result. A tool that returns sooner on more threads can still be spending more.

| op | GDAL, 1 | strata, 1 | GDAL, 12 | strata, 12 |
| --- | ---: | ---: | ---: | ---: |
| slope | 4.40 | 1.54 | 12.18 | 2.54 |
| aspect | 4.70 | 1.57 | 12.55 | 2.60 |
| hillshade | 2.07 | 1.35 | 9.86 | 2.53 |
| tri | 2.78 | 1.54 | 10.63 | 2.59 |
| tpi | 2.33 | 1.35 | 10.42 | 2.49 |
| roughness | 2.59 | 1.35 | 10.55 | 2.55 |
| mean3 | 6.19 | 1.38 | 6.71 | 2.58 |
| mean11 | 41.45 | 1.53 | 41.92 | 2.62 |
| min3 | 5.55 | 1.38 | 6.18 | 2.51 |
| max3 | 5.50 | 1.43 | 5.99 | 2.66 |
| gauss5 | 11.17 | 1.50 | 11.82 | 2.60 |
| conv5 | 11.17 | 1.55 | 11.66 | 2.61 |
| add | 5.71 | 2.25 | 6.50 | 3.94 |
| mul | 5.60 | 2.25 | 6.42 | 3.87 |
| min | 6.08 | 2.25 | 7.06 | 3.82 |
| max | 6.15 | 2.33 | 7.10 | 3.89 |
| clamp | 5.27 | 1.36 | 5.68 | 2.65 |
| stats | 1.86 | 1.11 | 1.83 | 1.49 |
| minmax | 1.18 | 0.98 | 1.20 | 1.36 |
| near-half | 1.99 | 1.15 | 2.38 | 1.99 |
| bilinear-half | 7.60 | 1.28 | 24.28 | 2.82 |
| cubic-half | 12.88 | 1.34 | 28.60 | 3.04 |
| lanczos-half | 16.93 | 1.42 | 22.22 | 3.16 |
| average-half | 3.52 | 1.19 | 4.29 | 2.19 |
| cubic-double | 49.87 | 3.79 | 71.89 | 6.98 |

### Floors: each tool's cost before it computes anything

| case | median | spread |
| --- | ---: | ---: |
| gdal startup, 1 thread | 24.0 ms | 4% |
| strata startup, 1 thread | 4.0 ms | 0% |
| gdal copy-envi, 1 thread | 1.17 s | 7% |
| gdal copy-gtiff, 1 thread | 1.01 s | 19% |
| gdal copy-cog, 1 thread | 1.28 s | 9% |
| gdal copy-cog, 12 threads | 611.0 ms | 19% |
| gdal mem-copy, 1 thread | 145.4 ms | 11% |

### Do they agree?

strata's output against GDAL's, from the same file, excluding each tool's differently defined border. `identical` is whether strata's whole flow from the COG on N workers wrote exactly the bytes of its raw path on one.

| op | cells compared | validity differs | max abs diff | mean abs diff | value range | COG == raw |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| slope | 123,785,590 (97.6%) | 0 | 7.63e-06 | 1.26e-06 | 87.4541 | yes |
| aspect | 99,492,495 (78.4%) | 24,293,095 | 3.05e-05 | 4.62e-06 | 359.971 | yes |
| hillshade | 123,785,590 (97.6%) | 0 | 1.5 | 0.614 | 254 | yes |
| tri | 123,785,590 (97.6%) | 0 | 0 | 0 | 1419.87 | yes |
| tpi | 123,785,590 (97.6%) | 0 | 0 | 0 | 1001 | yes |
| roughness | 123,785,590 (97.6%) | 0 | 0 | 0 | 515 | yes |
| mean3 | 123,785,590 (97.6%) | 10,078 | 0 | 0 | 502.778 | yes |
| mean11 | 123,584,126 (97.6%) | 50,370 | 0 | 0 | 491.62 | yes |
| min3 | 123,785,590 (97.6%) | 10,078 | 0 | 0 | 502 | yes |
| max3 | 123,785,590 (97.6%) | 10,078 | 0 | 0 | 515 | yes |
| gauss5 | 123,735,212 (97.6%) | 20,154 | 0 | 0 | 502.32 | yes |
| conv5 | 123,735,212 (97.6%) | 20,154 | 0 | 0 | 2941.08 | yes |
| add | 119,014,576 (93.8%) | 0 | 0 | 0 | 960 | yes |
| mul | 119,014,576 (93.8%) | 0 | 0 | 0 | 230400 | yes |
| min | 119,014,576 (93.8%) | 0 | 0 | 0 | 480 | yes |
| max | 119,014,576 (93.8%) | 0 | 0 | 0 | 515 | yes |
| clamp | 123,835,976 (97.6%) | 0 | 0 | 0 | 150 | yes |
| stats | all | – | relative 2e-13 | – | – | – |
| minmax | all | – | relative 0 | – | – | – |
| near-half | 30,798,464 (97.7%) | 0 | 0 | 0 | 515 | yes |
| bilinear-half | 30,798,464 (97.7%) | 0 | 0 | 0 | 502.062 | yes |
| cubic-half | 30,798,464 (97.7%) | 0 | 6.1e-05 | 2.77e-07 | 608.914 | yes |
| lanczos-half | 30,798,422 (97.7%) | 42 | 0.000153 | 6.15e-06 | 693.426 | yes |
| average-half | 30,800,410 (97.7%) | 0 | 0 | 0 | 506.75 | yes |
| cubic-double | 494,699,120 (97.6%) | 0 | 0 | 0 | 696.216 | yes |

### Since the baseline (strata: 3ed48ad)

`now/then` below 1 is faster now. The speedup columns are this run's and the baseline's, each against the GDAL of its own run, so a change in either tool or the machine shows up there.

| op | tier | strata 1, then | strata 1, now | now/then | ×GDAL then | ×GDAL now |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| slope | compute | 190.0 ms | 188.5 ms | 0.99 | 16.1× | 16.1× |
| slope | file to file | 883.0 ms | 567.0 ms | 0.64 | 4.5× | 6.7× |
| slope | whole flow from a COG | 2.25 s | 1.52 s | 0.67 | 1.9× | 2.9× |
| aspect | compute | 241.8 ms | 243.2 ms | 1.01 | 14.4× | 14.0× |
| aspect | file to file | 899.0 ms | 673.0 ms | 0.75 | 4.6× | 6.2× |
| aspect | whole flow from a COG | 2.31 s | 1.56 s | 0.68 | 2.0× | 3.0× |
| hillshade | compute | 155.7 ms | 117.4 ms | 0.75 | 7.5× | 10.0× |
| hillshade | file to file | 784.0 ms | 504.0 ms | 0.64 | 2.1× | 3.3× |
| hillshade | whole flow from a COG | 2.16 s | 1.35 s | 0.62 | 0.9× | 1.5× |
| tri | compute | 230.8 ms | 227.7 ms | 0.99 | 6.9× | 6.7× |
| tri | file to file | 899.0 ms | 662.0 ms | 0.74 | 2.5× | 3.4× |
| tri | whole flow from a COG | 2.26 s | 1.53 s | 0.68 | 1.2× | 1.8× |
| tpi | compute | 89.9 ms | 97.9 ms | 1.09 | 13.4× | 12.4× |
| tpi | file to file | 770.0 ms | 463.0 ms | 0.60 | 2.5× | 4.2× |
| tpi | whole flow from a COG | 2.13 s | 1.34 s | 0.63 | 1.1× | 1.7× |
| roughness | compute | 104.8 ms | 104.0 ms | 0.99 | 13.7× | 13.6× |
| roughness | file to file | 783.0 ms | 485.0 ms | 0.62 | 2.7× | 4.4× |
| roughness | whole flow from a COG | 2.18 s | 1.35 s | 0.62 | 1.1× | 1.9× |
| mean3 | compute | 109.2 ms | 105.5 ms | 0.97 | 43.8× | 46.2× |
| mean3 | file to file | 779.0 ms | 483.0 ms | 0.62 | 7.0× | 11.8× |
| mean3 | whole flow from a COG | 2.17 s | 1.36 s | 0.63 | 2.7× | 4.5× |
| mean11 | compute | 221.0 ms | 240.6 ms | 1.09 | 174× | 170× |
| mean11 | file to file | 892.0 ms | 618.0 ms | 0.69 | 44.2× | 66.1× |
| mean11 | whole flow from a COG | 2.25 s | 1.52 s | 0.68 | 17.8× | 27.3× |
| min3 | compute | 128.2 ms | 119.5 ms | 0.93 | 34.3× | 36.8× |
| min3 | file to file | 776.0 ms | 504.0 ms | 0.65 | 6.3× | 10.1× |
| min3 | whole flow from a COG | 2.15 s | 1.37 s | 0.64 | 2.5× | 4.1× |
| max3 | compute | 133.6 ms | 129.0 ms | 0.97 | 31.8× | 33.0× |
| max3 | file to file | 811.0 ms | 498.0 ms | 0.61 | 6.0× | 10.1× |
| max3 | whole flow from a COG | 2.20 s | 1.41 s | 0.64 | 2.4× | 3.9× |
| gauss5 | compute | 140.3 ms | 131.6 ms | 0.94 | 67.8× | 75.0× |
| gauss5 | file to file | 799.0 ms | 497.0 ms | 0.62 | 12.9× | 21.5× |
| gauss5 | whole flow from a COG | 2.20 s | 1.48 s | 0.67 | 4.9× | 7.5× |
| conv5 | compute | 241.1 ms | 255.7 ms | 1.06 | 39.7× | 38.6× |
| conv5 | file to file | 894.0 ms | 682.0 ms | 0.76 | 11.4× | 15.8× |
| conv5 | whole flow from a COG | 2.30 s | 1.54 s | 0.67 | 4.7× | 7.3× |
| add | compute | 87.0 ms | 79.8 ms | 0.92 | 40.5× | 46.4× |
| add | file to file | 1.07 s | 506.0 ms | 0.47 | 2.8× | 6.0× |
| add | whole flow from a COG | 3.79 s | 2.23 s | 0.59 | 1.4× | 2.6× |
| mul | compute | 87.1 ms | 80.9 ms | 0.93 | 41.7× | 46.0× |
| mul | file to file | 1.07 s | 501.0 ms | 0.47 | 2.8× | 5.8× |
| mul | whole flow from a COG | 3.76 s | 2.23 s | 0.59 | 1.4× | 2.5× |
| min | compute | 86.5 ms | 109.3 ms | 1.26 | 47.0× | 38.9× |
| min | file to file | 1.04 s | 492.0 ms | 0.47 | 2.9× | 6.2× |
| min | whole flow from a COG | 3.72 s | 2.23 s | 0.60 | 1.6× | 2.7× |
| max | compute | 87.4 ms | 80.0 ms | 0.92 | 48.4× | 53.8× |
| max | file to file | 1.06 s | 518.0 ms | 0.49 | 2.8× | 5.8× |
| max | whole flow from a COG | 3.75 s | 2.31 s | 0.62 | 1.6× | 2.7× |
| clamp | compute | 70.5 ms | 65.2 ms | 0.92 | 58.0× | 64.2× |
| clamp | file to file | 720.0 ms | 431.0 ms | 0.60 | 3.1× | 5.5× |
| clamp | whole flow from a COG | 2.10 s | 1.34 s | 0.64 | 2.4× | 3.9× |
| stats | compute | 227.8 ms | 225.0 ms | 0.99 | 5.2× | 5.4× |
| stats | file to file | 508.0 ms | 320.0 ms | 0.63 | 2.7× | 4.7× |
| stats | whole flow from a COG | 1.90 s | 1.11 s | 0.59 | 0.9× | 1.7× |
| minmax | compute | 117.4 ms | 111.3 ms | 0.95 | 4.3× | 4.7× |
| minmax | file to file | 397.0 ms | 173.0 ms | 0.44 | 1.9× | 4.4× |
| minmax | whole flow from a COG | 1.76 s | 974.0 ms | 0.55 | 0.7× | 1.2× |
| near-half | compute | 123.5 ms | 125.1 ms | 1.01 | 6.6× | 6.6× |
| near-half | file to file | 526.0 ms | 321.0 ms | 0.61 | 3.1× | 5.1× |
| near-half | whole flow from a COG | 1.90 s | 1.14 s | 0.60 | 1.1× | 1.7× |
| bilinear-half | compute | 227.6 ms | 238.7 ms | 1.05 | 27.4× | 27.0× |
| bilinear-half | file to file | 652.0 ms | 427.0 ms | 0.65 | 10.8× | 16.9× |
| bilinear-half | whole flow from a COG | 2.02 s | 1.27 s | 0.63 | 3.7× | 6.0× |
| cubic-half | compute | 312.2 ms | 326.1 ms | 1.04 | 37.2× | 35.8× |
| cubic-half | file to file | 722.0 ms | 530.0 ms | 0.73 | 16.8× | 23.6× |
| cubic-half | whole flow from a COG | 2.10 s | 1.34 s | 0.64 | 5.9× | 9.6× |
| lanczos-half | compute | 386.2 ms | 404.8 ms | 1.05 | 39.1× | 39.0× |
| lanczos-half | file to file | 816.0 ms | 620.0 ms | 0.76 | 19.4× | 26.7× |
| lanczos-half | whole flow from a COG | 2.18 s | 1.42 s | 0.65 | 7.4× | 11.9× |
| average-half | compute | 178.5 ms | 173.1 ms | 0.97 | 12.7× | 13.3× |
| average-half | file to file | 579.0 ms | 374.0 ms | 0.65 | 5.1× | 8.2× |
| average-half | whole flow from a COG | 1.94 s | 1.18 s | 0.61 | 1.7× | 3.0× |
| cubic-double | compute | 1.25 s | 1.32 s | 1.06 | 35.3× | 34.7× |
| cubic-double | file to file | 2.89 s | 2.67 s | 0.93 | 16.5× | 18.5× |
| cubic-double | whole flow from a COG | 4.38 s | 3.69 s | 0.84 | 10.9× | 13.5× |

<!-- summarize.py output end -->

## Do they agree

The table above is `agree.py` on every operation: strata's one-worker
output from the raw file against GDAL's ENVI output from the same file,
leaving out the border where the two define edges differently (1 cell
for terrain, r for a radius-r focal operation, 8 for resampling). It
measures and does not judge; [`acceptance/`](../../acceptance/) is
where strata is judged against definitions with derived tolerances.

Three kinds of cell differ in *validity*. The first two were checked
cell for cell on this raster's full-size outputs:

- **Aspect, 24,293,095 cells (19%).** Every one is a flat cell, where
  strata writes −1 (Esri's flat aspect, `terrain.AspectFlat`) and
  gdaldem writes NoData. This canopy-height grid has large flat areas
  at zero. `AspectOptions.ZeroForFlat` is gdaldem's `-zero_for_flat`.
- **The focal operations, 10,078 cells for 3×3 (20,154 for 5×5, 50,370
  for 11×11), under 0.05%.** Every one is a cell whose window touches
  NoData, never one that is NoData itself: `gdal raster neighbors`
  computes from the valid part of the window, strata requires all of it
  (its package documentation, and `acceptance/` check 5).
- **Lanczos ½, 42 of 30.8M cells.** gdalwarp dropped Lanczos's
  half-valid rule in GDAL 3.13.1 (OSGeo/gdal c9507793, OSGeo/gdal#14560): it keeps
  any cell whose centre is valid and whose valid weight is at least
  1e-6. strata keeps the rule, so it invalidates a few cells near NoData
  that gdalwarp now computes. DESIGN.md §54 records the departure, and
  `acceptance/gdalwarp_resample.py` counts these cells on GDAL 3.13.1
  and later. The first run left this unexplained; it is not chunking,
  since `-wm 2048` warps this raster in one chunk.

Everything else is values, on cells both tools computed: bit-identical
for 18 of the 25 operations, and within 1.5e-4 for the others except
hillshade, whose gdaldem output is rounded to a byte and floored at 1.
Every agree line is the same as in the first run. Between the two,
slope, aspect and hillshade were refactored to compute through one
shared gradient (#54, DESIGN.md §52); against gdaldem they did not
move.

The bytes strata wrote in the whole flow, from the COG on 12 workers,
were compared with its raw path on one worker, `cmp` for `cmp`: equal
for all 24 operations that write a raster.

## What this does not tell you

- **A quiet-machine verdict on the noisy rows.** Three timed runs, on a
  desktop that was not idle. The ⚠ rows are mostly GDAL's
  `gdalwarp -multi` and strata's sub-0.5 s 12-worker cases; they move
  12-thread speedups, and of the one-thread cases only TPI's kernel.
- **The cost of GDAL's formats for strata.** strata writes raw float32.
  A user who needs a GeoTIFF out still has to write one, which strata
  cannot yet (DESIGN.md §34: writing is open), so the whole flow here
  ends at a raw file for strata and at a real GeoTIFF or ENVI for GDAL.
- **Other compressions, a real disk, or a remote COG.** Only Deflate
  with predictor 3, on a tmpfs. [`../cog/`](../cog/RESULTS.md) has ZSTD,
  LZW and uncompressed for reading and slope, and ZSTD narrows the
  decode gap.
- **Anything GDAL does that strata does not**: reprojection, curvature
  in GDAL's sense, median and mode focal statistics, `-compute_edges`,
  multidirectional hillshade, every other format. Nothing here is a
  claim about GDAL as a whole, and 25 operations on one raster on one
  machine are not a claim about every raster or machine.
- **An old strata against this suite.** The history table is from the
  earlier runs' own harnesses. `stratasuite` uses APIs that did not
  exist at d46d082 or 9f658d3, so it cannot be built there.
