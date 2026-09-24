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

## Headline

- **The arithmetic is 24× GDAL's on one core, and 57× on twelve.** That
  is the geometric mean over all 25 operations at the compute tier. It
  ranges from 4.3× (min/max, a scan GDAL also does well) to 174× (the
  11×11 focal mean, which `gdal raster neighbors` computes as a general
  121-tap kernel: 38.5 s against strata's 0.22 s). By family, on one
  core: focal 53×, algebra 47×, resample 22×, terrain 11×, statistics
  4.7×.
- **The whole flow a user runs is 2.3× GDAL's on one core, and 4.5× on
  twelve.** From a Deflate COG to a file, geometric mean. The lead is a
  tenth of the compute tier's, and on three operations strata is
  *slower* than GDAL on one core: hillshade (0.9×), statistics (0.9×)
  and min/max (0.7×). TPI, roughness and nearest resampling are ties
  (1.1×).
- **The reason is one number: reading the COG.** strata's one-worker
  whole flow takes 1.8–2.3 s for every single-input operation, whatever
  it computes, from min/max to the 11×11 mean (3.7 s for the two-input
  algebra, which reads two COGs; 4.4 s for the 2× upsample, which
  writes four times the cells). Compute is 2–18% of it (29% for the 2×
  upsample). The rest is decoding Deflate and writing
  the output. For GDAL it is the other way round: compute is 41–96% of
  its flow. So strata wins the whole flow wherever GDAL's arithmetic is
  slow, and loses where GDAL's arithmetic is cheap and its faster
  decoder decides ([`../cog/`](../cog/RESULTS.md): GDAL inflates this
  file in 0.81 s, strata in 1.70 s). **A faster kernel would now move
  the whole flow very little; a faster Deflate decoder would move every
  operation.**
- **From a raw float32 file, where there is nothing to decode, the lead
  is 5.3× on one core and 9.0× on twelve.** strata's file-to-file time
  is flat too (0.72–0.90 s for every terrain and focal operation and
  clamp), and GDAL's
  command-line tools pay about 1.1 s just to copy this raster
  (`gdal_translate`, the floors table).
- **The SIMD kernels are about 3× of the 24×.** A default `go build`
  (no `GOEXPERIMENT=simd`) is still 8.4× GDAL at the compute tier and
  3.7× file to file, and faster than GDAL on all 25 operations at both.
  But not evenly: terrain drops to 2.0× and statistics to 1.4× without
  SIMD, while algebra (34×) and focal (15×) keep most of their lead,
  because there the gap is GDAL's general-purpose machinery
  (`muparser` expressions per cell, a generic kernel filter), not lanes.
- **Twelve threads help strata more than GDAL.** Of GDAL's tools here
  only `gdalwarp -multi` puts more than one thread on the arithmetic. On
  the COG, `GDAL_NUM_THREADS=12` makes `gdaldem` *slower* (slope 4.4 s →
  8.1 s, and 2.7× the CPU), so the one-thread GDAL configuration is the
  baseline there. strata's twelve-worker whole flow is 0.87–0.91 s for
  every terrain and focal operation.
- **Every speed is a speed at the same answer.** Ruggedness, all six
  focal operations, all five algebra operations, min/max and nearest,
  bilinear, average and 2× cubic resampling are **bit-identical** to
  GDAL on every cell both compute (up to 495M cells). Slope differs by at most
  7.6e-6°, aspect 3.1e-5°, cubic and Lanczos ½ by 6e-5 and 1.5e-4,
  statistics by 2e-13 relative. Hillshade differs by up to 1.5 of 254
  because gdaldem rounds to a byte. The cells where *validity* differs
  are two documented conventions, checked cell for cell below. And
  strata's whole flow from the COG on 12 workers wrote exactly the
  bytes of its raw path on one worker, for all 24 raster operations.

## How far it has come

The same window, machine, container and GDAL build were timed twice
before, so some of this run has a history. strata's own times:

| | 2026-09-21, [`../gdal/`](../gdal/RESULTS.md) (d46d082) | 2026-09-23, [`../cog/`](../cog/RESULTS.md), reader as first proposed (9f658d3) | 2026-09-23, same, after its decoder fixes (b3c31c2) | this run (3ed48ad) |
|---|---:|---:|---:|---:|
| slope, raw file to file, 1 worker | 0.863 s | – | 0.892 s | 0.883 s |
| aspect, raw file to file, 1 worker | 0.922 s | – | – | 0.899 s |
| hillshade, raw file to file, 1 worker | 0.789 s | – | – | 0.784 s |
| slope, raw file to file, 12 workers | 0.448 s | – | 0.463 s | 0.435 s |
| slope kernel alone, in memory | 190 ms | – | – | 190 ms |
| aspect kernel alone, in memory | 246 ms | – | – | 242 ms |
| hillshade kernel alone, in memory | 114 ms | – | – | 156 ms ⚠ |
| **slope, whole flow from the Deflate COG, 1 worker** | – | **4.55 s** | **2.28 s** | **2.25 s** |
| **… against gdaldem on the same COG** | – | **0.96×** | **1.9×** | **1.9×** |
| slope, whole flow from the COG, 12 workers | – | 1.70 s | 0.93 s | 0.91 s |

Read along a row and the kernels and the raw path have not moved since
they were first measured, with one exception: the hillshade kernel,
114 ms on 09-21 and 156 ms here. That case is the noisiest of this run
(131–166 ms over its three runs), so it may be the machine; it may
also be a regression, and it is worth a quiet rerun of
`benchmarks/terrain` before trusting either. What moved is the whole
flow. When the COG reader landed,
slope straight from a Deflate COG was a tie with gdaldem on one core.
The decoder fixes in #34 made it 2.0× faster, and it has held since. GDAL's own numbers are
stable across the three runs (gdaldem slope on GeoTIFF: 3.75, 3.96,
3.98 s), so the change is strata's.

This is the first run that covers all 25 operations, so it is the
baseline for the others. Rerun with `BASELINE=` pointing at the
committed timings and `summarize.py` adds a "since" table, per
operation and tier.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200; the Docker Desktop VM is given 24 CPUs and 31 GiB |
| Host OS | Windows 11 Home 10.0.22631, Docker Desktop 29.8.0 |
| Container | `ghcr.io/osgeo/gdal:ubuntu-small-latest`, Ubuntu 26.04, Linux 6.18 WSL2 |
| GDAL | 3.14.0dev-17759e56, released 2026/08/18, with libdeflate and numpy 2.3.5 |
| strata | 3ed48ad, cross-compiled `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, go1.27.0: once with `GOEXPERIMENT=simd` (every "strata" number) and once without (the `scalar` rows) |
| Raster | `HGV_leaf.tif`, a 12.5 m Swedish canopy-height grid, UInt16, NoData 65535; the window at (1024, 320), 11264 × 11264 = 126.9M cells, promoted to Float32: the window of `../gdal/` and `../cog/`. 97.6% of it carries data. The two-input algebra adds the window at (0, 0) as B |
| COG | `gdal_translate -of COG`, Deflate, predictor 3, 512 × 512 blocks, no overviews, no stored statistics (117 MB) |
| Working files | a 20 GB tmpfs inside the container |
| Runs | per case one untimed warm-up, then **3** timed; the median is reported, never the minimum |
| Duration | 2 h 01 min for the whole suite |
| Raw output | [`testdata/timings.txt`](testdata/timings.txt) |

```bash
./gdalsuite.sh /path/to/HGV_leaf.tif 1024 320 11264 11264   # REPEATS=3 for this run
python summarize.py testdata/timings.txt                     # the tables below
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
  number on that: a MEM-to-MEM copy of the raster costs GDAL 133 ms,
  under 5% of its compute time on 17 of the 25 operations and 6–26% on
  the eight cheapest (most for min/max, 504 ms, and nearest ½,
  816 ms). On those eight the compute speedup is overstated by up to
  that much. The algebra
  reads its inputs from uncompressed GeoTIFFs in `/vsimem/`, because
  `gdal raster calc` names inputs only by file.
- **3 timed runs, not 5, and the machine was in use during part of the
  run.** Cases with a spread over 15% are marked ⚠. Almost all of them
  are GDAL's own 12-thread `gdalwarp -multi`, which varies run to run
  by itself; they move the 12-thread resample speedups, and nothing on
  one thread. The one-thread cases behind the headline are within 10%
  except strata's hillshade kernel (23%, 131–166 ms). The copy floors,
  taken first, spread 22–43%.
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

# strata: 3ed48ad
# go: go1.27.0
# date: 2026-09-23T15:36Z
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
# note: agree.py was killed for memory on cubic-double in this run (it then held the whole 2x output in float64);
# the agree line below is from rerunning only that check with the row-streaming agree.py, same window, same build.
# note: the stats agree line had a count compared against one derived from GDAL's rounded STATISTICS_VALID_PERCENT;
# GDAL computes no count, so that comparison was removed here and from gdalcompute.py.

11264 × 11264 = 126.9M cells, 3 timed runs per case, median reported. N = 12.

### Scoreboard: how many times faster strata is than GDAL

`1` is one thread each; `12` is strata on 12 workers against GDAL's fastest configuration at any thread count. Above 1× strata is faster.

| op | compute, 1 | compute, 12 | file to file, 1 | file to file, 12 | whole flow from a COG, 1 | whole flow from a COG, 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 16.1× | 54.4× | 4.5× | 9.1× | 1.9× | 4.8× |
| aspect | 14.4× | 68.0× | 4.6× | 9.6× | 2.0× | 5.0× |
| hillshade | 7.5× | 20.2× | 2.1× | 3.5× | 0.9× | 2.2× |
| tri | 6.9× | 29.2× | 2.5× | 5.3× | 1.2× | 2.9× |
| tpi | 13.4× | 22.4× | 2.5× | 4.2× | 1.1× | 2.5× |
| roughness | 13.7× | 26.7× | 2.7× | 4.9× | 1.1× | 2.7× |
| **terrain** (geometric mean) | **11.4×** | **33.1×** | **3.0×** | **5.7×** | **1.3×** | **3.2×** |
| mean3 | 43.8× | 95.7× | 7.0× | 12.5× | 2.7× | 6.1× |
| mean11 | 174× | 532× | 44.2× | 91.9× | 17.8× | 43.5× |
| min3 | 34.3× | 86.4× | 6.3× | 11.6× | 2.5× | 5.2× |
| max3 | 31.8× | 85.0× | 6.0× | 11.1× | 2.4× | 5.1× |
| gauss5 | 67.8× | 172× | 12.9× | 23.7× | 4.9× | 11.6× |
| conv5 | 39.7× | 170× | 11.4× | 23.8× | 4.7× | 11.3× |
| **focal** (geometric mean) | **53.1×** | **149×** | **10.9×** | **20.9×** | **4.3×** | **9.9×** |
| add | 40.5× | 57.2× | 2.8× | 6.1× | 1.4× | 3.9× |
| mul | 41.7× | 61.0× | 2.8× | 5.9× | 1.4× | 3.9× |
| min | 47.0× | 67.4× | 2.9× | 6.0× | 1.6× | 4.4× |
| max | 48.4× | 70.3× | 2.8× | 6.0× | 1.6× | 4.5× |
| clamp | 58.0× | 90.1× | 3.1× | 5.2× | 2.4× | 5.9× |
| **algebra** (geometric mean) | **46.8×** | **68.3×** | **2.8×** | **5.8×** | **1.6×** | **4.5×** |
| stats | 5.2× | 51.3× | 2.7× | 13.3× | 0.9× | 4.2× |
| minmax | 4.3× | 38.0× | 1.9× | 8.1× | 0.7× | 2.9× |
| **reduce** (geometric mean) | **4.7×** | **44.1×** | **2.3×** | **10.4×** | **0.8×** | **3.5×** |
| near-half | 6.6× | 20.4× | 3.1× | 4.5× | 1.1× | 1.6× |
| bilinear-half | 27.4× | 69.6× | 10.8× | 13.9× | 3.7× | 4.5× |
| cubic-half | 37.2× | 40.6× | 16.8× | 16.6× | 5.9× | 4.7× |
| lanczos-half | 39.1× | 62.1× | 19.4× | 12.6× | 7.4× | 3.1× |
| average-half | 12.7× | 18.3× | 5.1× | 4.8× | 1.7× | 1.9× |
| cubic-double | 35.3× | 31.7× | 16.5× | 5.2× | 10.9× | 4.5× |
| **resample** (geometric mean) | **22.2×** | **35.7×** | **9.9×** | **8.3×** | **3.8×** | **3.1×** |
| **all 25 operations** | **23.9×** | **57.2×** | **5.3×** | **9.0×** | **2.3×** | **4.5×** |

### Where the time goes: compute against the whole flow, one thread

The compute tier times the call with the input already in memory; the whole flow times the process from a Deflate COG to a file. `compute share` is the first over the second: how much of what a user waits for is arithmetic. The last columns are the speedup at each end.

| op | strata compute | strata flow | compute share | GDAL compute | GDAL flow | compute share | × compute | × flow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 190.0 ms | 2.25 s | 8% | 3.07 s | 4.37 s | 70% | 16.1× | 1.9× |
| aspect | 241.8 ms | 2.31 s | 10% | 3.48 s | 4.60 s | 76% | 14.4× | 2.0× |
| hillshade | 155.7 ms | 2.16 s | 7% | 1.17 s | 2.02 s | 58% | 7.5× | 0.9× |
| tri | 230.8 ms | 2.26 s | 10% | 1.58 s | 2.67 s | 59% | 6.9× | 1.2× |
| tpi | 89.9 ms | 2.13 s | 4% | 1.21 s | 2.26 s | 53% | 13.4× | 1.1× |
| roughness | 104.8 ms | 2.18 s | 5% | 1.43 s | 2.45 s | 58% | 13.7× | 1.1× |
| mean3 | 109.2 ms | 2.17 s | 5% | 4.78 s | 5.91 s | 81% | 43.8× | 2.7× |
| mean11 | 221.0 ms | 2.25 s | 10% | 38.45 s | 39.92 s | 96% | 174× | 17.8× |
| min3 | 128.2 ms | 2.15 s | 6% | 4.40 s | 5.38 s | 82% | 34.3× | 2.5× |
| max3 | 133.6 ms | 2.20 s | 6% | 4.25 s | 5.23 s | 81% | 31.8× | 2.4× |
| gauss5 | 140.3 ms | 2.20 s | 6% | 9.52 s | 10.78 s | 88% | 67.8× | 4.9× |
| conv5 | 241.1 ms | 2.30 s | 10% | 9.58 s | 10.75 s | 89% | 39.7× | 4.7× |
| add | 87.0 ms | 3.79 s | 2% | 3.53 s | 5.41 s | 65% | 40.5× | 1.4× |
| mul | 87.1 ms | 3.76 s | 2% | 3.64 s | 5.31 s | 69% | 41.7× | 1.4× |
| min | 86.5 ms | 3.72 s | 2% | 4.07 s | 5.92 s | 69% | 47.0× | 1.6× |
| max | 87.4 ms | 3.75 s | 2% | 4.23 s | 5.92 s | 71% | 48.4× | 1.6× |
| clamp | 70.5 ms | 2.10 s | 3% | 4.09 s | 5.03 s | 81% | 58.0× | 2.4× |
| stats | 227.8 ms | 1.90 s | 12% | 1.17 s | 1.79 s | 65% | 5.2× | 0.9× |
| minmax | 117.4 ms | 1.76 s | 7% | 503.7 ms | 1.17 s | 43% | 4.3× | 0.7× |
| near-half | 123.5 ms | 1.90 s | 6% | 816.4 ms | 2.00 s | 41% | 6.6× | 1.1× |
| bilinear-half | 227.6 ms | 2.02 s | 11% | 6.24 s | 7.38 s | 85% | 27.4× | 3.7× |
| cubic-half | 312.2 ms | 2.10 s | 15% | 11.62 s | 12.51 s | 93% | 37.2× | 5.9× |
| lanczos-half | 386.2 ms | 2.18 s | 18% | 15.12 s | 16.25 s | 93% | 39.1× | 7.4× |
| average-half | 178.5 ms | 1.94 s | 9% | 2.26 s | 3.37 s | 67% | 12.7× | 1.7× |
| cubic-double | 1.25 s | 4.38 s | 29% | 44.20 s | 47.90 s | 92% | 35.3× | 10.9× |

### compute: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal mem, 1 | gdal mem, 12 | strata simd, 1 | strata simd, 12 | strata scalar, 1 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 3.07 s | – | 190.0 ms | 56.4 ms | 1.16 s | 668 | 6.1× | 10% |
| aspect | 3.48 s | – | 241.8 ms | 51.2 ms | 2.91 s | 525 | 12.0× | 4% |
| hillshade | 1.17 s | – | 155.7 ms | 58.0 ms | 1.02 s | 815 | 6.5× | 23% ⚠ |
| tri | 1.58 s | – | 230.8 ms | 54.2 ms | 581.2 ms | 550 | 2.5× | 4% |
| tpi | 1.21 s | – | 89.9 ms | 53.8 ms | 197.7 ms | 1411 | 2.2× | 10% |
| roughness | 1.43 s | – | 104.8 ms | 53.5 ms | 1.32 s | 1211 | 12.6× | 4% |
| mean3 | 4.78 s | – | 109.2 ms | 50.0 ms | 313.4 ms | 1162 | 2.9× | 7% |
| mean11 | 38.45 s | – | 221.0 ms | 72.2 ms | 872.5 ms | 574 | 3.9× | 2% |
| min3 | 4.40 s | – | 128.2 ms | 51.0 ms | 308.5 ms | 990 | 2.4× | 3% |
| max3 | 4.25 s | – | 133.6 ms | 50.0 ms | 443.8 ms | 949 | 3.3× | 5% |
| gauss5 | 9.52 s | – | 140.3 ms | 55.3 ms | 539.4 ms | 904 | 3.8× | 14% |
| conv5 | 9.58 s | – | 241.1 ms | 56.4 ms | 1.14 s | 526 | 4.7× | 5% |
| add | 3.53 s | – | 87.0 ms | 61.7 ms | 97.4 ms | 1459 | 1.1× | 13% |
| mul | 3.64 s | – | 87.1 ms | 59.6 ms | 98.8 ms | 1457 | 1.1× | 8% |
| min | 4.07 s | – | 86.5 ms | 60.4 ms | 103.7 ms | 1467 | 1.2× | 5% |
| max | 4.23 s | – | 87.4 ms | 60.2 ms | 133.9 ms | 1452 | 1.5× | 6% |
| clamp | 4.09 s | – | 70.5 ms | 45.4 ms | 145.2 ms | 1799 | 2.1× | 6% |
| stats | 1.17 s | 1.16 s | 227.8 ms | 22.6 ms | 890.4 ms | 557 | 3.9× | 9% |
| minmax | 503.7 ms | 496.9 ms | 117.4 ms | 13.1 ms | 348.6 ms | 1081 | 3.0× | 8% |
| near-half | 816.4 ms | 329.4 ms | 123.5 ms | 16.2 ms | 122.5 ms | 1028 | 1.0× | 8% |
| bilinear-half | 6.24 s | 2.07 s | 227.6 ms | 29.8 ms | 520.0 ms | 558 | 2.3× | 19% ⚠ |
| cubic-half | 11.62 s | 1.49 s | 312.2 ms | 36.6 ms | 801.0 ms | 406 | 2.6× | 40% ⚠ |
| lanczos-half | 15.12 s | 3.01 s | 386.2 ms | 48.4 ms | 999.2 ms | 329 | 2.6× | 23% ⚠ |
| average-half | 2.26 s | 483.1 ms | 178.5 ms | 26.4 ms | 429.9 ms | 711 | 2.4× | 5% |
| cubic-double | 44.20 s | 5.41 s | 1.25 s | 170.6 ms | 2.96 s | 101 | 2.4× | 6% |

### file to file: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal envi, 1 | gdal gdal_calc-envi, 1 | gdal gdal_calc-gtiff, 1 | gdal gtiff, 1 | gdal envi, 12 | gdal gtiff, 12 | strata simd, 1 | strata simd, 12 | strata scalar, 1 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 4.20 s | – | – | 3.98 s | – | – | 883.0 ms | 435.0 ms | 1.74 s | 144 | 2.0× | 5% |
| aspect | 4.44 s | – | – | 4.17 s | – | – | 899.0 ms | 435.0 ms | 3.54 s | 141 | 3.9× | 3% |
| hillshade | 1.77 s | – | – | 1.64 s | – | – | 784.0 ms | 463.0 ms | 1.61 s | 162 | 2.1× | 10% |
| tri | 2.52 s | – | – | 2.28 s | – | – | 899.0 ms | 430.0 ms | 1.18 s | 141 | 1.3× | 8% |
| tpi | 2.17 s | – | – | 1.89 s | – | – | 770.0 ms | 445.0 ms | 820.0 ms | 165 | 1.1× | 4% |
| roughness | 2.35 s | – | – | 2.11 s | – | – | 783.0 ms | 433.0 ms | 1.87 s | 162 | 2.4× | 11% |
| mean3 | 5.80 s | – | – | 5.43 s | – | – | 779.0 ms | 434.0 ms | 910.0 ms | 163 | 1.2× | 4% |
| mean11 | 39.82 s | – | – | 39.42 s | – | – | 892.0 ms | 429.0 ms | 1.48 s | 142 | 1.7× | 3% |
| min3 | 5.29 s | – | – | 4.88 s | – | – | 776.0 ms | 421.0 ms | 930.0 ms | 164 | 1.2× | 13% |
| max3 | 5.25 s | – | – | 4.85 s | – | – | 811.0 ms | 436.0 ms | 1.06 s | 156 | 1.3× | 2% |
| gauss5 | 10.67 s | – | – | 10.29 s | – | – | 799.0 ms | 434.0 ms | 1.13 s | 159 | 1.4× | 8% |
| conv5 | 10.67 s | – | – | 10.23 s | – | – | 894.0 ms | 429.0 ms | 1.75 s | 142 | 2.0× | 3% |
| add | 4.93 s | 3.24 s | 2.96 s | 4.57 s | – | – | 1.07 s | 484.0 ms | 992.0 ms | 119 | 0.9× | 5% |
| mul | 5.00 s | 3.04 s | 2.95 s | 4.54 s | – | – | 1.07 s | 504.0 ms | 988.0 ms | 119 | 0.9× | 9% |
| min | 5.50 s | 2.99 s | 2.99 s | 5.13 s | – | – | 1.04 s | 499.0 ms | 985.0 ms | 122 | 0.9× | 8% |
| max | 5.50 s | 3.11 s | 2.93 s | 5.12 s | – | – | 1.06 s | 487.0 ms | 1.03 s | 120 | 1.0× | 7% |
| clamp | 5.02 s | 2.51 s | 2.21 s | 4.61 s | – | – | 720.0 ms | 422.0 ms | 771.0 ms | 176 | 1.1× | 8% |
| stats | 1.41 s | – | – | 1.38 s | 1.44 s | 1.39 s | 508.0 ms | 104.0 ms | 1.12 s | 250 | 2.2× | 27% ⚠ |
| minmax | 748.0 ms | – | – | 766.0 ms | 748.0 ms | 777.0 ms | 397.0 ms | 92.0 ms | 589.0 ms | 320 | 1.5× | 4% |
| near-half | 1.61 s | – | – | 1.68 s | 1.15 s | 849.0 ms | 526.0 ms | 189.0 ms | 473.0 ms | 241 | 0.9× | 9% |
| bilinear-half | 7.04 s | – | – | 7.05 s | 5.33 s | 3.03 s | 652.0 ms | 218.0 ms | 924.0 ms | 195 | 1.4× | 19% ⚠ |
| cubic-half | 12.28 s | – | – | 12.16 s | 4.82 s | 3.86 s | 722.0 ms | 232.0 ms | 1.20 s | 176 | 1.7× | 10% |
| lanczos-half | 15.96 s | – | – | 15.87 s | 3.87 s | 2.93 s | 816.0 ms | 233.0 ms | 1.36 s | 155 | 1.7× | 53% ⚠ |
| average-half | 3.04 s | – | – | 2.96 s | 1.31 s | 1.00 s | 579.0 ms | 208.0 ms | 800.0 ms | 219 | 1.4× | 5% |
| cubic-double | 49.44 s | – | – | 47.68 s | 9.16 s | 8.63 s | 2.89 s | 1.65 s | 4.52 s | 44 | 1.6× | 8% |

### whole flow from a COG: medians

Throughput is source cells over the median time. `worst spread` is the largest (max − min)/median over the cases in the row.

| op | gdal to-envi, 1 | gdal to-gtiff, 1 | gdal to-envi, 12 | gdal to-gtiff, 12 | strata simd, 1 | strata simd, 12 | strata 1, M cells/s | scalar/SIMD | worst spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slope | 5.24 s | 4.37 s | 8.88 s | 8.14 s | 2.25 s | 908.0 ms | 56 | – | 6% |
| aspect | 5.56 s | 4.60 s | 9.36 s | 8.49 s | 2.31 s | 913.0 ms | 55 | – | 11% |
| hillshade | 2.12 s | 2.02 s | 6.07 s | 5.89 s | 2.16 s | 900.0 ms | 59 | – | 3% |
| tri | 3.25 s | 2.67 s | 7.20 s | 6.60 s | 2.26 s | 912.0 ms | 56 | – | 6% |
| tpi | 2.93 s | 2.26 s | 6.76 s | 6.18 s | 2.13 s | 904.0 ms | 60 | – | 6% |
| roughness | 3.08 s | 2.45 s | 6.93 s | 6.41 s | 2.18 s | 902.0 ms | 58 | – | 7% |
| mean3 | 6.47 s | 5.91 s | 5.89 s | 5.36 s | 2.17 s | 881.0 ms | 58 | – | 4% |
| mean11 | 40.67 s | 39.92 s | 39.75 s | 39.03 s | 2.25 s | 898.0 ms | 56 | – | 4% |
| min3 | 5.95 s | 5.38 s | 5.22 s | 4.70 s | 2.15 s | 897.0 ms | 59 | – | 5% |
| max3 | 5.75 s | 5.23 s | 5.09 s | 4.60 s | 2.20 s | 893.0 ms | 58 | – | 6% |
| gauss5 | 11.39 s | 10.78 s | 10.62 s | 10.10 s | 2.20 s | 869.0 ms | 58 | – | 12% |
| conv5 | 11.38 s | 10.75 s | 10.79 s | 10.11 s | 2.30 s | 896.0 ms | 55 | – | 5% |
| add | 6.14 s | 5.41 s | 4.49 s | 4.15 s | 3.79 s | 1.05 s | 34 | – | 5% |
| mul | 6.07 s | 5.31 s | 4.52 s | 4.17 s | 3.76 s | 1.06 s | 34 | – | 3% |
| min | 6.48 s | 5.92 s | 5.04 s | 4.67 s | 3.72 s | 1.07 s | 34 | – | 3% |
| max | 6.71 s | 5.92 s | 5.25 s | 4.77 s | 3.75 s | 1.07 s | 34 | – | 4% |
| clamp | 5.67 s | 5.03 s | 4.81 s | 4.44 s | 2.10 s | 747.0 ms | 60 | – | 4% |
| stats | 1.81 s | 1.79 s | 1.78 s | 1.82 s | 1.90 s | 426.0 ms | 67 | – | 5% |
| minmax | 1.20 s | 1.17 s | 1.21 s | 1.21 s | 1.76 s | 408.0 ms | 72 | – | 6% |
| near-half | 2.03 s | 2.00 s | 693.0 ms | 630.0 ms | 1.90 s | 405.0 ms | 67 | – | 3% |
| bilinear-half | 7.38 s | 7.57 s | 3.28 s | 3.86 s | 2.02 s | 732.0 ms | 63 | – | 18% ⚠ |
| cubic-half | 12.69 s | 12.51 s | 4.20 s | 3.45 s | 2.10 s | 733.0 ms | 60 | – | 20% ⚠ |
| lanczos-half | 16.25 s | 16.27 s | 3.30 s | 2.34 s | 2.18 s | 755.0 ms | 58 | – | 55% ⚠ |
| average-half | 3.43 s | 3.37 s | 855.0 ms | 791.0 ms | 1.94 s | 421.0 ms | 66 | – | 9% |
| cubic-double | 50.63 s | 47.90 s | 10.61 s | 8.48 s | 4.38 s | 1.86 s | 29 | – | 22% ⚠ |

### CPU spent: the whole flow from a COG, CPU-seconds (user + sys)

What each tool burns for the same result. A tool that returns sooner on more threads can still be spending more.

| op | GDAL, 1 | strata, 1 | GDAL, 12 | strata, 12 |
| --- | ---: | ---: | ---: | ---: |
| slope | 4.37 | 2.25 | 11.73 | 4.95 |
| aspect | 4.60 | 2.32 | 12.08 | 5.07 |
| hillshade | 2.02 | 2.17 | 9.48 | 4.95 |
| tri | 2.67 | 2.26 | 10.19 | 4.98 |
| tpi | 2.26 | 2.15 | 9.74 | 4.84 |
| roughness | 2.44 | 2.19 | 9.96 | 4.84 |
| mean3 | 5.90 | 2.19 | 6.46 | 4.79 |
| mean11 | 39.91 | 2.27 | 40.20 | 4.91 |
| min3 | 5.37 | 2.18 | 5.79 | 4.94 |
| max3 | 5.23 | 2.22 | 5.67 | 4.92 |
| gauss5 | 10.77 | 2.22 | 11.27 | 4.64 |
| conv5 | 10.74 | 2.32 | 11.24 | 4.87 |
| add | 5.41 | 3.81 | 6.16 | 6.19 |
| mul | 5.30 | 3.78 | 6.22 | 5.96 |
| min | 5.91 | 3.73 | 6.71 | 6.34 |
| max | 5.92 | 3.77 | 6.84 | 6.22 |
| clamp | 5.00 | 2.12 | 5.48 | 4.55 |
| stats | 1.79 | 1.89 | 1.78 | 2.28 |
| minmax | 1.17 | 1.76 | 1.21 | 2.15 |
| near-half | 2.00 | 1.91 | 2.29 | 3.02 |
| bilinear-half | 7.37 | 2.02 | 23.81 | 5.90 |
| cubic-half | 12.50 | 2.11 | 25.11 | 6.01 |
| lanczos-half | 16.25 | 2.19 | 20.74 | 6.11 |
| average-half | 3.37 | 1.94 | 4.00 | 3.15 |
| cubic-double | 47.89 | 4.43 | 65.52 | 11.32 |

### Floors: each tool's cost before it computes anything

| case | median | spread |
| --- | ---: | ---: |
| gdal startup, 1 thread | 23.0 ms | 9% |
| strata startup, 1 thread | 4.0 ms | 0% |
| gdal copy-envi, 1 thread | 2.14 s | 40% |
| gdal copy-gtiff, 1 thread | 1.12 s | 43% |
| gdal copy-cog, 1 thread | 1.34 s | 22% |
| gdal copy-cog, 12 threads | 661.0 ms | 1% |
| gdal mem-copy, 1 thread | 133.1 ms | 18% |

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


<!-- summarize.py output end -->

## Do they agree

The table above is `agree.py` on every operation: strata's one-worker
output from the raw file against GDAL's ENVI output from the same file,
leaving out the border where the two define edges differently (1 cell
for terrain, r for a radius-r focal operation, 8 for resampling). It
measures and does not judge; [`acceptance/`](../../acceptance/) is
where strata is judged against definitions with derived tolerances.

Two kinds of cell differ in *validity*, and both were checked cell for
cell on this raster's full-size outputs:

- **Aspect, 24,293,095 cells (19%).** Every one is a flat cell, where
  strata writes −1 (Esri's flat aspect, `terrain.AspectFlat`) and
  gdaldem writes NoData. This canopy-height grid has large flat areas
  at zero. `AspectOptions.ZeroForFlat` is gdaldem's `-zero_for_flat`.
- **The focal operations, 10,078 cells for 3×3 (20,154 for 5×5, 50,370
  for 11×11), under 0.05%.** Every one is a cell whose window touches
  NoData, never one that is NoData itself: `gdal raster neighbors`
  computes from the valid part of the window, strata requires all of it
  (its package documentation, and `acceptance/` check 5).
- **Lanczos ½, 42 of 30.8M cells: not explained.** DESIGN.md §54 says
  strata's validity matches gdalwarp's exactly, including Lanczos's
  half-valid rule, and `acceptance/gdalwarp_resample.py` shows it on
  small grids. On this raster it does not, for 42 cells. One candidate
  is in §54 itself: gdalwarp's kernel width depends on how it chunks a
  large warp, and a 127M-cell warp is chunked, which would move the
  half-valid count for cells near NoData. That is a hypothesis, not a
  finding; it is the one open question this run raised about
  correctness.

Everything else is values, on cells both tools computed: bit-identical
for 18 of the 25 operations, and within 1.5e-4 for the others except
hillshade, whose gdaldem output is rounded to a byte and floored at 1.

The bytes strata wrote in the whole flow, from the COG on 12 workers,
were compared with its raw path on one worker, `cmp` for `cmp`: equal
for all 24 operations that write a raster.

## What this does not tell you

- **A quiet-machine verdict on the noisy rows.** Three timed runs, on a
  desktop in use for part of them. The ⚠ rows are mostly GDAL's
  `gdalwarp -multi` and only move 12-thread resample speedups, but the
  hillshade kernel row is strata's and is worth rerunning.
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
