# Against gdaldem: results

strata and GDAL's `gdaldem` computing slope, aspect and hillshade from
the same 11264 × 11264 raster, in the same container, on the same
machine, under the same timer. The rest of `benchmarks/` measures strata
against strata: the SIMD kernels against the scalar ones, many workers
against one. That says how much each part is worth and nothing about
whether the whole thing is fast. This directory is the outside number.

It is the speed companion to [`acceptance/`](../../acceptance/), which
asks whether strata gets the same answer as gdaldem. The files timed
here are the files that directory checks: the last thing `gdalbench.sh`
does is run `acceptance/gdalcompare.py` over them, and this run passed
all 7 of its checks. Every speed below is a speed at the same answer.

## Headline

- **strata computes all three operations in 2.57 s where gdaldem takes
  9.39 s, single-threaded on one core: 3.6× faster.** Per operation,
  against the fastest of three gdaldem configurations: slope 4.3×,
  aspect 4.4×, hillshade 2.0×. With 12 workers, which gdaldem has no
  equivalent of, the three take 1.33 s: 7.1× overall, 8.3× / 9.2× / 3.7×
  per operation.
- **Take the SIMD kernels away and the advantage nearly vanishes.** On
  the scalar kernels, which is what a build without `GOEXPERIMENT=simd`
  gets, one worker is 2.1× gdaldem on slope, 1.1× on aspect and 1.0× on
  hillshade. So the win is not Go beating C++, and it is not the engine:
  it is AVX2 against gdaldem's scalar arithmetic. strata's remaining
  structural advantage on one core, with the lanes turned off, is about
  1.3× across the three.
- **strata does the same job for a quarter of the CPU.** Slope costs
  gdaldem 3.72 CPU-seconds and strata 0.93, both including their file
  IO, for results that agree to 7.6e-06 degrees. At 12 workers strata
  spends about what gdaldem spends (3.69 CPU-seconds) and returns in an
  eighth of the wall time: the extra CPU buys latency, not work.
- **At this size both tools spend most of their time moving bytes.**
  strata's SIMD slope kernel is 190 ms of its 863 ms; `gdal_translate`
  copying the same raster with no computation at all takes 840 ms of
  gdaldem's 3722 ms. The arithmetic gap is far wider than the end-to-end
  gap, roughly 15× on slope, but the end-to-end gap is what a user waits
  for, so it is the one quoted above.
- **12 workers is the ceiling; 24 is slower.** 283 M cells/s at 12, 250
  at 24, on 12 physical cores. The same shape the chunked suite found
  ([`../chunked/RESULTS.md`](../chunked/RESULTS.md)): past one worker per
  core the tile buffers, not the kernels, are the limit.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200; the Docker Desktop VM is given 24 CPUs and 31 GiB |
| Host OS | Windows 11 Home 10.0.22631, Docker Desktop 29.8.0 |
| Container | `ghcr.io/osgeo/gdal:ubuntu-small-latest`, Ubuntu 26.04, Linux 6.18.33.2-microsoft-standard-WSL2 |
| GDAL | 3.14.0dev-17759e56, released 2026/08/18 |
| Go | go1.27.0, cross-compiled `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| Raster | `HGV_leaf.tif`, a 12.5 m Swedish canopy-height grid, UInt16, NoData 65535; the window at (1024, 320), 11264 × 11264 = 126.9M cells, promoted to Float32. 97.6% of it carries data |
| Working files | a 14 GB tmpfs inside the container |
| Runs | one untimed warm-up, then 5 timed; the median is reported |
| Raw output | [`testdata/timings.txt`](testdata/timings.txt) |

```bash
./gdalbench.sh /path/to/HGV_leaf.tif 1024 320 11264 11264
python summarize.py testdata/timings.txt 11264 11264   # the tables below
```

The tables between the markers are exactly that command's output for the
committed `testdata/timings.txt`. Unlike the Go suites, this category has
no `TestResultsMatch`: the summary is produced by a Python script, and
adding a Python dependency to `go test ./...` would cost more than it
buys. Re-running the one line above is the check.

### What is controlled, and what is not

Both tools run in one container, so the OS, the CPU, the filesystem and
the timer are shared. Beyond that:

- **The timer is the whole process**, bash's `time` on the command a user
  would type. It includes process start, which is 23 ms for a GDAL tool
  and 3 ms for the Go binary: under 1% of either, and measured in the
  floors table rather than assumed away.
- **The files are a tmpfs.** On a bind mount to NTFS, Docker Desktop's
  filesystem cost more than either tool's computation and swamped the
  comparison. A memory filesystem takes the host's disk out of a
  measurement that is about the two tools. It flatters both.
- **gdaldem gets three configurations**: raw float32 in and out (the file
  shape strata reads), GeoTIFF (its own default and home format), and
  GeoTIFF with a 4 GB block cache. Every speedup is quoted against
  whichever of the three was *fastest* for that operation. The
  raw-float32 row is the closest shape match but never the fastest, so
  quoting it would have flattered strata by about 10%.
- **gdaldem is single-threaded**, so the one-worker row is the
  like-for-like one. The 12- and 24-worker rows are what strata offers,
  not what GDAL failed at.
- **GDAL does strictly more work in its IO layer.** It decodes a real
  format, carries a geotransform and a projection, and goes through a
  block cache; strata reads a headerless file whose shape it is told on
  the command line. That is the trade strata makes by design, since it is
  not a file-format project, but it means part of the gap is a difference
  in scope rather than in speed. The `gdal_translate` floors are there to
  size that part: they are the most GDAL-favourable reading of how much
  of gdaldem's time is not arithmetic.
- **Hillshade is measured against strata**, not for it. gdaldem writes
  one byte per cell; strata writes a float32, so it writes four times as
  many bytes for the same picture. Its 2.0× is an understatement by
  however much that costs.
- **Throughput counts every cell**, valid or not. 2.4% of this window is
  NoData.

## The numbers

`M cells/s` is 126.9M cells over the median wall time, so it measures the
whole invocation, not a kernel. `CPU/wall` is `(user + sys) / real`: 1.0
is one core saturated, and it is how the claim that gdaldem is
single-threaded was checked rather than assumed. `spread` is
(max − min)/median over the five timed runs.

<!-- summarize.py output begin -->

11264 x 11264 = 126.9M cells, 5 timed runs per case, median reported. Throughput counts every cell, valid or not.

### slope

Baseline: gdaldem, 4 GB block cache, GeoTIFF, the fastest of 3 gdaldem configurations.

| case | wall s | M cells/s | CPU s | CPU/wall | spread | vs gdaldem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdaldem, raw float32 | 4.038 | 31 | 4.03 | 1.0 | 2% | 0.9x |
| gdaldem, GeoTIFF | 3.748 | 34 | 3.74 | 1.0 | 2% | 1.0x |
| gdaldem, 4 GB block cache, GeoTIFF | 3.722 | 34 | 3.72 | 1.0 | 3% | 1.0x |
| strata chunked, raw float32, 1 worker | 0.863 | 147 | 0.93 | 1.1 | 3% | 4.3x |
| strata chunked, raw float32, 12 workers | 0.448 | 283 | 3.69 | 8.2 | 3% | 8.3x |
| strata chunked, raw float32, 24 workers | 0.508 | 250 | 6.46 | 12.7 | 4% | 7.3x |
| strata chunked, scalar kernels, raw float32, 1 worker | 1.811 | 70 | 1.83 | 1.0 | 2% | 2.1x |
| strata, whole raster in memory | 1.046 | 121 | 1.14 | 1.1 | 5% | 3.6x |

### aspect

Baseline: gdaldem, GeoTIFF, the fastest of 3 gdaldem configurations.

| case | wall s | M cells/s | CPU s | CPU/wall | spread | vs gdaldem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdaldem, raw float32 | 4.495 | 28 | 4.51 | 1.0 | 3% | 0.9x |
| gdaldem, GeoTIFF | 4.079 | 31 | 4.04 | 1.0 | 1% | 1.0x |
| gdaldem, 4 GB block cache, GeoTIFF | 4.106 | 31 | 4.11 | 1.0 | 4% | 1.0x |
| strata chunked, raw float32, 1 worker | 0.922 | 138 | 0.98 | 1.1 | 2% | 4.4x |
| strata chunked, raw float32, 12 workers | 0.443 | 286 | 3.68 | 8.3 | 2% | 9.2x |
| strata chunked, raw float32, 24 workers | 0.520 | 244 | 6.93 | 13.3 | 4% | 7.8x |
| strata chunked, scalar kernels, raw float32, 1 worker | 3.667 | 35 | 3.70 | 1.0 | 1% | 1.1x |
| strata, whole raster in memory | 1.088 | 117 | 1.14 | 1.0 | 2% | 3.7x |

### hillshade

Baseline: gdaldem, 4 GB block cache, GeoTIFF, the fastest of 3 gdaldem configurations.

| case | wall s | M cells/s | CPU s | CPU/wall | spread | vs gdaldem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdaldem, raw float32 | 1.691 | 75 | 1.69 | 1.0 | 4% | 0.9x |
| gdaldem, GeoTIFF | 1.611 | 79 | 1.60 | 1.0 | 6% | 1.0x |
| gdaldem, 4 GB block cache, GeoTIFF | 1.590 | 80 | 1.59 | 1.0 | 4% | 1.0x |
| strata chunked, raw float32, 1 worker | 0.789 | 161 | 0.85 | 1.1 | 2% | 2.0x |
| strata chunked, raw float32, 12 workers | 0.435 | 292 | 3.57 | 8.2 | 5% | 3.7x |
| strata chunked, raw float32, 24 workers | 0.505 | 251 | 6.55 | 13.0 | 3% | 3.1x |
| strata chunked, scalar kernels, raw float32, 1 worker | 1.665 | 76 | 1.71 | 1.0 | 1% | 1.0x |
| strata, whole raster in memory | 0.967 | 131 | 1.00 | 1.0 | 13% | 1.6x |

### floors: the same bytes moved, and the process started, without either tool computing anything

| case | wall s | M cells/s | CPU s | CPU/wall | spread | vs gdaldem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| process start, nothing else | 0.023 | - | 0.02 | 1.0 | 13% | - |
| process start, nothing else, Go | 0.003 | - | 0.00 | 1.3 | 33% | - |
| gdal_translate, copy only, raw float32 | 1.133 | 112 | 1.11 | 1.0 | 6% | - |
| gdal_translate, copy only, GeoTIFF | 0.840 | 151 | 0.85 | 1.0 | 13% | - |

<!-- summarize.py output end -->

## Where strata's time goes

The wall times above are whole processes, so they cannot separate the
kernel from the file. These are strata's own timings of the same work on
one core, from [`testdata/timings.txt`](testdata/timings.txt). The kernel
columns are the median of the two warm runs of three — the first touches
the destination's pages and is 2.5× slower — and the file-to-file column
the median of three:

| | kernel, SIMD | kernel, scalar | SIMD/scalar | file to file, SIMD |
|---|---:|---:|---:|---:|
| slope | 190 ms (667 M cells/s) | 1146 ms (111) | 6.0× | 767 ms |
| aspect | 246 ms (515 M cells/s) | 2996 ms (42) | 12.2× | 841 ms |
| hillshade | 114 ms (1112 M cells/s) | 996 ms (127) | 8.7× | 703 ms |

Two things follow.

The kernels agree with the terrain suite, which measured 706, 543 and
1248 M cells/s for the same three on this CPU
([`../terrain/RESULTS.md`](../terrain/RESULTS.md)): 4–11% faster there,
on Windows, at 16384², with the raster already in memory. So this is the
same code running at the same speed, not a differently tuned build.

And the kernel is a quarter of the streamed run. Slope computes in 190 ms
and takes 767 ms once the cells have to come out of a file and go back
into one. That is the cost of the tile buffers and the tmpfs, and it is
why the end-to-end advantage over gdaldem (4.3×) is so much smaller than
the arithmetic advantage (roughly 15×, taking `gdal_translate` as
gdaldem's IO floor). Both tools are paying it. It is also where the
remaining headroom is: a faster kernel would now move the end-to-end
number very little.

## What this does not tell you

- **Correctness.** [`acceptance/`](../../acceptance/) does, and its
  checks ran on these files. On this raster strata is very slightly the
  *less* accurate of the two: against a float64 reference, gdaldem is
  closer on 17.8% of cells and strata on 6.5%, tied on 75.7%, mean error
  1.44e-06 against 1.91e-06 degrees. Both are far inside float32.
- **A default Go build.** Every SIMD row needs `GOEXPERIMENT=simd`
  ([`docs/adr/0001-simd-backend.md`](../../docs/adr/0001-simd-backend.md)).
  The scalar row is what an ordinary `go build` gets, and against gdaldem
  it is roughly a tie.
- **Anything but these three operations**, this one raster, this one size
  and this one machine. gdaldem has options strata has no counterpart for
  (`-compute_edges`, `-alg ZevenbergenThorne`, `-multidirectional`, Igor
  shading) and formats strata cannot read at all. Nothing here is a claim
  about GDAL as a whole.
- **Real disk.** Everything timed here was in memory. On a raster larger
  than RAM the comparison becomes a comparison of IO patterns, which
  neither this nor [`../chunked/RESULTS.md`](../chunked/RESULTS.md)
  measures against another tool.
