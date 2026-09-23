# The cog reader against GDAL: results

How fast strata's GeoTIFF/COG reader (the [`cog`](../../cog/) module)
decodes, measured against GDAL reading the same files, in the same
container, on the same machine, under the same timer as
[`../gdal/`](../gdal/RESULTS.md). [`acceptance/cogcheck.sh`](../../acceptance/cogcheck.sh)
already shows the reader gets GDAL's answer, bit for bit, on 98 files.
This directory asks how long it takes to get there.

The first version of the reader had never been timed. Profiling it put
its own loops ahead of the decompressors, and the fixes those profiles
called for are part of this change. The tables show the reader before
and after them, and GDAL.

## Headline

- **On a compressed COG, GDAL still reads faster: 1.4–2.1× on one
  core, 1.5–2× on twelve.** The whole 508 MB float32 raster from a
  Deflate COG with predictor 3 takes strata 1.70 s on one core and GDAL
  0.81 s. From ZSTD with predictor 3 it is 1.10 s against 0.80 s; from
  LZW, 2.21 s against 1.23 s. Only an uncompressed COG is roughly a tie:
  0.47 s against 0.41 s on one core, and strata 1.1× ahead on twelve.
- **The fixes made the reader 1.3–2.6× faster.** Deflate went from 3.98 s
  to 1.70 s on one core, ZSTD from 2.84 s to 1.10 s, LZW from 3.23 s to
  2.21 s and uncompressed from 0.71 s to 0.47 s (1.3× on 12 threads, the
  smallest gain). Before them, GDAL was 4.9× faster on Deflate. The
  biggest single cost was not a decompressor but the loop that undoes
  the floating-point predictor: 37% of a Deflate read and 54% of a ZSTD
  read.
- **Slope over a compressed COG is still 1.8–2.6× faster than gdaldem's
  on one core, and 4.3–5.6× on twelve workers** (3.7× and 6.7× on the
  uncompressed COG). gdaldem has to decode the same blocks, and
  strata's slope kernel is so much faster than gdaldem's (see
  [`../gdal/`](../gdal/RESULTS.md)) that it covers the slower decode. On the Deflate COG, strata takes 2.28 s on one worker
  and gdaldem 4.39 s.
- **But the format now costs more than the computation.** From the raw
  float32 file, slope takes 0.89 s on one worker. From the Deflate COG
  it takes 2.28 s, so reading the format is about 60% of the run. From
  ZSTD it is 1.68 s and from an uncompressed COG 1.07 s. On twelve
  workers it is 0.46 s raw, and 0.93 s, 0.78 s and 0.59 s from the same
  three COGs.
- **The block cache does its job on one worker and falls short on many.**
  Halos and tiles that do not line up with blocks ask for the same block
  several times. With the default 64 MiB cache, one worker decodes every
  block exactly once at tile heights of 16, 64 and 256 rows. Without the
  cache it decodes each block 3.9, 9.9 and 33.9 times, and 16-row tiles
  take 54 s instead of 2.3 s. With 12 workers and 256-row tiles, though,
  the default cache decodes each block 1.57 times. That run is 26% slower
  than with a cache large enough to hold everything (0.85 s against
  0.68 s). See [The cache](#the-cache).

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200; the Docker Desktop VM is given 24 CPUs and 31 GiB |
| Host OS | Windows 11 Home 10.0.22631, Docker Desktop 29.8.0 |
| Container | `ghcr.io/osgeo/gdal:ubuntu-small-latest`, Ubuntu 26.04, Linux 6.18.33.2-microsoft-standard-WSL2 |
| GDAL | 3.14.0dev-17759e56, released 2026/08/18, linked to libtiff 6, **libdeflate** and libzstd |
| Go | go1.27.0, cross-compiled `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, `GOAMD64=v1`, `GOEXPERIMENT=simd` (the slope kernel uses it; the decoder has no SIMD code) |
| strata | this change (b3c31c2) for "strata"; 9f658d3, the reader as PR #32 first proposed it, for "strata before" |
| Raster | `HGV_leaf.tif`, a 12.5 m Swedish canopy-height grid, UInt16, NoData 65535; the window at (1024, 320), 11264 × 11264 = 126.9M cells, promoted to Float32: the same window as [`../gdal/`](../gdal/RESULTS.md). 97.6% of it carries data |
| Files | written in the container by `gdal_translate -of COG`, 512 × 512 blocks, no overviews: Deflate + predictor 3 (117 MB), ZSTD + predictor 3 (113 MB), LZW with GDAL's default of no predictor (149 MB), uncompressed (508 MB) |
| Working files | a 16 GB tmpfs inside the container |
| Runs | one untimed warm-up, then 5 timed (3 for the cache sweep); the median is reported, never the minimum |
| Raw output | [`testdata/timings.txt`](testdata/timings.txt) |

```bash
./cogbench.sh /path/to/HGV_leaf.tif 1024 320 11264 11264
python summarize.py testdata/timings.txt 11264 11264   # the tables below
```

`cogbench.sh` builds [`main.go`](main.go) (`cogbench`) twice, from this
checkout and from `BEFORE_REF`. It starts the container and runs
[`runbench.sh`](runbench.sh) there, and that script writes the files and
times every case. Like `../gdal/`, this directory has no
`TestResultsMatch`. The tables between the markers are exactly what
[`summarize.py`](summarize.py) prints for the committed timings. To check
them, rerun that one line.

### What is measured, and how

- **Pure read** is the whole of band 1, level 0, decoded to float32 and
  handed to the caller in full-width strips of 256 rows. These are the
  tiles the engine asks a source for. Each thread reuses one buffer.
  strata is `Source.ReadWindow` on N goroutines (`cogbench -mode read`),
  with its default cache. GDAL is `ReadAsArray` on the same strips from
  Python ([`gdalread.py`](gdalread.py)), with `GDAL_NUM_THREADS=N`,
  which lets GDAL's GTiff driver decode the blocks one request touches
  in parallel. Both are timed **in process**, from the first strip to
  the last. That leaves out Python's 104 ms to import GDAL and numpy,
  which would otherwise be an eighth of GDAL's time. The whole-process
  numbers are in the full table. `gdal_translate -of MEM` is timed there
  too, as a GDAL-native reader with no Python.
- **End to end** is `terrain.SlopeChunked` from each COG to a raw float32
  file, 256-row tiles, 1 and 12 workers (`cogbench -mode slope`). It is
  compared with two things. One is `gdaldem slope` on the same COG,
  writing an uncompressed GeoTIFF. The other is strata's own raw-file
  path (`engine.RawSource`), the one [`../gdal/`](../gdal/RESULTS.md)
  timed. Everything here is whole-process wall time, as in that
  directory.
- **The cache sweep** is slope over the Deflate COG at tile heights 16,
  64 and 256, with `SourceOptions.CacheBytes` set to -1 (no cache), 4 MiB,
  0 (the default, 64 MiB) and 1 GiB.
- **Decodes per block** come from `cogbench`, which counts the source's
  `ReadAt` calls after `Open`. The reader reads each block it decodes with
  one `ReadAt` when the block is at most 1 MiB, as every block here is,
  and makes no other reads. So the count divided by the 484 blocks is
  decodes per block.

### Every speed is a speed at the same answer

The last part of `runbench.sh` checks the files it timed. It does not
time these checks. From [`testdata/timings.txt`](testdata/timings.txt):

- strata's slope through each of the four COGs is **byte-identical** to
  its slope through the raw file. That holds on 1 worker and on 12, both
  with 256-row tiles and the default cache. It also holds on 12 workers
  with 16-row tiles and a 4 MiB cache that evicts constantly (12 checks,
  12 identical).
- That slope agrees with `gdaldem slope` by `acceptance/gdalcompare.py`'s
  criteria. The same cells carry data (0 differ), and the largest
  difference is 7.63e-06 degrees over 123,785,590 cells.

The decoder changes were checked against GDAL, not only against the
benchmark:

```
acceptance/cogcheck.sh:
GDAL 3.14.0dev-17759e56cbe7d693a39c3d5ccf339364de844ee7: 98/98 files identical to GDAL's reading, 58,535,874 cells compared, 9,587,169 of them NoData
acceptance/cogsabotage.py:
8/8 sabotages caught
```

## Where the time went, and what was changed

A CPU profile of one-core reads of each file (`cogbench -cpuprofile`,
5 reads per profile, in the container), before and after. The figures are
shares of the flat time:

| file | before: top of the profile | after: top of the profile |
|---|---|---|
| Deflate, pred. 3 | predictor 37%, `compress/flate` 28%, `convert` 7%, memmove/memclr 10% | `klauspost/compress/flate` 51%, predictor 19%, memmove 11%, NoData test 6% |
| ZSTD, pred. 3 | predictor 54%, `convert` + `toFloat32` 14%, memmove 9%, zstd 9% | predictor 33%, zstd 40%, NoData test 11%, memmove 6% |
| LZW | `x/image/tiff/lzw` 58%, memmove/memclr 19%, `convert` 8% | `klauspost/compress/lzw` 82%, copy 5%, NoData test 4% |
| uncompressed | `convert` + `toFloat32` 59%, memclr/memmove 21%, `pread` 9% | copy 27%, NoData test 25%, `pread` 20%, memmove 15% |

The changes, all in [`cog/decode.go`](../../cog/decode.go):

1. **The floating-point predictor.** It used to difference bytes in place
   with a bounds check per byte, copy the row, and put the byte planes
   back one byte at a time through a computed index. Now one pass keeps
   the running sum in a register. For the usual case, one band of
   float32, a second pass assembles each sample from its four planes and
   writes it straight into the block's `[]float32`, with nothing written
   back to the byte buffer. The generic path, for other sizes and
   several bands, also assembles four-byte samples whole. A bytewise
   SWAR prefix sum was tried as well and measured no faster, so it is
   not in.
2. **float32 samples skip `convert`.** They are copied bit for bit, which
   is how GDAL copies a Float32 band into a Float32 buffer. The general
   path converts every sample through float64, with a type switch per
   sample. The NoData test is now one branch-free pass over the sample
   bits, comparing them as integers, and it allocates the mask only if a
   cell is invalid. Before, a block with NoData was converted twice. For
   integer sample types nothing changed.
3. **LZW and Deflate come from `klauspost/compress`**, already the
   module's ZSTD dependency. Its `lzw` in `SetAldusCompatible` mode
   decodes libtiff's early-change LZW and replaces
   `golang.org/x/image/tiff/lzw`, which dropped that dependency from the
   module. Its `flate` replaces `compress/zlib`. The two-byte zlib header
   is now checked by hand and the stream inflated raw. Decoding stops at
   the block's size and never reaches the Adler-32 trailer, so a zlib
   reader computed a checksum it never compared (5–7% of a Deflate read).
   That was true before this change too.
4. **Fewer allocations.** Decoders and the read and decompression
   buffers are pooled. Blocks now decompress into a buffer of exactly the
   block's size instead of a `bytes.Buffer` that grew from zero. The
   decoded block's own `[]float32` is still allocated per block, because
   the cache holds it.

Along the way, each change was measured on one core with fewer runs,
mostly on the Windows host, so these figures are indicative only. The
predictor rewrite took ZSTD from 4.1 s to 1.7 s and Deflate from 4.2 s
to 2.9 s. The new decompressors, buffers and float32 copy took Deflate
to 2.1 s and LZW from 3.3 s to 2.3 s. In the container, fusing the
predictor with the copy and the integer NoData test took ZSTD from 1.34 s
to 1.10 s and Deflate from 2.0 s to 1.7 s.

What is left is mostly the decompressors, and they are not strata's
code. GDAL inflates Deflate with libdeflate, which is C with SIMD. All of
GDAL's one-core read takes 0.81 s. In strata's 1.70 s, `klauspost/compress/flate`
alone takes about 0.87 s (51%). So the inflater, not the reader around
it, is most of the remaining 2.1×. ZSTD is the closest compressed result
(1.38×). There the decompressor is the smaller share, and the largest
remaining cost is the predictor loop (33%), the same kind of scalar loop
libtiff runs. Two more things would help:
decoding straight into the caller's buffer when a window covers whole
blocks, which would skip the cache's copy, and a SIMD predictor. Neither
is attempted here.

## The numbers

<!-- summarize.py output begin -->

11264 x 11264 = 126.9M cells, 508 MB as float32. 5 timed runs per case (3 in the cache sweep) after a warm-up, median reported.

### Pure read: the whole raster, decoded to float32

In-process time, from the first strip to the last, of full-width strips of 256 rows into a reused buffer. MB/s is float32 output. GDAL is `ReadAsArray` from Python with `GDAL_NUM_THREADS`; strata is `Source.ReadWindow` on that many goroutines. *before* is the reader as PR #32 first proposed it (9f658d3).

| compression | threads | GDAL s | GDAL MB/s | strata before s | strata s | strata MB/s | strata vs GDAL | before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deflate, predictor 3 | 1 | 0.807 | 629 | 3.979 | 1.695 | 299 | 0.48x | 2.35x |
| Deflate, predictor 3 | 12 | 0.178 | 2854 | 0.811 | 0.348 | 1457 | 0.51x | 2.33x |
| ZSTD, predictor 3 | 1 | 0.801 | 633 | 2.835 | 1.104 | 460 | 0.73x | 2.57x |
| ZSTD, predictor 3 | 12 | 0.169 | 3001 | 0.607 | 0.246 | 2063 | 0.69x | 2.47x |
| LZW, no predictor | 1 | 1.229 | 413 | 3.226 | 2.207 | 230 | 0.56x | 1.46x |
| LZW, no predictor | 12 | 0.230 | 2207 | 0.683 | 0.450 | 1127 | 0.51x | 1.52x |
| uncompressed | 1 | 0.405 | 1254 | 0.705 | 0.467 | 1087 | 0.87x | 1.51x |
| uncompressed | 12 | 0.158 | 3208 | 0.190 | 0.143 | 3537 | 1.10x | 1.32x |

Every read case in full, with spreads and whole-process numbers. `gdal_translate -of MEM` has no in-process timer, so it is compared by wall time only; its system time is the kernel faulting in a fresh 507 MB dataset, which the strip readers do not pay.

| case | in-process s | spread | wall s | CPU s | CPU/wall | reads per block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deflate, predictor 3, GDAL, ReadAsArray strips, 1 thread | 0.807 | 12% | 0.968 | 1.03 | 1.1 | - |
| Deflate, predictor 3, GDAL, gdal_translate -of MEM, 1 thread | - | 10% | 1.261 | 1.28 | 1.0 | - |
| Deflate, predictor 3, strata before, 1 thread | 3.979 | 3% | 4.006 | 4.02 | 1.0 | 1.00 |
| Deflate, predictor 3, strata, 1 thread | 1.695 | 8% | 1.723 | 1.75 | 1.0 | 1.00 |
| Deflate, predictor 3, GDAL, ReadAsArray strips, 12 threads | 0.178 | 4% | 0.354 | 1.51 | 4.3 | - |
| Deflate, predictor 3, GDAL, gdal_translate -of MEM, 12 threads | - | 5% | 0.599 | 1.69 | 2.8 | - |
| Deflate, predictor 3, strata before, 12 threads | 0.811 | 5% | 0.862 | 4.60 | 5.3 | 1.00 |
| Deflate, predictor 3, strata, 12 threads | 0.348 | 3% | 0.409 | 2.10 | 5.1 | 1.00 |
| ZSTD, predictor 3, GDAL, ReadAsArray strips, 1 thread | 0.801 | 6% | 0.964 | 0.96 | 1.0 | - |
| ZSTD, predictor 3, GDAL, gdal_translate -of MEM, 1 thread | - | 12% | 1.204 | 1.20 | 1.0 | - |
| ZSTD, predictor 3, strata before, 1 thread | 2.835 | 4% | 2.861 | 2.86 | 1.0 | 1.00 |
| ZSTD, predictor 3, strata, 1 thread | 1.104 | 2% | 1.131 | 1.13 | 1.0 | 1.00 |
| ZSTD, predictor 3, GDAL, ReadAsArray strips, 12 threads | 0.169 | 10% | 0.352 | 1.57 | 4.5 | - |
| ZSTD, predictor 3, GDAL, gdal_translate -of MEM, 12 threads | - | 2% | 0.589 | 1.69 | 2.9 | - |
| ZSTD, predictor 3, strata before, 12 threads | 0.607 | 7% | 0.667 | 3.47 | 5.2 | 1.00 |
| ZSTD, predictor 3, strata, 12 threads | 0.246 | 7% | 0.310 | 1.51 | 4.9 | 1.00 |
| LZW, no predictor, GDAL, ReadAsArray strips, 1 thread | 1.229 | 7% | 1.386 | 1.44 | 1.0 | - |
| LZW, no predictor, GDAL, gdal_translate -of MEM, 1 thread | - | 3% | 1.698 | 1.70 | 1.0 | - |
| LZW, no predictor, strata before, 1 thread | 3.226 | 2% | 3.249 | 3.26 | 1.0 | 1.00 |
| LZW, no predictor, strata, 1 thread | 2.207 | 5% | 2.230 | 2.28 | 1.0 | 1.00 |
| LZW, no predictor, GDAL, ReadAsArray strips, 12 threads | 0.230 | 7% | 0.408 | 1.96 | 4.8 | - |
| LZW, no predictor, GDAL, gdal_translate -of MEM, 12 threads | - | 4% | 0.669 | 2.14 | 3.2 | - |
| LZW, no predictor, strata before, 12 threads | 0.683 | 3% | 0.733 | 3.91 | 5.3 | 1.00 |
| LZW, no predictor, strata, 12 threads | 0.450 | 8% | 0.506 | 2.63 | 5.2 | 1.00 |
| uncompressed, GDAL, ReadAsArray strips, 1 thread | 0.405 | 13% | 0.560 | 0.55 | 1.0 | - |
| uncompressed, GDAL, gdal_translate -of MEM, 1 thread | - | 14% | 0.834 | 0.82 | 1.0 | - |
| uncompressed, strata before, 1 thread | 0.705 | 12% | 0.729 | 0.74 | 1.0 | 1.00 |
| uncompressed, strata, 1 thread | 0.467 | 4% | 0.490 | 0.49 | 1.0 | 1.00 |
| uncompressed, GDAL, ReadAsArray strips, 12 threads | 0.158 | 5% | 0.330 | 1.09 | 3.3 | - |
| uncompressed, GDAL, gdal_translate -of MEM, 12 threads | - | 5% | 0.595 | 1.29 | 2.2 | - |
| uncompressed, strata before, 12 threads | 0.190 | 4% | 0.243 | 1.21 | 5.0 | 1.00 |
| uncompressed, strata, 12 threads | 0.143 | 6% | 0.198 | 0.96 | 4.9 | 1.00 |

### End to end: slope, file to file

Wall time of the whole process, as in benchmarks/gdal. gdaldem writes an uncompressed GeoTIFF, strata a raw float32 file. `gdaldem s` is gdaldem's fastest time on that input over both `GDAL_NUM_THREADS` settings (its compute is single-threaded either way), and `vs gdaldem` is against it. `vs raw` is strata's time on the raw file over its time on this input, at the same worker count, so 1.00 means the format cost nothing.

| input | threads | gdaldem s | strata before s | strata s | spread | M cells/s | vs gdaldem | vs raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| striped GeoTIFF, uncompressed (gdaldem) / raw float32 (strata) | 1 | 3.857 | - | 0.892 | 3% | 142 | 4.32x | 1.00 |
| Deflate, predictor 3 COG | 1 | 4.386 | 4.546 | 2.283 | 5% | 56 | 1.92x | 0.39 |
| ZSTD, predictor 3 COG | 1 | 4.357 | 3.419 | 1.683 | 2% | 75 | 2.59x | 0.53 |
| LZW, no predictor COG | 1 | 4.817 | 3.828 | 2.759 | 3% | 46 | 1.75x | 0.32 |
| uncompressed COG | 1 | 3.941 | 1.269 | 1.065 | 8% | 119 | 3.70x | 0.84 |
| striped GeoTIFF, uncompressed (gdaldem) / raw float32 (strata) | 12 | 3.857 | - | 0.463 | 4% | 274 | 8.33x | 1.00 |
| Deflate, predictor 3 COG | 12 | 4.386 | 1.700 | 0.925 | 5% | 137 | 4.74x | 0.50 |
| ZSTD, predictor 3 COG | 12 | 4.357 | 1.386 | 0.777 | 10% | 163 | 5.61x | 0.60 |
| LZW, no predictor COG | 12 | 4.817 | 1.489 | 1.113 | 6% | 114 | 4.33x | 0.42 |
| uncompressed COG | 12 | 3.941 | 0.644 | 0.592 | 6% | 214 | 6.66x | 0.78 |

gdaldem's own spreads and CPU, which the table above leaves out:

| gdaldem on | threads | wall s | spread | CPU s | CPU/wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| striped GeoTIFF, uncompressed | 1 | 3.958 | 3% | 3.95 | 1.0 |
| Deflate, predictor 3 | 1 | 4.386 | 3% | 4.36 | 1.0 |
| ZSTD, predictor 3 | 1 | 4.357 | 4% | 4.35 | 1.0 |
| LZW, no predictor | 1 | 4.817 | 3% | 4.80 | 1.0 |
| uncompressed | 1 | 3.941 | 1% | 3.92 | 1.0 |
| striped GeoTIFF, uncompressed | 12 | 3.857 | 3% | 3.86 | 1.0 |
| Deflate, predictor 3 | 12 | 8.653 | 2% | 12.54 | 1.4 |
| ZSTD, predictor 3 | 12 | 8.835 | 2% | 12.77 | 1.4 |
| LZW, no predictor | 12 | 8.815 | 1% | 13.11 | 1.5 |
| uncompressed | 12 | 8.769 | 3% | 12.39 | 1.4 |

### The block cache under halos: slope over the Deflate COG

In-process time and block decodes per block (484 blocks of 512 × 512). 1.00 is every block decoded once.

| tile rows | CacheBytes | 1 thread: s | decodes/block | 12 threads: s | decodes/block |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | -1 (no cache) | 54.047 | 33.91 | 6.141 | 33.91 |
| 16 | 4 MiB | 53.618 | 33.91 | 4.730 | 4.84 |
| 16 | 0 (default, 64 MiB) | 2.275 | 1.00 | 1.919 | 1.00 |
| 16 | 1024 MiB | 2.421 | 1.00 | 2.067 | 1.00 |
| 64 | -1 (no cache) | 16.069 | 9.91 | 1.918 | 9.91 |
| 64 | 4 MiB | 16.004 | 9.91 | 1.638 | 3.31 |
| 64 | 0 (default, 64 MiB) | 2.237 | 1.00 | 1.135 | 1.00 |
| 64 | 1024 MiB | 2.439 | 1.00 | 1.226 | 1.00 |
| 256 | -1 (no cache) | 6.838 | 3.91 | 1.025 | 3.91 |
| 256 | 4 MiB | 6.764 | 3.91 | 0.921 | 2.32 |
| 256 | 0 (default, 64 MiB) | 2.261 | 1.00 | 0.854 | 1.57 |
| 256 | 1024 MiB | 2.435 | 1.00 | 0.679 | 1.00 |

### Floors

| case | wall s | spread |
| --- | ---: | ---: |
| cogbench, start and stop | 0.003 | 33% |
| python3: import gdal and numpy, open the COG | 0.104 | 6% |

<!-- summarize.py output end -->

## The cache

The default cache is 64 MiB. A row of this file's blocks decodes to
22 × 1 MiB. One worker walks down the raster, so the blocks a tile needs
are mostly the ones the previous tile just used, plus the next row of
blocks. 64 MiB holds almost three rows, and the table shows exactly one
decode per block at every tile height.

Two settings break that:

- **A cache smaller than one row of blocks is no cache.** With 4 MiB,
  one worker decodes each block as often as with none: 33.9 times at
  16-row tiles. The LRU evicts a block before the next tile comes back
  for it. With 12 workers, 4 MiB does help, because tiles running at the
  same time share the blocks they overlap.
- **Many workers need more than 64 MiB.** Twelve workers on 256-row
  tiles have 3072 rows in flight, six block rows or 132 MiB, and the
  default cache holds fewer than three. The result is 1.57 decodes per
  block and a run 26% slower than with 1 GiB. Shorter tiles keep fewer
  rows in flight, so 16- and 64-row tiles on 12 workers still decode
  once.

This change leaves the default alone. It is a memory bound that callers
choose, and the right size depends on the worker count, which the
source cannot see. A rule of thumb consistent with these numbers is at
least
`(workers × TileHeight / blockHeight + 2)` rows of blocks, each
`blocksAcross × blockBytes`. For this file on 12 workers with 256-row
tiles that is 8 × 22 MiB, about 180 MiB. A default that scales with the
file's block-row size would be the obvious follow-up, if one is wanted.

One more observation, not investigated: **gdaldem gets twice as slow on a
COG when `GDAL_NUM_THREADS=12`** (8.7 s against 4.4 s, and 12.5 CPU-seconds
against 4.4). On the striped GeoTIFF it makes no difference. gdaldem
reads a few lines at a time, and parallel decoding of such small requests
seems to cost more than it saves. Every "vs gdaldem" figure uses
gdaldem's faster setting.

## What this does not tell you

- **Other rasters.** This is one raster, canopy height, which compresses
  4.3× under Deflate, with 2.4% NoData. Data that compresses less makes
  the decompressor a larger share, and very smooth data makes it a
  smaller one. There is one block size (512), one band, level 0 only,
  no overviews read, and no JPEG, WebP or LERC, which the reader
  refuses anyway.
- **Integer sample types.** Every file here is float32. UInt16 or Int16
  COGs, common for DEMs, go through the general `convert` path, which
  this change did not touch and this suite does not time. The acceptance
  suite still checks them for correctness.
- **Disk, network, cold caches.** Every file was on a tmpfs, and the
  warm-up run had already read it. Reading from disk, or over HTTP
  through `cog.HTTPReaderAt` (which arrived after these runs and is not
  timed here), would add IO that neither tool is charged for here.
- **GDAL's mask.** `gdalread.py` reads values only. strata's read also
  produces the validity mask, which GDAL would build in a separate
  `GetMaskBand().ReadRaster`. So GDAL's read-time figures are a lighter
  job than strata's, and favour GDAL.
- **GDAL at its most tuned.** GDAL ran with its defaults apart from
  `GDAL_NUM_THREADS`. It has other knobs (`GDAL_CACHEMAX`,
  `VSI_CACHE`, reading with its block-aligned `ReadBlock`) that were not
  explored for the pure-read case.
- **A second machine.** Zen 2, AVX2, and Docker Desktop's WSL2 VM, as in
  `../gdal/`.
- **Races.** The pooled decoders and buffers are covered by `go test
  -race`, which CI runs on Linux. It was not run on the Windows host
  where this change was made, which has no cgo.
