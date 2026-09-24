# The cog reader against GDAL: results

How fast strata's GeoTIFF/COG reader (the [`cog`](../../cog/) module)
decodes, measured against GDAL reading the same files, in the same
container, on the same machine, under the same timer as
[`../gdal/`](../gdal/RESULTS.md). [`acceptance/cogcheck.sh`](../../acceptance/cogcheck.sh)
already shows the reader gets GDAL's answer, bit for bit, on 98 files.
This directory asks how long it takes to get there.

The first version of the reader had never been timed. Profiling it put
its own loops ahead of the decompressors, and the fixes those profiles
called for came with the first version of this page. The tables show
the reader as it first was ("before", 9f658d3), the reader now, and
GDAL. They were last regenerated with the SIMD row kernels of
[The SIMD row kernels](#the-simd-row-kernels), so "strata" includes
every change since the first version of this page, not only those.

## Headline

- **On one core, strata now reads a predictor-3 COG about as fast as
  GDAL: ZSTD is a tie and Deflate 1.17× behind.** The whole 508 MB
  float32 raster from ZSTD with predictor 3 takes strata 0.79 s and GDAL
  0.78 s; from Deflate with predictor 3, 0.93 s against 0.80 s. LZW,
  whose decoder is not strata's, is still 1.5× behind (2.00 s against
  1.30 s). An uncompressed COG is read 1.37× faster than GDAL reads it
  (0.29 s against 0.40 s). Twelve threads give the same picture: ZSTD
  and Deflate within 1.05× and 1.14× of GDAL, LZW 1.8× behind,
  uncompressed 1.36× ahead.
- **Against the reader as first proposed, that is 2.5–4.3× faster.**
  Deflate went from 3.98 s to 0.93 s on one core, ZSTD from 2.82 s to
  0.79 s, LZW from 3.29 s to 2.00 s and uncompressed from 0.73 s to
  0.29 s. Most of that came before the SIMD kernels: the fixes of the
  first version of this page, the whole-block inflater (#44, #49) and a
  larger default cache.
- **Undoing the floating-point predictor now costs about 96 ms of a
  one-core read, down from about 116 ms (Deflate) and 105 ms (ZSTD)
  with the previous kernel** (30-read profiles), and from 37–54% of the
  read before the first fixes. The latest kernel alone makes a one-core
  read 3–4% faster (Deflate 883 → 854 ms, ZSTD 753 → 726 ms, medians of
  10 interleaved runs). See [The SIMD row kernels](#the-simd-row-kernels).
- **Slope over a COG is 1.9–3.1× faster than gdaldem's on one core, and
  6.0–7.9× on twelve workers** (4.4× and 7.4× on the uncompressed COG).
  On the Deflate COG, strata takes 1.51 s on one worker and gdaldem
  4.31 s.
- **On one core the format still costs more than the computation; on
  twelve, little.** From the raw float32 file, slope takes 0.84 s on
  one worker; from the Deflate COG 1.51 s, from ZSTD 1.39 s, from an
  uncompressed COG 0.83 s. On twelve workers it is 0.49 s raw, and
  0.56 s, 0.54 s and 0.50 s from the same three COGs.
- **The default block cache now does its job on many workers too.** It
  holds 8 rows of blocks (182 MiB for this file) rather than 64 MiB, and
  every block is decoded exactly once at every tile height, on one
  worker and on twelve. With 64 MiB, twelve workers on 256-row tiles
  decoded each block 1.57 times. See [The cache](#the-cache).

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200; the Docker Desktop VM is given 24 CPUs and 31 GiB |
| Host OS | Windows 11 Home 10.0.22631, Docker Desktop 29.8.0 |
| Container | `ghcr.io/osgeo/gdal:ubuntu-small-latest`, Ubuntu 26.04, Linux 6.18.33.2-microsoft-standard-WSL2 |
| GDAL | 3.14.0dev-17759e56, released 2026/08/18, linked to libtiff 6, **libdeflate** and libzstd |
| Go | go1.27.0, cross-compiled `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, `GOAMD64=v1`, `GOEXPERIMENT=simd` (the slope kernel and the reader's row kernels, `cog/internal/kern`, use it: AVX2 here) |
| strata | the fused SIMD predictor (on top of 8ad43da) for "strata"; 9f658d3, the reader as PR #32 first proposed it, for "strata before" |
| Raster | `HGV_leaf.tif`, a 12.5 m Swedish canopy-height grid, UInt16, NoData 65535; the window at (1024, 320), 11264 × 11264 = 126.9M cells, promoted to Float32: the same window as [`../gdal/`](../gdal/RESULTS.md). 97.6% of it carries data |
| Files | written in the container by `gdal_translate -of COG`, 512 × 512 blocks, no overviews: Deflate + predictor 3 (117 MB), ZSTD + predictor 3 (113 MB), LZW with GDAL's default of no predictor (149 MB), uncompressed (508 MB) |
| Working files | a 16 GB tmpfs inside the container |
| Runs | one untimed warm-up, then 5 timed (3 for the cache sweep); the median is reported, never the minimum |
| Raw output | [`testdata/timings.txt`](testdata/timings.txt). Two runs are missing from it: bash's `time` printed a garbled field for each (`real=2.:00` in the fifth "strata before" ZSTD read on one thread, `user=0.:00` in the first 1-thread, 256-row, default-cache run of the cache sweep), so they were dropped rather than guessed, and those two cases have 4 and 2 runs |

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
  0 (the default: 8 rows of blocks, 182 MiB here) and 1 GiB.
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

The rest of this section is the state of things when this page was
first written; [The SIMD row kernels](#the-simd-row-kernels) follows on
from its last paragraph.

What was left then was mostly the decompressors, and they are not strata's
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

## The SIMD row kernels

The first version of this page ended on a SIMD predictor not being
attempted. It since has been, in two steps, both in
[`cog/internal/kern`](../../cog/internal/kern/): a kernel package of the
cog module with a scalar form of each kernel that every build runs, and
AVX2 (amd64) and NEON (arm64) forms that `GOEXPERIMENT=simd` builds
install, as strata's own `internal/vec` and `internal/stencil` do
([ADR 0001](../../docs/adr/0001-simd-backend.md)). The cog module cannot
import strata's internal packages, so it has its own. Every SIMD form is
held bit for bit to its scalar form on every row width from 0 to 300
samples and some wider ragged ones (`simd_test.go`) and by a fuzz test
(`FuzzKernels`), and `acceptance/cogcheck.sh` finds all 98 files
identical to GDAL's reading in both the scalar and the SIMD build.

1. **48532c0, f021657:** the planes put back together sixteen samples a
   step, after the byte running sum, sixteen bytes a step (three
   shifted adds, then the previous vector's carry), in a pass of its own
   that writes the row back.
2. **The fused kernel:** one pass over the four planes, 32 bytes of
   each a step, with no running sum written back. A 256-bit register
   holds two streams, its lower half on the first half of the row and
   its upper half on the second, so the prefix sum needs no cross-lane
   step. Each plane's two streams start from the total of every byte
   before them, found first by one sequential pass of wrapping byte adds
   (reduced with `VPSADBW`). The four planes then have four independent
   carry chains of one add each, and the bytes are interleaved into
   samples with masks, shifts and 16- and 32-bit unpacks (archsimd has
   no AVX2 byte unpack). The same package now also undoes the predictor
   for the rows the fast path does not take (float64, float16, one byte
   apart: `SumBytes`) and reads single-band 8-bit rows with or without
   predictor 2 (`Uint8Row`). Multi-band chunky predictor rows and 32-bit
   integers stay scalar.

One 512-sample row (`go test -bench PlanesRow ./internal/kern` on the
Windows host, cache-hot): scalar about 1,500 ns, the first AVX2 kernel
about 450 ns, the fused one about 248 ns. Over a whole 512 × 512 block
just written, as the inflater leaves it, the fused kernel takes about
308 ns a row against the first AVX2 kernel's 475: a block is 1 MiB,
more than a Zen 2 core's 512 KiB of L2, so its early rows come back from
L3. That, not arithmetic, is now most of the predictor's cost, and why
it still takes about 96 ms of a read when its arithmetic alone would
take about 61 ms. Undoing the predictor while a block's rows are still
in L2, as they are inflated, would be the next step; it would change
the inflater, and is not attempted here.

In the container, the fused kernel against 8ad43da (the first AVX2
kernel), one core, 10 rounds, each running both builds in turn:

| case | 8ad43da median | fused median | spreads |
| --- | ---: | ---: | ---: |
| read, Deflate predictor 3 | 883 ms | 854 ms | 9%, 3% |
| read, ZSTD predictor 3 | 753 ms | 726 ms | 9%, 5% |
| slope, Deflate predictor 3 COG, 1 worker | 1,403 ms | 1,383 ms | 3%, 5% |

The predictor's share, from 30-read CPU profiles of each build (about
300 samples in the kernel each): Deflate 116 → 96 ms a read, ZSTD
105 → 96 ms. The 5-read profiles `runbench.sh` takes are too short to
tell these apart; at about 60 samples they put the kernel within ±10%
or more of either figure.

## The numbers

<!-- summarize.py output begin -->

11264 x 11264 = 126.9M cells, 508 MB as float32. 5 timed runs per case (2 in the cache sweep) after a warm-up, median reported.

### Pure read: the whole raster, decoded to float32

In-process time, from the first strip to the last, of full-width strips of 256 rows into a reused buffer. MB/s is float32 output. GDAL is `ReadAsArray` from Python with `GDAL_NUM_THREADS`; strata is `Source.ReadWindow` on that many goroutines. *before* is the reader as PR #32 first proposed it (9f658d3).

| compression | threads | GDAL s | GDAL MB/s | strata before s | strata s | strata MB/s | strata vs GDAL | before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deflate, predictor 3 | 1 | 0.795 | 638 | 3.984 | 0.929 | 546 | 0.86x | 4.29x |
| Deflate, predictor 3 | 12 | 0.167 | 3034 | 0.803 | 0.191 | 2664 | 0.88x | 4.22x |
| ZSTD, predictor 3 | 1 | 0.779 | 651 | 2.818 | 0.787 | 645 | 0.99x | 3.58x |
| ZSTD, predictor 3 | 12 | 0.165 | 3085 | 0.592 | 0.173 | 2934 | 0.95x | 3.42x |
| LZW, no predictor | 1 | 1.302 | 390 | 3.292 | 1.995 | 254 | 0.65x | 1.65x |
| LZW, no predictor | 12 | 0.221 | 2302 | 0.725 | 0.391 | 1297 | 0.56x | 1.85x |
| uncompressed | 1 | 0.395 | 1286 | 0.728 | 0.288 | 1762 | 1.37x | 2.53x |
| uncompressed | 12 | 0.163 | 3123 | 0.211 | 0.119 | 4261 | 1.36x | 1.77x |

Every read case in full, with spreads and whole-process numbers. `gdal_translate -of MEM` has no in-process timer, so it is compared by wall time only; its system time is the kernel faulting in a fresh 507 MB dataset, which the strip readers do not pay.

| case | in-process s | spread | wall s | CPU s | CPU/wall | reads per block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deflate, predictor 3, GDAL, ReadAsArray strips, 1 thread | 0.795 | 15% | 0.950 | 0.95 | 1.0 | - |
| Deflate, predictor 3, GDAL, gdal_translate -of MEM, 1 thread | - | 9% | 1.210 | 1.22 | 1.0 | - |
| Deflate, predictor 3, strata before, 1 thread | 3.984 | 3% | 4.009 | 4.04 | 1.0 | 1.00 |
| Deflate, predictor 3, strata, 1 thread | 0.929 | 8% | 0.958 | 0.96 | 1.0 | 1.00 |
| Deflate, predictor 3, GDAL, ReadAsArray strips, 12 threads | 0.167 | 3% | 0.342 | 1.42 | 4.2 | - |
| Deflate, predictor 3, GDAL, gdal_translate -of MEM, 12 threads | - | 8% | 0.591 | 1.67 | 2.8 | - |
| Deflate, predictor 3, strata before, 12 threads | 0.803 | 2% | 0.856 | 4.59 | 5.4 | 1.00 |
| Deflate, predictor 3, strata, 12 threads | 0.191 | 5% | 0.240 | 1.22 | 5.1 | 1.00 |
| ZSTD, predictor 3, GDAL, ReadAsArray strips, 1 thread | 0.779 | 11% | 0.939 | 0.96 | 1.0 | - |
| ZSTD, predictor 3, GDAL, gdal_translate -of MEM, 1 thread | - | 10% | 1.179 | 1.18 | 1.0 | - |
| ZSTD, predictor 3, strata before, 1 thread | 2.818 | 10% | 2.844 | 2.84 | 1.0 | 1.00 |
| ZSTD, predictor 3, strata, 1 thread | 0.787 | 10% | 0.816 | 0.83 | 1.0 | 1.00 |
| ZSTD, predictor 3, GDAL, ReadAsArray strips, 12 threads | 0.165 | 1% | 0.344 | 1.48 | 4.3 | - |
| ZSTD, predictor 3, GDAL, gdal_translate -of MEM, 12 threads | - | 3% | 0.608 | 1.68 | 2.8 | - |
| ZSTD, predictor 3, strata before, 12 threads | 0.592 | 3% | 0.652 | 3.46 | 5.3 | 1.00 |
| ZSTD, predictor 3, strata, 12 threads | 0.173 | 5% | 0.219 | 1.14 | 5.2 | 1.00 |
| LZW, no predictor, GDAL, ReadAsArray strips, 1 thread | 1.302 | 10% | 1.478 | 1.47 | 1.0 | - |
| LZW, no predictor, GDAL, gdal_translate -of MEM, 1 thread | - | 12% | 1.709 | 1.77 | 1.0 | - |
| LZW, no predictor, strata before, 1 thread | 3.292 | 5% | 3.316 | 3.32 | 1.0 | 1.00 |
| LZW, no predictor, strata, 1 thread | 1.995 | 5% | 2.025 | 2.02 | 1.0 | 1.00 |
| LZW, no predictor, GDAL, ReadAsArray strips, 12 threads | 0.221 | 6% | 0.403 | 1.89 | 4.7 | - |
| LZW, no predictor, GDAL, gdal_translate -of MEM, 12 threads | - | 6% | 0.665 | 2.06 | 3.1 | - |
| LZW, no predictor, strata before, 12 threads | 0.725 | 12% | 0.774 | 4.13 | 5.3 | 1.00 |
| LZW, no predictor, strata, 12 threads | 0.391 | 23% | 0.453 | 2.39 | 5.3 | 1.00 |
| uncompressed, GDAL, ReadAsArray strips, 1 thread | 0.395 | 16% | 0.569 | 0.56 | 1.0 | - |
| uncompressed, GDAL, gdal_translate -of MEM, 1 thread | - | 12% | 0.841 | 0.83 | 1.0 | - |
| uncompressed, strata before, 1 thread | 0.728 | 8% | 0.754 | 0.77 | 1.0 | 1.00 |
| uncompressed, strata, 1 thread | 0.288 | 5% | 0.316 | 0.33 | 1.0 | 1.00 |
| uncompressed, GDAL, ReadAsArray strips, 12 threads | 0.163 | 8% | 0.341 | 1.11 | 3.3 | - |
| uncompressed, GDAL, gdal_translate -of MEM, 12 threads | - | 19% | 0.606 | 1.38 | 2.3 | - |
| uncompressed, strata before, 12 threads | 0.211 | 7% | 0.266 | 1.35 | 5.1 | 1.00 |
| uncompressed, strata, 12 threads | 0.119 | 7% | 0.167 | 0.91 | 5.5 | 1.00 |

### End to end: slope, file to file

Wall time of the whole process, as in benchmarks/gdal. gdaldem writes an uncompressed GeoTIFF, strata a raw float32 file. `gdaldem s` is gdaldem's fastest time on that input over both `GDAL_NUM_THREADS` settings (its compute is single-threaded either way), and `vs gdaldem` is against it. `vs raw` is strata's time on the raw file over its time on this input, at the same worker count, so 1.00 means the format cost nothing.

| input | threads | gdaldem s | strata before s | strata s | spread | M cells/s | vs gdaldem | vs raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| striped GeoTIFF, uncompressed (gdaldem) / raw float32 (strata) | 1 | 3.913 | - | 0.843 | 2% | 151 | 4.64x | 1.00 |
| Deflate, predictor 3 COG | 1 | 4.305 | 4.661 | 1.510 | 3% | 84 | 2.85x | 0.56 |
| ZSTD, predictor 3 COG | 1 | 4.277 | 3.481 | 1.389 | 5% | 91 | 3.08x | 0.61 |
| LZW, no predictor COG | 1 | 4.539 | 3.623 | 2.412 | 3% | 53 | 1.88x | 0.35 |
| uncompressed COG | 1 | 3.692 | 1.224 | 0.831 | 4% | 153 | 4.44x | 1.01 |
| striped GeoTIFF, uncompressed (gdaldem) / raw float32 (strata) | 12 | 3.913 | - | 0.489 | 14% | 259 | 8.00x | 1.00 |
| Deflate, predictor 3 COG | 12 | 4.305 | 1.710 | 0.564 | 7% | 225 | 7.63x | 0.87 |
| ZSTD, predictor 3 COG | 12 | 4.277 | 1.328 | 0.542 | 5% | 234 | 7.89x | 0.90 |
| LZW, no predictor COG | 12 | 4.539 | 1.439 | 0.758 | 11% | 167 | 5.99x | 0.65 |
| uncompressed COG | 12 | 3.692 | 0.676 | 0.498 | 6% | 255 | 7.41x | 0.98 |

gdaldem's own spreads and CPU, which the table above leaves out:

| gdaldem on | threads | wall s | spread | CPU s | CPU/wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| striped GeoTIFF, uncompressed | 1 | 3.913 | 2% | 3.91 | 1.0 |
| Deflate, predictor 3 | 1 | 4.305 | 3% | 4.31 | 1.0 |
| ZSTD, predictor 3 | 1 | 4.277 | 3% | 4.28 | 1.0 |
| LZW, no predictor | 1 | 4.539 | 6% | 4.54 | 1.0 |
| uncompressed | 1 | 3.692 | 3% | 3.71 | 1.0 |
| striped GeoTIFF, uncompressed | 12 | 4.568 | 19% | 4.48 | 1.0 |
| Deflate, predictor 3 | 12 | 8.411 | 21% | 11.83 | 1.4 |
| ZSTD, predictor 3 | 12 | 8.270 | 1% | 11.85 | 1.4 |
| LZW, no predictor | 12 | 8.426 | 1% | 12.44 | 1.5 |
| uncompressed | 12 | 8.361 | 2% | 11.66 | 1.4 |

### The block cache under halos: slope over the Deflate COG

In-process time and block decodes per block (484 blocks of 512 × 512). 1.00 is every block decoded once.

| tile rows | CacheBytes | 1 thread: s | decodes/block | 12 threads: s | decodes/block |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | -1 (no cache) | 24.609 | 33.91 | 2.674 | 33.91 |
| 16 | 4 MiB | 57.350 | 33.91 | 2.285 | 4.06 |
| 16 | 0 (default, 8 block rows) | 1.378 | 1.00 | 1.114 | 1.00 |
| 16 | 1024 MiB | 1.438 | 1.00 | 1.200 | 1.00 |
| 64 | -1 (no cache) | 7.295 | 9.91 | 0.892 | 9.91 |
| 64 | 4 MiB | 7.785 | 9.91 | 0.899 | 4.25 |
| 64 | 0 (default, 8 block rows) | 1.341 | 1.00 | 0.692 | 1.00 |
| 64 | 1024 MiB | 1.467 | 1.00 | 0.745 | 1.00 |
| 256 | -1 (no cache) | 3.254 | 3.91 | 0.572 | 3.91 |
| 256 | 4 MiB | 3.313 | 3.91 | 0.589 | 2.16 |
| 256 | 0 (default, 8 block rows) | 1.391 | 1.00 | 0.485 | 1.00 |
| 256 | 1024 MiB | 1.489 | 1.00 | 0.517 | 1.00 |

### Floors

| case | wall s | spread |
| --- | ---: | ---: |
| cogbench, start and stop | 0.004 | 0% |
| python3: import gdal and numpy, open the COG | 0.105 | 17% |

<!-- summarize.py output end -->

## The cache

A row of this file's blocks decodes to 22 × 1 MiB. When this page was
first written the default cache was 64 MiB, almost three rows: enough for
one worker, which walks down the raster so that the blocks a tile needs
are mostly the ones the previous tile just used, plus the next row. Not
enough for twelve workers on 256-row tiles, which have 3072 rows in
flight, six block rows or 132 MiB: they decoded each block 1.57 times,
and ran 26% slower than with 1 GiB. That page suggested at least
`(workers × TileHeight / blockHeight + 2)` rows of blocks, each
`blocksAcross × blockBytes`, about 180 MiB here.

The default has since become 8 rows of blocks (`cog.DefaultCacheRows`,
between 64 MiB and 1 GiB), 182 MiB for this file, and the table now
shows exactly one decode per block at every tile height on one worker
and on twelve, within 7% of the 1 GiB cache's time or faster.

One setting still breaks it: **a cache smaller than one row of blocks
is no cache.** With 4 MiB, one worker decodes each block as often as
with none: 33.9 times at 16-row tiles. The LRU evicts a block before the
next tile comes back for it. With 12 workers, 4 MiB does help, because
tiles running at the same time share the blocks they overlap. (In this
run, one worker with 16-row tiles took 57.4 s with 4 MiB against 24.6 s
with no cache, at the same 33.9 decodes per block. The first version of
this page had them within 1% of each other, at 53.6 s and 54.0 s. This
run was not repeated to find out why; a second session's GDAL container
ran briefly alongside it at about that point.)

One more observation, not investigated: **gdaldem gets twice as slow on a
COG when `GDAL_NUM_THREADS=12`** (8.4 s against 4.3 s, and 11.8 CPU-seconds
against 4.3). On the striped GeoTIFF it makes no difference. gdaldem
reads a few lines at a time, and parallel decoding of such small requests
seems to cost more than it saves. Every "vs gdaldem" figure uses
gdaldem's faster setting.

## What this does not tell you

- **NEON.** The arm64 kernels are tested bit for bit (under QEMU on the
  Windows host, and on CI's arm64 macOS runner) but never timed. Word,
  the NoData test, has no NEON form and runs scalar there.

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
