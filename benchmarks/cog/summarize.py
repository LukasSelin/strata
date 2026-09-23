#!/usr/bin/env python3
"""Turn runbench.sh's timing lines into the RESULTS.md tables.

Reads lines of the form

    tool=strata op=read codec=deflate threads=12 tile=256 cache=0 run=3
      real=0.41 user=3.9 sys=0.6 ms=380.2 rpb=1.00

and prints the median of each case's timed runs, never the minimum, with
its spread, (max - min) / median. As in benchmarks/gdal, the medians are
of whole runs, and nothing is dropped.

    python summarize.py timings.txt <width> <height>
"""

import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else "out/timings.txt"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 11264
H = int(sys.argv[3]) if len(sys.argv) > 3 else W
CELLS = W * H
MB = CELLS * 4 / 1e6  # the raster as float32, what every reader hands back

runs, order, notes = {}, [], []
for line in open(path):
    line = line.rstrip("\n")
    if line.startswith("#"):
        notes.append(line)
        continue
    if not line.startswith("tool="):
        continue
    f = dict(kv.split("=", 1) for kv in line.split())
    key = (f["tool"], f["op"], f["codec"], int(f["threads"]), f["tile"], f["cache"])
    if key not in runs:
        runs[key] = []
        order.append(key)
    runs[key].append(f)


def col(key, name):
    v = [r[name] for r in runs[key]]
    return None if "-" in v else [float(x) for x in v]


def med(key, name):
    v = col(key, name)
    return statistics.median(v) if v else None


def spread(key, name):
    v = sorted(col(key, name))
    m = statistics.median(v)
    return (v[-1] - v[0]) / m if m else 0.0


def cpu(key):
    return med(key, "user") + med(key, "sys")


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


CODECS = {"deflate": "Deflate, predictor 3", "zstd": "ZSTD, predictor 3",
          "lzw": "LZW, no predictor", "none": "uncompressed",
          "tif": "striped GeoTIFF, uncompressed", "raw": "raw float32"}


def threads(n):
    return f"{n} thread" + ("s" if n != 1 else "")


# The notes (files, versions, the correctness checks) stay in the timings
# file: printed into Markdown, a leading # would make each a heading.
per_case = len(runs[order[0]])
per_cache = min((len(v) for k, v in runs.items() if k[0] == "cache"), default=per_case)
print(f"\n{W} x {H} = {CELLS / 1e6:.1f}M cells, {MB:.0f} MB as float32. "
      f"{per_case} timed runs per case ({per_cache} in the cache sweep) after a "
      f"warm-up, median reported.\n")

# --- 1. pure read -------------------------------------------------------

read = [k for k in order if k[1] == "read"]
codecs = list(dict.fromkeys(k[2] for k in read))
nthreads = sorted({k[3] for k in read})


def find(tool, op, codec, n, tile=None, cache=None):
    for k in order:
        if k[:4] == (tool, op, codec, n) and (tile is None or k[4] == tile) \
                and (cache is None or k[5] == cache):
            return k
    return None


print("### Pure read: the whole raster, decoded to float32\n")
print("In-process time, from the first strip to the last, of full-width "
      "strips of 256 rows into a reused buffer. MB/s is float32 output. "
      "GDAL is `ReadAsArray` from Python with `GDAL_NUM_THREADS`; strata is "
      "`Source.ReadWindow` on that many goroutines. *before* is the reader "
      "as PR #32 first proposed it (9f658d3).\n")
print("| compression | threads | GDAL s | GDAL MB/s | strata before s | strata s | "
      "strata MB/s | strata vs GDAL | before → after |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for c in codecs:
    for n in nthreads:
        g, b, s = (find(t, "read", c, n) for t in ("gdalpy", "before", "strata"))
        gm, bm, sm = (med(k, "ms") / 1000 for k in (g, b, s))
        print(f"| {CODECS[c]} | {n} | {gm:.3f} | {MB / gm:.0f} | {bm:.3f} | {sm:.3f} | "
              f"{MB / sm:.0f} | {gm / sm:.2f}x | {bm / sm:.2f}x |")
print()

print("Every read case in full, with spreads and whole-process numbers. "
      "`gdal_translate -of MEM` has no in-process timer, so it is compared "
      "by wall time only; its system time is the kernel faulting in a "
      "fresh 507 MB dataset, which the strip readers do not pay.\n")
print("| case | in-process s | spread | wall s | CPU s | CPU/wall | reads per block |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
NAMES = {"gdalpy": "GDAL, ReadAsArray strips", "gdalmem": "GDAL, gdal_translate -of MEM",
         "before": "strata before", "strata": "strata"}
for c in codecs:
    for n in nthreads:
        for t in ("gdalpy", "gdalmem", "before", "strata"):
            k = find(t, "read", c, n)
            if not k:
                continue
            ms = med(k, "ms")
            sp = spread(k, "ms") if ms else spread(k, "real")
            real = med(k, "real")
            print(f"| {CODECS[c]}, {NAMES[t]}, {threads(n)} | "
                  f"{fmt(ms and ms / 1000, '.3f')} | {sp * 100:.0f}% | {real:.3f} | "
                  f"{cpu(k):.2f} | {cpu(k) / real:.1f} | {fmt(med(k, 'rpb'), '.2f')} |")
print()

# --- 2. slope, end to end -----------------------------------------------

slope = [k for k in order if k[1] == "slope" and k[0] != "cache"]


def best_gdaldem(codec):
    return min(med(k, "real") for k in slope if k[0] == "gdaldem" and k[2] == codec)

print("### End to end: slope, file to file\n")
print("Wall time of the whole process, as in benchmarks/gdal. gdaldem writes "
      "an uncompressed GeoTIFF, strata a raw float32 file. `gdaldem s` is "
      "gdaldem's fastest time on that input over both `GDAL_NUM_THREADS` "
      "settings (its compute is single-threaded either way), and `vs "
      "gdaldem` is against it. `vs raw` is strata's time on the raw file "
      "over its time on this input, at the same worker count, so 1.00 "
      "means the format cost nothing.\n")
print("| input | threads | gdaldem s | strata before s | strata s | spread | "
      "M cells/s | vs gdaldem | vs raw |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for n in sorted({k[3] for k in slope}):
    raw = find("raw", "slope", "raw", n)
    rw = med(raw, "real")
    gw = best_gdaldem("tif")
    print(f"| {CODECS['tif']} (gdaldem) / {CODECS['raw']} (strata) | {n} | "
          f"{gw:.3f} | - | {rw:.3f} | {spread(raw, 'real') * 100:.0f}% | "
          f"{CELLS / rw / 1e6:.0f} | {gw / rw:.2f}x | 1.00 |")
    for c in codecs:
        b, s = (find(t, "slope", c, n) for t in ("before", "strata"))
        gw, bw, sw = best_gdaldem(c), med(b, "real"), med(s, "real")
        print(f"| {CODECS[c]} COG | {n} | {gw:.3f} | {bw:.3f} | {sw:.3f} | "
              f"{spread(s, 'real') * 100:.0f}% | {CELLS / sw / 1e6:.0f} | {gw / sw:.2f}x | "
              f"{rw / sw:.2f} |")
print()

print("gdaldem's own spreads and CPU, which the table above leaves out:\n")
print("| gdaldem on | threads | wall s | spread | CPU s | CPU/wall |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for k in slope:
    if k[0] == "gdaldem":
        print(f"| {CODECS[k[2]]} | {k[3]} | {med(k, 'real'):.3f} | "
              f"{spread(k, 'real') * 100:.0f}% | {cpu(k):.2f} | {cpu(k) / med(k, 'real'):.1f} |")
print()

# --- 3. the cache -------------------------------------------------------

cache = [k for k in order if k[0] == "cache"]
if cache:
    print("### The block cache under halos: slope over the Deflate COG\n")
    print("In-process time and block decodes per block (484 blocks of 512 × 512). "
          "1.00 is every block decoded once.\n")
    ns = sorted({k[3] for k in cache})
    head = " | ".join(f"{threads(n)}: s | decodes/block" for n in ns)
    print(f"| tile rows | CacheBytes | {head} |")
    print("| ---: | --- |" + " ---: | ---: |" * len(ns))
    CN = {"-1": "-1 (no cache)", "0": "0 (default, 64 MiB)"}
    for tile in dict.fromkeys(k[4] for k in cache):
        for cb in dict.fromkeys(k[5] for k in cache if k[4] == tile):
            label = CN.get(cb, f"{int(cb) >> 20} MiB")
            cells = []
            for n in ns:
                k = find("cache", "slope", "deflate", n, tile, cb)
                cells.append(f"{med(k, 'ms') / 1000:.3f} | {med(k, 'rpb'):.2f}")
            print(f"| {tile} | {label} | " + " | ".join(cells) + " |")
    print()

# --- floors -------------------------------------------------------------

print("### Floors\n")
print("| case | wall s | spread |")
print("| --- | ---: | ---: |")
for k in order:
    if k[1] == "none":
        name = {"startup-go": "cogbench, start and stop",
                "startup-py": "python3: import gdal and numpy, open the COG"}[k[0]]
        print(f"| {name} | {med(k, 'real'):.3f} | {spread(k, 'real') * 100:.0f}% |")
print()
