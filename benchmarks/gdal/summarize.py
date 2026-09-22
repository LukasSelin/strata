#!/usr/bin/env python3
"""Turn gdalbench.sh's timing lines into the RESULTS.md tables.

Reads lines of the form

    tool=strata op=slope fmt=envi workers=12 run=3 real=0.412 user=3.9 sys=0.6

and prints, per operation, the median wall time of each case, the
throughput it implies, and the speedup over gdaldem. The median of the
timed runs is reported, never the minimum: the fastest run of a
file-backed operation is the one that got luckiest with the cache.

The speedup is measured against the *fastest* gdaldem configuration for
that operation, not the slowest and not the default, so that the number
is what strata gains over GDAL at its best on this machine.

    python summarize.py timings.txt <width> <height>
"""

import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "out/timings.txt"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 11264
H = int(sys.argv[3]) if len(sys.argv) > 3 else W
CELLS = W * H

NAMES = {
    "gdaldem": "gdaldem",
    "gdaldemcache": "gdaldem, 4 GB block cache",
    "strata": "strata chunked",
    "stratascalar": "strata chunked, scalar kernels",
    "stratamem": "strata, whole raster in memory",
    "copy": "gdal_translate, copy only",
    "startup": "process start, nothing else",
}
FMTS = {"envi": "raw float32", "tif": "GeoTIFF", "go": "Go", "none": ""}

runs, order, notes = {}, [], []
for line in open(path):
    line = line.rstrip("\n")
    if line.startswith("#"):
        notes.append(line)
        continue
    if not line.startswith("tool="):
        continue
    f = dict(kv.split("=", 1) for kv in line.split())
    key = (f["op"], f["tool"], f["fmt"], int(f["workers"]))
    if key not in runs:
        runs[key] = []
        order.append(key)
    runs[key].append((float(f["real"]), float(f["user"]), float(f["sys"])))


def med(key, i=0):
    return statistics.median(r[i] for r in runs[key])


def spread(key):
    v = sorted(r[0] for r in runs[key])
    m = statistics.median(v)
    return (v[-1] - v[0]) / m if m else 0.0


def label(op, tool, fmt, w):
    parts = [NAMES[tool]]
    if FMTS[fmt]:
        parts.append(FMTS[fmt])
    if tool in ("strata", "stratascalar"):
        parts.append(f"{w} worker" + ("s" if w != 1 else ""))
    return ", ".join(parts)


def table(keys, baseline):
    print("| case | wall s | M cells/s | CPU s | CPU/wall | spread | vs gdaldem |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in keys:
        real, cpu = med(key), med(key, 1) + med(key, 2)
        rel = f"{baseline / real:.1f}x" if baseline else "-"
        # A process that touches no cells has no throughput.
        rate = "-" if key[1] == "startup" else f"{CELLS / real / 1e6:.0f}"
        print(f"| {label(*key)} | {real:.3f} | {rate} | {cpu:.2f} | "
              f"{cpu / real:.1f} | {spread(key) * 100:.0f}% | {rel} |")
    print()


for n in notes:
    print(n)
per_case = len(next(iter(runs.values())))
print(f"\n{W} x {H} = {CELLS / 1e6:.1f}M cells, {per_case} timed runs per case, "
      f"median reported. Throughput counts every cell, valid or not.\n")

ops = []
for op, *_ in order:
    if op != "none" and op not in ops:
        ops.append(op)

for op in ops:
    keys = [k for k in order if k[0] == op]
    gdal = [k for k in keys if k[1].startswith("gdaldem")]
    best = min((med(k) for k in gdal), default=None)
    which = min(gdal, key=med) if gdal else None
    print(f"### {op}\n")
    if which:
        print(f"Baseline: {label(*which)}, the fastest of "
              f"{len(gdal)} gdaldem configurations.\n")
    table(keys, best)

print("### floors: the same bytes moved, and the process started, "
      "without either tool computing anything\n")
table([k for k in order if k[0] == "none"], None)
