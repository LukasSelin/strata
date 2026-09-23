#!/usr/bin/env python3
"""How far apart strata's and GDAL's results for one operation are.

A speed comparison means little unless both tools computed the same
thing, so runsuite.sh runs this on every operation's output, and
summarize.py puts the result next to the timings. It measures and does
not judge: the verdicts on whether strata is *right* live in acceptance/,
which checks against definitions and against GDAL with derived
tolerances. This says how close the two answers are on this raster.

    python3 agree.py raster <op> <strata.raw> <gdal file> <w> <h> <border>
    python3 agree.py reduce <op> "<strata result>" "<gdal result>"

raster: strata's raw float32 with -9999 under invalid cells, against any
file GDAL reads (its NoData, and NaN, are invalid). The <border> cells
along each edge are left out, where the tools define edges differently.
Prints one line:

    agree op=<op> kind=raster compared=<cells> of=<interior cells>
      validity_mismatch=<cells> maxdiff=<abs> meandiff=<abs>
      range=<gdal's value range> worst=<x>,<y>:<strata>/<gdal>

reduce: the key=value results each tool printed; prints the absolute and
relative difference of every key both have.
"""

import math
import sys

import numpy as np


def raster(op, spath, gpath, w, h, b):
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(gpath)
    if (ds.RasterXSize, ds.RasterYSize) != (w, h):
        sys.exit(f"agree: {op}: GDAL wrote {ds.RasterXSize}x{ds.RasterYSize}, strata {w}x{h}")
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    strata = np.memmap(spath, "<f4", mode="r", shape=(h, w))
    # In blocks of rows, so that a 2x upsample's half-billion cells fit.
    n = mismatch = 0
    total = maxd = 0.0
    lo, hi = math.inf, -math.inf
    worst = (0, 0, 0.0, 0.0)
    step = 512
    for y0 in range(b, h - b, step):
        rows = min(step, h - b - y0)
        s = np.asarray(strata[y0:y0 + rows, b:w - b], np.float64)
        g = band.ReadAsArray(b, y0, w - 2 * b, rows).astype(np.float64)
        sv = s != -9999
        gv = ~np.isnan(g)
        if nd is not None:
            gv &= g != nd
        both = sv & gv
        mismatch += int((sv != gv).sum())
        k = int(both.sum())
        if k == 0:
            continue
        n += k
        d = np.where(both, np.abs(s - g), -1.0)
        total += float(d[both].sum())
        gb = g[both]
        lo, hi = min(lo, float(gb.min())), max(hi, float(gb.max()))
        i = int(np.argmax(d))
        if d.flat[i] > maxd or worst == (0, 0, 0.0, 0.0):
            y, x = divmod(i, d.shape[1])
            maxd = max(maxd, float(d.flat[i]))
            worst = (x + b, y + y0, s[y, x], g[y, x])
    of = (w - 2 * b) * (h - 2 * b)
    if n == 0:
        print(f"agree op={op} kind=raster compared=0 of={of} validity_mismatch={mismatch}")
        return
    print(f"agree op={op} kind=raster compared={n} of={of} validity_mismatch={mismatch} "
          f"maxdiff={maxd:.3g} meandiff={total / n:.3g} range={hi - lo:.6g} "
          f"worst={worst[0]},{worst[1]}:{worst[2]:.9g}/{worst[3]:.9g}")


def kv(line):
    return dict(p.split("=", 1) for p in line.split() if "=" in p)


def reduce(op, sline, gline):
    s, g = kv(sline), kv(gline)
    parts = []
    for k in s:
        if k in g:
            a, c = float(s[k]), float(g[k])
            rel = abs(a - c) / abs(c) if c else abs(a - c)
            parts.append(f"{k}={a:.9g}/{c:.9g}:rel={rel:.2g}")
    print(f"agree op={op} kind=reduce " + " ".join(parts))


if __name__ == "__main__":
    kind, op = sys.argv[1], sys.argv[2]
    if kind == "raster":
        raster(op, sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    else:
        reduce(op, sys.argv[3], sys.argv[4])
