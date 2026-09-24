#!/usr/bin/env python3
"""Check grasscompare.py's bounds: break strata's results on purpose and
confirm that the comparison against GRASS r.param.scale goes red.

The bounds in grasscompare.py are derived to admit strata's float32
rounding and GRASS's float64 rounding and nothing more. Bounds that
wide enough to pass real data could in principle be wide enough to pass
a real defect; this is the test that they are not. Each sabotage copies
one radius's results, damages one strata product the way a plausible
bug would, reruns the comparison on that radius and requires the named
check to fail.

Usage:  python grasssabotage.py [dir] [width] [height] [nodata] [cell] [radii]

Same arguments as grasscompare.py; run it after grasscheck.sh. Every
radius in radii is sabotaged in turn.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ARGS = sys.argv[1:]
SRC = ARGS[0] if ARGS else "out-grass"
REST = ARGS[1:5]
RADII = [int(v) for v in (ARGS[5] if len(ARGS) > 5 else "1,4,8").split(",")]
HERE = os.path.dirname(os.path.abspath(__file__))
FILL = np.float32(-9999)


def scale(k):
    def f(v, has):
        v[has] *= np.float32(k)
    return f


def rotate(deg):
    def f(v, has):
        v[has] = np.mod(v[has] + np.float32(deg), np.float32(360))
    return f


def trig(v, has):
    # A bearing written as a mathematical angle, counter-clockwise from east.
    v[has] = np.mod(np.float32(90) - v[has], np.float32(360))


def shift(v, has):
    # The window one column off: every cell takes its western neighbour's value.
    v[:, 1:] = v[:, :-1].copy()


def drop_one(v, has):
    cells = np.flatnonzero(has)
    v.flat[cells[len(cells) // 2]] = FILL


# (product, sabotage, description, the check that must fail)
SABOTAGES = [
    ("slope", scale(1.0005), "slope x 1.0005", "slope == GRASS slope"),
    ("slope", scale(1.00002), "slope x 1.00002", "slope == GRASS slope"),
    ("slope", shift, "slope one column off", "slope == GRASS slope"),
    ("slope", drop_one, "one slope cell invalid", "same cells carry data as GRASS"),
    ("aspect", rotate(0.01), "aspect + 0.01 deg", "aspect == (270 + GRASS aspect)"),
    ("aspect", trig, "aspect ccw from east", "aspect == (270 + GRASS aspect)"),
    ("profile", scale(1.0005), "profile x 1.0005", "profile == GRASS profc"),
    ("plan", scale(-1), "plan x -1 (GRASS's sign)", "plan == GRASS -planc"),
    ("plan", scale(1.0005), "plan x 1.0005", "plan == GRASS -planc"),
    ("mean", scale(1.0005), "mean x 1.0005", "mean == GRASS"),
    ("mean", scale(-1), "mean x -1", "mean == GRASS"),
]


def place(src, dst):
    """Hard-link src to dst where the file system allows, else copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


missed = total = 0
for r in RADII:
    k = 2 * r + 1
    files = ["dem.raw"] + [f"grass-{m}-{k}.raw" for m in ("slope", "aspect", "profc", "planc", "crosc", "minic", "maxic")]
    files += [f"strata-{m}-{r}.raw" for m in ("slope", "aspect", "profile", "plan", "mean")]
    for product, fn, what, check in SABOTAGES:
        total += 1
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            for f in files:
                place(os.path.join(SRC, f), os.path.join(d, f))
            p = os.path.join(d, f"strata-{product}-{r}.raw")
            v = np.fromfile(p, dtype="<f4")  # a fresh copy: the link is replaced below
            os.remove(p)
            W = int(REST[0]) if REST else 4096
            v = v.reshape(-1, W)
            fn(v, v != FILL)
            v.tofile(p)
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "grasscompare.py"), d, *REST, str(r)],
                capture_output=True, text=True,
            ).stdout
        line = next((l for l in out.splitlines() if f"r={r}: {check}" in l), "")
        caught = line.startswith("FAIL")
        missed += not caught
        detail = line[6:].split("  ", 1)[-1].strip() if line else "(check not found)"
        print(f"{'caught' if caught else 'MISSED'}  r={r} {what:<26} {detail}", flush=True)

print(f"\n{total - missed}/{total} sabotages caught")
sys.exit(1 if missed else 0)
