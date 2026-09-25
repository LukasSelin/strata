#!/usr/bin/env python3
"""Check gdalcompare.py's float32-rounding bound: drift strata's slope on
purpose and confirm the check goes red. Then nudge one ruggedness cell by
a single ulp and confirm the bit-for-bit ruggedness check goes red too.

The bound in gdalcompare.py's section 5 is derived to admit two correct
float32 roundings of Horn's slope and nothing more. A bound that wide
enough to pass nearly flat, high terrain could in principle be wide
enough to pass a real defect too; this is the test that it is not. Each
run copies the results, scales every valid strata slope by a constant
factor - a units or constant-factor slip - and reruns the comparison.

Usage:  python gdalsabotage.py [dir] [width] [height] [nodata] [cell]

Same arguments as gdalcompare.py; run it after gdalcheck.sh.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ARGS = sys.argv[1:]
SRC = ARGS[0] if ARGS else "out-gdal"
HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = "strata and gdaldem within float32 rounding"

# 1.0005 is sabotage.py's drift. 1.00002 is small enough that the
# absolute "slope == gdaldem slope" check (1e-3 deg) misses it on slopes
# below 50 deg, so only the derived bound stands between it and a pass.
FACTORS = (1.0005, 1.00002)

FILES = ("dem.raw", "gdal-slope.raw", "gdal-aspect.raw", "gdal-hillshade.raw",
         "strata-slope.raw", "strata-aspect.raw", "strata-hillshade.raw",
         "strata-northness.raw", "strata-eastness.raw", "strata-northness_unweighted.raw",
         "weight.raw", "gdal-wslope.raw", "strata-wslope.raw") + tuple(
    f"{tool}-{op}.raw" for tool in ("gdal", "strata") for op in ("tri", "triwilson", "tpi", "roughness"))


def compare(d):
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "gdalcompare.py"), d, *ARGS[1:]],
        capture_output=True, text=True,
    ).stdout


missed = 0
for k in FACTORS:
    with tempfile.TemporaryDirectory() as d:
        for f in FILES:
            shutil.copy(os.path.join(SRC, f), d)
        p = os.path.join(d, "strata-slope.raw")
        s = np.fromfile(p, dtype="<f4")
        s[s != -9999] *= np.float32(k)
        s.tofile(p)
        out = compare(d)
        line = next((l for l in out.splitlines() if CHECK in l), "")
        caught = line.startswith("FAIL")
        missed += not caught
        print(f"{'caught' if caught else 'MISSED'}  slope x {k:<8}  {line[6:].strip()}")

# Ruggedness is compared bit for bit, so one ulp in one cell must fail it.
RUGGEDNESS = ("tri", "triwilson", "tpi", "roughness")
for op in RUGGEDNESS:
    with tempfile.TemporaryDirectory() as d:
        for f in FILES:
            shutil.copy(os.path.join(SRC, f), d)
        p = os.path.join(d, f"strata-{op}.raw")
        s = np.fromfile(p, dtype="<f4")
        cells = np.flatnonzero(s != -9999)
        i = cells[len(cells) // 2]
        s[i] = np.nextafter(s[i], np.float32(np.inf))
        s.tofile(p)
        line = next((l for l in compare(d).splitlines() if f"{op} == gdaldem {op}" in l), "")
        caught = line.startswith("FAIL")
        missed += not caught
        print(f"{'caught' if caught else 'MISSED'}  {op + ' +1 ulp':<17} {line[6:].strip()}")

# Northness and eastness: a sign, a swap and the wrong weighting must each
# fail the comparison with gdaldem's slope and aspect.
ORIENTATION = (
    ("eastness sign flipped", "eastness", lambda d: -rd(d, "eastness")),
    ("northness is eastness", "northness", lambda d: rd(d, "eastness")),
    ("northness unweighted", "northness", lambda d: rd(d, "northness_unweighted")),
)


def rd(d, op):
    return np.fromfile(os.path.join(d, f"strata-{op}.raw"), dtype="<f4")


for label, op, bad in ORIENTATION:
    with tempfile.TemporaryDirectory() as d:
        for f in FILES:
            shutil.copy(os.path.join(SRC, f), d)
        s = bad(d)
        s[rd(d, op) == -9999] = -9999
        s.astype("<f4").tofile(os.path.join(d, f"strata-{op}.raw"))
        line = next((l for l in compare(d).splitlines() if l[6:].startswith(f"{op} == ")), "")
        caught = line.startswith("FAIL")
        missed += not caught
        print(f"{'caught' if caught else 'MISSED'}  {label:<22} {line[6:].strip()}")

total = len(FACTORS) + len(RUGGEDNESS) + len(ORIENTATION)
print(f"\n{total - missed}/{total} sabotages caught")
sys.exit(1 if missed else 0)
