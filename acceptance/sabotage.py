#!/usr/bin/env python3
"""Check the checker: break the results on purpose, one bug at a time,
and confirm check.py notices.

A green check.py only means something if it would go red when the
library is wrong. Each mutation below is a plausible defect - a
transposed kernel, a tile seam, a cell size in the wrong axis, an
off-by-one count - applied to a copy of the results. Any mutation that
check.py fails to catch is a hole in check.py.

Usage:  python sabotage.py [dir]     (default: out)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "out"
MAN = json.load(open(os.path.join(SRC, "manifest.json")))
W, H = MAN["width"], MAN["height"]


def read(d, name):
    return np.fromfile(os.path.join(d, name), dtype="<f4").reshape(H, W)


def write(d, name, a):
    a.astype("<f4").tofile(os.path.join(d, name))


# Each mutation takes the copied directory and introduces one defect.


def drift(d):
    """Every slope 0.05% too large - a units or constant-factor slip."""
    for f in os.listdir(d):
        if "slope_deg" in f and f.endswith(".f32"):
            write(d, f, read(d, f) * 1.0005)


def transpose_kernel(d):
    """dx and dy swapped - Horn's kernel applied along the wrong axis."""
    for stem in ("hill", "plane", "noisy"):
        for form in ("plain", "tiled", "chunked"):
            dx = read(d, f"{stem}-gradient_dx-{form}.f32")
            dy = read(d, f"{stem}-gradient_dy-{form}.f32")
            write(d, f"{stem}-gradient_dx-{form}.f32", dy)
            write(d, f"{stem}-gradient_dy-{form}.f32", dx)


def mirror_aspect(d):
    """Aspect measured the other way round the compass."""
    for f in os.listdir(d):
        if "aspect" in f and f.endswith(".f32"):
            a = read(d, f)
            flat = a == -1
            a = (360.0 - a) % 360.0
            a[flat] = -1
            write(d, f, a)


def tile_seam(d):
    """One tiled result wrong in a single cell, as a seam bug would be."""
    a = read(d, "hill-hillshade-tiled.f32")
    a[23, 37] = 0.0
    write(d, "hill-hillshade-tiled.f32", a)


def shifted(d):
    """A chunked result off by one row - a tile origin mistake."""
    a = read(d, "hill-slope_deg-chunked.f32")
    write(d, "hill-slope_deg-chunked.f32", np.roll(a, 1, axis=0))


def leaky_nodata(d):
    """One NoData cell treated as data: its validity bit set again."""
    p = os.path.join(d, "noisy-slope_deg-plain.mask.u8")
    m = np.fromfile(p, dtype=np.uint8).reshape(H, W)
    bad = np.argwhere(m == 0)
    y, x = bad[len(bad) // 2]
    m[y, x] = 1
    m.tofile(p)


def miscount(d):
    """A reduction that counts one cell too many."""
    p = os.path.join(d, "manifest.json")
    man = json.load(open(p))
    for s in man["scalars"]:
        if s["name"] == "noisy-count-chunked":
            s["count"] += 1
    json.dump(man, open(p, "w"))


def wrong_extreme(d):
    """MinMax reporting the second-largest value."""
    p = os.path.join(d, "manifest.json")
    man = json.load(open(p))
    for s in man["scalars"]:
        if s["name"] == "hill-minmax-plain":
            s["max"] = str(np.float32(float(s["max"])) * np.float32(0.9999))
    json.dump(man, open(p, "w"))


MUTATIONS = [
    ("slope 0.05% too large", drift),
    ("dx and dy swapped", transpose_kernel),
    ("aspect mirrored", mirror_aspect),
    ("one bad cell on a tile seam", tile_seam),
    ("chunked result off by one row", shifted),
    ("one NoData cell leaking in", leaky_nodata),
    ("count one too many", miscount),
    ("max slightly wrong", wrong_extreme),
]


def run(d):
    p = subprocess.run(
        [sys.executable, "check.py", d], capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
    )
    caught = p.returncode != 0
    fails = [ln for ln in p.stdout.splitlines() if ln.startswith("FAIL")]
    return caught, len(fails)


missed = 0
print(f"{'injected defect':<32}  {'caught':<8} failing checks")
print("-" * 62)
with tempfile.TemporaryDirectory() as tmp:
    for name, mutate in MUTATIONS:
        d = os.path.join(tmp, name.replace(" ", "_"))
        shutil.copytree(SRC, d)
        mutate(d)
        caught, n = run(d)
        missed += not caught
        print(f"{name:<32}  {'yes' if caught else 'NO':<8} {n}")

print()
if missed:
    print(f"{missed} defect(s) went unnoticed: check.py has a blind spot there.")
    sys.exit(1)
print(f"all {len(MUTATIONS)} injected defects were caught")
