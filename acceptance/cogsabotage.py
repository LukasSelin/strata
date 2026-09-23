#!/usr/bin/env python3
"""Check cogcompare.py: break strata's reading on purpose and confirm the
comparison goes red.

A comparison that passes 98 files proves nothing if it would also pass a
broken reader. Each sabotage below copies one file's results, plants a
defect a GeoTIFF reader could plausibly have, and reruns cogcompare.py on
that file alone; it must fail. Run it after cogcheck.sh.

Usage:  python cogsabotage.py [dir]
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "out-cog"
HERE = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(os.path.join(SRC, "manifest.json")))
strata = json.load(open(os.path.join(SRC, "strata.json")))


def shift_block(d, name):
    """A block placed one cell off: its first 128-cell block row moves right."""
    p = os.path.join(d, f"{name}.b0.l0.strata.f32")
    v = np.fromfile(p, "<f4").reshape(517, 701)
    v[0:128, 1:128] = v[0:128, 0:127].copy()
    v.tofile(p)


def one_ulp(d, name):
    """A rounding slip: one valid cell one float32 ulp off."""
    p, m = (os.path.join(d, f"{name}.b0.l0.strata.{e}") for e in ("f32", "mask"))
    v, valid = np.fromfile(p, "<u4"), np.fromfile(m, np.uint8)
    v[int(np.argmax(valid > 0)) + 1000] += 1
    v.tofile(p)


def nodata_missed(d, name):
    """NoData compared after the float32 cast, or not at all: one NoData cell reads valid."""
    m = os.path.join(d, f"{name}.b0.l0.strata.mask")
    valid = np.fromfile(m, np.uint8)
    valid[int(np.argmax(valid == 0))] = 255
    valid.tofile(m)


def clamp_overflow(d, name):
    """float64 beyond float32's range held at MaxFloat32, not Inf (this reader's first draft)."""
    p = os.path.join(d, f"{name}.b0.l0.strata.f32")
    v = np.fromfile(p, "<f4")
    v[np.isinf(v)] = np.copysign(np.finfo(np.float32).max, v[np.isinf(v)])
    v.tofile(p)


def ignore_point(d, name):
    """PixelIsPoint ignored: the origin half a cell off."""
    j = os.path.join(d, "strata.json")
    r = json.load(open(j))
    gt = r[0]["levels"][0]["GT"]
    gt[0] += 0.5 * gt[1]
    gt[3] += 0.5 * gt[5]
    json.dump(r, open(j, "w"))


def swap_bands(d, name):
    """Chunky samples read with the wrong stride: bands 1 and 2 swapped."""
    a, b = (os.path.join(d, f"{name}.b{k}.l0.strata.f32") for k in (1, 2))
    tmp = a + ".tmp"
    os.replace(a, tmp)
    os.replace(b, a)
    os.replace(tmp, b)


def drop_overview(d, name):
    """An overview skipped, say a reduced image mistaken for a mask."""
    j = os.path.join(d, "strata.json")
    r = json.load(open(j))
    r[0]["levels"].pop()
    json.dump(r, open(j, "w"))


def sparse_as_zero(d, name):
    """Sparse blocks read as valid zeros although the file has NoData."""
    p, m = (os.path.join(d, f"{name}.b0.l0.strata.{e}") for e in ("f32", "mask"))
    v, valid = np.fromfile(p, "<f4").reshape(517, 701), np.fromfile(m, np.uint8).reshape(517, 701)
    v[384:512, 0:128], valid[384:512, 0:128] = 0, 255  # a block inside the empty corner
    v.tofile(p)
    valid.tofile(m)


SABOTAGES = [
    ("block one cell off", "cog-Float32-DEFLATE-FLOATING_POINT", shift_block),
    ("one ulp", "cog-Float64-ZSTD-FLOATING_POINT", one_ulp),
    ("one NoData cell valid", "cog-Float32-LZW-STANDARD", nodata_missed),
    ("overflow clamped", "gtiff-Float64-huge", clamp_overflow),
    ("PixelIsPoint ignored", "gtiff-Float32-point", ignore_point),
    ("bands swapped", "gtiff-Int16-strips-PIXEL", swap_bands),
    ("overview dropped", "cog-Int16-ZSTD-STANDARD", drop_overview),
    ("sparse block valid", "cog-UInt16-DEFLATE-STANDARD", sparse_as_zero),
]

missed = 0
for label, name, sabotage in SABOTAGES:
    with tempfile.TemporaryDirectory() as d:
        for f in os.listdir(SRC):
            if f.startswith(name + "."):
                shutil.copy(os.path.join(SRC, f), d)
        m = dict(manifest, files=[e for e in manifest["files"] if e["name"] == name])
        json.dump(m, open(os.path.join(d, "manifest.json"), "w"))
        json.dump([r for r in strata if r["name"] == name], open(os.path.join(d, "strata.json"), "w"))
        clean = subprocess.run([sys.executable, os.path.join(HERE, "cogcompare.py"), d],
                               capture_output=True, text=True)
        sabotage(d, name)
        out = subprocess.run([sys.executable, os.path.join(HERE, "cogcompare.py"), d],
                             capture_output=True, text=True)
        caught = clean.returncode == 0 and out.returncode != 0
        missed += not caught
        why = next((l.strip() for l in out.stdout.splitlines() if l.startswith("        ")), "")
        print(f"{'caught' if caught else 'MISSED'}  {label:<24} {name:<38} {why[:90]}")

print(f"\n{len(SABOTAGES) - missed}/{len(SABOTAGES)} sabotages caught")
sys.exit(1 if missed else 0)
