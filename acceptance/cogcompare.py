#!/usr/bin/env python3
"""Difference strata's cog reader against GDAL on the files cogmake.py wrote.

For every file, GDAL's reading (manifest.json and the .f32/.mask files
cogmake.py wrote through GDAL's API) is the truth, and strata's reading
(strata.json and the .strata.* files cog/main.go wrote) must match it:

  - strata opens the file, with the same bands, number of levels and
    level sizes;
  - every level's geotransform equals GDAL's, to 1e-9 of a cell;
  - the EPSG code equals GDAL's (that of the horizontal CRS);
  - strata reports validity (Masked) exactly when GDAL's mask is not
    all-valid: GDAL has a NoData value that the type can hold, or a mask;
  - every band of every level has the same validity mask, cell for cell;
  - every valid cell holds the same float32 bits (any NaN matching any
    NaN), since both sides convert the same sample to float32 and a
    reader has no rounding latitude at all.

The comparison is exact, so there is no tolerance to derive.

For a corpus of files other software wrote (cogcorpus.py), three more
outcomes are not failures:

  - REFUSED: GDAL reads the file and cog refuses it with a "not
    supported" error, as it must for what it does not read (JPEG, WebP,
    LERC, 16-bit floats, sub-byte samples, rotated grids, ...);
  - BOTH-REFUSE: GDAL cannot read the file either, and cog refuses it;
  - ALPHA: GDAL derives its mask from an alpha band, which cog reads as a
    band like any other, so validity is not compared; every cell's value
    still is.

and a file whose CRS GDAL identifies as an EPSG code and cog leaves
empty (cog names a code only when the GeoKeys do not redefine it) is
counted as NO-CRS but not failed. A code that differs from GDAL's fails.

Any other error, and any other difference, is a failure. It prints one
line per file with the cells compared, writes each file's outcome to
results.json, and exits non-zero on any failure.

Usage:  python3 cogcompare.py [dir]
"""

import json
import os
import re
import sys

import numpy as np

GMF_ALL_VALID = 0x01
GMF_ALPHA = 0x04
MAX_BANDS = 16  # as cogtruth.py
REFUSAL = re.compile(r"(is|are) not supported")

D = sys.argv[1] if len(sys.argv) > 1 else "out-cog"
gdal_side = json.load(open(os.path.join(D, "manifest.json")))
strata_side = {r["name"]: r for r in json.load(open(os.path.join(D, "strata.json")))}


def load(path, dtype):
    return np.fromfile(os.path.join(D, path), dtype=dtype)


failed = 0
cells_total = invalid_total = 0
results = []
counts = {}
for g in gdal_side["files"]:
    name = g["name"]
    s = strata_side.get(name)
    problems = []
    cells = invalid = 0
    status, note = "IDENTICAL", ""
    crs_unidentified = None
    alpha = bool(g.get("mask_flags", 0) & GMF_ALPHA)
    if s is None:
        problems.append("strata did not read it")
    elif g.get("gdal_error"):
        if s.get("error"):
            status, note = "BOTH-REFUSE", f"GDAL: {g['gdal_error']} | cog: {s['error']}"
        else:
            problems.append(f"GDAL cannot read it ({g['gdal_error']}), but cog read it")
    elif s.get("error"):
        if REFUSAL.search(s["error"]):
            status, note = "REFUSED", s["error"]
        else:
            problems.append(f"strata refused it: {s['error']}")
    else:
        if alpha:
            status, note = "ALPHA", "GDAL's mask is its alpha band, which cog reads as a band"
        if s["bands"] != g["bands"]:
            problems.append(f"{s['bands']} bands, GDAL {g['bands']}")
        if len(s["levels"]) != len(g["levels"]):
            problems.append(f"{len(s['levels'])} levels, GDAL {len(g['levels'])}")
        want_epsg = f"EPSG:{g['epsg']}" if g["epsg"] else ""
        if s["epsg"] == "" and want_epsg:
            crs_unidentified = want_epsg
        elif s["epsg"] != want_epsg:
            problems.append(f"CRS {s['epsg']!r}, GDAL {want_epsg!r}")
        # Masked exactly when GDAL's mask is not all-valid, apart from an
        # alpha band's; files recorded before mask flags were, by NoData.
        if "mask_flags" in g:
            want_masked = not g["mask_flags"] & GMF_ALL_VALID and not alpha
        else:
            want_masked = g["nodata"] is not None
        if s["masked"] != want_masked:
            problems.append(f"Masked {s['masked']}, GDAL mask flags {g.get('mask_flags')}, NoData {g['nodata']}")
        for lvl, (sl, gl) in enumerate(zip(s["levels"], g["levels"])):
            if (sl["W"], sl["H"]) != (gl["w"], gl["h"]):
                problems.append(f"level {lvl}: {sl['W']}x{sl['H']}, GDAL {gl['w']}x{gl['h']}")
                continue
            if "win" in gl and sl["Win"] != gl["win"]:
                problems.append(f"level {lvl}: window {sl['Win']}, GDAL {gl['win']}")
                continue
            ww = gl.get("win", [0, 0, gl["w"], gl["h"]])[2]
            sgt, ggt = np.array(sl["GT"]), np.array(gl["gt"])
            cell = max(abs(ggt[1]), abs(ggt[5]))
            if np.max(np.abs(sgt - ggt)) > 1e-9 * cell:
                problems.append(f"level {lvl}: geotransform {sgt.tolist()}, GDAL {ggt.tolist()}")
            for b in range(min(g["bands"], s["bands"], MAX_BANDS)):
                stem = f"{name}.b{b}.l{lvl}"
                gv, sv = load(stem + ".f32", "<u4"), load(stem + ".strata.f32", "<u4")
                gm, sm = load(stem + ".mask", np.uint8) > 0, load(stem + ".strata.mask", np.uint8) > 0
                if gv.size != sv.size or gm.size != sm.size:
                    problems.append(f"{stem}: {sv.size} cells, GDAL {gv.size}")
                    continue
                if alpha:
                    gm = np.ones_like(sm)  # compare the value of every cell
                if (gm != sm).any():
                    k = int(np.argmax(gm != sm))
                    problems.append(f"{stem}: {int((gm != sm).sum())} cells differ in validity, "
                                    f"first at ({k % ww}, {k // ww})")
                gf, sf = gv.view("<f4"), sv.view("<f4")
                both_nan = np.isnan(gf) & np.isnan(sf)
                diff = gm & sm & (gv != sv) & ~both_nan
                if diff.any():
                    k = int(np.argmax(diff))
                    problems.append(f"{stem}: {int(diff.sum())} valid cells differ, first at "
                                    f"({k % ww}, {k // ww}): {sf[k]!r}, GDAL {gf[k]!r}")
                cells += gv.size
                invalid += int((~gm).sum())
    cells_total += cells
    invalid_total += invalid
    if problems:
        failed += 1
        status, note = "FAIL", "; ".join(problems[:6])
        print(f"FAIL  {name}")
        for p in problems[:6]:
            print(f"        {p}")
    elif status in ("IDENTICAL", "ALPHA"):
        print(f"PASS  {name:<48} {len(g['levels'])} levels x {g['bands']} bands, "
              f"{cells:>9,} cells ({invalid:,} NoData) identical" + (" (alpha not applied)" if alpha else "")
              + (f" (CRS: GDAL identifies {crs_unidentified}, cog none)" if crs_unidentified else ""))
    else:
        print(f"{status:<5} {name:<48} {note[:120]}")
    counts[status] = counts.get(status, 0) + 1
    if crs_unidentified and status != "FAIL":
        counts["NO-CRS"] = counts.get("NO-CRS", 0) + 1
    results.append({"name": name, "path": g.get("path"), "status": status, "note": note,
                    "crs_unidentified": crs_unidentified,
                    "cells": cells, "nodata_cells": invalid, "features": g.get("features"),
                    "software": g.get("software")})

with open(os.path.join(D, "results.json"), "w") as f:
    json.dump(results, f, indent=1)
n = len(gdal_side["files"])
same = counts.get("IDENTICAL", 0) + counts.get("ALPHA", 0)
print(f"\nGDAL {gdal_side['gdal']}: {same}/{n} files identical to GDAL's reading, "
      f"{cells_total:,} cells compared, {invalid_total:,} of them NoData")
others = [f"{counts[k]} {k.lower()}" for k in ("ALPHA", "NO-CRS", "REFUSED", "BOTH-REFUSE", "FAIL") if counts.get(k)]
if others:
    print("  " + ", ".join(others))
sys.exit(1 if failed else 0)
