#!/usr/bin/env python3
"""Difference strata's cog reader against GDAL on the files cogmake.py wrote.

For every file, GDAL's reading (manifest.json and the .f32/.mask files
cogmake.py wrote through GDAL's API) is the truth, and strata's reading
(strata.json and the .strata.* files cog/main.go wrote) must match it:

  - strata opens the file, with the same bands, number of levels and
    level sizes;
  - every level's geotransform equals GDAL's, to 1e-9 of a cell;
  - the EPSG code equals GDAL's;
  - strata reports validity (Masked) exactly when GDAL has a NoData value;
  - every band of every level has the same validity mask, cell for cell;
  - every valid cell holds the same float32 bits (any NaN matching any
    NaN), since both sides convert the same sample to float32 and a
    reader has no rounding latitude at all.

The comparison is exact, so there is no tolerance to derive. It prints
one line per file with the cells compared, and exits non-zero on any
failure.

Usage:  python3 cogcompare.py [dir]
"""

import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "out-cog"
gdal_side = json.load(open(os.path.join(D, "manifest.json")))
strata_side = {r["name"]: r for r in json.load(open(os.path.join(D, "strata.json")))}


def load(path, dtype):
    return np.fromfile(os.path.join(D, path), dtype=dtype)


failed = 0
cells_total = invalid_total = 0
for g in gdal_side["files"]:
    name = g["name"]
    s = strata_side.get(name)
    problems = []
    cells = invalid = 0
    if s is None:
        problems.append("strata did not read it")
    elif s.get("error"):
        problems.append(f"strata refused it: {s['error']}")
    else:
        if s["bands"] != g["bands"]:
            problems.append(f"{s['bands']} bands, GDAL {g['bands']}")
        if len(s["levels"]) != len(g["levels"]):
            problems.append(f"{len(s['levels'])} levels, GDAL {len(g['levels'])}")
        want_epsg = f"EPSG:{g['epsg']}" if g["epsg"] else ""
        if s["epsg"] != want_epsg:
            problems.append(f"CRS {s['epsg']!r}, GDAL {want_epsg!r}")
        if s["masked"] != (g["nodata"] is not None):
            problems.append(f"Masked {s['masked']}, GDAL NoData {g['nodata']}")
        for lvl, (sl, gl) in enumerate(zip(s["levels"], g["levels"])):
            if (sl["W"], sl["H"]) != (gl["w"], gl["h"]):
                problems.append(f"level {lvl}: {sl['W']}x{sl['H']}, GDAL {gl['w']}x{gl['h']}")
                continue
            sgt, ggt = np.array(sl["GT"]), np.array(gl["gt"])
            cell = max(abs(ggt[1]), abs(ggt[5]))
            if np.max(np.abs(sgt - ggt)) > 1e-9 * cell:
                problems.append(f"level {lvl}: geotransform {sgt.tolist()}, GDAL {ggt.tolist()}")
            for b in range(g["bands"]):
                stem = f"{name}.b{b}.l{lvl}"
                gv, sv = load(stem + ".f32", "<u4"), load(stem + ".strata.f32", "<u4")
                gm, sm = load(stem + ".mask", np.uint8) > 0, load(stem + ".strata.mask", np.uint8) > 0
                if gv.size != sv.size or gm.size != sm.size:
                    problems.append(f"{stem}: {sv.size} cells, GDAL {gv.size}")
                    continue
                if (gm != sm).any():
                    k = int(np.argmax(gm != sm))
                    problems.append(f"{stem}: {int((gm != sm).sum())} cells differ in validity, "
                                    f"first at ({k % gl['w']}, {k // gl['w']})")
                gf, sf = gv.view("<f4"), sv.view("<f4")
                both_nan = np.isnan(gf) & np.isnan(sf)
                diff = gm & sm & (gv != sv) & ~both_nan
                if diff.any():
                    k = int(np.argmax(diff))
                    problems.append(f"{stem}: {int(diff.sum())} valid cells differ, first at "
                                    f"({k % gl['w']}, {k // gl['w']}): {sf[k]!r}, GDAL {gf[k]!r}")
                cells += gv.size
                invalid += int((~gm).sum())
    cells_total += cells
    invalid_total += invalid
    if problems:
        failed += 1
        print(f"FAIL  {name}")
        for p in problems[:6]:
            print(f"        {p}")
    else:
        print(f"PASS  {name:<48} {len(g['levels'])} levels x {g['bands']} bands, "
              f"{cells:>9,} cells ({invalid:,} NoData) identical")

n = len(gdal_side["files"])
print(f"\nGDAL {gdal_side['gdal']}: {n - failed}/{n} files identical to GDAL's reading, "
      f"{cells_total:,} cells compared, {invalid_total:,} of them NoData")
sys.exit(1 if failed else 0)
