#!/usr/bin/env python3
"""Difference strata's resampling against gdalwarp, the outside oracle.

For every plain case of resample.json it loads the source into an
in-memory GDAL dataset (NoData = the manifest's fill under invalid
cells), warps it onto the case's destination grid with the same method,
exactly as

    gdalwarp -tr <rx> <ry> -te <extent> -r <method> -et 0 -wt Float64 \
             -srcnodata <fill> -dstnodata <fill>

would, and compares: validity cell for cell, and values within twice the
float32 tolerance check_resample.py derives for the cell (gdalwarp sums in
float64, strata in float32, so their difference is bounded by strata's
error plus the rounding of the float32 result).

Excluded, and counted rather than judged, are the cells DESIGN.md §54
records as gdalwarp's own departures from its definitions:

  * Average: output cells that extend past the source's edge, where
    gdalwarp's weights differ by about 1%;
  * a stretched kernel whose stretch gdalwarp takes from pixel counts
    (the clipped source window over the destination's cells) where they
    differ from the resolution ratio, as for the 1.37 grids, which
    overhang the source: the values are not compared (nor, for Lanczos,
    whose half-valid window depends on the stretch, validity). The 1.25
    grids are laid out so the two agree, and are compared in full;
  * Lanczos downsampling by an odd integer factor (none in these cases).

Needs GDAL's Python bindings (osgeo), not numpy.

Usage:  python3 gdalwarp_resample.py [dir]      (default: out)
"""

import json
import math
import os
import struct
import sys

from osgeo import gdal

import check_resample as ref

gdal.UseExceptions()

ALG = {"Nearest": "near", "Bilinear": "bilinear", "Cubic": "cubic", "Lanczos": "lanczos", "Average": "average"}


def warp(case, src, valid, fill):
    sg, dg = case["src_grid"], case["dst_grid"]
    drv = gdal.GetDriverByName("MEM")
    s = drv.Create("", sg["width"], sg["height"], 1, gdal.GDT_Float32)
    s.SetGeoTransform((sg["ox"], sg["rx"], 0, sg["oy"], 0, sg["ry"]))
    s.SetProjection("EPSG:32633")
    b = s.GetRasterBand(1)
    vals = [v if ok else fill for v, ok in zip(src, valid)]
    if case.get("src_mask"):
        b.SetNoDataValue(fill)
    b.WriteRaster(0, 0, sg["width"], sg["height"], struct.pack("<%df" % len(vals), *vals))
    d = drv.Create("", dg["width"], dg["height"], 1, gdal.GDT_Float32)
    d.SetGeoTransform((dg["ox"], dg["rx"], 0, dg["oy"], 0, dg["ry"]))
    d.SetProjection("EPSG:32633")
    db = d.GetRasterBand(1)
    db.SetNoDataValue(fill)
    db.Fill(fill)
    gdal.Warp(d, s, resampleAlg=ALG[case["method"]], errorThreshold=0, workingType=gdal.GDT_Float64)
    n = dg["width"] * dg["height"]
    return list(struct.unpack("<%df" % n, db.ReadRaster(0, 0, dg["width"], dg["height"])))


def past_edge(case, c, r):
    """Whether output cell (c, r) extends past the source."""
    sg, dg = case["src_grid"], case["dst_grid"]
    x0, x1 = sorted((dg["ox"] + c * dg["rx"], dg["ox"] + (c + 1) * dg["rx"]))
    y0, y1 = sorted((dg["oy"] + r * dg["ry"], dg["oy"] + (r + 1) * dg["ry"]))
    sx0, sx1 = sorted((sg["ox"], sg["ox"] + sg["width"] * sg["rx"]))
    sy0, sy1 = sorted((sg["oy"], sg["oy"] + sg["height"] * sg["ry"]))
    return x0 < sx0 or x1 > sx1 or y0 < sy0 or y1 > sy1


def pixel_stretch(case):
    """gdalwarp's stretch per axis: source window cells over destination
    cells, and whether it differs from the resolution ratio on an axis
    the method stretches."""
    sg, dg, m = case["src_grid"], case["dst_grid"], case["method"]

    def axis(n, o, r, sn, so, sr):
        u0, u1 = (o - so) / sr, (o + n * r - so) / sr
        lo, hi = min(u0, u1), max(u0, u1)
        win = (min(sn, math.ceil(hi - 1e-9)) - max(0, math.floor(lo + 1e-9))) / n
        s = abs(r / sr)
        return ref.stretch(m, s) > 1 and abs(win - s) > 1e-9, win

    dx, wx = axis(dg["width"], dg["ox"], dg["rx"], sg["width"], sg["ox"], sg["rx"])
    dy, wy = axis(dg["height"], dg["oy"], dg["ry"], sg["height"], sg["oy"], sg["ry"])
    return m in ("Bilinear", "Cubic", "Lanczos") and (dx or dy), wx, wy


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "out"
    with open(os.path.join(d, "resample.json")) as f:
        man = json.load(f)
    fill = man["fill"]
    print("gdalwarp: GDAL", gdal.__version__)
    failures, checks = [], 0
    for case in man["cases"]:
        if case["form"] != "plain":
            continue
        sg, dg = case["src_grid"], case["dst_grid"]
        ns, nd = sg["width"] * sg["height"], dg["width"] * dg["height"]
        src = ref.load_f32(os.path.join(d, case["src"]), ns)
        valid = ref.load_mask(os.path.join(d, case["src_mask"]), ns) if case.get("src_mask") else [True] * ns
        ours = ref.load_f32(os.path.join(d, case["out"]), nd)
        mask = ref.load_mask(os.path.join(d, case["out_mask"]), nd)
        theirs = warp(case, src, valid, fill)
        tols = ref.reference(case, src, valid)
        differs, wx, wy = pixel_stretch(case)
        bad_valid = bad_val = excluded = 0
        worst = 0.0
        for r in range(dg["height"]):
            for c in range(dg["width"]):
                i = r * dg["width"] + c
                if case["method"] == "Average" and past_edge(case, c, r):
                    excluded += 1
                    continue
                gv = theirs[i] != fill
                if differs and case["method"] == "Lanczos":
                    excluded += 1
                    continue
                if gv != mask[i]:
                    bad_valid += 1
                    continue
                if not gv:
                    continue
                if differs:
                    excluded += 1
                    continue
                t = tols[r][c]
                tol = 2 * t[1] + 2 * ref.EPS * abs(theirs[i]) if t is not None else 1e-6 * abs(theirs[i])
                err = abs(ours[i] - theirs[i])
                if err > tol:
                    bad_val += 1
                    worst = max(worst, err / max(tol, 1e-300))
        checks += 1
        ok = bad_valid == 0 and bad_val == 0
        if differs:
            line = "%-5s %-40s gdalwarp stretches by pixel counts (%.4f × %.4f): %s" % (
                "ok" if ok else "FAIL", case["name"], wx, wy,
                "nothing compared" if case["method"] == "Lanczos" else
                "validity %s, values not compared" % ("exact" if bad_valid == 0 else "%d cells differ" % bad_valid))
        else:
            line = "%-5s %-40s validity %s, values %s%s" % (
                "ok" if ok else "FAIL", case["name"],
                "exact" if bad_valid == 0 else "%d cells differ" % bad_valid,
                "within tolerance" if bad_val == 0 else "%d cells out (worst %.3g x tolerance)" % (bad_val, worst),
                ", %d edge cells excluded" % excluded if excluded else "")
        print(line)
        if not ok:
            failures.append(line)
    print("\n%d cases against gdalwarp, %d failed" % (checks, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
