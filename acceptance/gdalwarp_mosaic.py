#!/usr/bin/env python3
"""Difference strata's mosaics against gdalwarp given the same sources.

For each method of mosaic.json it loads every source into an in-memory
GDAL dataset (NoData = the manifest's fill under invalid cells) and warps
them, in order, onto the case's destination grid, exactly as

    gdalwarp -tr 10 10 -te <extent> -r <method> -et 0 -wt Float64 \
             -srcnodata <fill> -dstnodata <fill> a b c d out

would. It then makes four checks:

  (a) gdalwarp's own rule. The warp of all sources must equal, bit for
      bit, the warps of each source alone laid over one another in order,
      a cell taking the last source's value where that source's warp is
      valid. This is GDAL judged against itself; it is the evidence that
      "the last valid source wins, with no blending at partly valid
      seams" is what gdalwarp does, which is the rule strata implements.
  (b) strata against gdalwarp: validity cell for cell, and values within
      twice the float32 tolerance check_resample.py derives for the cell
      from the source that wins it (as gdalwarp_resample.py does).
  (c) strata against the float64 reference of check_resample.py, overlaid
      the same way: validity exactly, values within the derived tolerance.
      This needs no GDAL, and holds strata to the definitions where (b)
      holds it to GDAL.
  (d) plain == tiled == chunked, bit for bit.

Excluded from (b), and counted rather than judged, are the cells whose
winning source falls under a departure DESIGN.md §54 records:

  * Average: output cells that straddle or touch the edge of any source.
    Where a cell straddles it, gdalwarp's weights for that source differ
    by about 1%. Where a cell only touches it, with no area in common,
    gdalwarp nonetheless makes that source valid there, with its edge
    cell's value, whenever the source's resolution differs from the
    output's (GDAL 3.12.2 to 3.14-dev); a later source then wins a cell
    that lies wholly inside an earlier one. strata keeps Average's
    definition, weights by area, and leaves such a source out of the
    cell (DESIGN.md §54);
  * before GDAL 3.13.0, cells a source could be valid in (by the
    reference or by gdalwarp) whose widened kernel gdalwarp stretches by
    pixel counts where they differ from the resolution ratio: its values
    differ, and for Lanczos so does its half-valid window, and with it
    which source wins;
  * where gdalwarp's rules for the winning source differ from strata's
    (from 3.13.0 Bilinear and Cubic unwidened below 2x, from 3.13.1 no
    Lanczos half-valid rule), cells where the two rules pick a different
    winner or validity; where they agree the value is judged against the
    reference under gdalwarp's rule.

--sabotage lays the sources over one another in the wrong order, the
first valid source on top: in (a) the overlay of gdalwarp's single
warps, in (b) the mosaic judged against gdalwarp (the reference overlaid
that way stands in for strata's), in (c) the reference. Every one of
those checks must then fail: they can tell the rule from its mirror
image.

Needs GDAL's Python bindings (osgeo), not numpy.

Usage:  python3 gdalwarp_mosaic.py [dir] [--sabotage]      (default: out)
"""

import json
import os
import struct
import sys

from osgeo import gdal

import check_resample as ref
from gdalwarp_resample import ALG, FOUR_SAMPLE_TO_HALF, NO_HALF_VALID, past_edge, pixel_stretch

gdal.UseExceptions()

GEOMETRY_STRETCH = int(gdal.VersionInfo("VERSION_NUM")) >= 3130000


def mem_source(grid, vals, valid, masked, fill):
    drv = gdal.GetDriverByName("MEM")
    s = drv.Create("", grid["width"], grid["height"], 1, gdal.GDT_Float32)
    s.SetGeoTransform((grid["ox"], grid["rx"], 0, grid["oy"], 0, grid["ry"]))
    s.SetProjection("EPSG:32633")
    b = s.GetRasterBand(1)
    if masked:
        b.SetNoDataValue(fill)
    out = [v if ok else fill for v, ok in zip(vals, valid)]
    b.WriteRaster(0, 0, grid["width"], grid["height"], struct.pack("<%df" % len(out), *out))
    return s


def warp(dg, sources, method, fill):
    drv = gdal.GetDriverByName("MEM")
    d = drv.Create("", dg["width"], dg["height"], 1, gdal.GDT_Float32)
    d.SetGeoTransform((dg["ox"], dg["rx"], 0, dg["oy"], 0, dg["ry"]))
    d.SetProjection("EPSG:32633")
    db = d.GetRasterBand(1)
    db.SetNoDataValue(fill)
    db.Fill(fill)
    gdal.Warp(d, sources, resampleAlg=ALG[method], errorThreshold=0, workingType=gdal.GDT_Float64)
    n = dg["width"] * dg["height"]
    return list(struct.unpack("<%df" % n, db.ReadRaster(0, 0, dg["width"], dg["height"])))


def winners(layers, n_rows, n_cols, first_wins):
    """Per cell, the index of the layer that wins it, or None. layers[i]
    is a per-cell predicate grid: True where layer i is valid."""
    order = range(len(layers)) if first_wins else range(len(layers) - 1, -1, -1)
    out = [[None] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            for i in order:
                if layers[i][r][c]:
                    out[r][c] = i
                    break
    return out


def straddle(case, c, r):
    """Whether output cell (c, r) overlaps or touches the source and
    extends past it."""
    sg, dg = case["src_grid"], case["dst_grid"]
    x0, x1 = sorted((dg["ox"] + c * dg["rx"], dg["ox"] + (c + 1) * dg["rx"]))
    y0, y1 = sorted((dg["oy"] + r * dg["ry"], dg["oy"] + (r + 1) * dg["ry"]))
    sx0, sx1 = sorted((sg["ox"], sg["ox"] + sg["width"] * sg["rx"]))
    sy0, sy1 = sorted((sg["oy"], sg["oy"] + sg["height"] * sg["ry"]))
    return x0 <= sx1 and x1 >= sx0 and y0 <= sy1 and y1 >= sy0 and past_edge(case, c, r)


def bits(v):
    return struct.pack("<f", v)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sabotage = "--sabotage" in sys.argv
    d = args[0] if args else "out"
    with open(os.path.join(d, "mosaic.json")) as f:
        man = json.load(f)
    fill = man["fill"]
    print("gdalwarp: GDAL", gdal.__version__)
    failures, checks = [], 0
    groups = {}
    for case in man["cases"]:
        groups.setdefault(case["method"], {})[case["form"]] = case
    for method, forms in groups.items():
        case = forms["plain"]
        dg = case["dst_grid"]
        W, H = dg["width"], dg["height"]
        nd = W * H
        srcs = []
        for s in case["sources"]:
            sg = s["src_grid"]
            ns = sg["width"] * sg["height"]
            vals = ref.load_f32(os.path.join(d, s["src"]), ns)
            valid = ref.load_mask(os.path.join(d, s["src_mask"]), ns) if s.get("src_mask") else [True] * ns
            one = {"src_grid": sg, "dst_grid": dg, "method": method, "src_mask": s.get("src_mask")}
            srcs.append((one, vals, valid, mem_source(sg, vals, valid, bool(s.get("src_mask")), fill)))

        # The reference of each source alone: under strata's rules, and
        # under the rules of the GDAL at hand.
        refs, grefs, flats = [], [], []
        for one, vals, valid, _ in srcs:
            r = ref.reference(one, vals, valid)
            g, flat = r, False
            if NO_HALF_VALID and method == "Lanczos" and one["src_mask"]:
                g = ref.reference(one, vals, valid, half_valid=False)
            sg = one["src_grid"]
            widens = any(ref.stretch(method, abs(x)) > 1 for x in (dg["rx"] / sg["rx"], dg["ry"] / sg["ry"]))
            if FOUR_SAMPLE_TO_HALF and widens and ref.unwidened_to_half(one):
                g, flat = ref.reference(one, vals, valid, four_sample_to_half=True), True
            refs.append(r)
            grefs.append(g)
            flats.append(flat)
        win = winners([[[v is not None for v in row] for row in r] for r in refs], H, W, False)
        gwin = winners([[[v is not None for v in row] for row in g] for g in grefs], H, W, False)
        # The reference overlaid in the order under test.
        rwin = winners([[[v is not None for v in row] for row in r] for r in refs], H, W, sabotage)

        theirs = warp(dg, [s[3] for s in srcs], method, fill)
        alone = [warp(dg, [s[3]], method, fill) for s in srcs]
        ours = ref.load_f32(os.path.join(d, case["out"]), nd)
        mask = ref.load_mask(os.path.join(d, case["out_mask"]), nd)
        # What (b) judges: strata's mosaic, or under --sabotage the
        # reference laid first-on-top.
        cand, cmask = ours, mask
        if sabotage:
            cmask = [rwin[i // W][i % W] is not None for i in range(nd)]
            cand = [refs[rwin[i // W][i % W]][i // W][i % W][0] if cmask[i] else fill for i in range(nd)]
        straddles = [[any(method == "Average" and straddle(s[0], c, r) for s in srcs) for c in range(W)] for r in range(H)]

        # (a) gdalwarp's rule, GDAL against itself.
        layered = [[[alone[i][r * W + c] != fill for c in range(W)] for r in range(H)] for i in range(len(srcs))]
        lw = winners(layered, H, W, sabotage)
        bad = 0
        for r in range(H):
            for c in range(W):
                i = r * W + c
                want = fill if lw[r][c] is None else alone[lw[r][c]][i]
                if bits(want) != bits(theirs[i]):
                    bad += 1
        checks += 1
        line = "%-5s %-34s %s" % ("ok" if bad == 0 else "FAIL", "gdalwarp rule, " + method,
                                   "all sources == each alone, last valid on top, bit for bit (%d cells)" % nd
                                   if bad == 0 else "%d cells differ from the overlay" % bad)
        (failures.append(line) if bad else None)
        print(line)

        # (b) strata against gdalwarp.
        stretch_differs = [not GEOMETRY_STRETCH and pixel_stretch(s[0])[0] for s in srcs]
        bad_valid = bad_val = excluded = departed = 0
        worst = 0.0
        for r in range(H):
            for c in range(W):
                i = r * W + c
                w, gw = win[r][c], gwin[r][c]
                if w != gw:
                    departed += 1
                    continue
                if straddles[r][c] or any(stretch_differs[k] and (refs[k][r][c] is not None or alone[k][i] != fill)
                                          for k in range(len(srcs))):
                    excluded += 1
                    continue
                gv = theirs[i] != fill
                if gv != cmask[i]:
                    bad_valid += 1
                    continue
                if not gv:
                    continue
                t, want = refs[w][r][c], cand[i]
                if flats[w]:
                    t = grefs[w][r][c]
                    want = t[0]
                tol = 2 * t[1] + 2 * ref.EPS * abs(theirs[i])
                err = abs(want - theirs[i])
                if tol > 0:
                    worst = max(worst, err / tol)
                if err > tol:
                    bad_val += 1
        checks += 1
        ok = bad_valid == 0 and bad_val == 0
        line = "%-5s %-34s validity %s, values %s%s%s" % (
            "ok" if ok else "FAIL", "strata vs gdalwarp, " + method,
            "exact" if bad_valid == 0 else "%d cells differ" % bad_valid,
            "within tolerance (worst %.2f of it)" % worst if bad_val == 0 else
            "%d cells out (worst %.3g x tolerance)" % (bad_val, worst),
            ", %d cells excluded (§54 departures)" % excluded if excluded else "",
            ", %d cells where GDAL's rules pick another winner" % departed if departed else "")
        (failures.append(line) if not ok else None)
        print(line)

        # (c) strata against the reference.
        bad_valid = bad_val = 0
        worst = 0.0
        for r in range(H):
            for c in range(W):
                i = r * W + c
                w = rwin[r][c]
                if (w is not None) != mask[i]:
                    bad_valid += 1
                    continue
                if w is None:
                    continue
                v, tol = refs[w][r][c]
                err = abs(ours[i] - v)
                if tol > 0:
                    worst = max(worst, err / tol)
                if err > tol:
                    bad_val += 1
        checks += 1
        ok = bad_valid == 0 and bad_val == 0
        line = "%-5s %-34s validity %s, values %s" % (
            "ok" if ok else "FAIL", "strata vs reference, " + method,
            "exact" if bad_valid == 0 else "%d cells differ" % bad_valid,
            "within tolerance (worst %.2f of it)" % worst if bad_val == 0 else
            "%d cells out (worst %.3g x tolerance)" % (bad_val, worst))
        (failures.append(line) if not ok else None)
        print(line)

        # (d) the forms.
        for form in ("tiled", "chunked"):
            fc = forms[form]
            v = ref.load_f32(os.path.join(d, fc["out"]), nd)
            mk = ref.load_mask(os.path.join(d, fc["out_mask"]), nd)
            same = mk == mask and all(bits(a) == bits(b) for a, b, ok in zip(ours, v, mask) if ok)
            checks += 1
            line = "%-5s %-34s %s == plain, bit for bit" % ("ok" if same else "FAIL", "forms, " + method, form)
            (failures.append(line) if not same else None)
            print(line)

        cover = sum(1 for row in win for w in row if w is not None)
        counts = [sum(1 for row in win for w in row if w == k) for k in range(len(srcs))]
        print("      %s: %d of %d cells valid; won by source %s" % (
            method, cover, nd, ", ".join("%s %d" % ("abcd"[k] if k < 4 else k, n) for k, n in enumerate(counts))))

    if sabotage:
        # (d) does not depend on the order; every other check must fail.
        judged = [f for f in failures if not f.split(None, 1)[1].startswith("forms")]
        want = 3 * len(groups)
        if len(judged) == want:
            print("\nok    sabotage: laying the sources first-on-top fails all %d order-dependent checks" % want)
            return 0
        print("\nFAIL  sabotage: only %d of %d order-dependent checks noticed the wrong order" % (len(judged), want))
        return 1
    print("\n%d checks, %d failed" % (checks, len(failures)))
    for f in failures:
        print(f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
