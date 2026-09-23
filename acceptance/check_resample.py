#!/usr/bin/env python3
"""Judge the resampling files written by `go run .` (resample.go) without
reading strata's source.

The reference is written from the published definitions and gdalwarp's
documented and measured conventions (DESIGN.md §54), in float64, in plain
Python so that it runs where numpy is not installed; the rasters are
80x60, so plain loops are fast enough.

  * Pixel is area. Output cell (c, r) is its centre, at world
    X = ox + (c + 0.5) rx; its source pixel coordinate is
    u = inv0 + X inv1 with inv1 = 1 / src_rx, inv0 = -src_ox inv1 (the
    inverse geotransform), and source cell i covers [i, i + 1).
  * Nearest takes cell floor(u).
  * Bilinear, cubic and Lanczos weigh source cell i by K((i + 0.5 - u)/s)
    over |.| < support: the tent (support 1), Keys' cubic with a = -0.5
    (support 2), and sinc(x) sinc(x/3) (support 3). s is 1, or the source
    cells per output cell when the axis downsamples: for bilinear and
    cubic when 1/s < 0.95, for Lanczos when s > 1.
  * Average weighs each source cell by its overlap with the output cell.
  * Weights are the product of the two axes' weights, and the value is
    the weighted mean over the valid source cells inside the source.
  * An output cell is valid if its centre cell is inside the source and
    valid (not required under Average), and the valid weight is positive.
    Lanczos over a masked source also needs half of the source cells its
    window reaches valid, unless its centre lies exactly on a source
    centre on both axes (a copy). Cubic with neither axis widened falls
    back to unwidened bilinear for a cell whose 4x4 taps of non-zero
    weight include an invalid cell or one outside the source.

Checks, per case: validity exactly equal to the reference; valid values
within a tolerance derived from float32 rounding (below); Nearest exactly
equal; and plain, Tiled and Chunked giving the same bits on valid cells
and the same validity.

TOLERANCE. strata rounds each weight to float32 (relative error EPS) and
accumulates products in float32 over n = taps_x + taps_y roundings of
partial sums no larger than A = sum |w| |x| over the valid taps. The
unnormalised sum N is then off by at most (n + 2) EPS A, and the valid
weight D by (n + 2) EPS B with B = sum |w|. The value N / D is off by
(n + 2) EPS (A / |D| + |N| B / D^2) + EPS |N / D|, which is what this
script allows, times 1.5 for the second-order terms. Where negative lobes
make D small, the bound grows with them, as the error does.

Usage:  python3 check_resample.py [dir]             (default: out)
        python3 check_resample.py out --sabotage   the checker must fail
                                                   when the reference is
                                                   shifted by half a cell
"""

import json
import math
import os
import struct
import sys

EPS = 2.0**-24


def load_f32(path, n):
    with open(path, "rb") as f:
        b = f.read()
    return list(struct.unpack("<%df" % n, b[: 4 * n]))


def load_mask(path, n):
    with open(path, "rb") as f:
        return [v != 0 for v in f.read()[:n]]


def tent(x):
    return 1 - abs(x)


def keys(x):
    a = -0.5
    x = abs(x)
    if x < 1:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1
    return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a


def lanczos(x):
    if x == 0:
        return 1.0
    if x == math.trunc(x):
        return 0.0
    px = math.pi * x
    return 3 * math.sin(px) * math.sin(px / 3) / (px * px)


KERNELS = {"Bilinear": (tent, 1), "Cubic": (keys, 2), "Lanczos": (lanczos, 3)}


def stretch(method, s):
    if method in ("Bilinear", "Cubic") and 1 / s < 0.95:
        return s
    if method == "Lanczos" and s > 1:
        return s
    return 1.0


def axis(method, n, o, r, sn, so, sr, shift=0.0):
    """Per output index: centre cell (or None), taps [(i, w)] inside the
    source, whether a tap of non-zero weight was outside it, and the
    in-source cells the kernel reaches (the half-valid window)."""
    inv1 = 1 / sr
    inv0 = -so * inv1
    s = abs(r / sr)
    out = []
    for c in range(n):
        x = o + (c + 0.5 + shift) * r
        u = inv0 + x * inv1
        ci = math.floor(u)
        centre = ci if 0 <= ci < sn else None
        if method == "Nearest":
            out.append((centre, [(ci, 1.0)] if centre is not None else [], False, []))
            continue
        if method == "Average":
            e0 = inv0 + (o + (c + shift) * r) * inv1
            e1 = inv0 + (o + (c + 1 + shift) * r) * inv1
            lo, hi = min(e0, e1), max(e0, e1)
            taps = []
            for i in range(max(0, math.floor(lo)), min(sn, math.ceil(hi))):
                w = min(hi, i + 1) - max(lo, i)
                if w > 0:
                    taps.append((i, w))
            out.append((None, taps, False, []))
            continue
        k, sup = KERNELS[method]
        st = stretch(method, s)
        taps, clipped, window = [], False, []
        for i in range(math.floor(u - 0.5 - sup * st) - 1, math.ceil(u - 0.5 + sup * st) + 2):
            d = (i + 0.5 - u) / st
            if abs(d) >= sup:
                continue
            w = k(d)
            if 0 <= i < sn:
                window.append(i)
                if w != 0:
                    taps.append((i, w))
            elif w != 0:
                clipped = True
        out.append((centre, taps, clipped, window))
    return out, s


def reference(case, src, valid, shift=0.0, half_valid=True):
    """Per output cell, (value, tolerance) or None where it is invalid.
    half_valid=False leaves out Lanczos's half-valid rule, which gdalwarp
    dropped in GDAL 3.13.1 (DESIGN.md §54)."""
    sg, dg, m = case["src_grid"], case["dst_grid"], case["method"]
    ax, sx = axis(m, dg["width"], dg["ox"], dg["rx"], sg["width"], sg["ox"], sg["rx"], shift)
    ay, sy = axis(m, dg["height"], dg["oy"], dg["ry"], sg["height"], sg["oy"], sg["ry"])
    sw = sg["width"]
    widened = stretch(m, sx) > 1 or stretch(m, sy) > 1
    cubic4 = m == "Cubic" and not widened
    masked = case.get("src_mask") is not None
    if cubic4:
        bx, _ = axis("Bilinear", dg["width"], dg["ox"], dg["rx"], sg["width"], sg["ox"], sg["rx"], shift)
        by, _ = axis("Bilinear", dg["height"], dg["oy"], dg["ry"], sg["height"], sg["oy"], sg["ry"])
        # four-sample mode never widens
        bx = [(c, t, cl, w) for (c, t, cl, w) in bx]
    def centres(n, o, rr, so, sr, sh):
        return [(-so / sr) + (o + (i + 0.5 + sh) * rr) * (1 / sr) for i in range(n)]
    ux = centres(dg["width"], dg["ox"], dg["rx"], sg["ox"], sg["rx"], shift)
    uy = centres(dg["height"], dg["oy"], dg["ry"], sg["oy"], sg["ry"], 0.0)
    res = []
    for r, (cy, ty, cly, wy_) in enumerate(ay):
        row = []
        for c, (cx, tx, clx, wx_) in enumerate(ax):
            if m == "Nearest":
                if cx is None or cy is None or not valid[cy * sw + cx]:
                    row.append(None)
                else:
                    row.append((src[cy * sw + cx], 0.0))
                continue
            if m != "Average" and (cx is None or cy is None or not valid[cy * sw + cx]):
                row.append(None)
                continue

            def sums(txs, tys):
                sx_ = sum(w for _, w in txs) or 1.0
                sy_ = sum(w for _, w in tys) or 1.0
                N = D = A = B = 0.0
                full = True
                for j, wj in tys:
                    for i, wi in txs:
                        w = (wi / sx_) * (wj / sy_)
                        if not valid[j * sw + i]:
                            full = False
                            continue
                        x = src[j * sw + i]
                        N += w * x
                        D += w
                        A += abs(w * x)
                        B += abs(w)
                return N, D, A, B, full

            N, D, A, B, full = sums(tx, ty)
            n = len(tx) + len(ty)
            if cubic4 and (clx or cly or not full):
                N, D, A, B, _ = sums(bx[c][1], by[r][1])
                n = len(bx[c][1]) + len(by[r][1])
            if not D > 0:
                row.append(None)
                continue
            exact = abs(((ux[c] - 0.5) % 1.0)) < 1e-9 and abs(((uy[r] - 0.5) % 1.0)) < 1e-9
            if half_valid and m == "Lanczos" and masked and not exact:
                nv = sum(1 for j in wy_ for i in wx_ if valid[j * sw + i])
                if 2 * nv < len(wx_) * len(wy_):
                    row.append(None)
                    continue
            v = N / D
            tol = 1.5 * ((n + 2) * EPS * (A / abs(D) + abs(N) * B / (D * D)) + EPS * abs(v))
            row.append((v, tol))
        res.append(row)
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sabotage = "--sabotage" in sys.argv
    d = args[0] if args else "out"
    with open(os.path.join(d, "resample.json")) as f:
        man = json.load(f)
    failures, checks = [], 0
    groups = {}
    for case in man["cases"]:
        groups.setdefault(case["name"].rsplit("-", 1)[0], []).append(case)
    for base, cases in sorted(groups.items()):
        c0 = cases[0]
        sg, dg = c0["src_grid"], c0["dst_grid"]
        ns, nd = sg["width"] * sg["height"], dg["width"] * dg["height"]
        src = load_f32(os.path.join(d, c0["src"]), ns)
        valid = load_mask(os.path.join(d, c0["src_mask"]), ns) if c0.get("src_mask") else [True] * ns
        ref = reference(c0, src, valid, 0.5 if sabotage else 0.0)
        outs = {}
        for case in cases:
            vals = load_f32(os.path.join(d, case["out"]), nd)
            mask = load_mask(os.path.join(d, case["out_mask"]), nd)
            outs[case["form"]] = (vals, mask)
            bad_valid = bad_val = 0
            worst = 0.0
            for r in range(dg["height"]):
                for c in range(dg["width"]):
                    i = r * dg["width"] + c
                    want = ref[r][c]
                    if (want is not None) != mask[i]:
                        bad_valid += 1
                        continue
                    if want is None:
                        continue
                    v, tol = want
                    err = abs(vals[i] - v)
                    if tol > 0:
                        worst = max(worst, err / tol)
                    if err > tol:
                        bad_val += 1
            checks += 1
            ok = bad_valid == 0 and bad_val == 0
            line = "%-5s %-40s validity %s, values %s" % (
                "ok" if ok else "FAIL", case["name"],
                "exact" if bad_valid == 0 else "%d cells differ" % bad_valid,
                "within tolerance (worst %.2f of it)" % worst if bad_val == 0 else
                "%d cells out (worst %.3g x tolerance)" % (bad_val, worst))
            (failures.append(line) if not ok else None)
            if not sabotage:
                print(line)
        # plain == tiled == chunked, bit for bit on valid cells.
        pv, pm = outs["plain"]
        for form in ("tiled", "chunked"):
            v, mk = outs[form]
            same = pm == mk and all(
                struct.pack("<f", a) == struct.pack("<f", b) for a, b, ok in zip(pv, v, pm) if ok)
            checks += 1
            line = "%-5s %-40s %s == plain, bit for bit" % ("ok" if same else "FAIL", base, form)
            if not same:
                failures.append(line)
            if not sabotage:
                print(line)
    if sabotage:
        if failures:
            print("ok    sabotage: shifting the reference by half a cell fails %d of %d checks" % (len(failures), checks))
            return 0
        print("FAIL  sabotage: a half-cell shift went unnoticed")
        return 1
    print("\n%d checks, %d failed" % (checks, len(failures)))
    for f in failures:
        print(f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
