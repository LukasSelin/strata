#!/usr/bin/env python3
"""Judge the files written by `go run .` without reading strata's source.

Every expected value here is computed from a published definition in
float64 numpy:

  * Horn's 3x3 gradient, as gdaldem documents its kernel:
        dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
        dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)
    over the neighbourhood  a b c / d e f / g h i.
  * slope     = atan(hypot(dx, dy)), in degrees, radians or 100*tan.
  * aspect    = compass bearing of the downslope direction, which in
                (east, north) components is (-dx, dy), so atan2(-dx, dy).
  * hillshade = 255 * max(0, cos of the angle between the light and the
                surface normal). With the light at compass azimuth `az`
                and altitude `alt`, the unit vector towards it is
                (cos(alt)sin(az), cos(alt)cos(az), sin(alt)) in
                (east, north, up), and the surface normal is
                (-dx, dy, 1) / sqrt(1 + dx^2 + dy^2).
  * curvature, from the Zevenbergen & Thorne (1987) quadratic through the
    centre and its four edge neighbours:
        p = (f - d) / (2 cx)          q = (h - b) / (2 cy)
        r = (d - 2e + f) / cx^2       t = (b - 2e + h) / cy^2
        s = (a - c - g + i) / (4 cx cy)
    and Florinsky's (2016) normal-section curvatures, positive where the
    surface is convex, with g = p^2 + q^2:
        profile = -(p^2 r + 2pqs + q^2 t) / (g (1 + g)^1.5)
        plan    = -(q^2 r - 2pqs + p^2 t) / g^1.5
        mean    = -((1 + q^2) r - 2pqs + (1 + p^2) t) / (2 (1 + g)^1.5)
    Profile and plan are undefined on flat cells (p = q = 0); strata
    documents 0 there.
  * ruggedness, over the eight neighbours of the centre e:
        TRI (Riley)    = sqrt(sum (n - e)^2)
        TRI (Wilson)   = sum |n - e| / 8
        TPI            = e - sum n / 8
        roughness      = max - min of all nine cells
    These are checked for exact equality, not within a tolerance: strata
    documents that it rounds exactly as GDAL's gdaldem does
    (apps/gdaldem_lib.cpp, float32 input), so the reference is that
    source's arithmetic in numpy: each n - e rounded to float32, sums
    folded left to right in row-major order in float32, "/ 8" as
    "* 0.125f", and Riley's squares, sum and root in float64 before one
    rounding to float32. The plane also checks all four against their
    closed forms in float64 (check 3).

The focal operations are checked against their definitions as shifted
sums of the float32 input, in float64:

  * correlate: out(x, y) = sum_j sum_c w[j][c] * z(x + c - r, y + j - r),
    with the (2r+1)x(2r+1) weights row-major, row 0 above the cell;
  * convolve:  the same with the weights rotated by 180 degrees, which is
    scipy.ndimage.convolve's definition;
  * separable: correlate with the weights col[j] * row[c];
  * mean:      the box sum divided by (2r+1)^2;
  * min, max:  of the float32 neighbourhood, exactly.

If scipy is installed, the reference itself is cross-checked against
scipy.ndimage.correlate and convolve, an implementation with nothing in
common with either side.

Min-max normalisation is the one float32 exception: it is checked for
exact equality against the textbook (z - z.min()) / (z.max() - z.min())
evaluated in float32 numpy over the valid cells, the way the pointwise
algebra is.

Tolerances are derived, not tuned: see TOLERANCES below.

Usage:  python check.py [dir]        (default: out)
        python check.py out --png    also writes PNGs to look at
"""

import json
import os
import sys

import numpy as np

# TOLERANCES
#
# strata computes in float32; this reference computes in float64 from the
# same float32 elevations, so the two differ by float32 rounding alone.
# Horn's numerator is a sum of six weighted elevations, each of magnitude
# at most zmax, accumulated in at most six roundings of a partial sum of
# magnitude at most 4*zmax. With a half-ulp relative error of 2^-24, the
# numerator is off by at most 6 * 2^-24 * 4 * zmax, so the gradient is off
# by at most that over 8*cellsize:
#
#       gtol = 3 * 2^-24 * zmax / cellsize
#
# Every other tolerance follows from how the operation propagates gtol,
# plus one more float32 rounding of the result itself (EPS * |result|).
# A real defect - a transposed kernel, a sign flip, degrees for radians,
# a cell size used in the wrong axis - is wrong by a fraction of the
# signal, which is thousands of times larger than these bounds.
#
# Curvature is built on its own differences. Each is a sum of at most
# four elevations: p's one rounding of magnitude 2*zmax, r's and t's two
# (4*zmax), s's three (4*zmax), then a rounded scale factor and product
# (2 * EPS * |result|). That gives per-derivative bounds, which are
# carried to first order through each formula with its partial
# derivatives, plus about one EPS per operation of the formula on the
# magnitude of its terms. First order is only valid while the gradient
# is known well: cells where the error of p and q exceeds 1% of the
# gradient's length are too flat for a direction of curvature, and are
# reported rather than judged, as aspect's are.

# The focal weighted sums are dot products of m = (2r+1)^2 terms (a
# separable sum is two of 2r+1 terms each), accumulated in float32 one
# rounding per product and per sum. The classic bound for a recursively
# summed dot product is gamma_m * sum |w_i z_i|, gamma_m = m*u/(1 - m*u)
# with u = 2^-24; for two nested passes it is gamma_{2k} over the outer
# product's |weights|. The mean adds one rounding for its division. So
#
#       ftol = (m + 1) * 2^-24 * sum |w_i| |z_i|  +  2^-24 * |result|
#
# which for the smooth test surfaces is a few thousandths of an ulp of
# the signal per term; a flipped kernel or a shifted window is wrong by a
# large fraction of the signal. Min and max involve no arithmetic, so
# they must be exact.
EPS = 2.0**-24  # float32 half-ulp, 5.96e-8
DEG = 180.0 / np.pi

OUT = "out"
WANT_PNG = False
for arg in sys.argv[1:]:
    if arg == "--png":
        WANT_PNG = True
    else:
        OUT = arg

MAN = json.load(open(os.path.join(OUT, "manifest.json")))
W, H = MAN["width"], MAN["height"]
FILL = MAN["fill"]

results = []

# Operations that are not a stencil: no border, no eroded validity.
POINTWISE = ("normalize",)


def record(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def load(name):
    """A result raster as float64, and its raw bits."""
    raw = np.fromfile(os.path.join(OUT, name), dtype="<f4")
    if raw.size != W * H:
        raise SystemExit(f"{name}: {raw.size} cells, expected {W * H}")
    return raw.astype(np.float64).reshape(H, W), raw.view(np.uint32).reshape(H, W)


def load_mask(name):
    if not name:
        return None
    m = np.fromfile(os.path.join(OUT, name), dtype=np.uint8).reshape(H, W)
    return m.astype(bool)


def horn(z, cx, cy):
    """Horn's gradient of z. Returns full-size arrays, NaN on the border."""
    dx = np.full(z.shape, np.nan)
    dy = np.full(z.shape, np.nan)
    a, b, c = z[0:-2, 0:-2], z[0:-2, 1:-1], z[0:-2, 2:]
    d, f = z[1:-1, 0:-2], z[1:-1, 2:]
    g, h, i = z[2:, 0:-2], z[2:, 1:-1], z[2:, 2:]
    dx[1:-1, 1:-1] = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cx)
    dy[1:-1, 1:-1] = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cy)
    return dx, dy


def zt(z, cx, cy):
    """Zevenbergen-Thorne p, q, r, s, t of z, NaN on the border."""
    out = [np.full(z.shape, np.nan) for _ in range(5)]
    a, b, c = z[0:-2, 0:-2], z[0:-2, 1:-1], z[0:-2, 2:]
    d, e, f = z[1:-1, 0:-2], z[1:-1, 1:-1], z[1:-1, 2:]
    g, h, i = z[2:, 0:-2], z[2:, 1:-1], z[2:, 2:]
    out[0][1:-1, 1:-1] = (f - d) / (2 * cx)
    out[1][1:-1, 1:-1] = (h - b) / (2 * cy)
    out[2][1:-1, 1:-1] = (d - 2 * e + f) / (cx * cx)
    out[3][1:-1, 1:-1] = (a - c - g + i) / (4 * cx * cy)
    out[4][1:-1, 1:-1] = (b - 2 * e + h) / (cy * cy)
    return out


CURVATURES = ("curvature_profile", "curvature_plan", "curvature_mean")

RUGGEDNESS = ("ruggedness_tri", "ruggedness_triwilson", "ruggedness_tpi", "ruggedness_roughness")


def ruggedness(op, z):
    """A ruggedness measure of z, with gdaldem's float32 arithmetic, as
    float64 holding float32 values, NaN on the border."""
    z = z.astype(np.float32)
    out = np.full(z.shape, np.nan)
    win = [z[j : j + z.shape[0] - 2, i : i + z.shape[1] - 2] for j in range(3) for i in range(3)]
    e, nb = win[4], win[:4] + win[5:]
    if op == "ruggedness_tri":
        s = np.zeros(e.shape)
        for n in nb:
            d = (n - e).astype(np.float64)  # the difference rounds in float32
            s = s + d * d
        v = np.sqrt(s).astype(np.float32)
    elif op == "ruggedness_triwilson":
        s = np.abs(nb[0] - e)
        for n in nb[1:]:
            s = s + np.abs(n - e)
        v = s * np.float32(0.125)
    elif op == "ruggedness_tpi":
        s = nb[0]
        for n in nb[1:]:
            s = s + n
        v = e - s * np.float32(0.125)
    else:
        v = np.max(win, axis=0) - np.min(win, axis=0)
    assert v.dtype == np.float32, v.dtype
    out[1:-1, 1:-1] = v
    return out


def curvature(op, z, cx, cy):
    """A curvature, its tolerance, and which cells are flat and which too
    flat to judge (see TOLERANCES)."""
    p, q, r, s, t = zt(z, cx, cy)
    zmax = np.nanmax(np.abs(z))
    ep = EPS * zmax / cx + 2 * EPS * np.abs(p)
    eq = EPS * zmax / cy + 2 * EPS * np.abs(q)
    er = 6 * EPS * zmax / (cx * cx) + 2 * EPS * np.abs(r)
    et = 6 * EPS * zmax / (cy * cy) + 2 * EPS * np.abs(t)
    es = 12 * EPS * zmax / (4 * cx * cy) + 2 * EPS * np.abs(s)
    g = p * p + q * q
    w = 1 + g
    flat = g == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "curvature_profile":
            num = p * p * r + 2 * p * q * s + q * q * t
            terms = np.abs(p * p * r) + np.abs(2 * p * q * s) + np.abs(q * q * t)
            den = g * w**1.5
            dn = (2 * p * r + 2 * q * s, 2 * p * s + 2 * q * t, p * p, 2 * p * q, q * q)
            dd = (2 * p * np.sqrt(w) * (w + 1.5 * g), 2 * q * np.sqrt(w) * (w + 1.5 * g))
        elif op == "curvature_plan":
            num = q * q * r - 2 * p * q * s + p * p * t
            terms = np.abs(q * q * r) + np.abs(2 * p * q * s) + np.abs(p * p * t)
            den = g**1.5
            dn = (2 * p * t - 2 * q * s, 2 * q * r - 2 * p * s, q * q, -2 * p * q, p * p)
            dd = (3 * p * np.sqrt(g), 3 * q * np.sqrt(g))
        else:
            num = (1 + q * q) * r - 2 * p * q * s + (1 + p * p) * t
            terms = np.abs((1 + q * q) * r) + np.abs(2 * p * q * s) + np.abs((1 + p * p) * t)
            den = 2 * w**1.5
            dn = (2 * p * t - 2 * q * s, 2 * q * r - 2 * p * s, 1 + q * q, -2 * p * q, 1 + p * p)
            dd = (6 * p * np.sqrt(w), 6 * q * np.sqrt(w))
            flat = np.zeros_like(flat)
        ref = -num / den
        # f = -N/D, so |df| <= (|dN| + |f| |dD|) / |D| for each variable.
        errs = (ep, eq, er, es, et)
        tol = sum(np.abs(dn[k]) * errs[k] for k in range(5))
        tol = tol + np.abs(ref) * (np.abs(dd[0]) * ep + np.abs(dd[1]) * eq)
        tol = tol / np.abs(den) + 8 * EPS * (terms / np.abs(den) + np.abs(ref))
        # The second-order remainder is under 5% of the first-order term
        # while p and q are known to 1%.
        tol = 1.05 * tol
        vague = ~flat & ((ep + eq) > 0.01 * np.sqrt(g))
        if op == "curvature_mean":
            vague = np.zeros_like(vague)
    ref = np.where(flat, 0.0, ref)
    return ref, tol, flat, vague


def radius(case):
    """The neighbourhood radius of a case's operation: 0 for pointwise."""
    if case["op"] in POINTWISE or case["op"].startswith("algebra_"):
        return 0
    return case.get("radius") or 1


def focal_weights(case):
    """The (2r+1)x(2r+1) weights a weighted focal case applies, as
    correlation weights, in float64 holding the exact float32 values."""
    r = case["radius"]
    k = 2 * r + 1
    op = case["op"]
    if op.startswith("focal_correlate") or op.startswith("focal_convolve"):
        w = np.array(case["weights"], np.float32).astype(np.float64).reshape(k, k)
        return w[::-1, ::-1] if op.startswith("focal_convolve") else w
    if op.startswith("focal_separable") or op.startswith("focal_gaussian"):
        row = np.array(case["row"], np.float32).astype(np.float64)
        col = np.array(case["col"], np.float32).astype(np.float64)
        return np.outer(col, row)
    if op.startswith("focal_mean"):
        return np.ones((k, k))
    return None


def shifted(z, r):
    """Yield (j, c, view) for every offset of a radius-r neighbourhood:
    view[y, x] = z[y + j, x + c], over the interior."""
    h, w = z.shape
    k = 2 * r + 1
    for j in range(k):
        for c in range(k):
            yield j, c, z[j : j + h - 2 * r, c : c + w - 2 * r]


def focal_expected(z, case):
    """A focal operation from its definition, full size, NaN on the
    border, with its tolerance."""
    r = case["radius"]
    op = case["op"]
    ref = np.full(z.shape, np.nan)
    tol = np.full(z.shape, np.nan)
    inner = (slice(r, z.shape[0] - r), slice(r, z.shape[1] - r))
    if op.startswith("focal_min") or op.startswith("focal_max"):
        # In float32, where min and max are exact: NaN (NoData) wins, as
        # it must, and every such cell is invalid anyway.
        pick = np.fmin if op.startswith("focal_min") else np.fmax
        acc = None
        for _, _, v in shifted(z.astype(np.float32), r):
            acc = v.copy() if acc is None else np.where(np.isnan(acc) | np.isnan(v), np.nan, pick(acc, v))
        ref[inner] = acc
        tol[inner] = 0.0
        return ref, tol
    w = focal_weights(case)
    k = 2 * r + 1
    total = np.zeros_like(z[inner])
    scale = np.zeros_like(z[inner])
    for j, c, v in shifted(z, r):
        total += w[j, c] * v
        scale += abs(w[j, c]) * np.abs(v)
    m = k * k
    if op.startswith("focal_mean"):
        total /= m
        scale /= m
    ref[inner] = total
    tol[inner] = (m + 1) * EPS * scale + EPS * np.abs(total)
    return ref, tol


def expected(op, z, case):
    """The reference result and its tolerance, or (None, None)."""
    if op.startswith("focal_"):
        return focal_expected(z, case)
    cx = case["cell_size"]
    cy = case["cell_size_y"] or cx
    dx, dy = horn(z, cx, cy)
    zmax = np.nanmax(np.abs(z))
    tolx = 3 * EPS * zmax / cx
    toly = 3 * EPS * zmax / cy
    gtol = np.hypot(tolx, toly)

    if op == "gradient_dx":
        return dx, tolx + EPS * np.abs(dx)
    if op == "gradient_dy":
        return dy, toly + EPS * np.abs(dy)

    m = np.hypot(dx, dy)
    if op == "slope_rad":
        # |d atan(m) / dm| <= 1.
        r = np.arctan(m)
        return r, gtol + EPS * r
    if op == "slope_deg":
        r = DEG * np.arctan(m)
        return r, DEG * gtol + EPS * r
    if op == "slope_pct":
        return 100.0 * m, 100.0 * gtol + EPS * 100.0 * m
    if op == "aspect":
        asp = (DEG * np.arctan2(-dx, dy)) % 360.0
        asp[(dx == 0) & (dy == 0)] = -1.0  # AspectFlat
        # A rotation of the gradient vector by gtol turns the bearing by
        # gtol/m radians, which blows up as the cell flattens.
        with np.errstate(divide="ignore", invalid="ignore"):
            return asp, DEG * gtol / m + EPS * 360.0
    if op in CURVATURES:
        ref, tol, _, _ = curvature(op, z, cx, cy)
        return ref, tol
    if op == "hillshade":
        az = np.radians(case["azimuth"])
        alt = np.radians(case["altitude"])
        cos_i = (np.sin(alt) + np.cos(alt) * (dy * np.cos(az) - dx * np.sin(az))) / np.sqrt(
            1 + dx * dx + dy * dy
        )
        hs = np.clip(255.0 * cos_i, 0.0, 255.0)
        # |d cos_i / d(dx or dy)| <= 1 for a unit light vector.
        return hs, 255.0 * 2 * gtol + EPS * 255.0
    return None, None


def defined(out_mask, r=1):
    """Cells that must hold a defined value: at least r from the edge,
    and valid."""
    keep = np.zeros((H, W), bool)
    keep[r : H - r, r : W - r] = True
    if out_mask is not None:
        keep &= out_mask
    return keep


def angular_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# --------------------------------------------------------------------
# 1. Each result against the independent reference.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    op = case["op"]
    if op.startswith("algebra_"):
        continue
    z, _ = load(case["dem"])
    dem_mask = load_mask(case.get("dem_mask"))
    if dem_mask is not None:
        z = np.where(dem_mask, z, np.nan)  # NoData must not enter the maths
    got, _ = load(case["out"])
    out_mask = load_mask(case.get("out_mask"))
    if op in RUGGEDNESS:
        ref = ruggedness(op, z)
        keep = defined(out_mask) & np.isfinite(ref)
        bad = int((got[keep] != ref[keep]).sum())
        record(f"{case['name']} == gdaldem's arithmetic", bad == 0,
               f"{bad} differing cells over {int(keep.sum())}")
        continue
    ref, tol = expected(op, z, case)
    if ref is None:
        continue

    keep = defined(out_mask, radius(case)) & np.isfinite(ref)

    if op == "aspect":
        dx, dy = horn(z, case["cell_size"], case["cell_size_y"] or case["cell_size"])
        flat = (dx == 0) & (dy == 0)
        # Where the tolerance has grown past a few degrees the cell is too
        # flat for its direction to mean anything: report, do not judge.
        vague = keep & ~flat & (tol > 5.0)
        judge = keep & ~flat & (tol <= 5.0)
        margin = (angular_diff(got, ref) / tol)[judge]
        worst = margin.max() if judge.any() else 0.0
        flat_ok = bool(np.all(got[keep & flat] == -1.0)) if (keep & flat).any() else True
        record(
            f"{case['name']} vs Horn reference",
            worst <= 1.0 and flat_ok,
            f"{worst:.2f}x tolerance over {int(judge.sum())} cells, "
            f"{int(vague.sum())} too flat to judge"
            + ("" if flat_ok else ", FLAT CELLS NOT -1"),
        )
        continue

    if op in CURVATURES:
        cx = case["cell_size"]
        _, _, flat, vague = curvature(op, z, cx, case["cell_size_y"] or cx)
        judge = keep & ~flat & ~vague
        err = np.abs(got - ref)
        worst = (err[judge] / tol[judge]).max() if judge.any() else 0.0
        flat_ok = bool(np.all(got[keep & flat] == 0.0))
        record(
            f"{case['name']} vs Zevenbergen-Thorne reference",
            worst <= 1.0 and flat_ok,
            f"{worst:.2f}x tolerance over {int(judge.sum())} cells, "
            f"{int((keep & vague).sum())} too flat to judge, {int((keep & flat).sum())} flat"
            + ("" if flat_ok else ", FLAT CELLS NOT 0"),
        )
        continue

    err = np.abs(got - ref)
    tol = np.broadcast_to(np.asarray(tol, float), err.shape)
    if op.startswith("focal_min") or op.startswith("focal_max"):
        bad = int((got[keep] != ref[keep]).sum())
        record(f"{case['name']} == float32 {op[6:9]} of the neighbourhood", bad == 0,
               f"{bad} differing cells over {int(keep.sum())}")
        continue
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(err == 0, 0.0, err / tol)
    worst = ratio[keep].max() if keep.any() else 0.0
    record(
        f"{case['name']} vs {'definition' if op.startswith('focal_') else 'Horn reference'}",
        worst <= 1.0,
        f"max error {err[keep].max():.3e}, tolerance {tol[keep].max():.3e} "
        f"({worst:.2f}x) over {int(keep.sum())} cells",
    )


# --------------------------------------------------------------------
# 2. The border carries no data: Horn needs all eight neighbours, and a
#    radius-r focal operation the whole (2r+1)x(2r+1) square, so a cell
#    within r of the edge is NaN, or - in a file with a NoData value -
#    the fill, with its validity bit cleared.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    r = radius(case)
    if r == 0:
        continue
    got, _ = load(case["out"])
    border = np.ones((H, W), bool)
    border[r : H - r, r : W - r] = False
    out_mask = load_mask(case.get("out_mask"))
    if out_mask is None:
        bad = int((~np.isnan(got[border])).sum())
        detail = f"{bad} border cells are not NaN"
    else:
        bad = int(out_mask[border].sum())
        detail = f"{bad} border cells are marked valid"
    record(f"{case['name']} border carries no data", bad == 0, detail)


# --------------------------------------------------------------------
# 3. The plane, against the analytic answer: one slope and one aspect
#    for the whole raster, from the surface's own coefficients.
# --------------------------------------------------------------------

PLANE_A, PLANE_B = 0.3, -0.7  # rise per column, rise per row (see main.go)

for case in MAN["rasters"]:
    if case["surface"] != "plane" or case["op"] not in ("slope_deg", "aspect") + CURVATURES + RUGGEDNESS:
        continue
    cx = case["cell_size"]
    cy = case["cell_size_y"] or cx
    tdx, tdy = PLANE_A / cx, PLANE_B / cy
    got, _ = load(case["out"])
    keep = defined(None)
    if case["op"] in RUGGEDNESS:
        # Neighbour (i, j) differs from the centre by a*i + b*j, so the
        # squares sum to 6a^2 + 6b^2, the absolute differences are |a|,
        # |b|, |a + b| and |a - b| twice each, the neighbours average to
        # the centre, and the window spans 2|a| + 2|b|. Each float32
        # difference is off by at most an ulp of the elevations, and the
        # sums add seven roundings each.
        a, b = PLANE_A, PLANE_B
        want = {
            "ruggedness_tri": np.sqrt(6 * a * a + 6 * b * b),
            "ruggedness_triwilson": 2 * (abs(a) + abs(b) + abs(a + b) + abs(a - b)) / 8,
            "ruggedness_tpi": 0.0,
            "ruggedness_roughness": 2 * abs(a) + 2 * abs(b),
        }[case["op"]]
        z, _ = load(case["dem"])
        tol = 16 * EPS * np.nanmax(np.abs(z))
        err = np.abs(got[keep] - want).max()
        record(f"{case['name']} = analytic {want:.4f}", err <= tol, f"max error {err:.2e}, tolerance {tol:.2e}")
        continue
    if case["op"] in CURVATURES:
        # A plane has no curvature: whatever strata reports is the
        # rounding of the float32 elevations, which the derived tolerance
        # of the reference covers when the reference itself is taken as 0.
        z, _ = load(case["dem"])
        _, tol, _, _ = curvature(case["op"], z, cx, cy)
        worst = (np.abs(got[keep]) / tol[keep]).max()
        record(f"{case['name']} = analytic 0", worst <= 1.0, f"max {np.abs(got[keep]).max():.2e} ({worst:.2f}x)")
        continue
    if case["op"] == "slope_deg":
        want = DEG * np.arctan(np.hypot(tdx, tdy))
        err = np.abs(got[keep] - want).max()
        record(f"{case['name']} = analytic {want:.4f} deg", err < 0.05, f"max {err:.2e} deg")
    else:
        want = (DEG * np.arctan2(-tdx, tdy)) % 360.0
        err = angular_diff(got[keep], want).max()
        record(f"{case['name']} = analytic {want:.4f} deg", err < 0.2, f"max {err:.2e} deg")


# --------------------------------------------------------------------
# 4. plain == tiled == chunked, bit for bit (the README's promise).
# --------------------------------------------------------------------

groups = {}
for case in MAN["rasters"]:
    groups.setdefault((case["surface"], case["op"], case["dem"]), {})[case["form"]] = case

for forms in groups.values():
    if "plain" not in forms or len(forms) < 2:
        continue
    base = forms["plain"]
    base_vals, base_bits = load(base["out"])
    base_nan = np.isnan(base_vals)
    base_mask = load_mask(base.get("out_mask"))
    for form, case in sorted(forms.items()):
        if form == "plain":
            continue
        got, bits = load(case["out"])
        same = (bits == base_bits) | (np.isnan(got) & base_nan)
        if base_mask is not None:
            # Data under invalid cells is documented as unspecified.
            same |= ~base_mask
            mask_same = np.array_equal(base_mask, load_mask(case.get("out_mask")))
        else:
            mask_same = True
        bad = int((~same).sum())
        record(
            f"{base['name']} == {case['name']} bit for bit",
            bad == 0 and mask_same,
            f"{bad} differing cells" + ("" if mask_same else ", masks differ"),
        )


# --------------------------------------------------------------------
# 5. Validity: an output cell is valid iff it is at least r from the
#    edge and all (2r+1)^2 cells of its neighbourhood are valid (nine for
#    the terrain operations).
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    r = radius(case)
    if not case.get("out_mask") or r == 0 or case.get("weight"):
        continue  # a weighted case's validity is check 11's
    src = load_mask(case["dem_mask"])
    want = np.zeros((H, W), bool)
    eroded = np.ones((H - 2 * r, W - 2 * r), bool)
    for _, _, v in shifted(src, r):
        eroded &= v
    want[r : H - r, r : W - r] = eroded
    got = load_mask(case["out_mask"])
    bad = int((got != want).sum())
    record(f"{case['name']} validity", bad == 0, f"{bad} cells with the wrong validity")


# --------------------------------------------------------------------
# 6. Pointwise algebra, against numpy in float32.
# --------------------------------------------------------------------

PLANE = np.fromfile(os.path.join(OUT, "plane.f32"), dtype="<f4").reshape(H, W)
HILL = np.fromfile(os.path.join(OUT, "hill.f32"), dtype="<f4").reshape(H, W)
ALGEBRA = {
    "algebra_add": PLANE + HILL,
    "algebra_sub": PLANE - HILL,
    "algebra_mul": PLANE * HILL,
    "algebra_min": np.minimum(PLANE, HILL),
    "algebra_max": np.maximum(PLANE, HILL),
    "algebra_clamp": np.clip(HILL, np.float32(50), np.float32(300)),
}

for case in MAN["rasters"]:
    want = ALGEBRA.get(case["op"])
    if want is None:
        continue
    got = np.fromfile(os.path.join(OUT, case["out"]), dtype="<f4").reshape(H, W)
    bad = int((got != want).sum())
    record(f"{case['name']} == numpy float32", bad == 0, f"{bad} differing cells")


# --------------------------------------------------------------------
# 7. Reductions, against numpy over the valid cells.
# --------------------------------------------------------------------

for s in MAN["scalars"]:
    z = np.fromfile(os.path.join(OUT, s["dem"]), dtype="<f4").reshape(H, W)
    mask = load_mask(s.get("dem_mask"))
    vals = z[mask] if mask is not None else z.ravel()
    if s["op"] == "count":
        want = int(vals.size)
        record(f"{s['name']} = {want}", s["count"] == want, f"got {s['count']}")
    else:
        wmin, wmax = vals.min(), vals.max()
        gmin, gmax = np.float32(s["min"]), np.float32(s["max"])
        ok = gmin == wmin and gmax == wmax and s["count"] == vals.size
        record(
            f"{s['name']} = ({wmin:g}, {wmax:g}, {vals.size})",
            ok,
            f"got ({gmin:g}, {gmax:g}, {s['count']})",
        )


# --------------------------------------------------------------------
# 8. The three slope units agree with each other, over valid cells.
# --------------------------------------------------------------------

for surface in {c["surface"] for c in MAN["rasters"]} - {"algebra"}:
    try:
        deg, _ = load(f"{surface}-slope_deg-plain.f32")
        rad, _ = load(f"{surface}-slope_rad-plain.f32")
        pct, _ = load(f"{surface}-slope_pct-plain.f32")
    except (FileNotFoundError, OSError, SystemExit):
        continue
    keep = defined(load_mask(f"{surface}-slope_deg-plain.mask.u8")
                   if os.path.exists(os.path.join(OUT, f"{surface}-slope_deg-plain.mask.u8"))
                   else None)
    keep &= np.isfinite(deg) & np.isfinite(rad) & np.isfinite(pct)
    e1 = np.abs(np.radians(deg[keep]) - rad[keep]).max()
    # 100*tan(atan(m)) recovers 100*m, amplified by 1 + m^2 = 1 + (pct/100)^2.
    amp = 1 + (pct[keep] / 100.0) ** 2
    e2 = (np.abs(100 * np.tan(rad[keep]) - pct[keep]) / amp).max()
    record(f"{surface} slope radians == degrees", e1 < 1e-5, f"max {e1:.2e} rad")
    record(f"{surface} slope percent == 100*tan", e2 < 1e-3, f"max {e2:.2e} % (scaled)")


# --------------------------------------------------------------------
# 9. Min-max normalisation, against numpy in float32 over the valid
#    cells: the same bits, the input's validity, and the endpoints 0
#    and 1 hit exactly.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    if case["op"] != "normalize":
        continue
    z = np.fromfile(os.path.join(OUT, case["dem"]), dtype="<f4").reshape(H, W)
    dem_mask = load_mask(case.get("dem_mask"))
    valid = dem_mask if dem_mask is not None else np.ones((H, W), bool)
    lo, hi = z[valid].min(), z[valid].max()
    want = (z - lo) / (hi - lo)  # float32 throughout
    got = np.fromfile(os.path.join(OUT, case["out"]), dtype="<f4").reshape(H, W)
    bad = int((got[valid] != want[valid]).sum())
    out_mask = load_mask(case.get("out_mask"))
    mask_ok = out_mask is None if dem_mask is None else np.array_equal(out_mask, dem_mask)
    record(
        f"{case['name']} == numpy float32",
        bad == 0 and mask_ok,
        f"{bad} differing cells" + ("" if mask_ok else ", validity differs from the input's"),
    )
    g = got[valid]
    ends = g.min() == 0 and g.max() == 1
    record(
        f"{case['name']} spans [0, 1] exactly",
        ends,
        f"min {g.min():.9g}, max {g.max():.9g} over {int(valid.sum())} cells",
    )


# --------------------------------------------------------------------
# 10. The focal reference itself, against scipy.ndimage where it is
#     installed: correlate and convolve by an implementation that shares
#     nothing with check.py's shifted sums (or with strata), on the same
#     float64 data. mode="constant" is irrelevant: only interior cells,
#     whose neighbourhood is inside the raster, are compared.
# --------------------------------------------------------------------

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

if ndimage is not None:
    for case in MAN["rasters"]:
        op = case["op"]
        if case["form"] != "plain" or not (op.startswith("focal_correlate") or op.startswith("focal_convolve")):
            continue
        z, _ = load(case["dem"])
        r = case["radius"]
        k = 2 * r + 1
        w = np.array(case["weights"], np.float32).astype(np.float64).reshape(k, k)
        fn = ndimage.convolve if op.startswith("focal_convolve") else ndimage.correlate
        sp = fn(z, w, mode="constant", cval=0.0)
        ref, tol = focal_expected(z, case)
        inner = (slice(r, H - r), slice(r, W - r))
        # Both are float64 sums of the same terms in different orders.
        err = np.abs(sp[inner] - ref[inner]).max()
        scale = np.abs(ref[inner]).max()
        record(f"{case['name']} definition == scipy.ndimage.{fn.__name__}", err <= 1e-12 * max(scale, 1),
               f"max {err:.2e}")


# --------------------------------------------------------------------
# 11. Weighted slope: Horn's slope in degrees times a weight read at the
#     cell itself. Data against the float64 reference, with the slope's
#     tolerance scaled by the weight and one more rounding for the
#     product; validity the 3x3 erosion of the DEM's mask AND the
#     weight's own mask, not eroded. An implementation that eroded the
#     weight too would lose the 3x3 around every NoData weight.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    if not case.get("weight"):
        continue
    z, _ = load(case["dem"])
    dem_mask = load_mask(case.get("dem_mask"))
    if dem_mask is not None:
        z = np.where(dem_mask, z, np.nan)
    wt, _ = load(case["weight"])
    wmask = load_mask(case["weight_mask"])
    wt = np.where(wmask, wt, np.nan)
    slope, stol = expected("slope_deg", z, case)
    ref = slope * wt
    tol = stol * np.abs(wt) + EPS * np.abs(ref)
    got, _ = load(case["out"])
    out_mask = load_mask(case["out_mask"])

    want = np.zeros((H, W), bool)
    eroded = np.ones((H - 2, W - 2), bool)
    for _, _, v in shifted(dem_mask if dem_mask is not None else np.ones((H, W), bool), 1):
        eroded &= v
    want[1 : H - 1, 1 : W - 1] = eroded
    want &= wmask
    bad = int((out_mask != want).sum())
    record(f"{case['name']} validity == eroded DEM AND weight", bad == 0,
           f"{bad} cells with the wrong validity")
    # The eroded-everything reading is wrong here; make sure the case can
    # tell the two apart, or the check above proves nothing.
    wrong = want.copy()
    weroded = np.ones((H - 2, W - 2), bool)
    for _, _, v in shifted(wmask, 1):
        weroded &= v
    wrong[1 : H - 1, 1 : W - 1] &= weroded
    record(f"{case['name']} case distinguishes an eroded weight", int((wrong != want).sum()) > 0,
           f"{int((wrong != want).sum())} cells differ between the two readings")

    keep = defined(out_mask) & np.isfinite(ref)
    err = np.abs(got - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(err == 0, 0.0, err / tol)
    worst = ratio[keep].max() if keep.any() else 0.0
    record(f"{case['name']} vs Horn reference times weight", worst <= 1.0,
           f"max error {err[keep].max():.3e} ({worst:.2f}x tolerance) over {int(keep.sum())} cells")


# --------------------------------------------------------------------
# 12. Surface: every product it writes, several at once from one
#     gradient, is the standalone operation's file bit for bit, Data
#     and validity, in every form. The standalone files are judged
#     against their references above (and against gdaldem by
#     gdalcompare.py), so this carries those verdicts over to Surface.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    if not case["op"].startswith("surface_"):
        continue
    alone = f"{case['dem'][:-4]}-{case['op'][len('surface_'):]}-{case['form']}"
    got, gbits = load(case["out"])
    want, wbits = load(alone + ".f32")
    differ = int(((gbits != wbits) & ~(np.isnan(got) & np.isnan(want))).sum())
    mask_same = True
    if case.get("out_mask"):
        mask_same = np.array_equal(load_mask(case["out_mask"]), load_mask(alone + ".mask.u8"))
    record(f"{case['name']} == {alone} bit for bit", differ == 0 and mask_same,
           f"{differ} differing cells" + ("" if mask_same else ", masks differ"))


# --------------------------------------------------------------------
# Pictures, for the eyeball check.
# --------------------------------------------------------------------

if WANT_PNG:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_dir = os.path.join(OUT, "png")
    os.makedirs(png_dir, exist_ok=True)
    # Fixed ranges where the operation has one, so a picture cannot be
    # flattered by auto-scaling. NoData and the border show up magenta.
    for name, cmap, lo, hi in [
        ("hill.f32", "terrain", None, None),
        ("noisy.f32", "terrain", None, None),
        ("hill-hillshade-plain.f32", "gray", 0, 255),
        ("noisy-hillshade-plain.f32", "gray", 0, 255),
        ("hill-slope_deg-plain.f32", "magma", 0, 90),
        ("noisy-slope_deg-plain.f32", "magma", 0, 90),
        ("hill-aspect-plain.f32", "hsv", 0, 360),
        ("noisy-aspect-plain.f32", "hsv", 0, 360),
        ("hill-curvature_profile-plain.f32", "RdBu", None, None),
        ("hill-curvature_plan-plain.f32", "RdBu", None, None),
        ("hill-curvature_mean-plain.f32", "RdBu", None, None),
        ("noisy-ruggedness_tri-plain.f32", "viridis", None, None),
        ("noisy-ruggedness_tpi-plain.f32", "RdBu", None, None),
        ("noisy-ruggedness_roughness-plain.f32", "viridis", None, None),
    ]:
        a, _ = load(name)
        a = np.where(a == FILL, np.nan, a)
        # Under an invalid cell the data is unspecified, so the picture
        # has to be cut by the validity mask, not by the values.
        m = load_mask(name.replace(".f32", ".mask.u8")
                      if os.path.exists(os.path.join(OUT, name.replace(".f32", ".mask.u8")))
                      else None)
        if m is not None:
            a = np.where(m, a, np.nan)
        if lo is None:
            lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
        cm = matplotlib.colormaps[cmap].with_extremes(bad="magenta")
        plt.imsave(
            os.path.join(png_dir, name.replace(".f32", ".png")),
            np.ma.masked_invalid(a),
            cmap=cm,
            vmin=lo,
            vmax=hi,
        )
    print(f"wrote PNGs to {png_dir}\n")


# --------------------------------------------------------------------

failed = [r for r in results if not r[1]]
pad = max(len(r[0]) for r in results)
for name, ok, detail in results:
    if failed and ok:
        continue  # on a failure, show only what failed
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{pad}}  {detail}")
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
