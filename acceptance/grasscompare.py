#!/usr/bin/env python3
"""Difference strata's multi-scale derivatives against GRASS GIS
r.param.scale's, on a real raster.

r.param.scale is Jo Wood's own implementation of the method strata's
FitRadius documents: the quadratic z = ax^2 + by^2 + cxy + dx + ey + f
fitted by least squares to the (2r+1)^2 window around each cell. It is
a different program, in C, that solves the fit a different way: it
subtracts the centre elevation from the window, forms the six normal
equations in float64 and solves them by LU decomposition, where strata
applies closed-form integer taps to the raw window in float32. With
size = 2r+1 and exponent = 0 (no distance weighting) both compute the
same fit. What each output is was established from the GRASS 8.5.0
source (raster/r.param.scale, lib/gmath/lu.c), not assumed:

  * Frame. x = res*(col - edge), y = res*(row - edge) (find_normal.c):
    y increases southward, as strata's does, so GRASS's d and e are
    strata's p and q (Gradient's dx and dy), 2a and 2b are r and t, and
    c is s. res is the region's ns_res (the mean of the two if they
    differ by 1% or more), a ground distance only in a projected
    project; grassrun.sh refuses a lat/long one.

  * slope = atan(sqrt(d^2 + e^2)) in degrees (param.c): strata's
    SlopeDegrees.

  * aspect = atan2(e, d) in degrees, -180..180 (param.c). The manual
    calls it the downslope direction with West 0, North +90, East
    +/-180, South -90. strata's is the compass bearing of the downslope
    direction, clockwise from north, so strata = (270 + GRASS) mod 360.
    Where d = e = 0 GRASS writes atan2(0, 0) = 0 and strata writes -1.

  * profc = -2(ad^2 + be^2 + cde) / ((d^2+e^2)(1+d^2+e^2)^1.5)
          = -(p^2 r + 2pqs + q^2 t) / (g (1+g)^1.5),   g = p^2 + q^2:
    strata's CurvatureProfile, the same formula and sign.

  * planc = 2(bd^2 + ae^2 - cde) / (d^2+e^2)^1.5
          = (q^2 r - 2pqs + p^2 t) / g^1.5:
    strata's CurvaturePlan with the opposite sign. Compared as -planc.

  * longc, crosc, minic, maxic have no strata counterpart: longc and
    crosc are Wood's curvatures without the slope terms, minic and maxic
    the eigenvalues of -Hessian. strata's CurvatureMean is not one of
    GRASS's outputs, but it follows from three of them exactly:
        minic + maxic = -(r + t),  crosc = -(q^2 r - 2pqs + p^2 t) / g,
        mean = -((1+q^2) r - 2pqs + (1+p^2) t) / (2 (1+g)^1.5)
             = (minic + maxic + g crosc) / (2 (1+g)^1.5),  g = tan^2(slope),
    which holds at g = 0 too (GRASS's crosc is 0 there). longc is not
    used.

  * profc, planc, longc and crosc are 0 where d = e = 0 exactly; strata
    documents 0 for profile and plan where p = q = 0.

  * slope_tolerance and curvature_tolerance are read only by
    method=feature (feature.c); grassrun.sh sets both to 0 anyway.
    zscale multiplies a..e and is set to 1. exponent 0 makes every
    weight 1/(dist+1)^0 = 1.

  * Edges and NULL. The outer (size-1)/2 rows and columns are NULL, and
    a cell is NULL if any cell of its window is (process.c). strata
    clears the validity bit in the same places. The comparison requires
    the two sets to be identical.

  * Precision. GRASS computes in float64 and writes DCELL (float64);
    grassrun.sh exports float64, so GRASS's values arrive unrounded.

Tolerances. Each tool's error is bounded from its own arithmetic, and
the gap between them may be the sum of the two, carried through the
formula of each product:

  * strata, as check.py's wood() derives from what terrain/doc.go says
    strata computes: each derivative is a float32 sum over the window
    with integer taps (a column pass of 2r+1 terms, then a row pass),
    off by at most gamma_{2k} * zmax * sum|w|, then one float32 factor
    and product (2u of the result). Here zmax is the largest |z| in the
    cell's own window, not the raster's.

  * GRASS: the normal equations N c = obs, N exact for this cell size
    (checked below), obs summed in float64 from the centred window
    (|z - centre| <= the window's range R, so each obs is off by at most
    gamma_{m+2} R sum|basis|), then G_ludcmp/G_lubksb, which is backward
    stable: (N + dN) c^ = obs with |dN| <= gamma_18 P'|L||U| (Higham,
    Accuracy and Stability of Numerical Algorithms, Thm 9.4). So
        |c^ - c| <= |N^-1| (gamma_18 P'|L||U| |c| + |d obs|),
    with L, U and P those of G_ludcmp's own pivoting, replicated here.

  * The products' own arithmetic: strata's documented float32 atan
    (<= 1.5e-7 rad) and atan2 (<= 3e-7 rad), the roundings of |grad|
    and of each result, and for curvature check.py's first-order
    propagation and per-operation EPS terms. GRASS's formulas in
    float64 add ~1e-16 relative, included for completeness.

Cells whose gradient is known too poorly for a direction (aspect
tolerance over 5 degrees; for profile and plan the gradient error over
1% of its length) are reported, not judged, as in check.py.

Both tools are also scored against a float64 reference built from the
closed forms (exact for integer elevations: the sums are integers below
2^53), to say which of the two is closer, not only that they agree.

Usage:  python grasscompare.py [dir] [width] [height] [nodata] [cell] [radii]
"""

import os
import sys
from fractions import Fraction

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

D = sys.argv[1] if len(sys.argv) > 1 else "out-grass"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
DEM_NODATA = float(np.float32(sys.argv[4])) if len(sys.argv) > 4 else 65535.0
CELL = float(sys.argv[5]) if len(sys.argv) > 5 else 12.5
RADII = [int(v) for v in (sys.argv[6] if len(sys.argv) > 6 else "1,4,8").split(",")]

FILL = -9999.0  # what both sides write under NULL / invalid cells
EPS = 2.0**-24  # float32 half-ulp
U64 = 2.0**-53  # float64 half-ulp
DEG = 180.0 / np.pi
ATAN32 = 1.5e-7  # rad: Atan32's documented bound (TestAtan32Accuracy)
ATAN2F32 = 3e-7  # rad: Atan2F32's documented bound (TestAtan2F32Accuracy)


def gamma(n, u=EPS):
    return n * u / (1 - n * u)


results = []


def record(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44}  {detail}")


def f32(name):
    return np.fromfile(os.path.join(D, name), dtype="<f4").reshape(H, W).astype(np.float64)


def f64(name):
    return np.fromfile(os.path.join(D, name), dtype="<f8").reshape(H, W)


def worst_ratio(err, tol):
    """The largest err/tol; a zero error is within any bound, including
    the zero bound of a constant window."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(err == 0, 0.0, err / tol).max())


def angular_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# --------------------------------------------------------------------
# GRASS's normal equations and its LU, replicated.
# --------------------------------------------------------------------


def normal_matrix(r, res):
    """find_normal() for exponent 0, exactly (Fractions) and as GRASS's
    float64 sums would hold it."""
    k = 2 * r + 1
    res_f = Fraction(res)
    s = {n: Fraction(0) for n in "x4 x2y2 x3y x3 x2y x2 y4 xy3 xy2 y3 y2 xy x1 y1 N".split()}
    sf = {n: 0.0 for n in s}
    for row in range(k):
        for col in range(k):
            for acc, x, y in ((s, res_f * (col - r), res_f * (row - r)), (sf, res * (col - r), res * (row - r))):
                acc["x4"] += x * x * x * x; acc["x2y2"] += x * x * y * y; acc["x3y"] += x * x * x * y
                acc["x3"] += x * x * x; acc["x2y"] += x * x * y; acc["x2"] += x * x
                acc["y4"] += y * y * y * y; acc["xy3"] += x * y * y * y; acc["xy2"] += x * y * y
                acc["y3"] += y * y * y; acc["y2"] += y * y; acc["xy"] += x * y
                acc["x1"] += x; acc["y1"] += y; acc["N"] += 1

    def mat(a):
        return [[a["x4"], a["x2y2"], a["x3y"], a["x3"], a["x2y"], a["x2"]],
                [a["x2y2"], a["y4"], a["xy3"], a["xy2"], a["y3"], a["y2"]],
                [a["x3y"], a["xy3"], a["x2y2"], a["x2y"], a["xy2"], a["xy"]],
                [a["x3"], a["xy2"], a["x2y"], a["x2"], a["xy"], a["x1"]],
                [a["x2y"], a["y3"], a["xy2"], a["xy"], a["y2"], a["y1"]],
                [a["x2"], a["y2"], a["xy"], a["x1"], a["y1"], a["N"]]]

    exact, held = mat(s), mat(sf)
    same = all(float(exact[i][j]) == held[i][j] and Fraction(held[i][j]) == exact[i][j]
               for i in range(6) for j in range(6))
    return np.array(held), same


def grass_lu(n):
    """G_ludcmp (lib/gmath/lu.c, Crout with implicit scaled partial
    pivoting), serially: returns L (unit lower), U and the permutation
    matrix P with P N = L U."""
    a = [list(map(float, row)) for row in n]
    m = len(a)
    vv = [1.0 / max(abs(x) for x in row) for row in a]
    perm = list(range(m))
    for j in range(m):
        for i in range(j):
            s = a[i][j]
            for k in range(i):
                s -= a[i][k] * a[k][j]
            a[i][j] = s
        big, imax = 0.0, j
        for i in range(j, m):
            s = a[i][j]
            for k in range(j):
                s -= a[i][k] * a[k][j]
            a[i][j] = s
            if vv[i] * abs(s) >= big:
                big, imax = vv[i] * abs(s), i
        if j != imax:
            a[imax], a[j] = a[j], a[imax]
            vv[imax] = vv[j]
            perm[imax], perm[j] = perm[j], perm[imax]
        if a[j][j] == 0.0:
            a[j][j] = 1.0e-20
        for i in range(j + 1, m):
            a[i][j] /= a[j][j]
    a = np.array(a)
    L = np.tril(a, -1) + np.eye(m)
    U = np.triu(a)
    P = np.zeros((m, m))
    for i, pi in enumerate(perm):
        P[i, pi] = 1.0
    return L, U, P


# --------------------------------------------------------------------
# Windowed sums, in float64.
# --------------------------------------------------------------------


def colpass(z, taps):
    r = (len(taps) - 1) // 2
    out = np.zeros((z.shape[0] - 2 * r, z.shape[1]))
    for j, w in enumerate(taps):
        if w:
            out += w * z[j : j + out.shape[0]]
    return out


def rowpass(c, taps):
    r = (len(taps) - 1) // 2
    out = np.zeros((c.shape[0], c.shape[1] - 2 * r))
    for i, w in enumerate(taps):
        if w:
            out += w * c[:, i : i + out.shape[1]]
    return out


def full(inner, r):
    out = np.full((H, W), np.nan)
    out[r : H - r, r : W - r] = inner
    return out


def window_reduce(z, r, fn):
    k = 2 * r + 1
    a = fn(sliding_window_view(z, k, axis=0), axis=-1)
    return full(fn(sliding_window_view(a, k, axis=1), axis=-1), r)


# --------------------------------------------------------------------
# Curvature from derivatives, with check.py's first-order bound.
# --------------------------------------------------------------------


def curvature_from(op, p, q, r, s, t, ep, eq, er, es, et):
    """check.py's curvature_from: the curvature, its tolerance given
    bounds on the errors of p..t, which cells are flat and which too
    flat to judge."""
    g = p * p + q * q
    w = 1 + g
    flat = g == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "profile":
            num = p * p * r + 2 * p * q * s + q * q * t
            terms = np.abs(p * p * r) + np.abs(2 * p * q * s) + np.abs(q * q * t)
            den = g * w**1.5
            dn = (2 * p * r + 2 * q * s, 2 * p * s + 2 * q * t, p * p, 2 * p * q, q * q)
            dd = (2 * p * np.sqrt(w) * (w + 1.5 * g), 2 * q * np.sqrt(w) * (w + 1.5 * g))
        elif op == "plan":
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
        errs = (ep, eq, er, es, et)
        tol = sum(np.abs(dn[k]) * errs[k] for k in range(5))
        tol = tol + np.abs(ref) * (np.abs(dd[0]) * ep + np.abs(dd[1]) * eq)
        tol = tol / np.abs(den) + 8 * EPS * (terms / np.abs(den) + np.abs(ref))
        tol = 1.05 * tol
        vague = ~flat & ((ep + eq) > 0.01 * np.sqrt(g))
        if op == "mean":
            vague = np.zeros_like(vague)
    ref = np.where(flat, 0.0, ref)
    return ref, tol, flat, vague


# --------------------------------------------------------------------

dem = f32("dem.raw")
src_valid = dem != DEM_NODATA
z = np.where(src_valid, dem, 0.0)
integral = bool(np.all(z == np.round(z)))
print(
    f"{W}x{H} cells, NoData {DEM_NODATA:g}, cell {CELL:g}: {100 * src_valid.mean():.1f}% of the DEM is valid, "
    f"elevations {z[src_valid].min():g} to {z[src_valid].max():g}"
    + (", all whole numbers (the float64 reference is exact)" if integral else "")
    + "\n"
)

for r in RADII:
    k = 2 * r + 1
    m_cells = k * k
    print(f"-- FitRadius {r} / r.param.scale size={k} --")
    cx = cy = CELL

    # The float64 reference derivatives, from the closed forms.
    iv = np.arange(-r, r + 1, dtype=float)
    t2 = 3 * iv * iv - r * (r + 1)
    ones = np.ones(k)
    S = float((iv * iv).sum())
    Qs = float((t2 * t2).sum())
    c1, ci, ct = colpass(z, ones), colpass(z, iv), colpass(z, t2)
    P = full(rowpass(c1, iv), r)
    Qg = full(rowpass(ci, ones), r)
    Rr = full(rowpass(c1, t2), r)
    Tt = full(rowpass(ct, ones), r)
    Ss = full(rowpass(ci, iv), r)
    Zsum = full(rowpass(c1, ones), r)
    del c1, ci, ct
    p = P / (k * S * cx)
    q = Qg / (k * S * cy)
    rr = 6 * Rr / (k * Qs * cx * cx)
    tt = 6 * Tt / (k * Qs * cy * cy)
    ss = Ss / (S * S * cx * cy)
    exact_flat = (P == 0) & (Qg == 0)
    del P, Qg, Rr, Tt, Ss

    # Validity: the whole window valid (and inside the raster).
    bad = (~src_valid).astype(np.int32)
    window_ok = full(rowpass(colpass(bad, ones), ones), r) == 0  # NaN border compares False

    g_slope, s_slope = f64(f"grass-slope-{k}.raw"), f32(f"strata-slope-{r}.raw")
    g_has, s_has = g_slope != FILL, s_slope != FILL
    names = [("aspect", "aspect"), ("profc", "profile"), ("planc", "plan"),
             ("crosc", None), ("minic", None), ("maxic", None), (None, "mean")]
    g = {"slope": g_slope}
    s = {"slope": s_slope}
    null_mismatch = 0
    for gn, sn in names:
        if gn:
            g[gn] = f64(f"grass-{gn}-{k}.raw")
            null_mismatch += int(((g[gn] != FILL) != g_has).sum())
        if sn:
            s[sn] = f32(f"strata-{sn}-{r}.raw")
            null_mismatch += int(((s[sn] != FILL) != s_has).sum())
    record(
        f"r={r}: same cells carry data as GRASS",
        int((g_has != s_has).sum()) == 0 and null_mismatch == 0 and g_has.any(),
        f"{int((g_has != s_has).sum())} cells differ, {null_mismatch} differ between products; "
        f"{int(g_has.sum()):,} carry data",
    )
    record(
        f"r={r}: data iff the whole window is valid",
        int((s_has != window_ok).sum()) == 0,
        f"{int((s_has != window_ok).sum())} cells differ from the definition",
    )
    both = g_has & s_has

    # ---- Error bounds. ----
    zmax = window_reduce(np.abs(z), r, np.max)
    zrange = window_reduce(z, r, np.max) - window_reduce(z, r, np.min)
    # strata (check.py's wood(), with the window's own zmax).
    lin = 2 * k * EPS * zmax * k * r * (r + 1)
    quad = 2 * k * EPS * zmax * k * np.abs(t2).sum()
    cross = 2 * k * EPS * zmax * (r * (r + 1)) ** 2
    sp = lin / (k * S * cx) + 2 * EPS * np.abs(p)
    sq = lin / (k * S * cy) + 2 * EPS * np.abs(q)
    sr = 6 * quad / (k * Qs * cx * cx) + 2 * EPS * np.abs(rr)
    st = 6 * quad / (k * Qs * cy * cy) + 2 * EPS * np.abs(tt)
    sS = cross / (S * S * cx * cy) + 2 * EPS * np.abs(ss)
    # GRASS: backward-stable LU on exact N, plus its float64 obs sums.
    N, n_exact = normal_matrix(r, CELL)
    if not n_exact:
        print(f"      note: GRASS's normal matrix is not exact in float64 for cell {CELL:g}; "
              f"the GRASS bound below omits that rounding")
    L, Umat, Pm = grass_lu(N)
    Ninv = np.abs(np.linalg.inv(N))
    Gmat = gamma(18, U64) * Ninv @ (Pm.T @ (np.abs(L) @ np.abs(Umat)))
    jj, ii = np.mgrid[-r : r + 1, -r : r + 1]
    xs, ys = ii * CELL, jj * CELL
    basis = [np.abs(xs * xs).sum(), np.abs(ys * ys).sum(), np.abs(xs * ys).sum(),
             np.abs(xs).sum(), np.abs(ys).sum(), float(m_cells)]
    dobs = Ninv @ (gamma(m_cells + 2, U64) * np.array(basis))
    a_, b_ = rr / 2, tt / 2
    f_ = (Zsum - m_cells * z) / m_cells - (a_ + b_) * k * S * CELL * CELL / m_cells
    coef = [np.abs(v) for v in (a_, b_, ss, p, q, f_)]
    E = [sum(Gmat[i, j] * coef[j] for j in range(6)) + dobs[i] * zrange for i in range(6)]
    gp, gq, gr, gt, gs = E[3], E[4], 2 * E[0], 2 * E[1], E[2]
    ep, eq, er, et, es = sp + gp, sq + gq, sr + gr, st + gt, sS + gs
    del zmax, zrange, f_, Zsum, coef
    print(f"      per-derivative bound at the median cell: strata p {np.nanmedian(sp[both]):.2e}, "
          f"GRASS p {np.nanmedian(gp[both]):.2e}; strata r {np.nanmedian(sr[both]):.2e}, "
          f"GRASS r {np.nanmedian(gr[both]):.2e}")

    # ---- Flat cells: p = q = 0 exactly. ----
    fl = both & exact_flat
    flat_ok = (
        np.all(s["aspect"][fl] == -1.0) and np.all(g["aspect"][fl] == 0.0)
        and np.all(s["profile"][fl] == 0.0) and np.all(g["profc"][fl] == 0.0)
        and np.all(s["plan"][fl] == 0.0) and np.all(g["planc"][fl] == 0.0)
        and not np.any(s["aspect"][both & ~exact_flat] == -1.0)
    )
    record(
        f"r={r}: flat cells agree",
        flat_ok,
        f"{int(fl.sum()):,} cells with p = q = 0: strata aspect -1 and GRASS 0, "
        f"profile and plan 0 in both",
    )
    cmp_ = both & ~exact_flat

    # ---- Slope. ----
    mag = np.hypot(p, q)
    dg = np.hypot(ep, eq)
    ref_slope = DEG * np.arctan(mag)
    fac = 1.0 / (1.0 + np.maximum(mag - dg, 0.0) ** 2)
    tol = (DEG * dg * fac + DEG * (2.5 * EPS * mag / (1 + mag * mag) + ATAN32)
           + 2 * EPS * ref_slope + 4 * U64 * ref_slope)
    c = both
    diff = np.abs(g["slope"][c] - s["slope"][c])
    worst = worst_ratio(diff, tol[c])
    record(
        f"r={r}: slope == GRASS slope",
        worst <= 1.0,
        f"max {diff.max():.2e} deg, mean {diff.mean():.2e}, worst {worst:.3f}x the bound, "
        f"over {int(c.sum()):,} cells",
    )
    eg = np.abs(g["slope"][c] - ref_slope[c]).mean()
    es_ = np.abs(s["slope"][c] - ref_slope[c]).mean()
    print(f"      vs the float64 reference: mean error GRASS {eg:.2e} deg, strata {es_:.2e} deg")

    # ---- Aspect: strata = (270 + GRASS) mod 360. ----
    g_bearing = (270.0 + g["aspect"]) % 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        atol = DEG * np.arcsin(np.minimum(dg / mag, 1.0)) + DEG * ATAN2F32 + 2 * EPS * 360.0
    vague = cmp_ & (atol > 5.0)
    judge = cmp_ & ~vague
    ad = angular_diff(g_bearing[judge], s["aspect"][judge])
    worst = worst_ratio(ad, atol[judge])
    record(
        f"r={r}: aspect == (270 + GRASS aspect) mod 360",
        worst <= 1.0,
        f"max {ad.max():.2e} deg, mean {ad.mean():.2e}, worst {worst:.3f}x the bound, "
        f"over {int(judge.sum()):,} cells, {int(vague.sum())} too flat to judge",
    )

    # ---- Curvatures. ----
    tan2 = np.tan(np.radians(g["slope"])) ** 2
    with np.errstate(invalid="ignore"):
        g_mean = (g["minic"] + g["maxic"] + tan2 * g["crosc"]) / (2 * (1 + tan2) ** 1.5)
    derive_err = 64 * U64 * (np.abs(g["minic"]) + np.abs(g["maxic"]) + tan2 * np.abs(g["crosc"]))
    for op, gval, what in (("profile", g["profc"], "profc"), ("plan", -g["planc"], "-planc"),
                           ("mean", g_mean, "(minic+maxic+g crosc)/(2(1+g)^1.5)")):
        ref, ctol, _, cvague = curvature_from(op, p, q, rr, ss, tt, ep, eq, er, es, et)
        if op == "mean":
            ctol = ctol + derive_err
            judge = both
        else:
            judge = cmp_ & ~cvague
        d = np.abs(gval[judge] - s[op][judge])
        worst = worst_ratio(d, ctol[judge])
        record(
            f"r={r}: {op} == GRASS {what}"[:44],
            worst <= 1.0,
            f"max {d.max():.2e} /m, mean {d.mean():.2e}, worst {worst:.3f}x the bound, "
            f"over {int(judge.sum()):,} cells"
            + (f", {int((cmp_ & cvague).sum())} too flat to judge" if op != "mean" else ""),
        )
        eg = np.abs(gval[judge] - ref[judge]).mean()
        es_ = np.abs(s[op][judge] - ref[judge]).mean()
        print(f"      vs the float64 reference: mean error GRASS {eg:.2e}, strata {es_:.2e}")
    print()

failed = [x for x in results if not x[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
