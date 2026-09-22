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


def expected(op, z, case):
    """The reference result and its tolerance, or (None, None)."""
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


def defined(out_mask):
    """Cells that must hold a defined value: interior, and valid."""
    keep = np.zeros((H, W), bool)
    keep[1:-1, 1:-1] = True
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
    ref, tol = expected(op, z, case)
    if ref is None:
        continue

    keep = defined(out_mask) & np.isfinite(ref)

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

    err = np.abs(got - ref)
    tol = np.broadcast_to(np.asarray(tol, float), err.shape)
    worst = (err[keep] / tol[keep]).max() if keep.any() else 0.0
    record(
        f"{case['name']} vs Horn reference",
        worst <= 1.0,
        f"max error {err[keep].max():.3e}, tolerance {tol[keep].max():.3e} "
        f"({worst:.2f}x) over {int(keep.sum())} cells",
    )


# --------------------------------------------------------------------
# 2. The border carries no data: Horn needs all eight neighbours, so a
#    border cell is NaN, or - in a file with a NoData value - the fill,
#    with its validity bit cleared.
# --------------------------------------------------------------------

# Operations that are not a 3x3 stencil: no border, no eroded validity.
POINTWISE = ("normalize",)

for case in MAN["rasters"]:
    if case["op"].startswith("algebra_") or case["op"] in POINTWISE:
        continue
    got, _ = load(case["out"])
    border = np.ones((H, W), bool)
    border[1:-1, 1:-1] = False
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
    if case["surface"] != "plane" or case["op"] not in ("slope_deg", "aspect"):
        continue
    cx = case["cell_size"]
    cy = case["cell_size_y"] or cx
    tdx, tdy = PLANE_A / cx, PLANE_B / cy
    got, _ = load(case["out"])
    keep = defined(None)
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
# 5. Validity: an output cell is valid iff it is interior and all nine
#    cells of its neighbourhood are valid.
# --------------------------------------------------------------------

for case in MAN["rasters"]:
    if not case.get("out_mask") or case["op"] in POINTWISE:
        continue
    src = load_mask(case["dem_mask"])
    want = np.zeros((H, W), bool)
    eroded = np.ones((H - 2, W - 2), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            eroded &= src[dy : dy + H - 2, dx : dx + W - 2]
    want[1:-1, 1:-1] = eroded
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
