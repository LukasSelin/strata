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


def _normalize_inputs(d, stem):
    z = read(d, f"{stem}.f32")
    p = os.path.join(d, f"{stem}.mask.u8")
    valid = np.fromfile(p, dtype=np.uint8).reshape(H, W).astype(bool) if os.path.exists(p) else np.ones((H, W), bool)
    return z, valid


def reciprocal_normalize(d):
    """Normalize as one multiply-add with a precomputed 1/(hi-lo) - the
    obvious fast rewrite, which rounds differently."""
    for stem in ("hill", "plane", "noisy"):
        z, valid = _normalize_inputs(d, stem)
        lo, hi = z[valid].min(), z[valid].max()
        a = np.float32(1) / (hi - lo)
        for form in ("plain", "tiled", "chunked"):
            write(d, f"{stem}-normalize-{form}.f32", z * a - lo * a)


def normalize_over_nodata(d):
    """Normalize taking its range from every cell, NoData included."""
    z, _ = _normalize_inputs(d, "noisy")
    lo, hi = z.min(), z.max()
    write(d, "noisy-normalize-chunked.f32", (z - lo) / (hi - lo))


def flip_curvature(d):
    """Curvature with the opposite sign convention (concave positive)."""
    for f in os.listdir(d):
        if "curvature_" in f and f.endswith(".f32"):
            write(d, f, -read(d, f))


def swap_profile_plan(d):
    """Profile and plan curvature swapped - the two quadratic forms mixed up."""
    for stem in ("hill", "plane", "noisy"):
        for form in ("plain", "tiled", "chunked"):
            pr = read(d, f"{stem}-curvature_profile-{form}.f32")
            pl = read(d, f"{stem}-curvature_plan-{form}.f32")
            write(d, f"{stem}-curvature_profile-{form}.f32", pl)
            write(d, f"{stem}-curvature_plan-{form}.f32", pr)


def curvature_drift(d):
    """Mean curvature 0.05% too large - a constant-factor slip."""
    for f in os.listdir(d):
        if "curvature_mean" in f and f.endswith(".f32"):
            write(d, f, read(d, f) * 1.0005)


# The focal mutations rewrite every form of a result the same way, as a
# library bug would, so that only the reference, border and validity
# checks - not plain == tiled == chunked - can catch them.

STEMS = ("hill", "plane", "noisy")
FORMS = ("plain", "tiled", "chunked")


def _case(stem, op):
    for c in MAN["rasters"]:
        if c["surface"] == stem and c["op"] == op and c["form"] == "plain":
            return c
    raise KeyError(op)


def _focal_write(d, stem, op, a):
    for form in FORMS:
        write(d, f"{stem}-{op}-{form}.f32", a)


def _correlate(z, w):
    """float64 correlation over the interior, NaN on the border."""
    k = w.shape[0]
    r = k // 2
    out = np.full(z.shape, np.nan)
    acc = np.zeros((H - 2 * r, W - 2 * r))
    for j in range(k):
        for c in range(k):
            acc += w[j, c] * z[j : j + H - 2 * r, c : c + W - 2 * r]
    out[r : H - r, r : W - r] = acc
    return out


def flipped_correlate(d):
    """Correlate applying Convolve's rotation of the weights."""
    for stem in STEMS:
        for form in FORMS:
            shutil.copy(os.path.join(d, f"{stem}-focal_convolve_r2-{form}.f32"),
                        os.path.join(d, f"{stem}-focal_correlate_r2-{form}.f32"))


def swapped_passes(d):
    """CorrelateSeparable applying the row taps down the columns and the
    column taps along the rows."""
    for stem in STEMS:
        c = _case(stem, "focal_separable_r2")
        z = read(d, f"{stem}.f32").astype(np.float64)
        w = np.outer(np.array(c["row"], np.float32), np.array(c["col"], np.float32)).astype(np.float64)
        _focal_write(d, stem, "focal_separable_r2", _correlate(z, w))


def radius_one_short(d):
    """Min of radius 3 reading a radius-2 window, with a radius-2 border."""
    for stem in STEMS:
        z = read(d, f"{stem}.f32")
        r = 2
        out = np.full(z.shape, np.nan, np.float32)
        acc = None
        for j in range(2 * r + 1):
            for c in range(2 * r + 1):
                v = z[j : j + H - 2 * r, c : c + W - 2 * r]
                acc = v.copy() if acc is None else np.minimum(acc, v)
        out[r : H - r, r : W - r] = acc
        _focal_write(d, stem, "focal_min_r3", out)


def mean_wrong_count(d):
    """Mean dividing by the neighbours without the centre, 24 for 5x5."""
    for stem in STEMS:
        _focal_write(d, stem, "focal_mean_r2", read(d, f"{stem}-focal_mean_r2-plain.f32") * np.float32(25 / 24))


def erosion_3x3(d):
    """A radius-2 validity computed with a 3x3 erosion."""
    src = np.fromfile(os.path.join(d, "noisy.mask.u8"), dtype=np.uint8).reshape(H, W)
    m = np.zeros((H, W), np.uint8)
    e = np.ones((H - 2, W - 2), np.uint8)
    for j in range(3):
        for c in range(3):
            e &= src[j : j + H - 2, c : c + W - 2]
    m[1:-1, 1:-1] = e
    for form in FORMS:
        m.tofile(os.path.join(d, f"noisy-focal_mean_r2-{form}.mask.u8"))


def focal_leaky_nodata(d):
    """One output of a radius-3 Max whose neighbourhood holds NoData
    marked valid."""
    p = os.path.join(d, "noisy-focal_max_r3-plain.mask.u8")
    m = np.fromfile(p, dtype=np.uint8).reshape(H, W)
    inner = np.zeros((H, W), bool)
    inner[3 : H - 3, 3 : W - 3] = True
    y, x = np.argwhere((m == 0) & inner)[0]
    m[y, x] = 1
    m.tofile(p)


# The ruggedness mutations also rewrite every form. check.py requires
# gdaldem's bits exactly, so each of these must fail it, down to the
# last two: a single ulp, and a float32 sum where gdaldem sums in float64.


def _dem(d, stem):
    return np.fromfile(os.path.join(d, f"{stem}.f32"), dtype="<f4").reshape(H, W)


def _rewrite(d, op, f):
    """Replace every form of op on every surface by f(stem, result)."""
    for stem in STEMS:
        for form in FORMS:
            name = f"{stem}-{op}-{form}.f32"
            write(d, name, f(stem, read(d, name)))


def riley_wilson_swapped(d):
    """Riley's TRI and Wilson's swapped - the two definitions mixed up."""
    for stem in STEMS:
        for form in FORMS:
            ri = read(d, f"{stem}-ruggedness_tri-{form}.f32")
            wi = read(d, f"{stem}-ruggedness_triwilson-{form}.f32")
            write(d, f"{stem}-ruggedness_tri-{form}.f32", wi)
            write(d, f"{stem}-ruggedness_triwilson-{form}.f32", ri)


def tpi_sign(d):
    """TPI as the neighbours' mean minus the centre."""
    _rewrite(d, "ruggedness_tpi", lambda stem, a: -a)


def roughness_without_centre(d):
    """Roughness over the eight neighbours only, leaving out the centre."""
    def f(stem, a):
        z = _dem(d, stem)
        win = [z[j : j + H - 2, i : i + W - 2] for j in range(3) for i in range(3)]
        nb = win[:4] + win[5:]
        out = a.copy()
        out[1:-1, 1:-1] = np.max(nb, axis=0) - np.min(nb, axis=0)
        return out
    _rewrite(d, "ruggedness_roughness", f)


def tri_one_ulp(d):
    """One TRI cell of the hill one ulp too large."""
    for form in FORMS:
        name = f"hill-ruggedness_tri-{form}.f32"
        a = read(d, name)
        a[100, 100] = np.nextafter(a[100, 100], np.float32(np.inf))
        write(d, name, a)


def tri_float32_sum(d):
    """Riley's squares summed in float32 rather than gdaldem's float64."""
    def f(stem, a):
        z = _dem(d, stem)
        win = [z[j : j + H - 2, i : i + W - 2] for j in range(3) for i in range(3)]
        e = win[4]
        s = np.zeros(e.shape, np.float32)
        for n in win[:4] + win[5:]:
            s = s + (n - e) * (n - e)
        out = a.copy()
        out[1:-1, 1:-1] = np.sqrt(s)
        return out
    _rewrite(d, "ruggedness_tri", f)


# The larger windows. check.py holds them to their float32 arithmetic
# exactly and to the float64 definition within a bound, so a window that
# counts its centre, or is one ring short, must fail both.


def _windows(z, r):
    """Every (2r+1)x(2r+1) window of z, row-major, over the interior."""
    k = 2 * r + 1
    return [z[j : j + H - 2 * r, i : i + W - 2 * r] for j in range(k) for i in range(k)]


def tpi_counts_centre(d):
    """Radius-3 TPI with the centre in the mean: divided by 49, not 48."""
    def f(stem, a):
        win = _windows(_dem(d, stem), 3)
        s = win[0]
        for w in win[1:]:
            s = s + w
        out = a.copy()
        out[3:-3, 3:-3] = win[len(win) // 2] - s / np.float32(len(win))
        return out
    _rewrite(d, "ruggedness_tpi_r3", f)


def roughness_ring_short(d):
    """Radius-8 roughness taken over the radius-7 window."""
    def f(stem, a):
        win = _windows(_dem(d, stem), 7)
        out = a.copy()
        out[8:-8, 8:-8] = (np.max(win, axis=0) - np.min(win, axis=0))[1:-1, 1:-1]
        return out
    _rewrite(d, "ruggedness_roughness_r8", f)


def features_largest_erosion(d):
    """Features' 3x3 TPI given the validity of the radius-8 output next
    to it: one erosion for every output, by the largest window."""
    dem = np.fromfile(os.path.join(d, "noisy.mask.u8"), dtype=np.uint8).reshape(H, W).astype(bool)
    eroded = np.zeros((H, W), bool)
    eroded[8:-8, 8:-8] = np.logical_and.reduce([w for w in _windows(dem, 8)])
    for form in FORMS:
        eroded.astype(np.uint8).tofile(os.path.join(d, f"noisy-features_ruggedness_tpi-{form}.mask.u8"))


def features_ulp(d):
    """One Features output cell one ulp off its standalone operation."""
    name = "hill-features_ruggedness_tri_r3-chunked.f32"
    a = read(d, name).astype("<f4")
    a[60, 70] = np.nextafter(a[60, 70], np.float32(np.inf))
    write(d, name, a)


def _weighted(d, f):
    """Apply f(values, mask) -> (values, mask) to every weighted slope."""
    for c in MAN["rasters"]:
        if c.get("weight"):
            m = np.fromfile(os.path.join(d, c["out_mask"]), dtype=np.uint8).reshape(H, W)
            a, m = f(read(d, c["out"]), m)
            write(d, c["out"], a)
            m.astype(np.uint8).tofile(os.path.join(d, c["out_mask"]))


def weight_eroded(d):
    """The weight's validity eroded 3x3 like the DEM's - one erosion over
    every input, the engine's rule before per-input reach."""
    wm = np.fromfile(os.path.join(d, "weight.mask.u8"), dtype=np.uint8).reshape(H, W).astype(bool)
    er = np.zeros((H, W), bool)
    er[1:-1, 1:-1] = True
    for j in range(3):
        for i in range(3):
            er[1:-1, 1:-1] &= wm[j : H - 2 + j, i : W - 2 + i]
    _weighted(d, lambda a, m: (a, m & er))


def weight_ignored(d):
    """The weight's NoData cells marked valid, as if its mask were never
    read."""
    wm = np.fromfile(os.path.join(d, "weight.mask.u8"), dtype=np.uint8).reshape(H, W)
    interior = np.zeros((H, W), np.uint8)
    interior[1:-1, 1:-1] = 1
    _weighted(d, lambda a, m: (a, m | ((1 - wm) & interior)))


def weight_neighbour(d):
    """The weight read one cell to the right of the slope it multiplies."""
    wt = read(d, "weight.f32")
    shifted = np.roll(wt, -1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        _weighted(d, lambda a, m: (np.where(m == 1, a / wt * shifted, a), m))


def surface_ulp(d):
    """One Surface aspect cell one ulp off - a from-gradient kernel that
    rounds once differently from the fused one."""
    name = "hill-surface_aspect-tiled.f32"
    a = read(d, name).astype("<f4")
    a[40, 50] = np.nextafter(a[40, 50], np.float32(np.inf))
    write(d, name, a)


def surface_swapped(d):
    """Surface's dx and dy written to each other's output."""
    for form in ("plain", "tiled", "chunked"):
        dx = read(d, f"noisy-surface_gradient_dx-{form}.f32")
        dy = read(d, f"noisy-surface_gradient_dy-{form}.f32")
        write(d, f"noisy-surface_gradient_dx-{form}.f32", dy)
        write(d, f"noisy-surface_gradient_dy-{form}.f32", dx)


def focal_seam(d):
    """One tiled Gaussian cell wrong, as a halo bug at a seam would be:
    row 46 and column 74 are tile edges of the 37x23 tiling."""
    a = read(d, "hill-focal_gaussian_r3-tiled.f32")
    a[46, 74] += np.float32(0.01)
    write(d, "hill-focal_gaussian_r3-tiled.f32", a)


MUTATIONS = [
    ("slope 0.05% too large", drift),
    ("weight validity eroded 3x3", weight_eroded),
    ("weight NoData ignored", weight_ignored),
    ("weight read one cell over", weight_neighbour),
    ("Surface aspect one ulp off", surface_ulp),
    ("Surface dx and dy swapped", surface_swapped),
    ("dx and dy swapped", transpose_kernel),
    ("aspect mirrored", mirror_aspect),
    ("one bad cell on a tile seam", tile_seam),
    ("chunked result off by one row", shifted),
    ("one NoData cell leaking in", leaky_nodata),
    ("count one too many", miscount),
    ("max slightly wrong", wrong_extreme),
    ("normalize by a reciprocal", reciprocal_normalize),
    ("normalize range from NoData", normalize_over_nodata),
    ("curvature sign flipped", flip_curvature),
    ("profile and plan swapped", swap_profile_plan),
    ("mean curvature 0.05% too large", curvature_drift),
    ("correlate with convolve's flip", flipped_correlate),
    ("separable passes swapped", swapped_passes),
    ("focal radius one short", radius_one_short),
    ("mean divides by 24, not 25", mean_wrong_count),
    ("radius-2 validity eroded 3x3", erosion_3x3),
    ("focal NoData leaking in", focal_leaky_nodata),
    ("focal tile seam", focal_seam),
    ("Riley and Wilson TRI swapped", riley_wilson_swapped),
    ("TPI sign flipped", tpi_sign),
    ("roughness without the centre", roughness_without_centre),
    ("one TRI cell one ulp off", tri_one_ulp),
    ("TRI summed in float32", tri_float32_sum),
    ("r=3 TPI counts its centre", tpi_counts_centre),
    ("r=8 roughness one ring short", roughness_ring_short),
    ("Features erodes by the largest r", features_largest_erosion),
    ("Features output one ulp off", features_ulp),
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
