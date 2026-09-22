#!/usr/bin/env python3
"""Difference strata's results against gdaldem's, on a real GeoTIFF.

This is the strongest outside opinion available: a different program, by
different authors, in a different language, computing the same quantity
from the same bytes. Agreement is only meaningful once the two encodings
are reconciled, and the reconciliations are the interesting part:

  * NoData. gdaldem writes -9999 into any cell whose 3x3 window touches
    NoData, and into the one-cell border (no -compute_edges). strata
    clears the validity bit there and writes whatever fill the sink was
    given. The two sets of cells must be the same set.

  * Flat cells. gdaldem aspect writes its NoData value (-9999) where the
    gradient is zero; strata writes -1 (Esri's flat aspect) and keeps
    the cell valid. So a cell that is -9999 in gdaldem and -1 in strata
    is agreement, not a difference.

  * Hillshade. gdaldem writes a byte, reserving 0 for NoData:
    round(1 + 254*max(0, cos)). strata writes the unrounded
    255*max(0, cos) as a float, because validity lives in its mask.
    Converting strata's v with round(1 + v*254/255) puts them on the
    same scale. gdaldem also uses an approximate square root, so a
    difference of one count is expected and is not strata being wrong.

  * Cell size. gdaldem aspect differences raw elevations without
    dividing by the cell size; strata divides. They agree only for
    square cells, which this raster has (12.5 m both ways).

  * Ruggedness needs no reconciliation at all. strata documents that
    TRI (Riley and Wilson), TPI and roughness round exactly as gdaldem
    does, so those four are compared bit for bit: any difference in any
    cell that carries data fails.

Usage:  python gdalcompare.py [dir] [width] [height] [nodata] [cell]

nodata and cell are the DEM's own NoData value and cell size;
gdalcheck.sh reads both from the ENVI header GDAL wrote.
"""

import os
import warnings
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "out-gdal"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
# The DEM is float32, so compare against the fill as float32 holds it.
DEM_NODATA = float(np.float32(sys.argv[4])) if len(sys.argv) > 4 else 65535.0
CELL = float(sys.argv[5]) if len(sys.argv) > 5 else 12.5

GDAL_NODATA = -9999.0
STRATA_FILL = -9999.0

results = []


def record(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<46}  {detail}")


def f32(name):
    return np.fromfile(os.path.join(D, name), dtype="<f4").reshape(H, W).astype(np.float64)


def u8(name):
    return np.fromfile(os.path.join(D, name), dtype=np.uint8).reshape(H, W)


dem = f32("dem.raw")
src_valid = dem != DEM_NODATA
print(
    f"{W}x{H} cells, NoData {DEM_NODATA:g}, cell {CELL:g}: "
    f"{100 * src_valid.mean():.1f}% of the DEM is valid\n"
)

# Where both tools say a result exists.
gslope = f32("gdal-slope.raw")
sslope = f32("strata-slope.raw")
g_has = gslope != GDAL_NODATA
s_has = sslope != STRATA_FILL

# --------------------------------------------------------------------
# 1. The two tools discard the same cells.
# --------------------------------------------------------------------

disagree = int((g_has != s_has).sum())
record(
    "same cells carry data as gdaldem",
    disagree == 0,
    f"{disagree} cells differ; {int(g_has.sum())} carry data",
)
both = g_has & s_has
if not both.any():
    # Agreeing that there is no data anywhere says nothing about the kernels.
    record("some cells to compare", False, "neither tool produced data in this window")
    sys.exit(1)

# --------------------------------------------------------------------
# 2. Slope.
# --------------------------------------------------------------------

d = np.abs(gslope[both] - sslope[both])
record(
    "slope == gdaldem slope",
    d.max() < 1e-3,
    f"max {d.max():.2e} deg, mean {d.mean():.2e}, over {both.sum():,} cells",
)

# --------------------------------------------------------------------
# 3. Aspect, with the flat-cell sentinels reconciled.
# --------------------------------------------------------------------

gasp = f32("gdal-aspect.raw")
sasp = f32("strata-aspect.raw")
g_flat = (gasp == GDAL_NODATA) & both  # gdaldem: NoData means flat here
s_flat = (sasp == -1.0) & both  # strata: AspectFlat
record(
    "aspect flat cells agree",
    int((g_flat != s_flat).sum()) == 0,
    f"{int((g_flat != s_flat).sum())} disagree; {int(g_flat.sum()):,} flat cells",
)

sloped = both & ~g_flat & ~s_flat
if not sloped.any():
    record("some sloped cells to compare", False, "the window is entirely flat")
    sys.exit(1)
diff = np.abs(gasp[sloped] - sasp[sloped]) % 360.0
diff = np.minimum(diff, 360.0 - diff)
record(
    "aspect == gdaldem aspect",
    np.percentile(diff, 99.99) < 1e-2 and diff.max() < 1.0,
    f"max {diff.max():.2e} deg, 99.99th pct {np.percentile(diff, 99.99):.2e}, "
    f"over {sloped.sum():,} cells",
)

# --------------------------------------------------------------------
# 4. Hillshade, on gdaldem's byte scale.
# --------------------------------------------------------------------

ghs = u8("gdal-hillshade.raw").astype(np.int32)
shs = f32("strata-hillshade.raw")
# strata's float -> gdaldem's byte, as strata's own documentation states.
shs_byte = np.rint(1.0 + shs * 254.0 / 255.0).astype(np.int32)
hs_cells = both & (ghs != 0)  # gdaldem reserves byte 0 for NoData
delta = np.abs(ghs[hs_cells] - shs_byte[hs_cells])
off_by_one = int((delta == 1).sum())
worse = int((delta > 1).sum())
record(
    "hillshade == gdaldem hillshade",
    worse == 0,
    f"{int((delta == 0).sum()):,} of {delta.size:,} exact, {off_by_one:,} off by one "
    f"(gdaldem's approximate sqrt), {worse} off by more",
)

# --------------------------------------------------------------------
# 4b. Ruggedness: the same cells, and the same bits in every one.
# --------------------------------------------------------------------

for op in ("tri", "triwilson", "tpi", "roughness"):
    graw = np.fromfile(os.path.join(D, f"gdal-{op}.raw"), dtype="<f4").reshape(H, W)
    sraw = np.fromfile(os.path.join(D, f"strata-{op}.raw"), dtype="<f4").reshape(H, W)
    gh, sh = graw != GDAL_NODATA, sraw != STRATA_FILL
    cells = gh & sh
    differ = int((graw[cells].view(np.uint32) != sraw[cells].view(np.uint32)).sum())
    record(
        f"{op} == gdaldem {op}",
        (gh == sh).all() and differ == 0 and cells.any(),
        f"{differ:,} of {int(cells.sum()):,} cells differ in any bit; "
        f"{int((gh != sh).sum())} disagree on carrying data",
    )

# --------------------------------------------------------------------
# 5. Which of the two is closer to the truth? Neither tool is the
#    reference: recompute the slope in float64 here and score both.
#    This turns "they differ in the last digit" into a statement about
#    who is right, and it is the only way to tell a real defect from
#    two defensible float32 roundings.
# --------------------------------------------------------------------

zz = np.where(dem == DEM_NODATA, np.nan, dem)
a_, b_, c_ = zz[:-2, :-2], zz[:-2, 1:-1], zz[:-2, 2:]
d_, f_ = zz[1:-1, :-2], zz[1:-1, 2:]
g_, h_, i_ = zz[2:, :-2], zz[2:, 1:-1], zz[2:, 2:]
dx = ((c_ + 2 * f_ + i_) - (a_ + 2 * d_ + g_)) / (8 * CELL)
dy = ((g_ + 2 * h_ + i_) - (a_ + 2 * b_ + c_)) / (8 * CELL)
truth = np.full((H, W), np.nan)
truth[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dx, dy)))

# How far apart may two correct float32 implementations land? Not a
# fixed number of ulps of the slope: Horn's weighted sums cancel, so on
# high, nearly flat terrain (elevations near 500 m, slopes near 0.1 deg)
# a rounding of the sum moves the gradient by far more than an ulp of
# the result. Derive it the way check.py's TOLERANCES does. Horn's
# numerator is six weighted elevations of magnitude at most zmax (here
# the largest |z| in the cell's own 3x3 window), accumulated in at most
# six roundings of a partial sum of magnitude at most 4*zmax, so each
# gradient component is off by at most
#
#       gtol = 3 * 2^-24 * zmax / cellsize
#
# and |gradient| by at most hypot(gtol, gtol). atan has derivative at
# most 1, so the slope is off by at most DEG * that. Both tools round,
# so the gap between them is at most twice it, plus one float32 rounding
# of each tool's result (an ulp of the slope each). A real defect is
# wrong by a fraction of the signal, which dwarfs this on any cell with
# relief; on a cell with none, both tools are rightly allowed to differ.
EPS = 2.0**-24  # float32 half-ulp
win = np.stack([a_, b_, c_, d_, zz[1:-1, 1:-1], f_, g_, h_, i_])
zmax = np.full((H, W), np.nan)
zmax[1:-1, 1:-1] = np.abs(win).max(axis=0)
gtol = 3 * EPS * zmax / CELL
t = both & np.isfinite(truth)
ulp = np.spacing(np.abs(sslope[t]).astype(np.float32)).astype(np.float64)
tol = 2 * np.degrees(np.hypot(gtol[t], gtol[t])) + 2 * ulp
gap = np.abs(gslope[t] - sslope[t])
eg, es = np.abs(gslope[t] - truth[t]), np.abs(sslope[t] - truth[t])
worst = (gap / tol).max()
record(
    "strata and gdaldem within float32 rounding",
    worst <= 1.0,
    f"worst {worst:.2f}x the bound, {int((gap > tol).sum()):,} cells over, "
    f"{100 * (eg == es).mean():.2f}% bit-identical",
)
print(
    f"      both vs a float64 reference: mean error gdaldem {eg.mean():.2e} deg, "
    f"strata {es.mean():.2e} deg\n"
    f"      closer to truth: gdaldem {100 * (eg < es).mean():.2f}%, "
    f"strata {100 * (es < eg).mean():.2f}%, tied {100 * (eg == es).mean():.2f}%"
)

# --------------------------------------------------------------------
# 6. No tile seam: the streamed run must not be worse on the rows where
#    one chunk meets the next.
# --------------------------------------------------------------------

TILE = int(os.environ.get("STRATA_TILE", "256"))
err = np.where(both, np.abs(gslope - sslope), np.nan)
with warnings.catch_warnings():  # rows that are entirely NoData
    warnings.simplefilter("ignore", RuntimeWarning)
    rows = np.nanmean(err, axis=1)
seam = np.arange(TILE, H - 1, TILE)
other = np.setdiff1d(np.arange(1, H - 1), seam)
s_mean, o_mean = np.nanmean(rows[seam]), np.nanmean(rows[other])
record(
    f"no seam every {TILE} rows",
    s_mean < 2 * o_mean,
    f"mean error on chunk boundaries {s_mean:.3e} vs {o_mean:.3e} elsewhere",
)

# --------------------------------------------------------------------
# 7. A difference image, so a systematic error would be visible even if
#    it hid inside a tolerance.
# --------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = np.full((H, W), np.nan)
    err[both] = gslope[both] - sslope[both]
    png = os.path.join(D, "slope-difference.png")
    cm = matplotlib.colormaps["coolwarm"].with_extremes(bad="black")
    lim = np.nanpercentile(np.abs(err), 99.9) or 1e-6
    plt.imsave(png, np.ma.masked_invalid(err), cmap=cm, vmin=-lim, vmax=lim)
    print(f"\nwrote {png} (black = no data; structure here would mean a real bug)")
except ImportError:
    pass

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
