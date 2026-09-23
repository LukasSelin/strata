#!/usr/bin/env python3
"""Time one GDAL operation with its input already in memory.

The GDAL side of the suite's compute tier, the counterpart of
`stratasuite -mode memory`. It copies the input into a MEM dataset,
runs the operation once untimed, then REPEAT times, and prints

    run=<i> ms=<milliseconds> [result]

per timed run, as stratasuite does. Only the call is timed: GDAL's own
algorithm (gdaldem's, gdalwarp's, `gdal raster neighbors`' ...) reading
from a MEM dataset and writing a new one. That is as close to GDAL's
arithmetic as its public API reaches. Every call creates its output
dataset, which GDAL's API does not let a caller reuse; the `copy` op,
a MEM-to-MEM copy of the raster, is the floor that puts a number on it.
The algebra reads its inputs from uncompressed GeoTIFFs in /vsimem/
rather than MEM datasets, because `gdal raster calc` names its inputs
only by file name.

    python3 gdalcompute.py <op> <dem.tif> <dem2.tif> <repeat> <threads> [kernel]

<threads> sets GDAL_NUM_THREADS, and for the warps also -multi and
NUM_THREADS. Operations that GDAL runs on one thread whatever it says
ignore it. [kernel] is conv5's weights, from `stratasuite -kernel`.
"""

import sys
import time

from osgeo import gdal

gdal.UseExceptions()

op, path, path2 = sys.argv[1], sys.argv[2], sys.argv[3]
repeat, threads = int(sys.argv[4]), int(sys.argv[5])
kernel = sys.argv[6] if len(sys.argv) > 6 else None

gdal.SetConfigOption("GDAL_NUM_THREADS", str(threads))
gdal.SetConfigOption("GDAL_CACHEMAX", "4096")
MEM = gdal.GetDriverByName("MEM")


def load(p):
    ds = MEM.CreateCopy("", gdal.Open(p))
    ds.GetRasterBand(1).ReadRaster()  # touched, as stratasuite's input is
    return ds


src = load(path)
src2 = load(path2) if op in ("add", "mul", "min", "max") else None
W, H = src.RasterXSize, src.RasterYSize


def dem(name, **kw):
    return lambda: gdal.DEMProcessing("", src, name, format="MEM", **kw)


def neighbors(**kw):
    return lambda: gdal.Run("raster", "neighbors", input=src, output="", output_format="MEM",
                            datatype="Float32", **kw).Output()


def calc(expr, two):
    # calc takes several inputs only by name, and names only files, so the
    # inputs are uncompressed GeoTIFFs in GDAL's in-memory filesystem,
    # written on the first (untimed) call.
    inputs = ["A=/vsimem/a.tif"] + (["B=/vsimem/b.tif"] if two else [])

    def run():
        if gdal.VSIStatL("/vsimem/a.tif") is None:
            gdal.GetDriverByName("GTiff").CreateCopy("/vsimem/a.tif", src)
            if two:
                gdal.GetDriverByName("GTiff").CreateCopy("/vsimem/b.tif", src2)
        return gdal.Run("raster", "calc", input=inputs, output="", output_format="MEM",
                        calc=expr, datatype="Float32", propagate_nodata=True,
                        nodata=-9999, no_check_extent=True).Output()
    return run


def warp(alg, scale):
    return lambda: gdal.Warp("", src, format="MEM", width=round(W * scale), height=round(H * scale),
                             resampleAlg=alg, errorThreshold=0, outputType=gdal.GDT_Float32,
                             dstNodata=-9999, warpMemoryLimit=2048,
                             multithread=threads > 1, warpOptions=[f"NUM_THREADS={threads}"])


def stats():
    # ComputeStatistics returns no count, so there is none to compare.
    s = src.GetRasterBand(1).ComputeStatistics(False)
    return f"min={s[0]:.9g} max={s[1]:.9g} mean={s[2]:.17g} stddev={s[3]:.17g}"


def minmax():
    mn, mx = src.GetRasterBand(1).ComputeRasterMinMax(False)
    return f"min={mn:.9g} max={mx:.9g}"


OPS = {
    "copy": lambda: MEM.CreateCopy("", src),
    "slope": dem("slope"),
    "aspect": dem("aspect"),
    "hillshade": dem("hillshade"),
    "tri": dem("TRI"),
    "tpi": dem("TPI"),
    "roughness": dem("Roughness"),
    "mean3": neighbors(kernel="equal", size=3, method="mean"),
    "mean11": neighbors(kernel="equal", size=11, method="mean"),
    "min3": neighbors(kernel="equal", size=3, method="min"),
    "max3": neighbors(kernel="equal", size=3, method="max"),
    "gauss5": neighbors(kernel="gaussian", size=5, method="sum"),
    "conv5": neighbors(kernel=kernel, method="sum"),
    "add": calc("A+B", True),
    "mul": calc("A*B", True),
    "min": calc("min(A,B)", True),
    "max": calc("max(A,B)", True),
    "clamp": calc("min(max(A,50),200)", False),
    "stats": stats,
    "minmax": minmax,
    "near-half": warp("near", 0.5),
    "bilinear-half": warp("bilinear", 0.5),
    "cubic-half": warp("cubic", 0.5),
    "lanczos-half": warp("lanczos", 0.5),
    "average-half": warp("average", 0.5),
    "cubic-double": warp("cubic", 2),
}

f = OPS[op]
out = f()  # untimed: checks that it works, and warms whatever GDAL caches
if isinstance(out, gdal.Dataset) and out.GetDriver().ShortName != "MEM":
    sys.exit(f"gdalcompute: {op} returned a {out.GetDriver().ShortName} dataset, "
             "which may compute lazily; the timing would not include the work")
out = None
for i in range(repeat):
    t0 = time.perf_counter()
    out = f()
    ms = (time.perf_counter() - t0) * 1000
    print(f"run={i} ms={ms:.2f} {out if isinstance(out, str) else ''}".rstrip())
    out = None  # freed before the next run, as a caller would
