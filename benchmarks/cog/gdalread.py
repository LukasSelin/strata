#!/usr/bin/env python3
"""Read band 1 of a GeoTIFF through GDAL the way cogbench's read mode
reads it through strata: full-width strips of <tile> rows, into one
float32 buffer that is reused.

    python3 gdalread.py <file> <tile>

Prints `run=0 ms=<t>`, the time from the first strip to the last,
excluding interpreter start and gdal.Open, as cogbench excludes its own
process start and cog.Open. GDAL_NUM_THREADS, from the environment,
sets how many threads decode the blocks one ReadAsArray call touches.
Values only: GDAL's mask band is a separate read, which strata's read
includes. That leaves GDAL the lighter job.
"""

import sys
import time

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
path, tile = sys.argv[1], int(sys.argv[2])
ds = gdal.Open(path)
band = ds.GetRasterBand(1)
w, h = ds.RasterXSize, ds.RasterYSize
buf = np.empty((tile, w), np.float32)
t0 = time.perf_counter()
for y in range(0, h, tile):
    rows = min(tile, h - y)
    band.ReadAsArray(0, y, w, rows, buf_obj=buf[:rows])
d = time.perf_counter() - t0
print(f"run=0 ms={d * 1000:.1f} MBs={w * h * 4 / d / 1e6:.0f}")
