#!/usr/bin/env python3
"""Write GeoTIFFs with rasterio and rio-cogeo, from corpus inputs.

rasterio and rio-cogeo ship their own GDAL build inside their wheels, so
these files come from a different GDAL version and a different choice of
creation options than cogmake.py's. Runs in a python container
(generate.sh starts it, after pip install rio-cogeo); /in is the corpus,
/out the output.
"""

import os
import subprocess

import numpy as np
import rasterio
from rasterio.transform import from_origin

IN, OUT = "/in", "/out/generated-rasterio"
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "VERSION.txt"), "w") as f:
    import rio_cogeo
    f.write(f"rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}, rio-cogeo {rio_cogeo.__version__}\n")

DEM = f"{IN}/products/nasadem/NASADEM_HGT_n59e017.tif"
RGB = f"{IN}/gdal-autotest/gcore/data/rgbsmall.tif"
F32 = f"{IN}/gdal-autotest/gcore/data/float32.tif"


def cogeo(src, name, *args):
    subprocess.run(["rio", "cogeo", "create", src, f"{OUT}/{name}.tif", "--quiet", *args], check=True)


cogeo(DEM, "nasadem-cog-deflate", "--cog-profile", "deflate", "--blocksize", "256")
cogeo(DEM, "nasadem-cog-zstd-web", "--cog-profile", "zstd", "--web-optimized", "--blocksize", "256")
cogeo(DEM, "nasadem-cog-lzw-nodata", "--cog-profile", "lzw", "--nodata", "0", "--overview-level", "3")
cogeo(RGB, "rgb-cog-mask", "--cog-profile", "deflate", "--add-mask", "--blocksize", "16", "--overview-blocksize", "16")
cogeo(RGB, "rgb-cog-packbits-band1", "--cog-profile", "packbits", "-b", "1")
cogeo(F32, "float32-cog-raw", "--cog-profile", "raw", "--blocksize", "16", "--overview-blocksize", "16")

# The rasterio API directly, with profiles a user might pass.
h, w = 257, 131
y, x = np.mgrid[0:h, 0:w]
z = (100 * np.sin(x / 13) * np.cos(y / 17)).astype(np.float32)
z[40:80, 20:60] = np.nan
common = dict(driver="GTiff", width=w, height=h, crs="EPSG:3006",
              transform=from_origin(600000, 6700000, 10, 10))

with rasterio.open(f"{OUT}/float32-nan-nodata.tif", "w", count=1, dtype="float32", nodata=np.nan,
                   tiled=True, blockxsize=16, blockysize=16, compress="deflate", predictor=3, **common) as d:
    d.write(z, 1)
with rasterio.open(f"{OUT}/int16-band-interleave.tif", "w", count=3, dtype="int16", nodata=-32768,
                   interleave="band", compress="lzw", predictor=2, **common) as d:
    for k in range(3):
        a = np.nan_to_num(z * (k + 1), nan=-32768).astype(np.int16)
        d.write(a, k + 1)
with rasterio.open(f"{OUT}/uint8-dataset-mask.tif", "w", count=1, dtype="uint8",
                   tiled=True, blockxsize=32, blockysize=32, compress="deflate", **common) as d:
    d.write(np.nan_to_num(z + 128, nan=0).astype(np.uint8), 1)
    d.write_mask(~np.isnan(z))  # GDAL internal mask, no NoData value
with rasterio.open(f"{OUT}/uint16-rgba-alpha.tif", "w", count=4, dtype="uint16", photometric="RGB",
                   compress="zstd", **common) as d:
    for k in range(3):
        d.write(np.nan_to_num(z * 300 + 30000, nan=0).astype(np.uint16), k + 1)
    d.write(np.where(np.isnan(z), 0, 65535).astype(np.uint16), 4)
    d.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                     rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
with rasterio.open(f"{OUT}/float64-ungeoreferenced.tif", "w", driver="GTiff", width=w, height=h,
                   count=1, dtype="float64", compress="deflate", predictor=2) as d:
    d.write(z.astype(np.float64) * 1e-3, 1)
print(sorted(os.listdir(OUT)))
