#!/usr/bin/env python3
"""Write a matrix of GeoTIFFs with GDAL, and GDAL's own reading of each.

Runs inside the GDAL container (cogcheck.sh starts it). For every file
it writes, it also records what GDAL reads back from it, which is the
truth that strata's cog reader is judged against:

  <case>.tif                     the file, written by gdal.Translate
  <case>.b<band>.l<level>.f32    GDAL's cells converted to float32 by GDAL
  <case>.b<band>.l<level>.mask   GDAL's mask band
  manifest.json                  per file: bands, sizes and geotransform
                                 per level, EPSG code, NoData

cogtruth.py records the reading. Every value is GDAL's, read through its
public API; nothing is computed from strata's code or conventions.

Usage:  python3 cogmake.py <outdir> [source.tif]

Without a source it synthesises a 701x517 DEM (odd sizes, so every
layout has ragged edge blocks) with NoData holes, one of them large
enough to leave whole blocks empty, which COG writes as sparse blocks.
"""

import json
import os
import sys

import numpy as np
from osgeo import gdal, osr

from cogtruth import record

gdal.UseExceptions()

OUT = sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
DEM_ND = -9999.0


def synthetic_dem():
    w, h = 701, 517
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(42)
    z = (400 + 150 * np.sin(x / 57.0) * np.cos(y / 43.0) + 0.08 * x + 0.03 * y
         + rng.normal(0, 0.7, (h, w)))
    z[(x - 500) ** 2 + (y - 120) ** 2 < 30 ** 2] = DEM_ND  # a small hole
    z[300:517, 0:260] = DEM_ND                                 # whole blocks
    z[rng.random((h, w)) < 0.002] = DEM_ND                     # speckle
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((500000.0, 12.5, 0.0, 7000000.0, 0.0, -12.5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    ds.SetProjection(srs.ExportToWkt())
    b = ds.GetRasterBand(1)
    b.WriteArray(z.astype(np.float32))
    b.SetNoDataValue(DEM_ND)
    return ds


def load_source(path):
    ds = gdal.Open(path)
    mem = gdal.Translate("", ds, format="MEM", outputType=gdal.GDT_Float32,
                         srcWin=[0, 0, min(ds.RasterXSize, 1500), min(ds.RasterYSize, 1500)])
    b = mem.GetRasterBand(1)
    if b.GetNoDataValue() is None:
        b.SetNoDataValue(DEM_ND)
    return mem


dem = load_source(SRC) if SRC else synthetic_dem()
W, H = dem.RasterXSize, dem.RasterYSize
z = dem.GetRasterBand(1).ReadAsArray().astype(np.float64)
nd_in = dem.GetRasterBand(1).GetNoDataValue()
hole = (z == nd_in) | np.isnan(z)
zv = z[~hole]
zmin, zmax = float(zv.min()), float(zv.max())
unit = (z - zmin) / (zmax - zmin)  # 0..1 over the valid cells

# Each type gets the DEM mapped onto most of its range, so its extremes
# and sign are exercised, and a NoData value of its own.
TYPES = {
    "Byte":    (gdal.GDT_Byte,    lambda u: np.round(u * 254),                     255),
    "Int8":    (gdal.GDT_Int8,    lambda u: np.round(u * 254 - 127),               -128),
    "UInt16":  (gdal.GDT_UInt16,  lambda u: np.round(u * 65000),                   65535),
    "Int16":   (gdal.GDT_Int16,   lambda u: np.round(u * 65000 - 32500),           -32768),
    "UInt32":  (gdal.GDT_UInt32,  lambda u: np.round(u * 4.2e9),                   4294967295),
    "Int32":   (gdal.GDT_Int32,   lambda u: np.round(u * 4.2e9 - 2.1e9),           -2147483648),
    "Float32": (gdal.GDT_Float32, lambda u: zmin + u * (zmax - zmin),              DEM_ND),
    "Float64": (gdal.GDT_Float64, lambda u: (zmin + u * (zmax - zmin)) * 1.000001, DEM_ND),
}


def typed(name, bands=1, nodata=True, scale=1.0):
    """A MEM dataset of the DEM as type name, with bands variants."""
    gtype, f, nd = TYPES[name]
    ds = gdal.GetDriverByName("MEM").Create("", W, H, bands, gtype)
    ds.SetGeoTransform(dem.GetGeoTransform())
    ds.SetProjection(dem.GetProjection())
    for k in range(bands):
        # Later bands are the first reflected, so a band mix-up shows.
        u = unit if k % 2 == 0 else 1 - unit
        v = f(u) * scale
        v[hole] = nd if nodata else f(np.zeros(1))[0]
        b = ds.GetRasterBand(k + 1)
        b.WriteArray(v)
        if nodata:
            b.SetNoDataValue(nd)
    return ds


cases = []


def add(name, src, fmt, opts, metadata=None):
    path = os.path.join(OUT, name + ".tif")
    if metadata:
        src.SetMetadata(metadata)
    gdal.Translate(path, src, format=fmt, creationOptions=opts)
    cases.append(name)


# COGs: every type, compression and predictor that applies, 128-cell
# blocks so the 701x517 DEM gets overviews.
for tname in TYPES:
    isfloat = tname.startswith("Float")
    src = typed(tname)
    for comp in ("NONE", "LZW", "DEFLATE", "ZSTD", "PACKBITS"):
        preds = ["NO"] if comp in ("NONE", "PACKBITS") else ["NO", "STANDARD"]
        if isfloat and comp not in ("NONE", "PACKBITS"):
            preds.append("FLOATING_POINT")
        for pred in preds:
            opts = [f"COMPRESS={comp}", "BLOCKSIZE=128", "OVERVIEWS=AUTO", "SPARSE_OK=TRUE"]
            if pred != "NO":
                opts.append(f"PREDICTOR={pred}")
            add(f"cog-{tname}-{comp}-{pred}", src, "COG", opts)

# BigTIFF COGs.
add("cog-Float32-DEFLATE-FLOATING_POINT-bigtiff", typed("Float32"), "COG",
    ["COMPRESS=DEFLATE", "PREDICTOR=FLOATING_POINT", "BLOCKSIZE=256", "BIGTIFF=YES"])
add("cog-Int16-ZSTD-STANDARD-bigtiff", typed("Int16"), "COG",
    ["COMPRESS=ZSTD", "PREDICTOR=STANDARD", "BLOCKSIZE=128", "BIGTIFF=YES"])

# Plain GeoTIFFs: strips and tiles, pixel and band interleaving, three
# bands, big-endian, and no NoData at all.
for tname in ("Byte", "Int16", "UInt32", "Float32", "Float64"):
    src3 = typed(tname, bands=3)
    for inter in ("PIXEL", "BAND"):
        add(f"gtiff-{tname}-strips-{inter}", src3, "GTiff",
            [f"INTERLEAVE={inter}", "COMPRESS=DEFLATE", "PREDICTOR=2", "BLOCKYSIZE=7"])
        add(f"gtiff-{tname}-tiles-{inter}-be", src3, "GTiff",
            [f"INTERLEAVE={inter}", "TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=32",
             "COMPRESS=LZW", "ENDIANNESS=BIG"])
add("gtiff-Float32-strips-nonodata", typed("Float32", nodata=False), "GTiff", ["COMPRESS=NONE"])
add("gtiff-Float32-onestrip", typed("Float32"), "GTiff", [f"BLOCKYSIZE={H}"])
add("gtiff-Float32-tiles-be-fp", typed("Float32"), "GTiff",
    ["TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=3", "ENDIANNESS=BIG"])
add("gtiff-Float64-tiles-be-fp", typed("Float64"), "GTiff",
    ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "ENDIANNESS=BIG", "BLOCKXSIZE=48", "BLOCKYSIZE=16"])
# float64 values far beyond float32's range: how GDAL converts them.
add("gtiff-Float64-huge", typed("Float64", scale=1e290), "GTiff", ["TILED=YES"])
# PixelIsPoint: the tiepoint is a cell centre.
add("gtiff-Float32-point", typed("Float32"), "GTiff", ["TILED=YES"],
    metadata={"AREA_OR_POINT": "Point"})

manifest = [record(os.path.join(OUT, name + ".tif"), name, OUT) for name in cases]
for e in manifest:
    if "gdal_error" in e:
        sys.exit(f"GDAL cannot read back its own {e['name']}: {e['gdal_error']}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"gdal": gdal.__version__, "files": manifest}, f, indent=1)
print(f"GDAL {gdal.__version__}: {len(cases)} files, "
      f"{sum(len(e['levels']) * e['bands'] for e in manifest)} band-levels")
