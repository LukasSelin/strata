#!/usr/bin/env python3
"""Write GeoTIFFs with GDAL options that cogmake.py does not use.

cogmake.py covers the layouts a GDAL user gets by default. These are the
ones they get by asking: big-endian with the horizontal predictor,
internal masks and alpha, COG band and tile interleaving, rows per strip
of 1 or more than the height, sparse files without NoData, NoData values
the type cannot hold, and the sample types and codecs cog refuses. Runs
in the GDAL container (generate.sh starts it); writes to /out.
"""

import os

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
OUT = "/out/generated-gdal"
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "VERSION.txt"), "w") as f:
    f.write(f"GDAL {gdal.__version__}\n")

W, H = 203, 157
y, x = np.mgrid[0:H, 0:W].astype(np.float64)
Z = 500 + 200 * np.sin(x / 19) * np.cos(y / 23) + 0.1 * x
HOLE = (x - 120) ** 2 + (y - 60) ** 2 < 25 ** 2
srs = osr.SpatialReference()
srs.ImportFromEPSG(3006)


def mem(dtype, bands=1, nodata=None, values=None, gt=(600000.0, 10.0, 0.0, 6700000.0, 0.0, -10.0)):
    d = gdal.GetDriverByName("MEM").Create("", W, H, bands, dtype)
    d.SetGeoTransform(gt)
    d.SetProjection(srs.ExportToWkt())
    for k in range(bands):
        v = (Z * (k + 1)) if values is None else values(k)
        b = d.GetRasterBand(k + 1)
        if nodata is not None:
            v = np.where(HOLE, nodata, v)
            b.SetNoDataValue(nodata)
        b.WriteArray(v)
    return d


def write(name, src, opts, fmt="GTiff"):
    try:
        gdal.Translate(f"{OUT}/{name}.tif", src, format=fmt, creationOptions=opts)
    except RuntimeError as e:
        print(f"skipped {name}: {e}")


# Big-endian with the horizontal predictor, which cogmake.py never combines.
write("uint16-be-pred2-strips", mem(gdal.GDT_UInt16, nodata=0), ["ENDIANNESS=BIG", "COMPRESS=LZW", "PREDICTOR=2"])
write("int32-be-pred2-tiles", mem(gdal.GDT_Int32, 3, nodata=-1), ["ENDIANNESS=BIG", "COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BLOCKXSIZE=32", "BLOCKYSIZE=16"])
write("float64-be-pred2", mem(gdal.GDT_Float64, nodata=-9999), ["ENDIANNESS=BIG", "COMPRESS=ZSTD", "PREDICTOR=2"])
write("float32-pred2-pixel", mem(gdal.GDT_Float32, 3), ["COMPRESS=LZW", "PREDICTOR=2", "INTERLEAVE=PIXEL"])
write("byte-be-pred2-pixel", mem(gdal.GDT_Byte, 3, values=lambda k: (Z * (k + 1)) % 256), ["ENDIANNESS=BIG", "COMPRESS=LZW", "PREDICTOR=2"])
# Strip heights.
write("float32-rps1", mem(gdal.GDT_Float32, nodata=-1), ["BLOCKYSIZE=1", "COMPRESS=DEFLATE"])
write("float32-rps-over-height", mem(gdal.GDT_Float32), [f"BLOCKYSIZE={H + 50}"])
write("int16-planar-rps13", mem(gdal.GDT_Int16, 4, nodata=-32768), ["INTERLEAVE=BAND", "BLOCKYSIZE=13", "COMPRESS=ZSTD", "PREDICTOR=2"])
# Sparse without NoData: absent blocks read as 0.
d = gdal.GetDriverByName("GTiff").Create(f"{OUT}/float32-sparse-nonodata.tif", W, H, 1, gdal.GDT_Float32,
                                          ["SPARSE_OK=TRUE", "TILED=YES", "BLOCKXSIZE=32", "BLOCKYSIZE=32"])
d.SetGeoTransform((0, 1, 0, 0, 0, -1))
d.GetRasterBand(1).WriteArray(Z[:40, :70].astype(np.float32), 20, 30)
d = None
# NoData values the sample type cannot hold, and special floats.
write("byte-nodata-fraction", mem(gdal.GDT_Byte, values=lambda k: Z % 256), [])
d = gdal.Open(f"{OUT}/byte-nodata-fraction.tif", gdal.GA_Update)
d.GetRasterBand(1).SetNoDataValue(12.5)
d = None
write("float32-nodata-nan", mem(gdal.GDT_Float32, nodata=float("nan")), ["COMPRESS=DEFLATE", "PREDICTOR=3"])
write("float32-nodata-inf", mem(gdal.GDT_Float32, nodata=float("-inf")), ["TILED=YES"])
write("float32-nodata-1e-9", mem(gdal.GDT_Float32, nodata=1e-9), [])
write("float64-nodata-maxfloat", mem(gdal.GDT_Float64, nodata=1.7976931348623157e308), [])
write("uint32-nodata-max", mem(gdal.GDT_UInt32, nodata=4294967295), ["COMPRESS=LZW"])


def near(nodata, dtype, steps):
    """Cells at NoData and at the given float steps away from it, to
    find how close to NoData GDAL's mask still calls a cell NoData."""
    ulps = np.array(steps, dtype=np.int64)
    if dtype == gdal.GDT_Float32:
        base = np.float32(nodata).view(np.int32)
        v = (base + ulps).astype(np.int32).view(np.float32).astype(np.float64)
    else:
        base = np.float64(nodata).view(np.int64)
        v = (base + ulps).view(np.float64)
    return lambda k: np.resize(v, (H, W))


F32_STEPS = list(range(-12, 13))
F64_STEPS = [0, 1, -1, 1 << 20, -(1 << 20), 1 << 28, -(1 << 28), 1 << 29, -(1 << 29), 1 << 30, -(1 << 30), 1 << 32]
for nd, name in ((-9999.0, "m9999"), (1e-9, "1e-9"), (3.0e38, "3e38")):
    d = mem(gdal.GDT_Float32, values=near(nd, gdal.GDT_Float32, F32_STEPS))
    d.GetRasterBand(1).SetNoDataValue(nd)
    write(f"float32-nodata-near-{name}", d, [])
    d = mem(gdal.GDT_Float64, values=near(nd, gdal.GDT_Float64, F64_STEPS))
    d.GetRasterBand(1).SetNoDataValue(nd)
    write(f"float64-nodata-near-{name}", d, [])
# Integer NoData that is fractional or out of the type's range.
for dtype, name, nd, vals in ((gdal.GDT_Int16, "int16-nodata-m3.7", -3.7, [-4, -3, 3, 4]),
                              (gdal.GDT_Byte, "byte-nodata-300", 300, [44, 255, 0]),
                              (gdal.GDT_Int32, "int32-nodata-1e10", 1e10, [0, 2147483647, -2147483648])):
    d = mem(dtype, values=lambda k, vals=vals: np.resize(np.array(vals, dtype=np.float64), (H, W)))
    d.GetRasterBand(1).SetNoDataValue(nd)
    write(name, d, [])
# A float32 NoData beyond float32's range, which the driver rounds to Inf.
d = mem(gdal.GDT_Float32, values=lambda k: np.resize(np.array([np.inf, 1.0, -np.inf, 3e38]), (H, W)))
d.GetRasterBand(1).SetNoDataValue(1e39)
write("float32-nodata-1e39", d, [])
# Masks and alpha.
m = mem(gdal.GDT_Int16)
m.CreateMaskBand(gdal.GMF_PER_DATASET)
m.GetRasterBand(1).GetMaskBand().WriteArray(np.where(HOLE, 0, 255).astype(np.uint8))
write("int16-internal-mask", m, ["TILED=YES", "COMPRESS=DEFLATE"])
write("int16-internal-mask-cog", m, ["COMPRESS=DEFLATE", "BLOCKSIZE=64", "OVERVIEWS=AUTO"], fmt="COG")
a = mem(gdal.GDT_Byte, 2, values=lambda k: (Z % 256) if k == 0 else np.where(HOLE, 0, 255))
write("byte-alpha", a, ["ALPHA=YES", "COMPRESS=DEFLATE"])
# COG layouts cogmake.py does not ask for.
rgb = mem(gdal.GDT_UInt16, 3, nodata=0)
write("cog-uint16-interleave-band", rgb, ["INTERLEAVE=BAND", "BLOCKSIZE=64", "COMPRESS=ZSTD"], fmt="COG")
write("cog-uint16-interleave-tile", rgb, ["INTERLEAVE=TILE", "BLOCKSIZE=64", "COMPRESS=DEFLATE"], fmt="COG")
write("cog-float32-googlemaps", mem(gdal.GDT_Float32, nodata=-9999), ["TILING_SCHEME=GoogleMapsCompatible", "COMPRESS=DEFLATE"], fmt="COG")
# Overviews built into a plain GeoTIFF, compressed differently from it.
p = f"{OUT}/int16-internal-overviews.tif"
gdal.Translate(p, mem(gdal.GDT_Int16, nodata=-32768), creationOptions=["TILED=YES", "COMPRESS=LZW"])
with gdal.config_options({"COMPRESS_OVERVIEW": "ZSTD", "PREDICTOR_OVERVIEW": "2"}):
    ds = gdal.Open(p, gdal.GA_Update)
    ds.BuildOverviews("AVERAGE", [2, 4, 8])
    ds = None
# Other profiles and georeferencing.
write("float32-baseline-tfw", mem(gdal.GDT_Float32), ["PROFILE=BASELINE", "TFW=YES"])
write("float32-geotiff-1.1", mem(gdal.GDT_Float32), ["GEOTIFF_VERSION=1.1"])
write("float32-streamable", mem(gdal.GDT_Float32, nodata=-1), ["STREAMABLE_OUTPUT=YES"])
write("float32-rotated", mem(gdal.GDT_Float32, gt=(600000.0, 10.0, 1.0, 6700000.0, 1.0, -10.0)), [])
g = mem(gdal.GDT_Float32)
g.SetGCPs([gdal.GCP(600000 + 10 * i, 6700000 - 10 * j, 0, i, j) for i, j in ((0, 0), (W, 0), (0, H), (W, H))], srs.ExportToWkt())
write("float32-gcps", g, [])
one = gdal.GetDriverByName("MEM").Create("", 1, 1, 1, gdal.GDT_Float32)
one.SetGeoTransform((10, 1, 0, 20, 0, -1))
one.GetRasterBand(1).Fill(42)
write("float32-1x1", one, ["TILED=YES"])
write("palette-byte", mem(gdal.GDT_Byte, values=lambda k: Z % 256), ["PHOTOMETRIC=PALETTE"])
# Sample types and codecs cog refuses: each must be refused by name.
write("int64", mem(gdal.GDT_Int64), [])
write("uint64", mem(gdal.GDT_UInt64), [])
write("cint16", mem(gdal.GDT_CInt16), [])
write("float16", mem(gdal.GDT_Float32), ["NBITS=16"])
write("uint12", mem(gdal.GDT_UInt16, values=lambda k: Z % 4096), ["NBITS=12"])
write("bit1", mem(gdal.GDT_Byte, values=lambda k: Z > 500), ["NBITS=1"])
for c in ("JPEG", "WEBP", "LERC", "LERC_DEFLATE", "LZMA", "JXL"):
    bands = 3 if c == "WEBP" else 1  # WebP takes RGB or RGBA only
    write(f"byte-{c.lower()}", mem(gdal.GDT_Byte, bands, values=lambda k: Z % 256), [f"COMPRESS={c}"])
print(sorted(os.listdir(OUT)))
