"""GDAL's reading of a GeoTIFF, recorded as the truth cog is judged by.

Shared by cogmake.py (files GDAL writes) and cogcorpus.py (files it did
not). For one file it writes, through GDAL's public API only:

  <case>.b<band>.l<level>.f32    GDAL's cells converted to float32 by GDAL
                                 (ReadRaster with buf_type=Float32)
  <case>.b<band>.l<level>.mask   GDAL's mask band: nonzero valid, 0 invalid

and returns the file's manifest entry: bands, sizes and geotransform per
level, EPSG code, NoData, and GDAL's mask flags. Levels are the full
resolution and then each overview in GDAL's order.

A level larger than MAX_CELLS is recorded over one window of it rather
than whole, so a 36000² product does not need 5 GB per band: the window
starts a third of the way in on each axis, which lines up with no block
size, and is at most WINDOW cells on a side. cog/main.go applies the
same rule, and cogcompare.py checks that both read the same window.
Likewise at most MAX_BANDS bands are recorded.
"""

import math
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

MAX_CELLS = 1 << 24
WINDOW = 4096
MAX_BANDS = 16


def window(w, h):
    """The region of a w×h level that is compared: x, y, width, height."""
    if w * h <= MAX_CELLS:
        return [0, 0, w, h]
    x, y = w // 3, h // 3
    return [x, y, min(w - x, WINDOW), min(h - y, WINDOW)]


def jsonable(v):
    """v with every non-finite float spelled as a string ("nan", "-inf"),
    so that manifest.json is strict JSON that Go's decoder accepts."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    return v


def features(path, ds):
    """A short description of the file's layout, for the report. It is
    GDAL's view plus the header bytes, never used for judging."""
    with open(path, "rb") as f:
        head = f.read(4)
    b1 = ds.GetRasterBand(1)
    st = ds.GetMetadata("IMAGE_STRUCTURE") or {}
    bw, bh = b1.GetBlockSize()
    out = [gdal.GetDataTypeName(b1.DataType)]
    nbits = b1.GetMetadataItem("NBITS", "IMAGE_STRUCTURE")
    if nbits:
        out[0] += f"/{nbits}bit"
    if ds.RasterCount > 1:
        out.append(f"{ds.RasterCount} bands {st.get('INTERLEAVE', '').lower()}".strip())
    out.append(st.get("COMPRESSION", "none").lower())
    if st.get("PREDICTOR", "1") != "1":
        out.append(f"pred {st['PREDICTOR']}")
    out.append(f"strips of {bh}" if bw == ds.RasterXSize else f"tiles {bw}x{bh}")
    if head[:2] == b"MM":
        out.append("big-endian")
    if head[2:4] in (b"\x00\x2b", b"\x2b\x00"):
        out.append("BigTIFF")
    if st.get("LAYOUT") == "COG":
        out.append("COG")
    if b1.GetOverviewCount():
        out.append(f"{b1.GetOverviewCount()} overviews")
    flags = b1.GetMaskFlags()
    if flags & gdal.GMF_ALPHA:
        out.append("alpha")
    elif flags & gdal.GMF_PER_DATASET:
        out.append("mask")
    if b1.GetNoDataValue() is not None:
        out.append(f"NoData {b1.GetNoDataValue():g}")
    return ", ".join(out)


def record(path, name, out):
    """Record GDAL's reading of path as case name in out, and return the
    manifest entry. A file GDAL cannot open or read gets gdal_error."""
    entry = {"name": name}
    warnings = []

    def handler(cls, _no, msg):
        if cls >= gdal.CE_Warning:
            warnings.append(msg.strip())

    gdal.PushErrorHandler(handler)
    try:
        _record(path, name, out, entry)
    except (RuntimeError, ValueError, MemoryError) as e:
        entry["gdal_error"] = str(e).strip() or type(e).__name__
    finally:
        gdal.PopErrorHandler()
    if warnings:
        entry["gdal_warnings"] = warnings[:5]
    return jsonable(entry)


def _record(path, name, out, entry):
    ds = gdal.Open(path)
    b1 = ds.GetRasterBand(1)
    levels = 1 + b1.GetOverviewCount()
    epsg = None
    srs = ds.GetSpatialRef()
    if srs is not None:
        # strata's grid is 2D, so the code compared is that of the
        # horizontal CRS: a compound CRS loses its vertical part, and a
        # 3D geographic CRS (EPSG:4979 from GeoTIFF 1.1 keys) its height.
        srs = srs.Clone()
        if srs.IsCompound():
            srs.StripVertical()
        if srs.GetAxesCount() == 3:
            srs.DemoteTo2D()
        try:
            srs.AutoIdentifyEPSG()
        except RuntimeError:
            pass
        code = srs.GetAuthorityCode(None)
        epsg = int(code) if code and code.isdigit() else None
    software = ds.GetMetadataItem("TIFFTAG_SOFTWARE")
    entry.update({"bands": ds.RasterCount, "levels": [], "epsg": epsg,
                  "nodata": b1.GetNoDataValue(), "type": gdal.GetDataTypeName(b1.DataType),
                  "mask_flags": b1.GetMaskFlags(), "features": features(path, ds),
                  "software": software.strip() if software else None})
    for lvl in range(levels):
        # An overview's geotransform, as GDAL reports it when the
        # overview is opened as a dataset of its own.
        lds = ds if lvl == 0 else gdal.OpenEx(path, open_options=[f"OVERVIEW_LEVEL={lvl - 1}"])
        w, h = lds.RasterXSize, lds.RasterYSize
        x, y, ww, wh = win = window(w, h)
        entry["levels"].append({"w": w, "h": h, "gt": list(lds.GetGeoTransform()), "win": win})
        for k in range(min(ds.RasterCount, MAX_BANDS)):
            band = lds.GetRasterBand(k + 1)
            raw = band.ReadRaster(x, y, ww, wh, buf_type=gdal.GDT_Float32)
            with open(os.path.join(out, f"{name}.b{k}.l{lvl}.f32"), "wb") as f:
                f.write(np.frombuffer(raw, dtype=np.float32).astype("<f4").tobytes())
            mask = band.GetMaskBand().ReadRaster(x, y, ww, wh, buf_type=gdal.GDT_Byte)
            with open(os.path.join(out, f"{name}.b{k}.l{lvl}.mask"), "wb") as f:
                f.write(mask)
