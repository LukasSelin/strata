#!/usr/bin/env python3
"""Record GDAL's reading of every GeoTIFF in a directory tree.

The corpus counterpart of cogmake.py: instead of writing files, it takes
the ones it finds, which other software wrote. Runs inside the GDAL
container (cogcorpus.sh starts it). Every *.tif and *.tiff under <indir>
becomes a case named after its path (a/b/c.tif -> a__b__c), recorded by
cogtruth.py into <outdir>, and manifest.json lists them with their paths
relative to <indir>, so cog/main.go can open the same files.

GDAL reads each file alone: GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR keeps
it from picking up sidecars (.ovr, .msk, .aux.xml, .tfw) that strata does
not read either, and PAM is off so nothing is written next to the file.
A file GDAL cannot read is recorded with its error; cogcompare.py then
expects cog to refuse it too.

Usage:  python3 cogcorpus.py <indir> <outdir>
"""

import json
import os
import re
import sys

from osgeo import gdal

from cogtruth import record

gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

IN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

paths = []
for root, dirs, files in os.walk(IN):
    dirs.sort()
    for f in sorted(files):
        if re.search(r"\.tiff?$", f, re.I):
            paths.append(os.path.relpath(os.path.join(root, f), IN).replace(os.sep, "/"))

manifest = []
for rel in paths:
    name = re.sub(r"\.tiff?$", "", rel, flags=re.I).replace("/", "__")
    e = record(os.path.join(IN, rel), name, OUT)
    e["path"] = rel
    manifest.append(e)
    print(f"  {rel:<70} {e.get('gdal_error', e.get('features', ''))[:80]}", flush=True)

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"gdal": gdal.__version__, "files": manifest}, f, indent=1)
bad = sum("gdal_error" in e for e in manifest)
print(f"GDAL {gdal.__version__}: {len(manifest)} files, {bad} that GDAL cannot read")
