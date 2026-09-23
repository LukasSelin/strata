#!/usr/bin/env bash
# Write the generated part of the corpus: files from tiffcp (libtiff),
# rasterio and rio-cogeo, and GDAL with options cogmake.py does not use.
# Each writer runs in its own container, so nothing is installed locally.
# Run fetch.py first: some generators rewrite fetched files.
#
#   ./generate.sh [dir]        # default: files/
#
# These are not hash-pinned, since they depend on the writers' versions;
# each generated-* directory records the version that wrote it.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${1:-$HERE/files}
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
GDAL_IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}
M=(-v "$(win "$HERE"):/gen:ro" -v "$(win "$DIR"):/in:ro" -v "$(win "$DIR"):/out")

echo "== tiffcp =="
MSYS_NO_PATHCONV=1 docker run --rm "${M[@]}" debian:stable-slim sh /gen/gen_tiffcp.sh
echo "== rasterio / rio-cogeo =="
MSYS_NO_PATHCONV=1 docker run --rm "${M[@]}" python:3.12-slim \
  sh -c "apt-get update -qq >/dev/null && apt-get install -y -qq libexpat1 >/dev/null 2>&1 && pip install -q --root-user-action=ignore --disable-pip-version-check rio-cogeo >/dev/null && python /gen/gen_rasterio.py"
echo "== GDAL, unusual options =="
MSYS_NO_PATHCONV=1 docker run --rm "${M[@]}" "$GDAL_IMAGE" python3 /gen/gen_gdal.py
