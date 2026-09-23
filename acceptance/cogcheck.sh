#!/usr/bin/env bash
# Judge strata's GeoTIFF/COG reader (the cog module) against GDAL, using
# the GDAL suite in a Docker container so nothing has to be installed
# locally.
#
#   ./cogcheck.sh [geotiff]
#
# GDAL writes a matrix of files (cogmake.py): COGs of every sample type,
# compression and predictor, BigTIFF, stripped and tiled GeoTIFFs with
# pixel and band interleaving, big-endian, sparse blocks, overviews and
# PixelIsPoint. The files come from a synthetic DEM, or from the first
# 1500x1500 cells of the given GeoTIFF. GDAL also records its own reading
# of every file. strata then reads the same files through cog
# (cog/main.go), and cogcompare.py requires the two readings to be
# identical. Needs Docker and Go; the comparison runs in the container,
# so numpy is not needed locally.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out-cog"
rm -rf "$OUT"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}
MOUNTS=(-v "$(win "$HERE"):/acc:ro" -v "$(win "$OUT"):/out")
SRCARG=""
if [ $# -ge 1 ]; then
  SRC_DIR=$(win "$(cd "$(dirname "$1")" && pwd)")
  MOUNTS+=(-v "$SRC_DIR:/in:ro")
  SRCARG="/in/$(basename "$1")"
fi

echo "== GDAL ($IMAGE) writes the files and reads them =="
MSYS_NO_PATHCONV=1 docker run --rm "${MOUNTS[@]}" "$IMAGE" \
  python3 /acc/cogmake.py /out $SRCARG

echo
echo "== strata reads them through cog =="
(cd "$HERE" && go run ./cog -dir out-cog)

echo
echo "== difference =="
MSYS_NO_PATHCONV=1 docker run --rm "${MOUNTS[@]}" "$IMAGE" \
  python3 /acc/cogcompare.py /out
