#!/usr/bin/env bash
# Cross-check strata's multi-scale derivatives (FitRadius) against GRASS
# GIS r.param.scale on a real GeoTIFF, using GRASS in a Docker container
# so nothing has to be installed locally.
#
#   ./grasscheck.sh <geotiff> [xoff yoff size]
#
# Example (the window gdalcheck.sh is run on):
#   ./grasscheck.sh "C:/Users/you/Downloads/HGV_leaf.tif" 0 6127 4096
#
# It extracts a window, runs r.param.scale's slope, aspect, profc, planc,
# crosc, minic and maxic at window sizes 3, 9 and 17 (STRATA_FIT_SIZES)
# with exponent 0 (unweighted), runs strata's Slope, Aspect and
# Curvature (profile, plan, mean) at FitRadius 1, 4 and 8 through the
# bounded-memory Chunked path, and differences the results with
# grasscompare.py. Needs Docker, Go and numpy.
set -euo pipefail

SRC=${1:?usage: grasscheck.sh <geotiff> [xoff yoff size]}
XOFF=${2:-0}
YOFF=${3:-0}
SIZE=${4:-4096}
TILE=${STRATA_TILE:-256}
SIZES=${STRATA_FIT_SIZES:-3 9 17}

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out-grass"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
SRC_NAME=$(basename "$SRC")
OUT_WIN=$(win "$OUT")
HERE_WIN=$(win "$HERE")
IMAGE=${GRASS_IMAGE:-osgeo/grass-gis:8.5.0-ubuntu}

echo "== GRASS ($IMAGE) =="
# grassrun.sh is mounted from a checkout that may have CRLF line endings.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$SRC_DIR:/in:ro" -v "$OUT_WIN:/out" -v "$HERE_WIN/grassrun.sh:/grassrun.sh:ro" "$IMAGE" \
  bash -c "tr -d '\r' </grassrun.sh >/tmp/grassrun.sh && bash /tmp/grassrun.sh '$SRC_NAME' $XOFF $YOFF $SIZE '$SIZES'"

# The DEM's own NoData and cell size, read from the ENVI header GDAL
# wrote, as gdalcheck.sh does. No fallbacks.
FILL=$(sed -n 's/.*data ignore value[[:space:]]*=[[:space:]]*\([-0-9.eE+]*\).*/\1/p' "$OUT/dem.hdr" | head -1)
CELL=$(sed -n 's/.*map info[[:space:]]*=[[:space:]]*{[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p' "$OUT/dem.hdr" | tr -d ' ')
: "${FILL:?no NoData (data ignore value) in $OUT/dem.hdr}" "${CELL:?no cell size (map info) in $OUT/dem.hdr}"

RADII=$(for s in $SIZES; do echo $(((s - 1) / 2)); done | paste -sd, -)

echo
echo "== strata (cell size $CELL, NoData $FILL, FitRadius $RADII, tile height $TILE) =="
(cd "$HERE" && go run ./grass -dir out-grass -w "$SIZE" -h "$SIZE" -cell "$CELL" -fill "$FILL" -radii "$RADII" -tile "$TILE")

echo
echo "== difference =="
(cd "$HERE" && python grasscompare.py out-grass "$SIZE" "$SIZE" "$FILL" "$CELL" "$RADII")
