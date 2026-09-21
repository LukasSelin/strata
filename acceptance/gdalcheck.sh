#!/usr/bin/env bash
# Cross-check strata against gdaldem on a real GeoTIFF, using the GDAL
# suite in a Docker container so nothing has to be installed locally.
#
#   ./gdalcheck.sh <geotiff> [xoff yoff size]
#
# Example:
#   ./gdalcheck.sh "C:/Users/you/Downloads/dem.tif" 0 6127 4096
#
# It extracts a window, runs gdaldem slope/aspect/hillshade on it, runs
# the same three operations through strata's bounded-memory Chunked
# path, and differences the results. Needs Docker, Go and numpy.
set -euo pipefail

SRC=${1:?usage: gdalcheck.sh <geotiff> [xoff yoff size]}
XOFF=${2:-0}
YOFF=${3:-0}
SIZE=${4:-4096}
TILE=${STRATA_TILE:-256}

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out-gdal"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
SRC_NAME=$(basename "$SRC")
OUT_WIN=$(win "$OUT")
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}

echo "== GDAL ($IMAGE) =="
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$SRC_DIR:/in:ro" -v "$OUT_WIN:/out" "$IMAGE" bash -c "
set -e
cd /out
rm -f dem.tif gdal-*.tif *.raw *.hdr *.aux.xml
gdal_translate -q -ot Float32 -srcwin $XOFF $YOFF $SIZE $SIZE '/in/$SRC_NAME' dem.tif
gdaldem slope     dem.tif gdal-slope.tif     -q
gdaldem aspect    dem.tif gdal-aspect.tif    -q
gdaldem hillshade dem.tif gdal-hillshade.tif -q
for f in dem gdal-slope gdal-aspect; do
  gdal_translate -q -of ENVI -ot Float32 \$f.tif \$f.raw
done
gdal_translate -q -of ENVI -ot Byte gdal-hillshade.tif gdal-hillshade.raw
gdalinfo --version
gdalinfo dem.tif | grep -E 'Pixel Size|NoData'
"

# The DEM's own NoData and cell size, read from the ENVI header GDAL wrote.
# ENVI spells NoData "data ignore value = ..."; map info is
# {projection, x, y, easting, northing, xsize, ysize, ...}. No fallbacks:
# guessing either value would make the comparison meaningless.
FILL=$(sed -n 's/.*data ignore value[[:space:]]*=[[:space:]]*\([-0-9.eE+]*\).*/\1/p' "$OUT/dem.hdr" | head -1)
CELL=$(sed -n 's/.*map info[[:space:]]*=[[:space:]]*{[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p' "$OUT/dem.hdr" | tr -d ' ')
: "${FILL:?no NoData (data ignore value) in $OUT/dem.hdr}" "${CELL:?no cell size (map info) in $OUT/dem.hdr}"

echo
echo "== strata (cell size $CELL, NoData $FILL, tile height $TILE) =="
(cd "$HERE" && go run ./gdal -dir out-gdal -w "$SIZE" -h "$SIZE" -cell "$CELL" -fill "$FILL" -tile "$TILE")

echo
echo "== difference =="
(cd "$HERE" && STRATA_TILE="$TILE" python gdalcompare.py out-gdal "$SIZE" "$SIZE" "$FILL" "$CELL")
