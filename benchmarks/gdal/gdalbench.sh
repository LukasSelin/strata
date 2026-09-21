#!/usr/bin/env bash
# Time strata against gdaldem on a real raster.
#
#   ./gdalbench.sh <geotiff> [xoff yoff width height]
#
# Both tools run inside the same GDAL container, on the same machine,
# over the same bytes, under the same shell timer, so that the tool is
# the only thing that differs. strata is cross-compiled to a static
# linux/amd64 binary and mounted in; nothing has to be installed locally
# but Docker, Go and numpy.
#
# The window, and everything either tool reads or writes while it is
# timed, lives on a tmpfs inside the container. Docker Desktop's bind
# mount to NTFS costs more than either tool's computation, so leaving
# the files there would have measured the mount. The results are copied
# out afterwards and differenced against gdaldem's, so the run that was
# timed is also the run that is checked.
#
# Environment:
#   REPEATS=5          timed runs per case, after one warm-up run
#   TILE=256           strata's tile height
#   WORKERS="1 12 24"  strata's worker counts
#   OPS="slope ..."    which operations to time
#   TMPFS=16g          size of the in-container working filesystem
#   GDAL_IMAGE=...     the container image
set -euo pipefail

SRC=${1:?usage: gdalbench.sh <geotiff> [xoff yoff width height]}
XOFF=${2:-0}
YOFF=${3:-0}
WIDTH=${4:-11264}
HEIGHT=${5:-$WIDTH}

REPEATS=${REPEATS:-5}
TILE=${TILE:-256}
WORKERS=${WORKERS:-"1 12 24"}
OPS=${OPS:-"slope aspect hillshade"}
TMPFS=${TMPFS:-16g}
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$HERE/out"
mkdir -p "$OUT"
rm -f "$OUT"/*.raw "$OUT"/*.hdr

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
SRC_NAME=$(basename "$SRC")

echo "== building strata for linux/amd64 (GOEXPERIMENT=simd) =="
(cd "$ROOT" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOEXPERIMENT=simd \
  go build -o "$OUT/strataterrain" ./benchmarks/gdal)

echo "== timing ${WIDTH}x${HEIGHT} at ($XOFF, $YOFF), $REPEATS runs per case =="
MSYS_NO_PATHCONV=1 docker run --rm \
  --tmpfs "/work:size=$TMPFS,exec" \
  -v "$SRC_DIR:/in:ro" \
  -v "$(win "$OUT"):/out" \
  -v "$(win "$HERE")/runbench.sh:/runbench.sh:ro" \
  -v "$(win "$OUT")/strataterrain:/strataterrain:ro" \
  -e WORK=/work -e OUT=/out -e SRC="/in/$SRC_NAME" \
  -e XOFF="$XOFF" -e YOFF="$YOFF" -e W="$WIDTH" -e H="$HEIGHT" \
  -e REPEATS="$REPEATS" -e TILE="$TILE" -e WORKERS="$WORKERS" -e OPS="$OPS" \
  "$IMAGE" bash /runbench.sh | tee "$OUT/timings.txt"

echo
python "$HERE/summarize.py" "$OUT/timings.txt" "$WIDTH" "$HEIGHT" | tee "$OUT/summary.md"

echo
echo "== the timed run's output, differenced against gdaldem =="
(cd "$ROOT/acceptance" && STRATA_TILE="$TILE" \
  python gdalcompare.py "$OUT" "$WIDTH" "$HEIGHT")
