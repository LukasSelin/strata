#!/usr/bin/env bash
# Time strata's GeoTIFF/COG reader against GDAL on a real raster.
#
#   ./cogbench.sh <geotiff> [xoff yoff width height]
#
# Both run inside the same GDAL container, on the same machine, over the
# same files, under the same shell timer (runbench.sh). The container
# writes a COG per compression from the window; strata's cogbench is
# cross-compiled to a static linux/amd64 binary and mounted in.
#
# Two builds of cogbench are timed: this checkout's, and the cog reader
# as it was at BEFORE_REF, so that the effect of the decoder changes
# measured here is measured under the same timer as everything else.
#
# Environment:
#   REPEATS=5          timed runs per case, after one warm-up run
#   CACHE_REPEATS=3    timed runs per case of the cache sweep
#   TILE=256           tile height of the read and slope cases
#   N=12               the many-threads setting, for both tools
#   BEFORE_REF=9f658d3 the cog reader to compare against
#   TMPFS=16g          size of the in-container working filesystem
#   GDAL_IMAGE=...     the container image
set -euo pipefail

SRC=${1:?usage: cogbench.sh <geotiff> [xoff yoff width height]}
XOFF=${2:-0}
YOFF=${3:-0}
WIDTH=${4:-11264}
HEIGHT=${5:-$WIDTH}

REPEATS=${REPEATS:-5}
CACHE_REPEATS=${CACHE_REPEATS:-3}
TILE=${TILE:-256}
N=${N:-12}
BEFORE_REF=${BEFORE_REF:-9f658d3}
TMPFS=${TMPFS:-16g}
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$HERE/out"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling, and a
# Windows checkout may give runbench.sh CRLF line ends, which the
# container strips before running it (as ../gdalsuite does).
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
SRC_NAME=$(basename "$SRC")
build() { GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOEXPERIMENT=simd go build -o "$1" .; }

echo "== building cogbench for linux/amd64 (GOEXPERIMENT=simd) =="
(cd "$HERE" && build "$OUT/cogbench")

echo "== building cogbench against the cog reader at $BEFORE_REF =="
BEFORE=$(mktemp -d)
trap 'git -C "$ROOT" worktree remove --force "$BEFORE" >/dev/null 2>&1 || true' EXIT
git -C "$ROOT" worktree add --detach "$BEFORE" "$BEFORE_REF" >/dev/null
mkdir -p "$BEFORE/benchmarks/cog"
cp "$HERE/main.go" "$HERE/go.mod" "$BEFORE/benchmarks/cog/"
(cd "$BEFORE/benchmarks/cog" && go mod tidy >/dev/null 2>&1 && build "$OUT/cogbench-before")

echo "== timing ${WIDTH}x${HEIGHT} at ($XOFF, $YOFF), $REPEATS runs per case =="
MSYS_NO_PATHCONV=1 docker run --rm \
  --tmpfs "/work:size=$TMPFS,exec" \
  -v "$SRC_DIR:/in:ro" \
  -v "$(win "$OUT"):/out" \
  -v "$(win "$HERE")/runbench.sh:/runbench.sh:ro" \
  -v "$(win "$HERE")/gdalread.py:/gdalread.py:ro" \
  -v "$(win "$OUT")/cogbench:/cogbench:ro" \
  -v "$(win "$OUT")/cogbench-before:/cogbench-before:ro" \
  -e WORK=/work -e OUT=/out -e SRC="/in/$SRC_NAME" \
  -e XOFF="$XOFF" -e YOFF="$YOFF" -e W="$WIDTH" -e H="$HEIGHT" \
  -e REPEATS="$REPEATS" -e CACHE_REPEATS="$CACHE_REPEATS" -e TILE="$TILE" -e N="$N" \
  "$IMAGE" bash -c "tr -d '\r' </runbench.sh >/tmp/runbench.sh && bash /tmp/runbench.sh" |
  tee "$OUT/timings.txt"

echo
python "$HERE/summarize.py" "$OUT/timings.txt" "$WIDTH" "$HEIGHT" | tee "$OUT/summary.md"
