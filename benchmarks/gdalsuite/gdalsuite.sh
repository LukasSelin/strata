#!/usr/bin/env bash
# The full suite against GDAL: every strata operation with a GDAL
# counterpart, timed against it three ways (compute only, raw file to
# file, and the whole flow from a COG), with the two tools' answers
# compared on every operation.
#
#   ./gdalsuite.sh <geotiff> [xoff yoff width height]
#
# Both tools run inside the same GDAL container, on the same machine,
# over the same bytes, under the same timer (runsuite.sh). strata is
# cross-compiled twice to static linux/amd64 binaries and mounted in:
# with GOEXPERIMENT=simd, and without it, which is what an ordinary
# `go build` gets. Nothing has to be installed locally but Docker, Go
# and Python.
#
# Environment:
#   REPEATS=5            timed runs per case, after one warm-up run
#   TILE=256             strata's tile height
#   N=12                 the many-threads setting, for both tools
#   OPS="slope ..."      which operations (default: all; stratasuite -list)
#   TIERS="compute raw cog"  which tiers
#   BASELINE=<timings>   an earlier run's timings.txt, for a "since" column
#   TMPFS=20g            size of the in-container working filesystem
#   WORKVOL=<volume>     work in this Docker volume instead of the tmpfs:
#                        a real disk (ext4 in Docker Desktop's VM), which
#                        tmpfs flatters; runsuite.sh then syncs before
#                        every run, so no run pays for an earlier one's
#                        writeback
#   GDAL_IMAGE=...       the container image
set -euo pipefail

SRC=${1:?usage: gdalsuite.sh <geotiff> [xoff yoff width height]}
XOFF=${2:-0}
YOFF=${3:-0}
WIDTH=${4:-11264}
HEIGHT=${5:-$WIDTH}

REPEATS=${REPEATS:-5}
TILE=${TILE:-256}
N=${N:-12}
OPS=${OPS:-}
TIERS=${TIERS:-"compute raw cog"}
TMPFS=${TMPFS:-20g}
WORKVOL=${WORKVOL:-}
WORKMOUNT=(--tmpfs "/work:size=$TMPFS,exec")
[[ -n $WORKVOL ]] && WORKMOUNT=(-v "$WORKVOL:/work" -e SYNC=1)
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
SRC_NAME=$(basename "$SRC")

echo "== building stratasuite for linux/amd64, with and without GOEXPERIMENT=simd =="
(cd "$HERE" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOEXPERIMENT=simd go build -o "$OUT/stratasuite" .)
(cd "$HERE" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$OUT/stratasuite-scalar" .)
COMMIT=$(git -C "$HERE" rev-parse --short HEAD)
DIRTY=$(git -C "$HERE" status --porcelain -- ../.. | grep -qv '^??' && echo "+changes" || true)

echo "== timing ${WIDTH}x${HEIGHT} at ($XOFF, $YOFF), $REPEATS runs per case =="
{
  echo "# strata: $COMMIT$DIRTY"
  echo "# go: $(go version | cut -d' ' -f3)"
  echo "# date: $(date -u +%Y-%m-%dT%H:%MZ)"
  MSYS_NO_PATHCONV=1 docker run --rm \
    "${WORKMOUNT[@]}" \
    -v "$SRC_DIR:/in:ro" \
    -v "$(win "$OUT"):/out" \
    -v "$(win "$HERE")/runsuite.sh:/runsuite.sh:ro" \
    -v "$(win "$HERE")/gdalcompute.py:/gdalcompute.py:ro" \
    -v "$(win "$HERE")/agree.py:/agree.py:ro" \
    -v "$(win "$OUT")/stratasuite:/stratasuite:ro" \
    -v "$(win "$OUT")/stratasuite-scalar:/stratasuite-scalar:ro" \
    -e WORK=/work -e OUT=/out -e SRC="/in/$SRC_NAME" \
    -e XOFF="$XOFF" -e YOFF="$YOFF" -e W="$WIDTH" -e H="$HEIGHT" \
    -e REPEATS="$REPEATS" -e TILE="$TILE" -e N="$N" -e OPS="$OPS" -e TIERS="$TIERS" \
    "$IMAGE" bash -c "tr -d '\r' </runsuite.sh >/tmp/runsuite.sh && bash /tmp/runsuite.sh"
} | tee "$OUT/timings.txt"

echo
python "$HERE/summarize.py" "$OUT/timings.txt" ${BASELINE:+--baseline "$BASELINE"} | tee "$OUT/summary.md"
