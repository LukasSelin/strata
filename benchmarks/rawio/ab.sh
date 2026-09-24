#!/usr/bin/env bash
# Interleaved A/B of output writing: stratasuite (benchmarks/gdalsuite)
# built from an earlier commit and from the working tree, over the same
# 11264² window of a GeoTIFF as gdalsuite.sh uses, in the GDAL container,
# on a tmpfs and on a Docker volume (ext4 on a real disk), raw file and
# COG in, raw file out. See RESULTS.md.
#
#   ./ab.sh <geotiff> [base-commit]      (default base: origin/master)
#
# Environment: ROUNDS, OPS, SRCS, NS (see abrun.sh); VOLUME=strata-disk.
set -euo pipefail
SRC=${1:?usage: ab.sh <geotiff> [base-commit]}
BASE=${2:-origin/master}
VOLUME=${VOLUME:-strata-disk}
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out"
mkdir -p "$OUT"
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

echo "== building stratasuite at $BASE (old) and the working tree (new) =="
TREE=$(mktemp -d)
git -C "$HERE" worktree add --detach "$TREE/base" "$BASE" >/dev/null
trap 'git -C "$HERE" worktree remove --force "$TREE/base"' EXIT
build() { (cd "$1/benchmarks/gdalsuite" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOEXPERIMENT=simd go build -o "$2" .); }
build "$TREE/base" "$OUT/suite-old"
build "$HERE/../.." "$OUT/suite-new"

SRC_DIR=$(win "$(cd "$(dirname "$SRC")" && pwd)")
docker volume create "$VOLUME" >/dev/null
MSYS_NO_PATHCONV=1 docker run --rm -v "$VOLUME:/disk" -v "$SRC_DIR:/in:ro" "$IMAGE" bash -c "
  set -e; mkdir -p /disk/prep; cd /disk/prep
  [ -f dem-cog.tif ] && exit 0
  gdal_translate -q -ot Float32 -srcwin 1024 320 11264 11264 /in/$(basename "$SRC") dem.tif
  gdal_translate -q -of ENVI dem.tif dem.raw
  gdal_translate -q -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES -co BLOCKSIZE=512 \
    -co OVERVIEWS=NONE -co NUM_THREADS=ALL_CPUS dem.tif dem-cog.tif
  rm dem.tif"

echo "== timing =="
MSYS_NO_PATHCONV=1 docker run --rm --tmpfs /work:size=6g,exec -v "$VOLUME:/disk" \
  -v "$(win "$OUT"):/p" -v "$(win "$HERE")/abrun.sh:/abrun.sh:ro" \
  -e DIRS="/work /disk/ab" -e INPUTS=/disk/prep -e ROUNDS="${ROUNDS:-7}" \
  -e GMP="${GMP:-}" -e OPS="${OPS:-slope}" -e SRCS="${SRCS:-raw cog}" -e NS="${NS:-1 12}" \
  "$IMAGE" bash -c "mkdir -p /disk/ab && tr -d '\r' </abrun.sh >/tmp/abrun.sh && bash /tmp/abrun.sh" \
  | tee "$OUT/timings.txt"
python "$HERE/summarize.py" "$OUT/timings.txt" | tee "$OUT/summary.md"
