#!/usr/bin/env bash
# Judge strata's GeoTIFF reader (the cog module) against GDAL on GeoTIFFs
# that other software wrote: cogcheck.sh's comparison, over a directory
# of files instead of a matrix GDAL generates.
#
#   ./cogcorpus.sh [dir]        # default: corpus/files
#
# Build the default corpus first (see corpus/README.md):
#   python3 corpus/fetch.py && corpus/generate.sh
#
# GDAL records its reading of every *.tif and *.tiff under dir
# (cogcorpus.py, into out-corpus/), strata reads the same files through
# cog (cog/main.go), and cogcompare.py judges. A file cog refuses with a
# "not supported" error is refused by design; one GDAL cannot read either
# must be refused by cog too. Anything else that differs is a finding,
# and the script exits non-zero. Needs Docker and Go.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "${1:-$HERE/corpus/files}" && pwd)
OUT="$HERE/out-corpus"
rm -rf "$OUT"
mkdir -p "$OUT"

win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}
MOUNTS=(-v "$(win "$HERE"):/acc:ro" -v "$(win "$OUT"):/out" -v "$(win "$SRC"):/in:ro")

echo "== GDAL ($IMAGE) reads the corpus =="
MSYS_NO_PATHCONV=1 docker run --rm "${MOUNTS[@]}" "$IMAGE" \
  python3 /acc/cogcorpus.py /in /out

echo
echo "== strata reads it through cog =="
(cd "$HERE" && go run ./cog -dir out-corpus -src "$SRC")

echo
echo "== difference =="
MSYS_NO_PATHCONV=1 docker run --rm "${MOUNTS[@]}" "$IMAGE" \
  python3 /acc/cogcompare.py /out
