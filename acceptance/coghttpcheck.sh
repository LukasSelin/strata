#!/usr/bin/env bash
# Judge strata's GeoTIFF/COG reader over HTTP: the same 98 files and the
# same comparison as cogcheck.sh, but strata reads every file through
# cog.HTTPReaderAt, by range requests to nginx, instead of from disk.
#
#   ./coghttpcheck.sh [geotiff]
#
# It runs cogcheck.sh first if out-cog/ holds no files yet (the argument
# is passed on to it). Then it deletes strata's local reading, serves
# out-cog/ with nginx in a container, has strata read every file from it
# with 8 goroutines per source, and has GDAL judge the result with
# cogcompare.py. cogblocks.py then sets the HTTP requests each file took
# against the blocks GDAL says it holds. Needs Docker and Go.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out-cog"
if [ ! -f "$OUT/manifest.json" ]; then
  "$HERE/cogcheck.sh" "$@"
  echo
fi

win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
IMAGE=${GDAL_IMAGE:-ghcr.io/osgeo/gdal:ubuntu-small-latest}
NGINX=${NGINX_IMAGE:-nginx:alpine}

# strata's local reading must not be what gets compared.
rm -f "$OUT/strata.json" "$OUT"/*.strata.f32 "$OUT"/*.strata.mask

echo "== nginx ($NGINX) serves out-cog =="
CID=$(MSYS_NO_PATHCONV=1 docker run -d --rm -p 127.0.0.1::80 \
  -v "$(win "$OUT"):/usr/share/nginx/html:ro" "$NGINX")
trap 'docker stop "$CID" >/dev/null' EXIT
URL="http://$(docker port "$CID" 80/tcp | head -n1)"
for _ in $(seq 50); do
  curl -sf -o /dev/null "$URL/manifest.json" && break
  sleep 0.2
done
echo "$URL"

echo
echo "== strata reads them through cog.HTTPReaderAt =="
(cd "$HERE" && go run ./cog -dir out-cog -url "$URL" -workers 8)

echo
echo "== difference =="
MSYS_NO_PATHCONV=1 docker run --rm -v "$(win "$HERE"):/acc:ro" -v "$(win "$OUT"):/out" "$IMAGE" \
  sh -c 'python3 /acc/cogcompare.py /out && echo && python3 /acc/cogblocks.py /out'
