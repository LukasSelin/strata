#!/usr/bin/env bash
# Run Herbie on strata's kernel formulas (kernels.fpcore) with strata's
# platform (strata.rkt: no fma, archsimd operations only), in Docker so
# nothing has to be installed locally.
#
#   ./run.sh [fpcore-file]
#
# Writes out/report/ (HTML; serve it, e.g. python3 -m http.server -d
# out/report, then open index.html) and out/improved.fpcore (the best
# rewrite of each input). Triage the results in RESULTS.md.
#
# The image is built from ./Dockerfile at a pinned Herbie tag, not pulled:
# uwplse/herbie is amd64-only and its Racket does not start under Rosetta
# on Apple silicon. The first build takes several minutes.
set -euo pipefail

HERBIE_TAG=${HERBIE_TAG:-v2.3}
IMAGE=strata-herbie:${HERBIE_TAG#v}
SEED=${HERBIE_SEED:-1}
THREADS=${HERBIE_THREADS:-yes}

HERE=$(cd "$(dirname "$0")" && pwd)
IN=${1:-kernels.fpcore}
OUT="$HERE/out"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "== building $IMAGE (herbie-fp/herbie@$HERBIE_TAG) =="
  docker build -t "$IMAGE" --build-arg HERBIE_TAG="$HERBIE_TAG" "$HERE"
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# Docker needs Windows-style paths and no MSYS path mangling.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
herbie() {
  MSYS_NO_PATHCONV=1 docker run --rm -v "$(win "$HERE"):/work" "$IMAGE" "$@"
}
OPTS=(--platform /work/strata.rkt --seed "$SEED" --threads "$THREADS" --timeout 600)

echo "== herbie report ($IMAGE, seed $SEED) =="
herbie report "${OPTS[@]}" "/work/$IN" /work/out/report

echo "== herbie improve =="
herbie improve "${OPTS[@]}" "/work/$IN" /work/out/improved.fpcore

echo
echo "wrote $OUT/report/ and $OUT/improved.fpcore"
