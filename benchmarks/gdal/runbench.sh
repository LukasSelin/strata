#!/usr/bin/env bash
# The timed loop, run inside the GDAL container by gdalbench.sh. Both
# tools are timed here, by the same shell, on the same machine, over the
# same bytes, so that nothing but the tool differs.
#
# Everything happens in $WORK, which gdalbench.sh mounts as a tmpfs.
# Docker Desktop's bind mount to NTFS costs more than either tool's
# computation, and it costs it unevenly; a memory filesystem takes the
# host's disk out of a measurement that is about the two tools. The
# gdal_translate copy cases below are the floor it leaves behind.
#
# One line per run goes to stdout:
#
#   tool=<gdaldem|strata|...> op=<op> fmt=<envi|tif> workers=<n> run=<i>
#     real=<s> user=<s> sys=<s>
set -uo pipefail

WORK=${WORK:-/work}
OUT=${OUT:-/out}
SRC=${SRC:?}
XOFF=${XOFF:-0} YOFF=${YOFF:-0}
W=${W:?} H=${H:?}
REPEATS=${REPEATS:-5}
TILE=${TILE:-256}
WORKERS=${WORKERS:-"1 12 24"}
OPS=${OPS:-"slope aspect hillshade"}
CACHEMAX=${CACHEMAX:-4096}

mkdir -p "$WORK"
cd "$WORK"

echo "# gdal: $(gdalinfo --version)"
echo "# nproc: $(nproc)"
echo "# work: $(df -h "$WORK" | tail -1)"

# --- the window -------------------------------------------------------
echo "# extracting ${W}x${H} at ($XOFF, $YOFF) from $(basename "$SRC")" >&2
gdal_translate -q -ot Float32 -srcwin "$XOFF" "$YOFF" "$W" "$H" "$SRC" dem.tif
gdal_translate -q -of ENVI -ot Float32 dem.tif dem.raw
gdalinfo dem.tif | grep -E 'Size is|Pixel Size|NoData' | sed 's/^/# /'

CELL=$(sed -n 's/.*map info[[:space:]]*=[[:space:]]*{[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p' dem.hdr | tr -d ' ')
FILL=$(sed -n 's/.*data ignore value[[:space:]]*=[[:space:]]*\([-0-9.eE+]*\).*/\1/p' dem.hdr | head -1)
: "${CELL:?no cell size in dem.hdr}" "${FILL:?no NoData in dem.hdr}"
echo "# cell: $CELL  nodata: $FILL  tile: $TILE  repeats: $REPEATS"

cp /strataterrain . && chmod +x strataterrain
TIMEFORMAT='%R %U %S'

# time_run <label> <command...> — runs it once untimed, to warm the
# cache and to check that it works at all, then REPEATS times for the
# record. A benchmark that reports how long a crash took is worse than
# no benchmark, so a failing command stops the run and shows its error.
time_run() {
  local label=$1; shift
  local i t
  if ! "$@" >/dev/null 2>/tmp/bench-err; then
    echo "FAILED, so nothing is timed ($label):" >&2
    sed 's/^/  /' /tmp/bench-err >&2
    exit 1
  fi
  for ((i = 1; i <= REPEATS; i++)); do
    t=$( { time "$@" >/dev/null 2>&1; } 2>&1 | tail -1 )
    echo "$label run=$i real=$(cut -d' ' -f1 <<<"$t")" \
         "user=$(cut -d' ' -f2 <<<"$t") sys=$(cut -d' ' -f3 <<<"$t")"
  done
}

# --- the floors: what each tool costs before it computes anything -----
time_run "tool=startup op=none fmt=none workers=1" gdalinfo --version
time_run "tool=startup op=none fmt=go workers=1"   ./strataterrain -op none
time_run "tool=copy op=none fmt=envi workers=1" \
  gdal_translate -q -of ENVI -ot Float32 dem.raw copy.raw
time_run "tool=copy op=none fmt=tif workers=1" \
  gdal_translate -q -of GTiff -ot Float32 dem.tif copy.tif
rm -f copy.raw copy.tif copy.hdr

for op in $OPS; do
  # gdaldem on raw float32, the file shape strata reads and writes.
  time_run "tool=gdaldem op=$op fmt=envi workers=1" \
    gdaldem "$op" dem.raw "gdal-$op.raw" -of ENVI -q
  # gdaldem on GeoTIFF, its own default and its home format.
  time_run "tool=gdaldem op=$op fmt=tif workers=1" \
    gdaldem "$op" dem.tif "gdal-$op.tif" -q
  # ... and with a block cache big enough for the whole raster, so the
  # comparison is against GDAL configured well rather than by default.
  time_run "tool=gdaldemcache op=$op fmt=tif workers=1" \
    gdaldem "$op" dem.tif "gdal-$op-cached.tif" -q --config GDAL_CACHEMAX "$CACHEMAX"

  # strata, raw float32 in and out, by worker count. One worker runs
  # with GOMAXPROCS=1, so that it is as single-threaded as gdaldem is.
  for w in $WORKERS; do
    procs=$w
    time_run "tool=strata op=$op fmt=envi workers=$w" \
      env GOMAXPROCS=$procs ./strataterrain -dir "$WORK" -op "$op" \
        -w "$W" -h "$H" -cell "$CELL" -fill "$FILL" -tile "$TILE" -workers "$w"
  done

  # strata on its scalar kernels, which is what a build without
  # GOEXPERIMENT=simd gets. gdaldem is scalar C++, so this is the
  # closer like-for-like, and it separates what the lanes are worth
  # from what the rest of the design is worth.
  time_run "tool=stratascalar op=$op fmt=envi workers=1" \
    env GOMAXPROCS=1 ./strataterrain -dir "$WORK" -op "$op" -scalar \
      -w "$W" -h "$H" -cell "$CELL" -fill "$FILL" -tile "$TILE" -workers 1

  # strata with the whole raster in memory: the kernel without the file.
  time_run "tool=stratamem op=$op fmt=none workers=1" \
    env GOMAXPROCS=1 ./strataterrain -dir "$WORK" -op "$op" -mode memory \
      -w "$W" -h "$H" -cell "$CELL" -fill "$FILL" -workers 1
done

# --- what strata measured from the inside, for the same runs ----------
# The wall times above are whole processes, so they cannot say how much
# of strata's time is the kernel and how much is moving bytes. These
# runs can: the chunked ones time the streamed operation without the
# process around it, and the memory ones time the kernel alone, over a
# raster already in memory.
for op in $OPS; do
  for w in $WORKERS; do
    echo "# strata in-process, $op, chunked, $w worker(s):"
    GOMAXPROCS=$w ./strataterrain -dir "$WORK" -op "$op" -w "$W" -h "$H" \
      -cell "$CELL" -fill "$FILL" -tile "$TILE" -workers "$w" \
      -repeat 3 | sed 's/^/#   /'
  done
  for kernels in simd scalar; do
    echo "# strata in-process, $op, kernel only, 1 worker, $kernels:"
    GOMAXPROCS=1 ./strataterrain -dir "$WORK" -op "$op" -mode memory \
      -w "$W" -h "$H" -cell "$CELL" -fill "$FILL" -workers 1 \
      $([[ $kernels == scalar ]] && echo -scalar) \
      -repeat 3 | sed 's/^/#   /'
  done
done

# --- leave a set of results behind so they can be checked -------------
echo "# copying results to $OUT for the correctness check" >&2
for op in $OPS; do
  gdaldem "$op" dem.raw "gdal-$op.raw" -of ENVI -q
  GOMAXPROCS=1 ./strataterrain -dir "$WORK" -op "$op" -w "$W" -h "$H" \
    -cell "$CELL" -fill "$FILL" -tile "$TILE" -workers 1 >/dev/null
done
cp dem.raw dem.hdr "$OUT"/ 2>/dev/null || true
for op in $OPS; do cp "gdal-$op.raw" "strata-$op.raw" "$OUT"/ 2>/dev/null || true; done
echo "# cell=$CELL fill=$FILL" > "$OUT/window.txt"
