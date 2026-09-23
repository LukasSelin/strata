#!/usr/bin/env bash
# The timed loop, run inside the GDAL container by cogbench.sh. strata's
# cog reader and GDAL are timed here, by the same shell, on the same
# machine, over the same files, so that nothing but the reader differs.
#
# Everything happens in $WORK, a tmpfs, for the reason benchmarks/gdal
# gives: Docker Desktop's bind mount to NTFS costs more than either tool,
# and unevenly.
#
# One line per run goes to stdout:
#
#   tool=<t> op=<read|slope|none> codec=<c> threads=<n> tile=<rows>
#     cache=<bytes|-> run=<i> real=<s> user=<s> sys=<s> ms=<in-process|->
#     rpb=<block decodes per block|->
#
# real/user/sys are bash's time of the whole process; ms is the time the
# program measured itself, from its first read to its last, which leaves
# out process start and, for Python, importing GDAL and numpy.
set -uo pipefail

WORK=${WORK:-/work}
OUT=${OUT:-/out}
SRC=${SRC:?}
XOFF=${XOFF:-0} YOFF=${YOFF:-0}
W=${W:?} H=${H:?}
REPEATS=${REPEATS:-5}
CACHE_REPEATS=${CACHE_REPEATS:-3}
TILE=${TILE:-256}
N=${N:-12}
CODECS=${CODECS:-"deflate zstd lzw none"}
CACHE_TILES=${CACHE_TILES:-"16 64 256"}

mkdir -p "$WORK"
cd "$WORK"

echo "# gdal: $(gdalinfo --version)"
echo "# nproc: $(nproc)"
echo "# work: $(df -h "$WORK" | tail -1)"

# --- the files ----------------------------------------------------------
# The window as a float32 GeoTIFF, then as a raw file for strata's
# format-free path, then as a COG per compression. PREDICTOR=YES is the
# floating-point predictor (3) for float32 data; GDAL's COG driver uses
# no predictor for LZW unless asked, and that default is kept. No
# overviews: level 0 is what is read, and they would only take tmpfs.
echo "# extracting ${W}x${H} at ($XOFF, $YOFF) from $(basename "$SRC")" >&2
gdal_translate -q -ot Float32 -srcwin "$XOFF" "$YOFF" "$W" "$H" "$SRC" dem.tif
gdal_translate -q -of ENVI -ot Float32 dem.tif dem.raw
for c in $CODECS; do
  case $c in
    deflate) co="-co COMPRESS=DEFLATE -co PREDICTOR=YES" ;;
    zstd)    co="-co COMPRESS=ZSTD -co PREDICTOR=YES" ;;
    lzw)     co="-co COMPRESS=LZW" ;;
    none)    co="-co COMPRESS=NONE" ;;
  esac
  # shellcheck disable=SC2086 # $co is a list of options
  gdal_translate -q -of COG $co -co BLOCKSIZE=512 -co OVERVIEWS=NONE \
    -co NUM_THREADS=ALL_CPUS dem.tif "dem-$c.tif"
  echo "# dem-$c.tif: $(stat -c %s "dem-$c.tif") bytes," \
    "$(gdalinfo "dem-$c.tif" | grep -E '^ +(COMPRESSION|PREDICTOR|LAYOUT)=' | tr -d ' ' | paste -sd' ')," \
    "$(gdalinfo "dem-$c.tif" | grep -m1 'Block=' | sed 's/^ *//')"
done
gdalinfo dem.tif | grep -E 'Size is|Pixel Size|NoData' | sed 's/^/# /'

CELL=$(sed -n 's/.*map info[[:space:]]*=[[:space:]]*{[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p' dem.hdr | tr -d ' ')
FILL=$(sed -n 's/.*data ignore value[[:space:]]*=[[:space:]]*\([-0-9.eE+]*\).*/\1/p' dem.hdr | head -1)
: "${CELL:?no cell size in dem.hdr}" "${FILL:?no NoData in dem.hdr}"
echo "# cell: $CELL  nodata: $FILL  tile: $TILE  N: $N  repeats: $REPEATS"

cp /cogbench /cogbench-before /gdalread.py . && chmod +x cogbench cogbench-before
TIMEFORMAT='%R %U %S'

# time_run <reps> <label> <command...> — runs it once untimed, to warm
# the page cache and to check that it works at all, then <reps> times for
# the record. A failing command stops the run and shows its error.
time_run() {
  local reps=$1 label=$2; shift 2
  local i t ms rpb
  if ! "$@" >/tmp/bench-out 2>/tmp/bench-err; then
    echo "FAILED, so nothing is timed ($label):" >&2
    sed 's/^/  /' /tmp/bench-err /tmp/bench-out >&2
    exit 1
  fi
  for ((i = 1; i <= reps; i++)); do
    t=$( { time "$@" >/tmp/bench-out 2>&1; } 2>&1 | tail -1 )
    ms=$(sed -n 's/^run=0 ms=\([0-9.]*\).*/\1/p' /tmp/bench-out)
    rpb=$(sed -n 's/.*reads_per_block=\([0-9.]*\).*/\1/p' /tmp/bench-out)
    echo "$label run=$i real=$(cut -d' ' -f1 <<<"$t")" \
         "user=$(cut -d' ' -f2 <<<"$t") sys=$(cut -d' ' -f3 <<<"$t")" \
         "ms=${ms:--} rpb=${rpb:--}"
  done
}

# gdal_threads <n> — how many threads GDAL may decode blocks on. Its
# default is one; saying so explicitly keeps the 1-thread rows honest.
gdal_threads() { echo "GDAL_NUM_THREADS=$1"; }

# --- floors -------------------------------------------------------------
time_run "$REPEATS" "tool=startup-go op=none codec=- threads=1 tile=- cache=-" \
  ./cogbench -mode none
time_run "$REPEATS" "tool=startup-py op=none codec=- threads=1 tile=- cache=-" \
  python3 -c "from osgeo import gdal; import numpy; gdal.Open('dem-deflate.tif')"

# --- 1. pure read -------------------------------------------------------
for c in $CODECS; do
  for n in 1 "$N"; do
    time_run "$REPEATS" "tool=gdalpy op=read codec=$c threads=$n tile=$TILE cache=-" \
      env "$(gdal_threads "$n")" python3 gdalread.py "dem-$c.tif" "$TILE"
    time_run "$REPEATS" "tool=gdalmem op=read codec=$c threads=$n tile=- cache=-" \
      env "$(gdal_threads "$n")" gdal_translate -q -of MEM "dem-$c.tif" mem
    time_run "$REPEATS" "tool=before op=read codec=$c threads=$n tile=$TILE cache=0" \
      env GOMAXPROCS="$n" ./cogbench-before -mode read -file "dem-$c.tif" -tile "$TILE" -workers "$n"
    time_run "$REPEATS" "tool=strata op=read codec=$c threads=$n tile=$TILE cache=0" \
      env GOMAXPROCS="$n" ./cogbench -mode read -file "dem-$c.tif" -tile "$TILE" -workers "$n"
  done
done

# --- 2. end to end: slope -----------------------------------------------
# gdaldem computes on one thread whatever GDAL_NUM_THREADS says; the
# variable only lets it decode blocks in parallel, so it gets both.
for n in 1 "$N"; do
  time_run "$REPEATS" "tool=gdaldem op=slope codec=tif threads=$n tile=- cache=-" \
    env "$(gdal_threads "$n")" gdaldem slope dem.tif gdal-slope.tif -q
  time_run "$REPEATS" "tool=raw op=slope codec=raw threads=$n tile=$TILE cache=-" \
    env GOMAXPROCS="$n" ./cogbench -mode slope -src raw -file dem.raw -w "$W" -h "$H" \
      -fill "$FILL" -cell "$CELL" -tile "$TILE" -workers "$n" -dir "$WORK"
  for c in $CODECS; do
    time_run "$REPEATS" "tool=gdaldem op=slope codec=$c threads=$n tile=- cache=-" \
      env "$(gdal_threads "$n")" gdaldem slope "dem-$c.tif" gdal-slope.tif -q
    time_run "$REPEATS" "tool=before op=slope codec=$c threads=$n tile=$TILE cache=0" \
      env GOMAXPROCS="$n" ./cogbench-before -mode slope -file "dem-$c.tif" \
        -cell "$CELL" -tile "$TILE" -workers "$n" -dir "$WORK"
    time_run "$REPEATS" "tool=strata op=slope codec=$c threads=$n tile=$TILE cache=0" \
      env GOMAXPROCS="$n" ./cogbench -mode slope -file "dem-$c.tif" \
        -cell "$CELL" -tile "$TILE" -workers "$n" -dir "$WORK"
  done
done

# --- 3. the block cache under halos -------------------------------------
# Slope over the Deflate COG with short tiles, whose halos reach into
# the block rows above and below. rpb is block decodes per block.
for tile in $CACHE_TILES; do
  for cache in -1 $((4 << 20)) 0 $((1 << 30)); do
    for n in 1 "$N"; do
      time_run "$CACHE_REPEATS" "tool=cache op=slope codec=deflate threads=$n tile=$tile cache=$cache" \
        env GOMAXPROCS="$n" ./cogbench -mode slope -file dem-deflate.tif \
          -cell "$CELL" -tile "$tile" -workers "$n" -cache "$cache" -dir "$WORK"
    done
  done
done

# --- the timed outputs agree --------------------------------------------
# strata's slope through every COG must be the bytes its slope through
# the raw file is: the reader may not change a cell. Checked on the
# settings timed above, and on one that leans hard on the cache: 16-row
# tiles, a 4 MiB cache that evicts constantly, many workers.
./cogbench -mode slope -src raw -file dem.raw -w "$W" -h "$H" -fill "$FILL" \
  -cell "$CELL" -tile "$TILE" -dir "$WORK" >/dev/null
mv strata-slope.raw raw-slope.raw
for c in $CODECS; do
  for setting in "1 $TILE 0" "$N $TILE 0" "$N 16 $((4 << 20))"; do
    read -r n tile cache <<<"$setting"
    GOMAXPROCS=$n ./cogbench -mode slope -file "dem-$c.tif" -cell "$CELL" \
      -tile "$tile" -workers "$n" -cache "$cache" -dir "$WORK" >/dev/null
    if cmp -s strata-slope.raw raw-slope.raw; then
      echo "# identical: slope through dem-$c.tif, $n worker(s), tile $tile, cache $cache == slope through dem.raw"
    else
      echo "# DIFFERENT: slope through dem-$c.tif, $n worker(s), tile $tile, cache $cache != slope through dem.raw"
    fi
  done
done
# ... and that slope is gdaldem's, by acceptance/gdalcompare.py's slope
# criteria: the same cells carry data, and none differs by 1e-3 degrees.
gdaldem slope dem.raw gdal-slope.raw -of ENVI -q
python3 - "$W" "$H" <<'PY'
import sys
import numpy as np
w, h = int(sys.argv[1]), int(sys.argv[2])
g = np.fromfile("gdal-slope.raw", "<f4").reshape(h, w).astype(np.float64)
s = np.fromfile("raw-slope.raw", "<f4").reshape(h, w).astype(np.float64)
gh, sh = g != -9999, s != -9999
both = gh & sh
d = np.abs(g[both] - s[both])
ok = (gh == sh).all() and d.max() < 1e-3
print(f"# {'agrees' if ok else 'DISAGREES'} with gdaldem slope: {int((gh != sh).sum())} cells "
      f"differ in carrying data; max {d.max():.2e} deg over {int(both.sum()):,} cells")
PY

# --- profiles -----------------------------------------------------------
for c in $CODECS; do
  GOMAXPROCS=1 ./cogbench -mode read -file "dem-$c.tif" -tile "$TILE" -repeat 5 \
    -cpuprofile "$OUT/read-$c.prof" >/dev/null
  GOMAXPROCS=1 ./cogbench-before -mode read -file "dem-$c.tif" -tile "$TILE" -repeat 5 \
    -cpuprofile "$OUT/before-read-$c.prof" >/dev/null
done
GOMAXPROCS=1 ./cogbench -mode slope -file dem-deflate.tif -cell "$CELL" -tile "$TILE" \
  -repeat 3 -dir "$WORK" -cpuprofile "$OUT/slope-deflate.prof" >/dev/null
