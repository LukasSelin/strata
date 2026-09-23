#!/usr/bin/env bash
# The timed loop, run inside the GDAL container by gdalsuite.sh. Every
# operation is timed three ways, strata against GDAL each time:
#
#   compute  the input already in memory, only the call timed, in
#            process: stratasuite -mode memory against gdalcompute.py
#            (GDAL's algorithm on a MEM dataset). What the arithmetic
#            and its loops cost.
#   raw      file to file, the whole process under bash's `time`, from
#            a float32 file: strata's format-free raw path against the
#            same GDAL command line on ENVI and on GeoTIFF.
#   cog      the whole flow a user runs: from a Deflate COG (predictor
#            3) to a file, the whole process timed. strata reads it with
#            its cog module.
#
# Both tools run here, by the same shell, on the same machine, over the
# same bytes, so that nothing but the tool differs. Everything happens
# in $WORK, a tmpfs, for the reason benchmarks/gdal gives: Docker
# Desktop's bind mount to NTFS costs more than either tool, unevenly.
#
# One line per timed run goes to stdout:
#
#   tier=<t> op=<op> tool=<tool> cfg=<config> threads=<n> run=<i>
#     real=<s|-> user=<s|-> sys=<s|-> ms=<in-process ms|-> [result]
#
# and one `agree ...` line per operation (agree.py) saying how far the
# two tools' results are apart. Lines starting with # are notes.
set -uo pipefail

WORK=${WORK:-/work}
OUT=${OUT:-/out}
SRC=${SRC:?}
XOFF=${XOFF:-0} YOFF=${YOFF:-0}
W=${W:?} H=${H:?}
REPEATS=${REPEATS:-5}
TILE=${TILE:-256}
N=${N:-12}
TIERS=${TIERS:-"compute raw cog"}

mkdir -p "$WORK"
cd "$WORK" || exit 1
cp /stratasuite /stratasuite-scalar /gdalcompute.py /agree.py . && chmod +x stratasuite stratasuite-scalar
OPS=${OPS:-$(./stratasuite -list | tr '\n' ' ')}
KERNEL=$(./stratasuite -kernel)

echo "# gdal: $(gdalinfo --version)"
echo "# nproc: $(nproc)"
echo "# work: $(df -h "$WORK" | tail -1)"
echo "# ops: $OPS"
echo "# tiers: $TIERS  repeats: $REPEATS  tile: $TILE  N: $N"

# --- the files ----------------------------------------------------------
# The window as float32 GeoTIFF and ENVI; a second window for the two-input
# algebra, offset so that its cells differ; and the window as the COG a
# user would have, Deflate with the floating-point predictor, 512 blocks.
echo "# extracting ${W}x${H} at ($XOFF, $YOFF) from $(basename "$SRC")" >&2
gdal_translate -q -ot Float32 -srcwin "$XOFF" "$YOFF" "$W" "$H" "$SRC" dem.tif
gdal_translate -q -ot Float32 -srcwin 0 0 "$W" "$H" "$SRC" dem2.tif
gdal_translate -q -of ENVI dem.tif dem.raw
gdal_translate -q -of ENVI dem2.tif dem2.raw
gdal_translate -q -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES -co BLOCKSIZE=512 \
  -co OVERVIEWS=NONE -co NUM_THREADS=ALL_CPUS dem.tif dem-cog.tif
gdal_translate -q -of COG -co COMPRESS=DEFLATE -co PREDICTOR=YES -co BLOCKSIZE=512 \
  -co OVERVIEWS=NONE -co NUM_THREADS=ALL_CPUS dem2.tif dem2-cog.tif
gdalinfo dem.tif | grep -E 'Size is|Pixel Size|NoData' | sed 's/^/# /'
echo "# dem-cog.tif: $(stat -c %s dem-cog.tif) bytes; stored statistics: $(gdalinfo dem-cog.tif | grep -c STATISTICS_)"

CELL=$(sed -n 's/.*map info[[:space:]]*=[[:space:]]*{[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\([^,]*\),.*/\1/p' dem.hdr | tr -d ' ')
FILL=$(sed -n 's/.*data ignore value[[:space:]]*=[[:space:]]*\([-0-9.eE+]*\).*/\1/p' dem.hdr | head -1)
: "${CELL:?no cell size in dem.hdr}" "${FILL:?no NoData in dem.hdr}"
echo "# cell: $CELL  nodata: $FILL"

TIMEFORMAT='%R %U %S'
export GDAL_CACHEMAX=4096 # GDAL configured well, not by default
export GDAL_PAM_ENABLED=NO # no .aux.xml: statistics are computed every run

# emit <label> <file> — prints one line per `run=` line of a program's
# output, prefixed with the label.
emit() {
  sed -n "s/^run=\([0-9]*\) ms=\([0-9.]*\)\(.*\)/$1 run=\1 real=- user=- sys=- ms=\2\3/p" "$2"
}

# in_process <label> <command...> — a compute case: the program runs its
# own warm-up and timed runs and reports them.
in_process() {
  local label=$1; shift
  if ! "$@" >/tmp/bench-out 2>/tmp/bench-err; then
    echo "FAILED, so nothing is timed ($label):" >&2
    sed 's/^/  /' /tmp/bench-err /tmp/bench-out >&2
    exit 1
  fi
  emit "$label" /tmp/bench-out
}

# whole_process <label> <command...> — a file-to-file case: one untimed
# run, to warm the page cache and to check that it works at all, then
# REPEATS timed ones of the whole process. A failing command stops the
# run: a benchmark that reports how long a crash took is worse than none.
whole_process() {
  local label=$1; shift
  local i t ms
  if ! "$@" >/tmp/bench-out 2>/tmp/bench-err; then
    echo "FAILED, so nothing is timed ($label):" >&2
    sed 's/^/  /' /tmp/bench-err /tmp/bench-out >&2
    exit 1
  fi
  for ((i = 1; i <= REPEATS; i++)); do
    t=$( { time "$@" >/tmp/bench-out 2>&1; } 2>&1 | tail -1 )
    ms=$(sed -n 's/^run=0 ms=\([0-9.]*\).*/\1/p' /tmp/bench-out)
    echo "$label run=$i real=$(cut -d' ' -f1 <<<"$t")" \
         "user=$(cut -d' ' -f2 <<<"$t") sys=$(cut -d' ' -f3 <<<"$t") ms=${ms:--}"
  done
}

# --- what each operation is, to each tool --------------------------------
family() {
  case $1 in
    slope|aspect|hillshade|tri|tpi|roughness) echo terrain ;;
    mean3|mean11|min3|max3|gauss5|conv5) echo focal ;;
    add|mul|min|max|clamp) echo algebra ;;
    stats|minmax) echo reduce ;;
    *-half|*-double) echo resample ;;
  esac
}
inputs() { case $1 in add|mul|min|max) echo 2 ;; *) echo 1 ;; esac; }

# Cells of the border that the agreement check leaves out: where the two
# tools define edges differently (a partial window, a kernel that runs
# off the raster). Interior cells are compared in full.
border() {
  case $1 in
    mean11) echo 5 ;; gauss5|conv5) echo 2 ;;
    add|mul|min|max|clamp) echo 0 ;;
    *-half|*-double) echo 8 ;;
    *) echo 1 ;;
  esac
}

# Whether GDAL can put more than one thread on the operation itself, not
# only on decoding blocks: gdalwarp -multi, and the statistics.
gdal_mt() { case $(family "$1") in resample|reduce) return 0 ;; *) return 1 ;; esac; }

scale_of() { case $1 in *-half) echo 0.5 ;; *-double) echo 2 ;; *) echo 1 ;; esac; }
dim() { awk -v n="$1" -v s="$(scale_of "$2")" 'BEGIN { printf "%d", n * s + 0.5 }'; }

# gdal_cmd <op> <in> <in2> <out> <format> <threads> — sets GCMD to GDAL's
# command line for the op, as a user would type it.
gdal_cmd() {
  local op=$1 in=$2 in2=$3 out=$4 fmt=$5 n=$6
  case $op in
    slope|aspect|hillshade) GCMD=(gdaldem "$op" "$in" "$out" -of "$fmt" -q) ;;
    tri) GCMD=(gdaldem TRI "$in" "$out" -of "$fmt" -q) ;;
    tpi) GCMD=(gdaldem TPI "$in" "$out" -of "$fmt" -q) ;;
    roughness) GCMD=(gdaldem roughness "$in" "$out" -of "$fmt" -q) ;;
    mean3|mean11|min3|max3|gauss5|conv5)
      local k=equal s=3 m=mean
      case $op in
        mean11) s=11 ;; min3) m=min ;; max3) m=max ;;
        gauss5) k=gaussian s=5 m=sum ;; conv5) k=$KERNEL s=5 m=sum ;;
      esac
      GCMD=(gdal raster neighbors -i "$in" -o "$out" -f "$fmt" --ot Float32
        --kernel "$k" --size "$s" --method "$m" --overwrite -q) ;;
    add|mul|min|max|clamp)
      local e
      case $op in
        add) e='A+B' ;; mul) e='A*B' ;; min) e='min(A,B)' ;; max) e='max(A,B)' ;;
        clamp) e='min(max(A,50),200)' ;;
      esac
      GCMD=(gdal raster calc -i "A=$in")
      [[ $op == clamp ]] || GCMD+=(-i "B=$in2" --no-check-extent)
      GCMD+=(-o "$out" -f "$fmt" --calc "$e" --ot Float32 --propagate-nodata
        --nodata -9999 --overwrite -q) ;;
    stats) GCMD=(gdalinfo -stats "$in") ;;
    minmax) GCMD=(gdalinfo -mm "$in") ;;
    *-half|*-double)
      GCMD=(gdalwarp -q -overwrite -r "${op%-*}" -ts "$(dim "$W" "$op")" "$(dim "$H" "$op")"
        -et 0 -ot Float32 -dstnodata -9999 -wm 2048 -of "$fmt")
      [[ $n -gt 1 ]] && GCMD+=(-multi -wo "NUM_THREADS=$n")
      GCMD+=("$in" "$out") ;;
  esac
}

# gdal_calc_cmd <op> <in> <in2> <out> <format> — sets GCMD to the
# numpy-backed gdal_calc.py, GDAL's other raster calculator. Algebra only.
gdal_calc_cmd() {
  local op=$1 in=$2 in2=$3 out=$4 fmt=$5 e
  case $op in
    add) e='A+B' ;; mul) e='A*B' ;; min) e='numpy.minimum(A,B)' ;; max) e='numpy.maximum(A,B)' ;;
    clamp) e='numpy.clip(A,50,200)' ;;
  esac
  GCMD=(gdal_calc.py -A "$in")
  [[ $op == clamp ]] || GCMD+=(-B "$in2")
  GCMD+=(--outfile "$out" --calc "$e" --type Float32 --NoDataValue -9999
    --format "$fmt" --overwrite --quiet)
}

# strata_args <op> <src kind> <file> <file2> — sets SARGS to stratasuite's
# arguments for the op.
strata_args() {
  SARGS=(-op "$1" -src "$2" -file "$3" -file2 "$4" -w "$W" -h "$H" -fill "$FILL"
    -cell "$CELL" -tile "$TILE" -dir "$WORK")
}

has_tier() { [[ " $TIERS " == *" $1 "* ]]; }

# --- floors: what each tool costs before it computes anything ------------
whole_process "tier=floor op=none tool=gdal cfg=startup threads=1" gdalinfo --version
whole_process "tier=floor op=none tool=strata cfg=startup threads=1" ./stratasuite -mode none
whole_process "tier=floor op=none tool=gdal cfg=copy-envi threads=1" \
  gdal_translate -q -of ENVI dem.raw copy.raw
whole_process "tier=floor op=none tool=gdal cfg=copy-gtiff threads=1" \
  gdal_translate -q -of GTiff dem.tif copy.tif
for n in 1 "$N"; do
  whole_process "tier=floor op=none tool=gdal cfg=copy-cog threads=$n" \
    env GDAL_NUM_THREADS="$n" gdal_translate -q -of GTiff dem-cog.tif copy.tif
done
rm -f copy.*
has_tier compute && in_process "tier=floor op=none tool=gdal cfg=mem-copy threads=1" \
  python3 gdalcompute.py copy dem.tif dem2.tif "$REPEATS" 1

# --- the operations -------------------------------------------------------
for op in $OPS; do
  echo "# --- $op ($(family "$op"))" >&2
  two=$(inputs "$op")
  r2=$([[ $two == 2 ]] && echo dem2.raw || echo -)
  t2=$([[ $two == 2 ]] && echo dem2.tif || echo -)
  c2=$([[ $two == 2 ]] && echo dem2-cog.tif || echo -)

  if has_tier compute; then
    strata_args "$op" raw dem.raw "$r2"
    for n in 1 "$N"; do
      if [[ $n == 1 ]] || gdal_mt "$op"; then
        in_process "tier=compute op=$op tool=gdal cfg=mem threads=$n" \
          python3 gdalcompute.py "$op" dem.tif dem2.tif "$REPEATS" "$n" "$KERNEL"
      fi
      in_process "tier=compute op=$op tool=strata cfg=simd threads=$n" \
        env GOMAXPROCS="$n" ./stratasuite -mode memory -repeat "$REPEATS" -workers "$n" "${SARGS[@]}"
    done
    in_process "tier=compute op=$op tool=strata cfg=scalar threads=1" \
      env GOMAXPROCS=1 ./stratasuite-scalar -mode memory -repeat "$REPEATS" -workers 1 "${SARGS[@]}"
  fi

  if has_tier raw; then
    strata_args "$op" raw dem.raw "$r2"
    for n in 1 "$N"; do
      if [[ $n == 1 ]] || gdal_mt "$op"; then
        gdal_cmd "$op" dem.raw "$r2" "gdal-$op.raw" ENVI "$n"
        whole_process "tier=raw op=$op tool=gdal cfg=envi threads=$n" \
          env GDAL_NUM_THREADS="$n" "${GCMD[@]}"
        gdal_cmd "$op" dem.tif "$t2" "gdal-$op.tif" GTiff "$n"
        whole_process "tier=raw op=$op tool=gdal cfg=gtiff threads=$n" \
          env GDAL_NUM_THREADS="$n" "${GCMD[@]}"
      fi
      whole_process "tier=raw op=$op tool=strata cfg=simd threads=$n" \
        env GOMAXPROCS="$n" ./stratasuite -mode chunked -workers "$n" "${SARGS[@]}"
    done
    if [[ $(family "$op") == algebra ]]; then
      gdal_calc_cmd "$op" dem.raw "$r2" "gdal-calc-$op.raw" ENVI
      whole_process "tier=raw op=$op tool=gdal cfg=gdal_calc-envi threads=1" "${GCMD[@]}"
      gdal_calc_cmd "$op" dem.tif "$t2" "gdal-calc-$op.tif" GTiff
      whole_process "tier=raw op=$op tool=gdal cfg=gdal_calc-gtiff threads=1" "${GCMD[@]}"
    fi
    whole_process "tier=raw op=$op tool=strata cfg=scalar threads=1" \
      env GOMAXPROCS=1 ./stratasuite-scalar -mode chunked -workers 1 "${SARGS[@]}"
  fi

  if has_tier cog; then
    strata_args "$op" cog dem-cog.tif "$c2"
    for n in 1 "$N"; do
      # GDAL decodes blocks on GDAL_NUM_THREADS threads whatever the
      # operation, so every op gets the threaded row here.
      gdal_cmd "$op" dem-cog.tif "$c2" "gdal-cog-$op.tif" GTiff "$n"
      whole_process "tier=cog op=$op tool=gdal cfg=to-gtiff threads=$n" \
        env GDAL_NUM_THREADS="$n" "${GCMD[@]}"
      gdal_cmd "$op" dem-cog.tif "$c2" "gdal-cog-$op.raw" ENVI "$n"
      whole_process "tier=cog op=$op tool=gdal cfg=to-envi threads=$n" \
        env GDAL_NUM_THREADS="$n" "${GCMD[@]}"
      whole_process "tier=cog op=$op tool=strata cfg=simd threads=$n" \
        env GOMAXPROCS="$n" ./stratasuite -mode chunked -workers "$n" "${SARGS[@]}"
    done
  fi

  # --- do the two tools agree? -------------------------------------------
  # strata's one-worker output from the raw file against GDAL's ENVI
  # output from the same file. And the whole flow's output must be the
  # raw path's, bit for bit: reading a COG may not change a cell.
  strata_args "$op" raw dem.raw "$r2"
  if [[ $(family "$op") == reduce ]]; then
    s=$(GOMAXPROCS=1 ./stratasuite -mode memory -repeat 1 "${SARGS[@]}" | sed -n 's/^run=0 ms=[0-9.]* //p')
    g=$(python3 gdalcompute.py "$op" dem.tif dem2.tif 1 1 | sed -n 's/^run=0 ms=[0-9.]* //p')
    python3 agree.py reduce "$op" "$s" "$g"
  else
    GOMAXPROCS=1 ./stratasuite -mode chunked -workers 1 "${SARGS[@]}" >/dev/null
    gdal_cmd "$op" dem.raw "$r2" "gdal-$op.raw" ENVI 1
    "${GCMD[@]}" >/dev/null 2>&1
    python3 agree.py raster "$op" "strata-$op.raw" "gdal-$op.raw" \
      "$(dim "$W" "$op")" "$(dim "$H" "$op")" "$(border "$op")"
    if has_tier cog; then
      mv "strata-$op.raw" "strata-$op-fromraw.raw"
      strata_args "$op" cog dem-cog.tif "$c2"
      GOMAXPROCS="$N" ./stratasuite -mode chunked -workers "$N" "${SARGS[@]}" >/dev/null
      if cmp -s "strata-$op.raw" "strata-$op-fromraw.raw"; then
        echo "# identical: $op from the COG on $N workers == $op from the raw file on 1"
      else
        echo "# DIFFERENT: $op from the COG on $N workers != $op from the raw file on 1"
      fi
    fi
  fi
  rm -f gdal-* strata-*
done
