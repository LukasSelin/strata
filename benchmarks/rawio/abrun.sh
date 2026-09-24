#!/usr/bin/env bash
# The timed loop of ab.sh: interleaved runs of stratasuite builds over the
# same inputs, one line per run. Run inside the GDAL container (ab.sh) or
# natively on Windows from a directory holding the inputs and binaries
# (DIRS=. BINDIR=. EXE=.exe).
#
#   dir=<d> op=<op> src=<raw|cog> n=<workers> v=<variant> round=<r>
#     ms=<in-process ms> open=<ms> run=<ms> close=<ms> mapped=<bool>
#
# Rounds go over every case in turn, so that noise and warm-up spread
# over the variants alike. On a disk (not tmpfs) every run is preceded by
# an untimed sync, so no run pays for an earlier run's writeback.
#
# Environment:
#   DIRS="/work /disk"   where the inputs, and the outputs, are
#   INPUTS=/disk/prep    where to copy dem.raw and dem-cog.tif from first
#   VARIANTS="old new"   binaries <BINDIR>/suite-<variant><EXE>; or
#                        name:binary:flag,flag... runs suite-<binary> with
#                        extra flags, which win over the defaults above
#                        (e.g. reuse64:new:-reuse,-tile=64)
#   OPS="slope" SRCS="raw cog" NS="1 12" ROUNDS=7
#   GMP=2                GOMAXPROCS for every case (default: the worker count)
#   NOSYNC=1             never sync (Windows, whose lazy writer needs none)
set -uo pipefail
DIRS=${DIRS:-/work} INPUTS=${INPUTS:-} BINDIR=${BINDIR:-/p} EXE=${EXE:-}
VARIANTS=${VARIANTS:-"old new"} OPS=${OPS:-slope} SRCS=${SRCS:-"raw cog"}
NS=${NS:-"1 12"} ROUNDS=${ROUNDS:-7}
for d in $DIRS; do
  [[ -n $INPUTS && ! -f $d/dem-cog.tif ]] && cp "$INPUTS/dem.raw" "$INPUTS/dem-cog.tif" "$d/"
done
echo "# $(date -u +%FT%TZ) $(nproc 2>/dev/null || echo "$NUMBER_OF_PROCESSORS") cpus"
for d in $DIRS; do echo "# $d: $(df -hT "$d" 2>/dev/null | tail -1)"; done
for r in $(seq 1 "$ROUNDS"); do
  for d in $DIRS; do for op in $OPS; do for s in $SRCS; do for n in $NS; do for spec in $VARIANTS; do
    IFS=: read -r v bin flags <<<"$spec"
    extra=()
    [[ -n $flags ]] && IFS=, read -r -a extra <<<"$flags"
    f=$d/dem.raw; [[ $s == cog ]] && f=$d/dem-cog.tif
    [[ -z ${NOSYNC:-} && $(stat -f -c %T "$d" 2>/dev/null) != tmpfs ]] && sync
    out=$(GOMAXPROCS=${GMP:-$n} "$BINDIR/suite-${bin:-$v}$EXE" -mode chunked -op "$op" -src "$s" -file "$f" \
      -w 11264 -h 11264 -fill 65535 -cell 12.5 -tile 256 -workers "$n" -dir "$d" "${extra[@]}" 2>&1)
    ms=$(sed -n 's/^run=0 ms=\([0-9.]*\).*/\1/p' <<<"$out")
    ph=$(sed -n 's/^phases open ms=\([0-9.]*\) run ms=\([0-9.]*\) close ms=\([0-9.]*\) mapped=\(.*\)/open=\1 run=\2 close=\3 mapped=\4/p' <<<"$out")
    echo "dir=$d op=$op src=$s n=$n v=$v round=$r ms=${ms:-FAILED} ${ph:-open=- run=- close=- mapped=-}"
  done; done; done; done; done
done
