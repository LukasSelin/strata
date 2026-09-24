#!/usr/bin/env bash
# The write probe's matrix (probe/): every mode and preallocation, 1 and
# 12 writers, a new file each run and an overwritten one, in each of DIRS.
#   ./probe.sh <probe binary> <dir>...
set -u
P=$1; shift
for d in "$@"; do
  echo "# $d: $(df -hT "$d" 2>/dev/null | tail -1)"
  for n in 1 12; do
    for cfg in "pwrite none" "pwrite fallocate" "mmap truncate" "mmap fallocate"; do
      set -- $cfg
      "$P" -f "$d/probe.raw" -mode "$1" -pre "$2" -n "$n" -r 5 | sed "s|^|dir=$d |"
    done
    "$P" -f "$d/probe.raw" -mode pwrite -n "$n" -r 5 -keep | tail -4 | sed "s|^|dir=$d |"
    "$P" -f "$d/probe.raw" -mode mmap -pre truncate -n "$n" -r 5 -keep | tail -4 | sed "s|^|dir=$d |"
  done
  rm -f "$d/probe.raw"
done
