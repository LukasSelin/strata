#!/bin/sh
# Rewrite a few corpus files with libtiff's tiffcp, in layouts GDAL does
# not produce by default: 16-bit and float predictors from libtiff's own
# encoder, odd RowsPerStrip, planar configuration 2 in strips and tiles,
# big-endian, BigTIFF, and codecs the reader refuses. Runs in a Debian
# container (generate.sh starts it); /in is the corpus, /out the output.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq libtiff-tools >/dev/null 2>&1
IN=/in
OUT=/out/generated-tiffcp
mkdir -p "$OUT"
tiffinfo 2>&1 | head -1 >"$OUT/VERSION.txt" || true
dpkg-query -W libtiff-tools >>"$OUT/VERSION.txt"

g16=$IN/libtiff-test/minisblack-1c-16b.tiff
rgb8=$IN/libtiff-test/rgb-3c-8b.tiff
rgb16=$IN/libtiff-test/rgb-3c-16b.tiff
# tiffcp converts to separate planes only at 8 bits, so 16-bit planar
# output starts from a file that is planar already.
planar16=$IN/libtiff-pics/depth/flower-rgb-planar-16.tif
f32=$IN/gdal-autotest/gcore/data/float32.tif
f64=$IN/gdal-autotest/gcore/data/float64.tif
i32=$IN/gdal-autotest/gcore/data/int32.tif

tc() { tiffcp "$@"; }
tc -c lzw:2 "$g16" "$OUT/gray16-lzw-pred2.tif"
tc -c zip:2 -B "$g16" "$OUT/gray16-deflate-pred2-be.tif"
tc -c zip:2 -r 3 "$rgb16" "$OUT/rgb16-deflate-pred2-rps3.tif"
tc -c lzw -r 5 -p separate "$rgb8" "$OUT/rgb8-lzw-planar2-rps5.tif"
tc -c zip:2 -t -w 16 -l 32 "$planar16" "$OUT/rgb16-deflate-pred2-planar2-tiled.tif"
tc -c lzw:2 -B -t -w 32 -l 16 "$rgb8" "$OUT/rgb8-lzw-pred2-be-tiled.tif"
tc -c packbits -r 1 "$rgb8" "$OUT/rgb8-packbits-rps1.tif"
tc -c none -8 -r 7 "$g16" "$OUT/gray16-none-bigtiff-rps7.tif"
tc -c zip:3 "$f32" "$OUT/float32-deflate-pred3.tif"
tc -c zip:3 -B "$f64" "$OUT/float64-deflate-pred3-be.tif"
tc -c lzw:2 -B "$f64" "$OUT/float64-lzw-pred2-be.tif"
tc -c lzw:2 -B "$planar16" "$OUT/rgb16-lzw-pred2-be-planar2.tif"
tc -c zstd:2 "$i32" "$OUT/int32-zstd-pred2.tif" || true
tc -c zstd -8 -B -t -w 16 -l 16 "$f32" "$OUT/float32-zstd-bigtiff-be-tiled.tif" || true
# FillOrder 2: libtiff reverses the bits of every byte before decoding.
tc -f lsb2msb -c none "$g16" "$OUT/gray16-fillorder2.tif"
tc -f lsb2msb -c lzw "$rgb8" "$OUT/rgb8-lzw-fillorder2.tif"
# Codecs cog refuses: the refusal must name them.
tc -c lzma "$g16" "$OUT/gray16-lzma.tif" || true
tc -c jpeg "$rgb8" "$OUT/rgb8-jpeg.tif" || true
tc -c webp "$rgb8" "$OUT/rgb8-webp.tif" || true
ls "$OUT"
