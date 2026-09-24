#!/usr/bin/env bash
# Runs inside the GRASS GIS container for grasscheck.sh: extracts the
# window with GDAL, imports it into a temporary GRASS project that takes
# its CRS from the file, and runs r.param.scale for each window size and
# each parameter compared, exporting the results as raw float64.
#
#   grassrun.sh <source name> <xoff> <yoff> <size> "<window sizes>"
set -euo pipefail
SRC=$1 XOFF=$2 YOFF=$3 SIZE=$4 SIZES=$5

# GRASS 8.5.0's G_ludcmp (lib/gmath/lu.c) searches for its pivot in an
# OpenMP loop that races on the shared best pivot and its row, so with
# more than one thread r.param.scale sometimes factorises its normal
# equations with a wrong pivot and writes a wrong map (OSGeo/grass#7539,
# fixed after 8.5.0). On this machine 2 of 12 identical 24-thread runs of
# method=slope size=9 did. One thread makes the factorisation the serial
# algorithm the source describes.
export OMP_NUM_THREADS=1

cd /out
rm -f dem.tif dem.raw dem.hdr grass-*.raw grass-*.hdr *.aux.xml
gdal_translate -q -ot Float32 -srcwin "$XOFF" "$YOFF" "$SIZE" "$SIZE" "/in/$SRC" dem.tif
gdal_translate -q -of ENVI -ot Float32 dem.tif dem.raw
gdalinfo dem.tif | grep -E 'Pixel Size|NoData'

# --tmp-project creates a throwaway project whose CRS is the file's.
# r.param.scale takes region.ns_res as the ground distance between cell
# centres, so the project must be projected: in a lat/long one that would
# be degrees. r.in.gdal turns the file's NoData into GRASS NULL.
grass --tmp-project dem.tif --exec bash -c '
set -euo pipefail
eval "$(g.proj -g | grep -E "^(name|proj|units)=" | sed "s/=\(.*\)/=\"\1\"/")"
echo "GRASS project: ${name:-?}, proj=${proj:-?}, units=${units:-?}"
if [ "${proj:-}" = ll ] || [ "${proj:-}" = longlat ]; then
  echo "grassrun.sh: a lat/long project would make r.param.scale fit in degrees" >&2
  exit 1
fi
r.in.gdal --quiet input=dem.tif output=dem
g.region raster=dem
g.region -g | grep -E "^(nsres|ewres|rows|cols)="
r.info -g dem | grep -E "^datatype="
for size in '"$SIZES"'; do
  for m in slope aspect profc planc crosc minic maxic; do
    # exponent=0: unweighted, the fit strata computes. zscale=1: no
    # vertical scaling. The two tolerances only enter method=feature
    # (feature.c); they are set to 0 so that nothing else can depend on
    # them.
    r.param.scale --quiet input=dem output="${m}_$size" size="$size" method="$m" \
      exponent=0 zscale=1 slope_tolerance=0 curvature_tolerance=0
    r.out.gdal --quiet -c -f input="${m}_$size" output="grass-$m-$size.raw" \
      format=ENVI type=Float64 nodata=-9999
  done
  echo "size $size done"
done
r.info -g slope_3 | grep -E "^datatype="
'
grass --version | head -1
