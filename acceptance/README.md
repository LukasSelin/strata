# Black-box acceptance checks

Judging strata from the outside: does it compute the right numbers?

The test suite inside the repository is written by whoever wrote the
library, in the same language, against the same understanding of the
problem. If that understanding is wrong, the tests agree with the bug.
This directory is the independent opinion. It uses strata only through
its public API, writes every input and output to plain files, and then
has a **separate program, in a different language, written from
published definitions** decide whether the numbers are right.

It is its own Go module, so it stays out of the library's `go build`,
`go vet`, `go test` and lint runs.

## Running it

```bash
cd acceptance
go run .                        # strata produces results into out/
python check.py out             # numpy judges them
python check.py out --png       # ... and writes pictures to out/png
python sabotage.py out          # check the checker (see below)
python3 check_resample.py out   # resampling against its definitions (no numpy needed)
python3 check_resample.py out --sabotage   # ... which must fail on a half-cell shift
python3 gdalwarp_resample.py out           # resampling against gdalwarp (GDAL's Python bindings)
./gdalcheck.sh <some.tif>       # difference against gdaldem in Docker
python gdalsabotage.py out-gdal # ... and check that comparison's checker
./cogcheck.sh [some.tif]        # the GeoTIFF/COG reader against GDAL's reading, in Docker
python cogsabotage.py out-cog   # ... and check that comparison's checker
```

`check.py` needs numpy; with scipy installed it also cross-checks its
focal reference against `scipy.ndimage` (check 10), and `--png` needs
matplotlib. It prints one line
per check, exits non-zero if any failed, and on a failure prints only the
failures.

## What is checked

`main.go` builds three small DEMs — a tilted plane, a smooth hill on
rectangular cells, and a noisy surface with two NoData regions — and runs
every terrain, focal, algebra and reduce operation on each, in all three
forms (plain, `Tiled` with ragged 37×23 tiles on 3 workers, `Chunked`
through raw float32 files on 4 workers). The focal cases are Correlate
and Convolve with 5×5 weights asymmetric in both axes, CorrelateSeparable
with asymmetric taps and with Gaussian taps at radius 3, Mean at radius
2, and Min and Max at radii 1 and 3. 720 checks come out of that
(714 without scipy):

| # | Check | Why it would catch a defect |
| - | ----- | --------------------------- |
| 1 | Every result against Horn's gradient, or for curvature the Zevenbergen–Thorne derivatives, recomputed in float64 numpy; focal results against their definition as shifted sums, Min and Max exactly; ruggedness (TRI, Riley and Wilson; TPI; roughness) exactly, against gdaldem's own arithmetic in float32 numpy | A wrong kernel, a cell size used on the wrong axis, degrees for radians, a sign flip, profile and plan curvature mixed up, a flipped or transposed weight grid, Riley's TRI summed in float32 where gdaldem sums in float64 |
| 2 | The border carries no data: one cell for terrain, r cells for a radius-r focal operation | A stencil needs its whole neighbourhood; a border cell that holds a number is reading outside the raster |
| 3 | The plane against its analytic slope, aspect, zero curvature and closed-form ruggedness | The whole pipeline agrees with pen-and-paper on a surface whose answer is known exactly |
| 4 | plain == tiled == chunked, bit for bit | Tile seams, worker races, off-by-one tile origins — the README's central promise |
| 5 | Validity after a (2r+1)×(2r+1) erosion of the input mask, 3×3 for terrain | NoData leaking into a result, or valid cells wrongly discarded |
| 6 | Pointwise algebra against numpy in float32 | Exact equality is required here, so any drift shows |
| 7 | `Count` and `MinMax` against numpy over the valid cells | A reduction that misses a tile or double-counts one |
| 8 | Degrees, radians and percent agree with each other | A unit conversion applied twice, or not at all |
| 9 | `Normalize` against `(z - min) / (max - min)` in float32 numpy over the valid cells, with min and max landing on exactly 0 and 1 | A range taken over NoData, a rounding change such as multiplying by a reciprocal, an endpoint off by an ulp |
| 10 | The focal reference against `scipy.ndimage.correlate` and `convolve`, if scipy is installed | A reference that shares a misreading of the weight layout or the rotation with the library |

Resampling is judged separately, by `check_resample.py`: a float64
reference in plain Python (80×60 sources, so no numpy is needed) written
from the kernels' definitions and gdalwarp's measured conventions
(DESIGN.md §54), which requires validity to match exactly, values within
a tolerance derived per cell from the weighted sums (so renormalised
cells next to negative lobes get the looser bound their conditioning
earns), Nearest to match exactly, and plain, Tiled and Chunked to agree
bit for bit; and by `gdalwarp_resample.py`, which warps the same sources
with gdalwarp through GDAL's Python bindings and requires the same
validity and values within twice that tolerance, excluding only the
departures §54 records.

The reference implementations are derived in `check.py`'s docstring from
Horn's kernel as gdaldem documents it, from the definition of shaded
relief as the cosine between the light and the surface normal, for
curvature from Zevenbergen and Thorne's (1987) quadratic and
Florinsky's (2016) normal-section formulas, and from the textbook
definitions of correlation and convolution (the latter
scipy.ndimage's) — not from strata's code. `gdaldem` has no curvature
mode, so for curvature this numpy reference is the only outside opinion.
Ruggedness is the exception to "definitions, not code": strata documents
that it rounds exactly as gdaldem does, so the reference is the
arithmetic of gdaldem's source (`apps/gdaldem_lib.cpp`) transcribed into
numpy, and the check is equality, not a tolerance. The plane's closed
forms (check 3) and `gdalcheck.sh` judge the same results against the
definitions and against gdaldem itself.

## Tolerances

strata computes in float32 and the reference in float64, so they differ
by float32 rounding alone. Each tolerance is **derived, not tuned**:
Horn's numerator is six weighted elevations accumulated in at most six
roundings of a partial sum of magnitude ≤ 4·zmax, so the gradient is off
by at most `3 · 2⁻²⁴ · zmax / cellsize`, and every other tolerance
follows from how the operation propagates that. A focal weighted sum of
m = (2r+1)² products is bounded by the classic dot-product bound,
`(m + 1) · 2⁻²⁴ · Σ|w||z|` plus a rounding of the result. Each line
reports the observed error, the bound, and their ratio — currently 0.04×
to 0.25×, so the results sit comfortably inside a bound that is itself
tight. Focal Min and Max do no arithmetic and must match exactly.

Aspect gets a per-cell tolerance, because the direction of a nearly flat
cell is genuinely undefined: a float32 rounding in the gradient can swing
the bearing by degrees. Cells whose tolerance exceeds 5° are counted and
reported rather than judged.

Curvature gets per-cell tolerances too. Its own differences have their
own rounding bounds (p: one rounding of 2·zmax; r and t: two of 4·zmax;
s: three of 4·zmax), which are carried to first order through each
formula with its partial derivatives, plus about one float32 epsilon per
operation on the magnitude of the formula's terms. Profile and plan
curvature divide by the gradient, so like aspect they have cells too
flat to judge: where the error of p and q exceeds 1% of the gradient's
length, first order no longer holds, and those cells are counted rather
than judged (none, on the three DEMs here). Flat cells must hold 0.

## Checking the checker

A green result is worth nothing if the checks would stay green on a
broken library. `sabotage.py` copies the results, introduces one
plausible defect at a time, and confirms `check.py` goes red:

```
injected defect                   caught   failing checks
--------------------------------------------------------------
slope 0.05% too large             yes      12
dx and dy swapped                 yes      18
aspect mirrored                   yes      12
one bad cell on a tile seam       yes      2
chunked result off by one row     yes      3
one NoData cell leaking in        yes      3
count one too many                yes      1
max slightly wrong                yes      1
normalize by a reciprocal         yes      9
normalize range from NoData       yes      3
curvature sign flipped            yes      18
profile and plan swapped          yes      18
mean curvature 0.05% too large    yes      3
correlate with convolve's flip    yes      9
separable passes swapped          yes      9
focal radius one short            yes      15
mean divides by 24, not 25        yes      9
radius-2 validity eroded 3x3      yes      7
focal NoData leaking in           yes      3
focal tile seam                   yes      2
Riley and Wilson TRI swapped      yes      24
TPI sign flipped                  yes      9
roughness without the centre      yes      3
one TRI cell one ulp off          yes      3
TRI summed in float32             yes      9
```

One plausible focal defect is out of reach: a Mean that multiplies by a
rounded 1/(2r+1)² instead of dividing is off by about an ulp, well
inside the float64 reference's derived bound, where `Normalize`'s is
caught because it has an exact float32 reference. The unit tests in
`focal/` pin Mean's division bit for bit instead.

A 0.05% slope error — far smaller than any plausible real bug — is
caught 34× over its tolerance.

## Against GDAL, on a real raster

The strongest outside opinion available: a different program, by
different authors, in a different language, computing the same quantity
from the same bytes. `gdalcheck.sh` runs the GDAL suite in Docker, so
nothing has to be installed locally.

```bash
./gdalcheck.sh "C:/Users/you/Downloads/some.tif" 0 6127 4096
```

It extracts a window, promotes it to Float32, runs `gdaldem
slope/aspect/hillshade` and `gdaldem TRI` (Riley, and `-alg Wilson`),
`TPI` and `roughness`, runs the same operations through strata's
bounded-memory `Chunked` path ([gdal/main.go](gdal/main.go)), and
differences them ([gdalcompare.py](gdalcompare.py)). Cell size and
NoData come from GDAL's own header, not from a hard-coded guess.

Agreement is only meaningful once three documented encoding differences
are reconciled, and those reconciliations are the interesting part:

| | gdaldem | strata |
| - | ------- | ------ |
| NoData in the 3×3 window, or the border | writes -9999 | clears the validity bit |
| Flat cell in `aspect` | writes -9999 (its NoData) | writes -1, cell stays valid |
| `hillshade` | byte, `round(1 + 254·max(0,cos))`, 0 reserved | float, `255·max(0,cos)` |
| `aspect` on non-square cells | differences raw elevations | divides by each cell size |

Last run, on a 4096² window of a 12.5 m Swedish canopy-height raster
with 23% NoData (12.85M cells carrying data):

```
same cells carry data as gdaldem    0 cells differ
slope == gdaldem slope              max 7.63e-06 deg
aspect flat cells agree             0 disagree; 2,681,171 flat cells
aspect == gdaldem aspect            max 3.05e-05 deg
hillshade == gdaldem hillshade      12,849,447 of 12,849,874 exact, 427 off by one, 0 worse
within float32 rounding             worst 0.09× the bound, 76.21% bit-identical
no seam every 256 rows              1.222e-06 on chunk boundaries vs 1.203e-06 elsewhere
```

Check 5 asks how far apart two *correct* float32 implementations of
Horn's slope may land, and the answer is not a fixed number of ulps of
the slope. Horn's weighted sums cancel, so on high, nearly flat terrain
each tool's rounding legitimately moves the gradient by about
`2⁻²⁴·zmax / (|gradient|·cellsize)` relative — far more than an ulp of
the result. The bound is derived the same way as `check.py`'s
[tolerances](#tolerances): a gradient tolerance
`gtol = 3 · 2⁻²⁴ · zmax / cellsize`, with zmax the largest |z| in the
cell's own 3×3 window, propagated to slope (d atan(m)/dm ≤ 1), doubled
because both tools round, plus an ulp of each tool's result.

A fixed 2-ulp bound fails on exactly that terrain: the `noisy`
acceptance DEM (257×193, 30 m cells, elevations up to ~520 m, slopes
near 0.1°), written as a GeoTIFF and run with `STRATA_TILE=64
./gdalcheck.sh noisy.tif 0 0 193`, differed by up to 1664 ulps on
11,812 cells although gdaldem and strata were equally close to a
float64 Horn slope (mean errors 5.0e-06 and 4.5e-06°). Under the derived
bound it passes at 0.28×.

Check 5 also does not stop at "they differ": it recomputes the slope in
float64 and scores **both** tools against it. On the canopy run gdaldem
was closer on 17.4% of cells and strata on 6.4%, tied on 76.2%, with
mean errors of 1.40e-06 and 1.85e-06 degrees. So strata is very slightly
the less accurate of the two there, by an amount inside its own
documented float32 bounds and about a millionth of a degree — worth
knowing, not worth fixing. (On `noisy` strata was the closer one.)

A bound wide enough to admit flat, high terrain must still reject a real
defect. `gdalsabotage.py`, run with the same arguments after
`gdalcheck.sh`, scales strata's slope by a constant factor and confirms
check 5 goes red:

```
                 canopy 4096²          noisy 193²
slope × 1.0005   caught, 371× bound    caught, 709× bound
slope × 1.00002  caught, 15× bound     caught, 28× bound
```

On `noisy`, the 1.00002 drift passes the absolute `slope == gdaldem
slope` check (max 7.9e-04° < 1e-3°); only the derived bound catches it.

Ruggedness needs no reconciliation and gets no tolerance: strata's TRI,
TRI Wilson, TPI and roughness must match gdaldem's in every bit of every
cell that carries data. They did, with GDAL 3.14.0dev in the
`ubuntu-small-latest` image, for the scalar and the AVX2 build, on a
4096² window at (6000, 107000) of `Bok_andel.tif` (a 12.5 m Swedish
beech-share raster, Int16, NoData -1; 16,760,836 cells) and on `noisy`
(32,812 cells). `gdalsabotage.py` also moves one cell of each by a
single ulp and confirms the comparison fails.

`out-gdal/slope-difference.png` shows where the two disagree. It should
be featureless speckle following the terrain texture; horizontal bands
every `TileHeight` rows would mean a seam bug that a tolerance could
have hidden.

## Reading GeoTIFFs: the cog module against GDAL

The `cog` module (DESIGN.md §34, [ADR 0002](../docs/adr/0002-cog-adapter.md))
is strata's own GeoTIFF parser. Its unit tests build their files with a
test-only TIFF writer by the same author, so they would share any
misreading of the TIFF, GeoTIFF or libtiff predictor specifications.
`cogcheck.sh` takes the author out of it:

1. `cogmake.py` runs in the GDAL container. It writes 98 files from a
   synthetic 701×517 DEM with NoData holes (or from the first 1500² cells
   of a GeoTIFF you pass):
   - COGs in 128² blocks with overviews, for all eight sample types,
     every compression (none, LZW, Deflate, ZSTD, PackBits) and every
     predictor that applies (none, horizontal, floating point);
   - sparse blocks where the DEM is empty;
   - two BigTIFF COGs;
   - three-band stripped and tiled GeoTIFFs, pixel- and band-interleaved,
     some big-endian;
   - a file without NoData, a single-strip file, a PixelIsPoint file, and
     a float64 file with values near 1e300.

   For each band of each level, GDAL records what it reads:
   - the cells as float32, converted by GDAL (`ReadRaster` with a
     Float32 buffer);
   - its mask band;
   - the level's geotransform, the EPSG code and the NoData value.
2. `go run ./cog` reads the same files through `cog.Source.ReadWindow`,
   in 100×77 windows that line up with no block size.
3. `cogcompare.py` requires the two readings to be **identical**:
   - the same levels and sizes;
   - the geotransform to 1e-9 of a cell;
   - the same EPSG code;
   - `Masked` exactly when GDAL has NoData;
   - every validity bit;
   - the float32 bits of every valid cell.

   A reader has no rounding latitude, so there is no tolerance.

With GDAL 3.14 (`ghcr.io/osgeo/gdal:ubuntu-small-latest`), all 98 files
are identical to GDAL's reading: 58.5M cells, 9.6M of them NoData, in
about 35 s.

The first run failed one file. The reader held float64 values beyond
float32's range at ±MaxFloat32, but GDAL reads them as ±Inf. The unit
tests had encoded the same wrong belief and passed; that is the case for
this directory in one line.

`cogsabotage.py` copies one file's results at a time, plants a defect a
reader could plausibly have, and requires the comparison to fail:

```
caught  block one cell off      a block's cells shifted right by one
caught  one ulp                 one valid cell off by one float32 ulp
caught  one NoData cell valid   NoData compared after the cast, or missed
caught  overflow clamped        the defect above, reintroduced
caught  PixelIsPoint ignored    the origin half a cell off
caught  bands swapped           chunky samples read at the wrong stride
caught  overview dropped        a reduced-resolution image skipped
caught  sparse block valid      sparse blocks read as valid zeros
8/8 sabotages caught
```

What it does not cover:
- files GDAL did not write: other writers' quirks, such as old-style LZW
  or odd strip layouts;
- compressions the reader refuses (JPEG, WebP, LERC);
- reading over HTTP;
- decode speed, which nothing here measures yet.

## What this does not tell you

* **Speed.** `benchmarks/` and `benchmarks/cmd/stratademo` measure that.
  [`benchmarks/gdal/`](../benchmarks/gdal/) times the same three
  operations against `gdaldem` on the same raster this directory checks
  them against, and re-runs these checks on the files it timed.
* **Ground elevation.** The raster used above is canopy height, not a
  DEM. `gdaldem` does not care what Z means, so the numerical
  cross-check is valid either way, but the slopes are not terrain
  slopes.
* **The SIMD path**, unless you produce the results twice and diff them:

  ```bash
  go run . -dir out
  GOEXPERIMENT=simd go run . -dir out-simd
  diff -r out out-simd          # must be empty
  ```

  On an AVX2 machine (checked on a Ryzen 9 3900X) all 759 files,
  curvature and ruggedness included, are byte-identical, which is the
  SIMD claim in the top-level README tested from outside the library. On
  a CPU without AVX2 the vector build falls back to scalar and the
  comparison proves nothing.
* **Anything outside these operations.** Extending it means adding a
  case to `main.go` and a reference to `expected()` in `check.py`.
