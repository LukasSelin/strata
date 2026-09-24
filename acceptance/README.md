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
python check_array.py out       # N-dimensional arrays against numpy and exact integers
python check_array.py out --sabotage   # ... which must catch every injected defect
python3 check_resample.py out   # resampling against its definitions (no numpy needed)
python3 check_resample.py out --sabotage   # ... which must fail on a half-cell shift
python3 gdalwarp_resample.py out           # resampling against gdalwarp (GDAL's Python bindings)
python3 gdalwarp_mosaic.py out             # mosaics against gdalwarp given the same sources
python3 gdalwarp_mosaic.py out --sabotage  # ... which must fail with the sources in the wrong order
./gdalcheck.sh <some.tif>       # difference against gdaldem in Docker
python gdalsabotage.py out-gdal # ... and check that comparison's checker
./cogcheck.sh [some.tif]        # the GeoTIFF/COG reader against GDAL's reading, in Docker
python cogsabotage.py out-cog   # ... and check that comparison's checker
./coghttpcheck.sh               # the same files read over HTTP range requests, from nginx
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
2, and Min and Max at radii 1 and 3. `WeightedSlope` runs on the noisy
DEM times a weight raster with NoData of its own, and `Surface` writes all
five of its products at once on each DEM. Each ruggedness measure also
runs at radius 3 and 8, and `Features` writes a stack of slope, plan
curvature, TPI at radius 1, 3 and 8, TRI at 3 and roughness at 8 in one
call, each output emitted as its own case. Slope, aspect, hillshade and
the three curvatures also run with the quadratic fit (`FitRadius` 1 and
4). `HeatLoad` runs four ways on each DEM: heat load and direct radiation
from Equation 1 at 45°N, Equation 3 heat load at 33.5°S, and Equation 2
radiation on the arithmetic scale. It also runs fitted at radius 1 and
4, and in the `Features` stack. It runs as well on small planes at the
points of McCune and Keon's test spreadsheet. 2085 checks come out of
that (2079 without scipy):

| # | Check | Why it would catch a defect |
| - | ----- | --------------------------- |
| 1 | Every result against Horn's gradient, or for curvature the Zevenbergen–Thorne derivatives, recomputed in float64 numpy; heat load against McCune and Keon's equation in angles on that gradient; fitted results against Wood's quadratic, solved by least squares with numpy's pseudo-inverse of the full design matrix; focal results against their definition as shifted sums, Min and Max exactly; ruggedness (TRI, Riley and Wilson; TPI; roughness) exactly, against gdaldem's own arithmetic in float32 numpy | A wrong kernel, a cell size used on the wrong axis, degrees for radians, a sign flip, profile and plan curvature mixed up, a flipped or transposed weight grid, Riley's TRI summed in float32 where gdaldem sums in float64 |
| 2 | The border carries no data: one cell for terrain, r cells for a radius-r focal operation | A stencil needs its whole neighbourhood; a border cell that holds a number is reading outside the raster |
| 3 | The plane against its analytic slope, aspect, zero curvature, closed-form ruggedness and heat load at its exact gradient | The whole pipeline agrees with pen-and-paper on a surface whose answer is known exactly |
| 4 | plain == tiled == chunked, bit for bit | Tile seams, worker races, off-by-one tile origins — the README's central promise |
| 5 | Validity after a (2r+1)×(2r+1) erosion of the input mask, 3×3 for terrain | NoData leaking into a result, or valid cells wrongly discarded |
| 6 | Pointwise algebra against numpy in float32 | Exact equality is required here, so any drift shows |
| 7 | `Count` and `MinMax` against numpy over the valid cells | A reduction that misses a tile or double-counts one |
| 8 | Degrees, radians and percent agree with each other | A unit conversion applied twice, or not at all |
| 9 | `Normalize` against `(z - min) / (max - min)` in float32 numpy over the valid cells, with min and max landing on exactly 0 and 1 | A range taken over NoData, a rounding change such as multiplying by a reciprocal, an endpoint off by an ulp |
| 10 | The focal reference against `scipy.ndimage.correlate` and `convolve`, if scipy is installed | A reference that shares a misreading of the weight layout or the rotation with the library |
| 12 | Every `Surface` product (dx, dy, slope, aspect, hillshade, all written in one call) is the standalone function's file bit for bit, Data and validity, in every form | A from-gradient kernel that rounds once differently from the fused one, products wired to the wrong output. The standalone files are judged by checks 1–5, so their verdicts carry over |
| 13 | Every `Features` output (3×3 derivatives next to ruggedness at radius 1, 3 and 8, all in one call) is the standalone function's file bit for bit, Data and validity, in every form | A fused pass that gives every output the largest window's border or erosion, an output wired to the wrong operation. The standalone files are judged by checks 1–5 and 14 |
| 14 | Ruggedness over a larger window against its definition in float64, taken with numpy's `sliding_window_view` rather than check 1's shifted sums, within derived bounds; roughness exactly | A misreading of the window that check 1's transcription could share: the centre counted, the wrong n, a shifted window |
| 11 | `WeightedSlope` against the float64 Horn slope times the weight, and its validity against the DEM's mask eroded 3×3 AND the weight's mask not eroded; and that the case tells the two readings apart | A weight's NoData wiping out its neighbours (one erosion over every input, the engine's rule before per-input reach, DESIGN.md §52), a weight's NoData ignored, a weight read from the wrong cell |
| 15 | Heat load on planes at the six latitude, slope and aspect points of McCune and Keon's own test spreadsheet (`testrad.xls`, published with the paper), transcribed rather than computed: all three equations, radiation and heat load, north of the equator and mirrored south of it, within a derived bound | A misread coefficient (the paper itself misprints one), heat load folded about the wrong axis, ln and arithmetic scales mixed up, the southern-hemisphere rule not applied. This compares against the authors' numbers, not only against a reading of their equation |

N-dimensional arrays are judged by `check_array.py` (DESIGN.md §10).
`array.go` builds a masked [time, y, x] = [5, 23, 300] float32 stack. Its
time steps span 13 orders of magnitude, and it holds a NaN, both
infinities, whole NoData series and a series of zeros with one −0. An
int16 stack sits near the type's limits. The two go through broadcasting
`Sub`, `Mul`, `Max` and `Min` (a [T,1,1] weight, a [Y,X] layer, a
transposed view against an expanded column, a slice into a window), int16
`Add` that wraps, `Convert`, and `Sum`, `Mean`, `Min`, `Max` and `Count`
over time, x, space, everything, and a sliced and transposed view: 34
cases. Every check is exact. numpy does the broadcasting and the IEEE
arithmetic. Sums and means are Python integers in units of 2⁻¹⁴⁹, divided
once, which Python rounds correctly, so no floating-point sum is involved.
Go's `min(−0, +0) = −0` is set explicitly, because numpy does not promise
it. `--sabotage` injects 11 plausible defects: a mean from the rounded sum,
a float64 running sum, a min that reads NoData, weights broadcast along
the wrong axis, a leaked NoData bit, one ulp, int16 saturation, +0 for a
−0 minimum, and a transposed view read as compact. Each one must fail. A
double-rounded mean and a float64 running sum are only visible where the
data can tell them apart, over time, so those mutants run on the
time-axis cases.

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

Mosaics are judged by `gdalwarp_mosaic.py`. It warps four overlapping
sources, at 10 m, 20 m, 4 m and 10 m south-up with NoData holes and gaps
between them, onto one 10 m grid with gdalwarp, in order. For each
method it checks:

- (a) gdalwarp's own rule: all four sources at once equal each source
  alone overlaid last-valid-wins, bit for bit. This is GDAL judged
  against itself, the evidence for the rule strata implements;
- (b) strata against gdalwarp: validity exactly, values within twice the
  tolerance of the source that wins the cell;
- (c) strata against `check_resample.py`'s reference overlaid the same
  way;
- (d) plain == tiled == chunked, bit for bit.

`--sabotage` puts the first valid source on top instead, and every
order-dependent check must fail. The Python bindings are not on the
desktop; the GDAL images are:

```bash
docker run --rm -v "$PWD:/w" -w /w ghcr.io/osgeo/gdal:ubuntu-small-3.13.0 python3 gdalwarp_mosaic.py out
```

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
numpy, and the check is equality, not a tolerance. The larger windows
are strata's own generalisation (the eight neighbours become the window's
other n cells, "/ 8" becomes "/ n"), so gdaldem has nothing to say about
them: check 1 transcribes that documented arithmetic the same way, and
check 14 judges it against the float64 definition by a separate route.
Heat load's reference is McCune and Keon's (2002) Table 2 in its
published form, in angles: aspect by `atan2`, folded by their formulas
(and McCune's 2004 southern-hemisphere supplement), then cos and sin. It
does not use strata's rewrite of it as a function of the gradient. The
tolerance is the gradient's error times a constant C, the sum of the
coefficients' magnitudes. Every term is linear in the unit surface
normal, which moves by at most as much as the gradient does. Check 15
then compares against the authors' own evaluated numbers.
The quadratic fit's reference does not use strata's closed forms for the
coefficients either: it solves the least-squares problem for every
window with the pseudo-inverse of the six-column design matrix, and
bounds the error by the documented separable float32 sums. The plane's closed
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
slope 0.05% too large             yes      33
weight validity eroded 3x3        yes      3
weight NoData ignored             yes      4
weight read one cell over         yes      3
Surface aspect one ulp off        yes      2
Surface dx and dy swapped         yes      6
dx and dy swapped                 yes      36
aspect mirrored                   yes      36
one bad cell on a tile seam       yes      3
chunked result off by one row     yes      5
one NoData cell leaking in        yes      5
count one too many                yes      1
max slightly wrong                yes      1
normalize by a reciprocal         yes      9
normalize range from NoData       yes      3
curvature sign flipped            yes      54
profile and plan swapped          yes      27
mean curvature 0.05% too large    yes      9
correlate with convolve's flip    yes      9
separable passes swapped          yes      9
focal radius one short            yes      15
mean divides by 24, not 25        yes      9
radius-2 validity eroded 3x3      yes      7
focal NoData leaking in           yes      3
focal tile seam                   yes      2
Riley and Wilson TRI swapped      yes      24
TPI sign flipped                  yes      18
roughness without the centre      yes      3
one TRI cell one ulp off          yes      3
TRI summed in float32             yes      9
r=3 TPI counts its centre         yes      24
r=8 roughness one ring short      yes      30
Features erodes by the largest r  yes      6
Features output one ulp off       yes      2
r=1 fit slope is Horn's           yes      6
r=4 fit curvature over 3x3        yes      12
r=4 fit slope 0.01% too large     yes      15
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
`TPI` and `roughness`, and `gdaldem slope` times a weight raster
through `gdal_calc.py` (a weight with NoData of its own, where the
elevation's tenths are 3 mod 7), runs the same operations
through strata's
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
weighted slope: same cells          0 cells differ; 11,644,877 carry data
weighted slope: data iff both       0 cells differ; 1,204,997 lose their weight and nothing else
weighted slope: gdal_calc is x      gdaldem slope × weight, rounded once, over 11,644,877 cells
weighted slope: float32 rounding    worst 0.11× the bound, 78.10% bit-identical
surface (gdal/main.go)              slope, aspect, hillshade in one call, each identical to its own run
no seam every 256 rows              1.222e-06 on chunk boundaries vs 1.203e-06 elsewhere
```

The weighted slope is the check on DESIGN.md §52's per-input reach:
strata computes it as one pipeline whose DEM is read over a 3×3 and whose
weight is read at the cell, and GDAL computes it as two programs. The
cells each keeps are the same set, so a NoData weight removes its own
cell and not the eight around it. The values differ by the slope's own
float32 bound scaled by the weight.

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
   - `Masked` exactly when GDAL's mask is not all-valid;
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
- files GDAL did not write, which "Reading GeoTIFFs other software
  wrote" below takes on;
- compressions the reader refuses (JPEG, WebP, LERC);
- decode speed, which [`benchmarks/cog/`](../benchmarks/cog/RESULTS.md)
  measures against GDAL.

### Over HTTP: coghttpcheck.sh

`cog.HTTPReaderAt` is an `io.ReaderAt` over HTTP range requests, so the
reader above runs unchanged on a COG in S3, GCS or behind any HTTPS URL.
Its unit tests (`cog/http_test.go`) use Go's own file server and inject
faults: 200 instead of 206, another range than the one asked for,
truncated bodies, dropped connections, 429 and 5xx, 412 and 416 from a
replaced file, and cancellation while waiting for a response, a retry or
a request slot. `coghttpcheck.sh` checks it from outside, against a
server strata did not write:

1. It runs `cogcheck.sh` first if `out-cog/` is empty, then deletes
   strata's local reading, so that reading cannot be what gets compared.
2. nginx (`nginx:alpine`, in Docker) serves `out-cog/` on a loopback
   port.
3. `go run ./cog -url <nginx> -workers 8` reads every file through
   `cog.NewHTTPReaderAt`, with eight goroutines reading windows of each
   source at once, and records each file's requests and bytes.
4. `cogcompare.py` judges the result exactly as before.
5. `cogblocks.py` counts, through GDAL's `BLOCK_OFFSET_x_y` and
   `BLOCK_SIZE_x_y` metadata, the blocks each file stores, the block
   reads strata's per-band sources make, and how many stored blocks end
   past the 64 KiB prefetch. It prints them next to the requests, and
   fails if any file was not read over HTTP.

With GDAL 3.14 and nginx 1.31, in about 25 s, all 98 files are identical to GDAL's
reading, as they are from disk. The request counts:

```
file                                              KiB blocks  reads fetches requests other fetched
cog-Float32-DEFLATE-FLOATING_POINT                958     39     39      34       35     0    100%
cog-Byte-ZSTD-STANDARD                            129     39     39      20       21     0    100%
gtiff-Float32-tiles-BAND-be                      3624    561    561     554      555     0    100%
gtiff-Float32-tiles-PIXEL-be                     3310    187    561     185      186     0    100%
...
98/98 files read over HTTP. The 97 counted: 8,344 requests for 8,789 blocks (11,399 block reads,
8,247 blocks past the prefetch) = 97 prefetches + 8,247 fetches + 0 other.
165.7 MiB fetched of 165.7 MiB
```

On every counted file, requests = 1 prefetch + 1 per stored block that
ends past the prefetch, and nothing else. Open never needed a request
beyond the prefetch, not even for the stripped GeoTIFFs. The caches meant
no block was fetched twice, although eight goroutines read 100×77
windows that cut every block and, in a pixel-interleaved file, three
band sources read each block: `reads` is 3 × `blocks` on those rows,
and `fetched` is still 100%, because the File keeps the compressed
blocks for its sources to share (up to 64 MiB, which holds every file
here). Before that cache, the PIXEL rows fetched every block once per
band: 10,814 requests and 217.0 MiB in total. A COG stores its smallest overviews
first, so those blocks sit inside the prefetch: a 129 KiB ZSTD COG with
39 blocks took 21 requests. Merging adjacent reads could only save
requests between blocks, and nothing issues those as one read, so the
reader does not merge.

Two costs remain:
- **A large pixel-interleaved file read band after band fetches again.**
  The shared cache holds 64 MiB of compressed blocks, least recently
  used out. Sources that read the same region at about the same time
  share every fetch, whatever the file's size; a source that reads a
  whole band of a file bigger than that before the next band starts
  finds the first blocks gone. None of the files here is that large.
- **Blocks over 1 MiB take one request per MiB.** `cog` reads a block in
  1 MiB pieces, so that a corrupt byte count cannot make it allocate
  before it reads. `gtiff-Float32-onestrip`, a single 1.4 MB strip,
  takes 3 requests: the prefetch and two pieces. GDAL reports that strip
  as virtual strips of 2 rows, so `cogblocks.py` prints it with a note
  and leaves it out of the counts.

A planted defect, one bit flipped in the middle of every fetched range,
makes 0 of 98 files identical, so the comparison sees the HTTP path.

## Reading GeoTIFFs other software wrote

`cogcheck.sh` only sees files GDAL wrote with the options it chose.
`cogcorpus.sh` runs the same comparison over a directory of files it
did not write:

```bash
python3 corpus/fetch.py       # 675 files, 840 MB, each checked by SHA-256
corpus/generate.sh            # 80 more, from tiffcp, rasterio and GDAL
./cogcorpus.sh                # GDAL records, strata reads, cogcompare.py judges
```

- `cogcorpus.py` has GDAL record its reading of every `*.tif` and
  `*.tiff`, through `cogtruth.py`, the recorder `cogmake.py` now shares.
  GDAL sees each file alone: no `.ovr`, `.msk`, `.aux.xml` or `.tfw`,
  which strata does not read either.
- A level over 2²⁴ cells is compared over a 4096² window a third of the
  way in, and at most 16 bands, so a 36000² product does not need 5 GB a
  band. `cog/main.go` applies the same rule and the comparison checks
  both read the same window.
- A file cog refuses with a "not supported" error is **refused**, not
  failed. So is one GDAL cannot read either (**refused, GDAL too**). A file
  whose mask GDAL takes from an alpha band has its values compared on
  every cell, since cog reads alpha as a band. The EPSG code compared is
  that of GDAL's horizontal 2D CRS; one that cog leaves empty and GDAL
  identifies through PROJ is counted (**no EPSG**) but not failed, while
  a code that differs from GDAL's fails. Any other error or difference
  fails.

### The corpus

`corpus/sources.tsv` lists every downloaded file with its URL, SHA-256
and licence. Nothing is committed.

- **GDAL autotest** (MIT, OSGeo/gdal@7617801f): every TIFF in
  `autotest/gcore/data`, `alg/data` and `gdrivers/data/gtiff`. Deliberately
  odd files: masks, SubIFDs, broken tags, truncations, every codec, and
  files from ArcGIS, ERDAS IMAGINE and VICAR.
- **libtiff** (BSD-style, libtiff@0962d91b): `test/images`, including
  old-style LZW, and the `pics-3.8.0` set: GraphicsMagick's bit-depth
  matrix (public domain), CMYK, YCbCr, LogLuv, and Photoshop, HP, Wang and
  camera output (no licence stated beyond libtiff's distribution).
- **Go** `golang.org/x/image` v0.45.0 testdata (BSD-3-Clause).
- **OSGeo GeoTIFF samples** (no licence stated): USGS DRGs and DEMs, and
  files from PCI, Intergraph, ERDAS, SPOT and Z/I Imaging, in 60-odd
  projections.
- **Products**: Copernicus DEM GLO-30 and GLO-90; USGS 3DEP 1″ (public
  domain); Sentinel-2 L2A COGs (Copernicus licence); Landsat 9 C2 L2
  (public domain); NASADEM and MODIS MOD13Q1 (NASA, no restrictions); JAXA
  AW3D30; ESA WorldCover (CC-BY 4.0); JRC surface water; AWS Terrain
  Tiles; Natural Earth shaded relief (public domain, Photoshop). The
  Planetary Computer ones are fetched with its free anonymous token.
- **Generated** (`corpus/generate.sh`, not hash-pinned since they
  depend on the writers' versions, which each directory records):
  `tiffcp` (libtiff 4.7) rewrites into 16-bit and float predictors, odd
  RowsPerStrip, planar strips and tiles, big-endian, BigTIFF and
  FillOrder 2; rasterio and rio-cogeo (their own GDAL build) write COGs
  with masks, web-optimised tiling and dataset masks; GDAL writes what
  `cogmake.py` does not ask for: big-endian with predictor 2, internal
  masks, alpha, COG band and tile interleaving, strips of 1 row or more
  than the height, NoData values near and beyond each type's limits, and
  the types and codecs cog refuses.

### Results

With GDAL 3.14, every one of the 755 files is identical to GDAL's
reading, refused by design, or refused by GDAL too:

| Source | Files | Identical | Refused | GDAL refuses too | Fixed | No EPSG | Failed | Cells compared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gdal-autotest | 434 | 279 | 109 | 46 | 55 | 16 | 0 | 122,993,952 |
| generated-gdal | 50 | 38 | 12 | 0 | 3 | 0 | 0 | 2,000,645 |
| generated-rasterio | 11 | 11 | 0 | 0 | 1 | 0 | 0 | 48,898,844 |
| generated-tiffcp | 19 | 16 | 3 | 0 | 2 | 0 | 0 | 471,267 |
| libtiff-pics | 61 | 26 | 34 | 1 | 7 | 0 | 0 | 4,054,245 |
| libtiff-test | 31 | 18 | 12 | 1 | 5 | 0 | 0 | 1,773,395 |
| osgeo-samples | 113 | 108 | 5 | 0 | 8 | 5 | 0 | 176,204,528 |
| products | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 451,276,700 |
| x-image | 17 | 12 | 5 | 0 | 0 | 0 | 0 | 340,470 |
| **all** | 755 | 527 | 180 | 48 | 81 | 21 | 0 | 808,014,046 |

[`corpus/RESULTS.md`](corpus/RESULTS.md) has a row per file: writer,
features, result and cells compared. A run takes about 5 minutes.

The first run failed 79 files; a second, after the fixes and with the
FillOrder files added, 2 more. Each finding was cut down to a unit test
built with `cog/writer_test.go` (in `cog/foreign_test.go`), fixed, and
the corpus, `cogcheck.sh` (98/98) and `cogsabotage.py` (8/8) rerun:

| Finding | Files | Example | Fix |
|---|---:|---|---|
| Internal masks ignored: masked cells read as valid | 12 | `test_with_mask_1bit.tif`, rio-cogeo `--add-mask` | Read transparency-mask IFDs (1- or 8-bit) as GDAL does, from the chain and SubIFDs, matched to levels by size; a mask replaces NoData |
| EPSG code GDAL does not give | 26 | ERDAS `erdas_spnad83.tif`: EPSG:26966 with US-foot units | Name a code only when no key or citation redefines it; units checked against a table generated from PROJ's `proj.db` (`cog/epsg_gen.py`); deprecated codes as their replacement |
| YCbCr, CMYK, CIELab read as stored | 15 | `ycbcr_22_lzw.tif`, `rgbsmall_cmyk.tif` | Refuse them: GDAL converts them to RGB (at 8 bits or fewer) |
| NoData parsed or matched differently | 5 | NoData 12.5 on bytes; `"-1.#INF"`; `""` | GDAL's `CPLAtofM`; integer NoData truncated, out of range none; floats matched with `ARE_REAL_EQUAL`; float32 NoData near ±MaxFloat32 snapped, beyond it Inf |
| Strip and tile tags libtiff accepts | 7 | `size_of_stripbytecount_lower_than_stripcount.tif`, `cramps-tile.tif` | Pad or cut offset lists; SLONG8 offsets; missing byte counts computed; tiles under StripOffsets |
| Old-style LZW | 3 | `quad-lzw-compat.tiff` | Recognise libtiff's pre-5.0 LSB-first LZW by its first code |
| IFD chain loops | 3 | `test_ifd_loop_to_self.tif` | Stop the chain, as libtiff does |
| Overviews in SubIFDs | 2 | `tiff_with_subifds.tif` | Read IFD0's SubIFDs before the chain; unscaled overview grids without a geotransform |
| Blocks with an offset but 0 bytes | 2 | `cog_strile_arrays_zeroified_when_possible.tif` | A 0-byte block is absent |
| Edge tile cut at the image's last row | 2 | `contig_tiled.tif` | Accept a tile that covers the rows inside the image |
| FillOrder 2 | 2 | tiffcp `-f lsb2msb` | Reverse each stored byte's bits, as libtiff does for every codec but JPEG |
| Negative ScaleY | 1 | `negative_scaley.tif` | North-up, GDAL's default |
| NODATA_VALUES | 1 | `test_nodatavalues.tif` | Refuse: GDAL masks a cell only where every band holds its value |

The NoData finding is the one to remember. The unit test
`TestNoDataNative` said three things: a fractional NoData on integers
matches nothing, a float64 cell that only rounds to NoData stays valid,
and 1e39 on float32 matches nothing. GDAL does none of these. Its mask
band truncates integer NoData and compares floats within
2·FLT_EPSILON·|a+b|. In float32 that sum overflows near ±MaxFloat32, so
every cell there matches. The corpus caught the byte case; GDAL's source
(`gdalnodatamaskband.cpp`) gave the rest, and `generated-gdal/*nodata*`
confirms each rule against GDAL, cell by cell. The test is now
`TestNoDataGDAL`.

What the corpus still does not cover:
- **Few current non-GDAL writers.** All 19 products turned out to be
  COGs made with GDAL (by ESA, USGS, Element 84, Microsoft), apart from
  Natural Earth's Photoshop files. The other writers come from test sets
  of 1995 to 2011: ERDAS, PCI, Intergraph, VICAR, GraphicsMagick,
  ImageMagick and libtiff's tools. No QGIS or current ArcGIS export is
  included (QGIS writes through GDAL anyway), and no current commercial
  tool (ENVI, Global Mapper, FME).
- **What cog refuses is checked only for the refusal**: JPEG, WebP,
  LERC, LZMA, JPEG XL, CCITT, 1- to 7- and 12-bit, 64-bit integer,
  complex and 16-bit float data; 180 files.
- **Alpha bands** are compared as values only (17 files), and the 21
  files whose CRS GDAL identifies through PROJ get no EPSG code from cog.
- **Sidecar files** (`.ovr`, `.msk`, `.aux.xml`, `.tfw`) are hidden from
  GDAL, because cog does not read them.
- **Very large levels** are compared over one 4096² window, not whole.
- **Generated files** depend on the writers' current versions, so a
  later `generate.sh` may produce different ones.

### The EPSG table against epsg.io: epsgcheck.py

cog's unit and deprecation table (`cog/epsg_table.go`) comes from PROJ's
`proj.db`, and so does GDAL's reading of the same keys. Every comparison
above would pass a mistake in it. [`epsgcheck.py`](epsgcheck.py)
checks the table against [epsg.io](https://epsg.io), MapTiler's import
of the same EPSG registry, which does not go through `proj.db`. It uses
only the free per-code pages (no API key) and needs no packages:

```
python3 epsgcheck.py              # judge the table
python3 epsgcheck.py --sabotage   # plant 4 defects; each must be caught
```

It checks every unit entry (first axis unit, by name), every replacement
(the old code is deprecated, and the new one is an EPSG CRS of the same
kind), and that 668 codes the table leaves out (every neighbour of a
listed code, plus 400 random ones) really are in metres or degrees.
About 3,000 pages are cached in `out-epsg/cache`, so only the first run
takes minutes. Last run, both on EPSG 12.029: no disagreements.

It reports, without failing:

- **26 replacements deprecated themselves**, e.g. 31265 → 31275 → 3907
  → 8677. The table follows one step, and so does GDAL: for all 419
  replaced codes, `ImportFromEPSG` in GDAL 3.14 gives the table's
  replacement, even where it is deprecated (despite its warning calling it
  "non-deprecated").
- **EPSG:8449 → 8860**, a 3D geographic CRS replaced by a 2D one.
- **Codes epsg.io answers with ESRI's definition**, such as 26761 (NAD27
  / Hawaii zone 1, which EPSG does not have), are skipped: they are not
  EPSG codes.

epsg.io does not publish which code replaces which, so it cannot check
the pairing itself; GDAL agreeing on all 419 is the evidence for that.

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
