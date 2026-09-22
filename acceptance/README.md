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
```

`check.py` needs numpy; `--png` also needs matplotlib. It prints one line
per check, exits non-zero if any failed, and on a failure prints only the
failures.

## What is checked

`main.go` builds three small DEMs — a tilted plane, a smooth hill on
rectangular cells, and a noisy surface with two NoData regions — and runs
every terrain, algebra and reduce operation on each, in all three forms
(plain, `Tiled` with ragged 37×23 tiles on 3 workers, `Chunked` through
raw float32 files on 4 workers). 351 checks come out of that:

| # | Check | Why it would catch a defect |
| - | ----- | --------------------------- |
| 1 | Every result against Horn's gradient, or for curvature the Zevenbergen–Thorne derivatives, recomputed in float64 numpy | A wrong kernel, a cell size used on the wrong axis, degrees for radians, a sign flip, profile and plan curvature mixed up |
| 2 | The one-cell border carries no data | Horn needs all eight neighbours; a border cell that holds a number is reading outside the raster |
| 3 | The plane against its analytic slope, aspect and zero curvature | The whole pipeline agrees with pen-and-paper on a surface whose answer is known exactly |
| 4 | plain == tiled == chunked, bit for bit | Tile seams, worker races, off-by-one tile origins — the README's central promise |
| 5 | Validity after a 3×3 erosion of the input mask | NoData leaking into a result, or valid cells wrongly discarded |
| 6 | Pointwise algebra against numpy in float32 | Exact equality is required here, so any drift shows |
| 7 | `Count` and `MinMax` against numpy over the valid cells | A reduction that misses a tile or double-counts one |
| 8 | Degrees, radians and percent agree with each other | A unit conversion applied twice, or not at all |
| 9 | `Normalize` against `(z - min) / (max - min)` in float32 numpy over the valid cells, with min and max landing on exactly 0 and 1 | A range taken over NoData, a rounding change such as multiplying by a reciprocal, an endpoint off by an ulp |

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
Horn's kernel as gdaldem documents it, and from the definition of
shaded relief as the cosine between the light and the surface normal,
and for curvature from Zevenbergen and Thorne's (1987) quadratic and
Florinsky's (2016) normal-section formulas — not from strata's code.
`gdaldem` has no curvature mode, so for curvature this numpy reference
is the only outside opinion.

## Tolerances

strata computes in float32 and the reference in float64, so they differ
by float32 rounding alone. Each tolerance is **derived, not tuned**:
Horn's numerator is six weighted elevations accumulated in at most six
roundings of a partial sum of magnitude ≤ 4·zmax, so the gradient is off
by at most `3 · 2⁻²⁴ · zmax / cellsize`, and every other tolerance
follows from how the operation propagates that. Each line reports the
observed error, the bound, and their ratio — currently 0.04× to 0.25×,
so the results sit comfortably inside a bound that is itself tight.

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
```

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
slope/aspect/hillshade`, runs the same three through strata's
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

`out-gdal/slope-difference.png` shows where the two disagree. It should
be featureless speckle following the terrain texture; horizontal bands
every `TileHeight` rows would mean a seam bug that a tolerance could
have hidden.

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

  On an AVX2 machine (checked on a Ryzen 9 3900X) all 101 files are
  byte-identical, which is the SIMD claim in the top-level README tested
  from outside the library. That run predates curvature, whose files
  have not been diffed on AVX2 hardware yet. On a CPU without AVX2 the vector build falls
  back to scalar and the comparison proves nothing.
* **Anything outside these operations.** Extending it means adding a
  case to `main.go` and a reference to `expected()` in `check.py`.
