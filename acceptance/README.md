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
./gdalcheck.sh <some.tif>       # difference against gdaldem in Docker
```

`check.py` needs numpy; `--png` also needs matplotlib. It prints one line
per check, exits non-zero if any failed, and on a failure prints only the
failures.

## What is checked

`main.go` builds three small DEMs — a tilted plane, a smooth hill on
rectangular cells, and a noisy surface with two NoData regions — and runs
every terrain, algebra and reduce operation on each, in all three forms
(plain, `Tiled` with ragged 37×23 tiles on 3 workers, `Chunked` through
raw float32 files on 4 workers). 237 checks come out of that:

| # | Check | Why it would catch a defect |
| - | ----- | --------------------------- |
| 1 | Every result against Horn's gradient recomputed in float64 numpy | A wrong kernel, a cell size used on the wrong axis, degrees for radians, a sign flip |
| 2 | The one-cell border carries no data | Horn needs all eight neighbours; a border cell that holds a number is reading outside the raster |
| 3 | The plane against its analytic slope and aspect | The whole pipeline agrees with pen-and-paper on a surface whose answer is known exactly |
| 4 | plain == tiled == chunked, bit for bit | Tile seams, worker races, off-by-one tile origins — the README's central promise |
| 5 | Validity after a 3×3 erosion of the input mask | NoData leaking into a result, or valid cells wrongly discarded |
| 6 | Pointwise algebra against numpy in float32 | Exact equality is required here, so any drift shows |
| 7 | `Count` and `MinMax` against numpy over the valid cells | A reduction that misses a tile or double-counts one |
| 8 | Degrees, radians and percent agree with each other | A unit conversion applied twice, or not at all |

The reference implementations are derived in `check.py`'s docstring from
Horn's kernel as gdaldem documents it, and from the definition of
shaded relief as the cosine between the light and the surface normal —
not from strata's code.

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
within 2 ulps of each other         76.21% bit-identical
no seam every 256 rows              1.222e-06 on chunk boundaries vs 1.203e-06 elsewhere
```

Check 5 does not stop at "they differ in the last digit": it recomputes
the slope in float64 and scores **both** tools against it. On that run
gdaldem was closer on 17.4% of cells and strata on 6.4%, tied on 76.2%,
with mean errors of 1.40e-06 and 1.85e-06 degrees. So strata is very
slightly the less accurate of the two, by an amount that is inside its
own documented float32 bounds and about a millionth of a degree — worth
knowing, not worth fixing.

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
  from outside the library. On a CPU without AVX2 the vector build falls
  back to scalar and the comparison proves nothing.
* **Anything outside these operations.** Extending it means adding a
  case to `main.go` and a reference to `expected()` in `check.py`.
