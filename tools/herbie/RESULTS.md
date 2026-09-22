# Herbie triage

Herbie 2.3 (`tools/herbie/Dockerfile`, native arm64) with the `strata.rkt`
platform, seed 1, 256 sample points, 4 iterations. The first run covered
14 single formulas and took 37 s on an Apple M4. A second run added 7
inputs: whole windows through to each terrain output, and all of
`Atan2F32`. The full set of 21 takes 3.5 min (see
[End to end](#end-to-end-and-atan2f32)). Herbie reports error as the
average number of bits wrong over its sample; 0 is
correctly rounded and 32 is garbage. Its costs use the platform's weights.
They count a shared subexpression once per use, so they overstate the cost
of any formula that has a `let` (the Atan32 branches).

Every candidate with a meaningful accuracy change was then checked by
hand in Go, against a float64 or `math/big` reference over realistic
data. Those numbers are in the Verdict column. Candidates that gain 0.05
bits or less were not checked further.

## Summary

**One rewrite is worth taking: the Horn difference.** Herbie found it on
its own, and it has landed (see below). Everything else is already as
accurate as float32 allows over real inputs. Herbie's other "improvements" either give up accuracy where
data lives in exchange for accuracy where it does not, or break
properties that strata relies on and Herbie cannot see.

| formula | error in, out (bits) | cost in, out | best alternative | verdict |
|---|---|---|---|---|
| hornDX·kx, one window (z in 1000..1100) | **4.71 → 0.00** | 1.650 → 1.625 | `(((z9-z7) + 2(z6-z4)) + (z3-z1))·kx` | **taken**, see below |
| hornDX·kx, any elevations (z in 0..9000) | 0.22 → 0.21 | 1.650 → 1.625 | same shape | the regrouping is no worse on unrelated elevations |
| slope magnitude `sqrt(gx²+gy²)` | 4.86 → 0.71 | 1.700 → 3.625 | `max + ½·min²/max` (a Taylor series) | **reject**: 6.1% wrong at gx = gy. The input's error is underflow of gx² for \|g\| < 1e-19, which no DEM produces. Over \|g\| ≤ 1e3 the current form is within 1.16 ulp |
| Atan32, x ≤ tan(π/8) | 0.00 → 0.00 | 3.875 → 3.550 | reassociated Horner | reject: no accuracy to gain, and the cost gain is noise |
| Atan32, tan(π/8) < x ≤ tan(3π/8) | 0.16 → 0.17 | 18.4 → 15.4 | divisions in place of the shared `t` | reject: the cost gain comes from `t` being computed once in the code, which Herbie's cost model does not see. It is less accurate |
| Atan32, x > tan(3π/8) | 0.00 → 0.00 | 13.1 → 6.75 | Horner in 1/x² with 5 divisions | reject: 5 divisions in place of 1 is slower on every SIMD unit. The cost win is again the unshared `t` |
| Atan2F32 reflection `π − a` | 0.00 → 0.00 | — | none | nothing to do |
| aspectDegrees fold | 0.00 → 0.00 | — | none | nothing to do |
| hillshade `(c + bx·gx + by·gy) / sqrt(1 + gx² + gy²)` | 0.12 → 0.12 | 3.725 → 3.725 | `gx² + (gy² + 1)` inside the sqrt | reject: no gain |
| hillshade light `bx` (float64, once per call) | 0.26 → 0.26 | — | none | nothing to do |
| Affine `v·a + b` | 0.03 → 0.03 | — | none | control: FMA is excluded, and nothing else helps |
| SubDiv `(v − lo) / span` | 0.09 → 0.09 | — | none | control: the form is deliberate (DESIGN.md §18) |
| Lookup lerp `y0 + ((v−x0)/(x1−x0))·(y1−y0)` | 4.01 → 0.25 | 1.850 → 3.100 | `(y1/d)·(v−x0) + ((x1−v)/d)·y0` | **reject**: see below |
| rescaleCoeffs offset `b` (float64) | 13.65 → 1.41 | 1.650 → 6.900 | three-way regime split | **reject**: see below |

## The Horn difference: taken

`hornDX` and `hornDY` (internal/stencil/horn.go) added the elevations up
first and subtracted second:

```go
((z3 + z9) + (z6 + z6)) - ((z1 + z7) + (z4 + z4))
```

Each partial sum is about 2 to 4 times the elevation, so it rounds to the
ulp of that sum: 1.2e-4 at 1000 m and 9.8e-4 at 8800 m. The subtraction
then cancels almost all of it and leaves that rounding error inside a
small difference. Subtracting neighbours first is exact whenever
neighbours are within a factor of two of each other, which is every
realistic DEM (Sterbenz). The small differences then add up with little
or no rounding:

```go
d := z6 - z4
((z3 - z1) + (z9 - z7)) + (d + d)
```

Herbie's version, `((z9 - z7) + 2·(z6 - z4)) + (z3 - z1)`, is the same
idea. Checked in Go against an exact float64 reference, on windows that
are a base elevation plus a plane plus noise, 200 000 windows per row:

| base, relief per cell | dx error, old (max) | regrouped | aspect error, old (max) | regrouped |
|---|---|---|---|---|
| 500 m, 1 m | 1.8e-4 | 0 | 0.12° | 0° |
| 1000 m, 0.1 m | 3.7e-4 | 0 | 2.6° | 0° |
| 1000 m, 0.01 m | 3.7e-4 | 0 | 135° | 0° |
| 4000 m, 1 m | 1.5e-3 | 0 | 1.2° | 0° |
| 8800 m, 10 m | 5.9e-3 | 0 | 1.8° | 0° |
| 8800 m, 0.1 m | 5.9e-3 | 0 | 45° | 0° |
| 10 m, 100 m (steep, near sea level) | 6.9e-5 | 6.8e-5 | 2.4e-5° | 6.1e-6° |

Gradient, slope and hillshade inherit the dx and dy errors directly.
Aspect suffers most, because on gentle terrain the error is a large part
of a small gradient. The regrouped form also costs one addition less per
axis: 6 operations instead of 7, the same in scalar and in both SIMD
backends.

## The Lookup lerp: reject

Herbie's form lowers the worst case: 111 → 2.8 ulp for same-sign knots,
and 3.8e4 → 1.3e4 ulp across a sign change, where the current form
cancels. But it raises the mean on same-sign data (0.17 → 0.23 ulp). It
also adds a division, and it is **not monotone in v**: it went backwards
between adjacent floats in 700 of 200 000 same-sign samples.
DESIGN.md §50 relies on a monotone curve staying bounded by its own
knots. The current form is monotone because each step rounds
monotonically. If the worst case ever matters, the two-sided lerp Herbie
also lists (`y1 − (1−t)·(y1−y0)` for t near 1, cost 3.4, 0.49 bits) is
the one to evaluate. It also needs a monotonicity proof. Lookup is scalar
only, so no SIMD code would change.

## The rescale offset: reject

Herbie measures the error of the float64 `b` before the code rounds it to
float32. Once it is rounded, the current `b` was already the correctly
rounded float32 of the exact `ol − a·il` in 1 000 000 of 1 000 000 random
intervals. The 13.65 bits sit where `b` is nearly 0, which Herbie's
sampling favours. There the float64 relative error cannot survive the
rounding to float32 anyway.

## End to end and Atan2F32

The first run checked each formula on its own, so it could miss rounding
that one step passes on to the next. The second run adds a raw 3×3
window carried through Horn to slope magnitude, slope in degrees, aspect
(once at 1000..1100 m and once at 8800..8801 m) and hillshade. It also
adds `Atan2F32` in full: the `|y|/|x|` division, `Atan32`, the 0/0 case,
the reflection and the sign.

Inside the window inputs, `Atan32` and `Atan2F32` appear as `atan` and
`atan2`, which the platform prices at 20 so that no rewrite adopts them.
Spelling them out with their branches does not work: Herbie inlines
every `let`, the expression tree grows past Docker's memory, and the run
dies without writing a report. Their own error is measured by the two
standalone `Atan2F32` inputs instead.

| input | error (bits) | best alternative | verdict |
|---|---|---|---|
| window → slope magnitude | 0.28 → 0.23 | multiplies by `kx` in a different order | reject: 0.05 bits on average is below what justifies changing output bits |
| window → slope degrees | 0.29 → 0.29 | reordering | nothing to do |
| window → aspect, 1000..1100 m | 0.23 → 0.23 | reordering | nothing to do |
| window → aspect, 8800..8801 m | 0.23 → 0.24 | none better | nothing to do |
| window → hillshade (with the clamp) | 0.37 → 0.36 | reordering | nothing to do |
| `Atan2F32` against the true atan2 | 0.11 → 0.11 | none | nothing to do: this is the polynomial's own approximation error |
| `Atan2F32`, rounding only | 0.01 → 0.02 | `\|y/x\|` for `\|y\|/\|x\|`, divisions in the polynomial | reject: the same value, and slower on SIMD. The cost gain is the unshared-subexpression effect again |

All of these are at the float32 floor. The same result holds on real
rasters, not just Herbie's samples. Slope, Aspect and Hillshade run
through the public API on 258² DEMs (a base elevation plus a tilted plane
plus noise, cell size 10) and are compared with a float64 reference from
the same float32 input, before and after the Horn fix (NEON build):

| base, relief per cell | slope° max, old → new | aspect° max, old → new | hillshade max, old → new |
|---|---|---|---|
| 10 m, 0.001 m | 5.4e-6 → 9.2e-10 | 0.13 → 2.8e-5 | 2.9e-5 → 1.1e-5 |
| 1000 m, 0.001 m | 3.6e-4 → 8.1e-10 | 8.4 → 1.6e-5 | 1.2e-3 → 1.1e-5 |
| 1000 m, 1 m | 6.9e-4 → 1.1e-6 | 0.013 → 2.7e-5 | 2.3e-3 → 3.4e-5 |
| 4000 m, 0.001 m | 1.5e-3 → 3.5e-10 | 179 → 2.4e-5 | 4.7e-3 → 3.9e-6 |
| 8800 m, 0.01 m | 5.9e-3 → 6.7e-9 | 32 → 2.3e-5 | 1.9e-2 → 1.8e-5 |
| 8800 m, 10 m | 4.2e-3 → 9.1e-6 | 9.5e-3 → 1.3e-5 | 1.9e-2 → 2.2e-5 |

After the fix, aspect stays within 2.8e-5° at every elevation and relief.
That matches `Atan2F32`'s documented bound (1.54e-5°) plus the degrees
conversion. Slope is within 1e-5° and hillshade within 5e-5 of 255.

## What landed

The Horn difference is now `((z3 - z1) + (z9 - z7)) + (d + d)` with
`d = z6 - z4`, and the same shape for dy. The change covers scalar
`hornDX`/`hornDY` (internal/stencil/horn.go), `hornDiff8`
(simd_amd64.go) and `hornDiff4` (simd_arm64.go).

- `TestHornGradientNearFlat` (internal/stencil/stencil_test.go) failed on
  the old grouping, with errors up to 2.9e-3 against tolerances near 1e-5,
  in both the scalar and NEON builds. It now passes.
- The bit-identity tests pass on NEON. The AVX2 kernel compiles and lints
  but was not run locally, because Rosetta has no AVX2. CI's Linux SIMD
  job runs it.
- The acceptance harness passes 261/261 in the scalar and NEON builds,
  and `sabotage.py` still catches all 10 injected defects. `gdalcheck.sh`
  needs a real GeoTIFF and was not run.
- Herbie re-run on the new form: 0.00 bits for one window and 0.21 bits
  for any elevations, with no further rewrite.
- `BenchmarkSlopeRow` on an Apple M4 (8 runs each, benchstat, p ≤ 0.03).
  Rows marked ~ show no significant change:

  | row | scalar | NEON |
  |---|---|---|
  | magnitude | −8.9% | −4.6% |
  | magnitude + atan32 | −4.1% | −3.7% |
  | atan | −4.2% | ~ |
  | magnitude + math.Atan | −3.1% | ~ |

The recorded suite numbers in `benchmarks/terrain` and `benchmarks/engine`
predate the change and were not re-run.
`benchmarks/nodata` keeps the old grouping on purpose: it is the
STRATA-3 NoData spike, its variants only need to agree with each other,
and its recorded results were measured with that grouping.
