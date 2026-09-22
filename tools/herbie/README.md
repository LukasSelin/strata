# Herbie

[Herbie](https://herbie.uwplse.org/) rewrites floating-point expressions
to be more accurate, faster, or both. This directory runs it on strata's
kernel formulas. It is a tool for finding candidates, not a check: nothing
here runs in CI, and a rewrite only lands after the checks below.
[RESULTS.md](RESULTS.md) is the triage of the latest run.

| file | what |
|---|---|
| `kernels.fpcore` | each kernel formula in FPCore, in the scalar kernel's evaluation order, with the float32 constants' exact values and a realistic `:pre` range |
| `strata.rkt` | the Herbie platform: the operations a rewrite may use (no `fma`, nothing archsimd lacks) and their relative costs |
| `Dockerfile` | Herbie at a pinned tag, native on amd64 and arm64 |
| `run.sh` | builds the image if needed and runs `report` and `improve` in Docker |

## Running

Needs Docker. The first run builds the image from `Dockerfile`,
which takes several minutes.

```bash
tools/herbie/run.sh
```

Results land in `tools/herbie/out/` (ignored by git). `HERBIE_SEED`,
`HERBIE_THREADS` and `HERBIE_TAG` override the defaults.

To add a formula, append an `FPCore` to `kernels.fpcore`. Name it after
the Go function and give its file and line. Write it in the exact
evaluation order of the scalar kernel, and bound its inputs with `:pre`.
Herbie samples float bit patterns over that range, so a range wider than
real data rewards rewrites that only matter at values no raster holds.

## Before a rewrite lands

- **Bit identity.** The scalar kernel is canonical, and the AVX2 and NEON
  kernels must match it bit for bit (DESIGN.md §15). A rewrite changes all
  three, in the same operation order, along with the benchmark copies in
  `benchmarks/fusion` and `benchmarks/nodata`.
- **No FMA.** The platform leaves out `fma`. Keep the explicit `float32(...)`
  conversions that stop the compiler fusing multiplies and adds, and run
  the FMADD check in DESIGN.md §50.
- **Deliberate forms stay.** Some forms are chosen for exact endpoints or
  numpy parity, such as Normalize's `(v - lo) / span` (§18). Herbie does
  not know about those constraints.
- **Check by hand.** Confirm the gain in Go against a float64 or `math/big`
  reference over real data, and benchmark it. Herbie's error is averaged
  over its sample, and its costs are estimates.
