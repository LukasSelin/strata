# ADR 0001: SIMD backend technology — Go `simd` vs hand-written assembly

- **Status:** Accepted (all-in on `simd/archsimd`; asm backend removed)
- **Date:** 2026-09-16
- **Ticket:** STRATA-2
- **Related:** STRATA-11 (ARM64 backend), DESIGN.md §13–17, §20, §29

## Context

DESIGN.md assumes Go's SIMD API, but `internal/vec` shipped as AVX2 Plan 9
assembly: 11 kernels in `simd_amd64.s`, Go stubs plus scalar tails in
`simd_amd64.go`, CPUID detection in `cpu_amd64.{s,go}`, and function-variable
dispatch in `dispatch.go`. Before adding multi-row terrain kernels (§20), the
ARM64 backend (STRATA-11), and eventually fused kernels (§29), we need to
decide which technology new SIMD code is written in.

### State of Go SIMD (verified against Go 1.27.1 source and release notes)

| | Go 1.26 | Go 1.27 (Aug 2026) |
|---|---|---|
| Gate | `GOEXPERIMENT=simd` | `GOEXPERIMENT=simd` — **still experimental**, not in the default experiment baseline |
| Packages | `simd/archsimd` | `simd/archsimd` + new portable `simd` |
| amd64 | AVX, AVX2, AVX-512 (VEX/EVEX only) | same, API revised |
| arm64 | — | NEON, 128-bit (`Float32x4`) |
| wasm | — | 128-bit |
| Compat promise | none | none ("not yet considered stable") |

- Fixed-width float32 types: `Float32x4` / `Float32x8` / `Float32x16`, with
  `Load…`, `Load…Array`, `Load…Part`, `Store`, `StoreArray`, `StorePart`,
  `Broadcast…`, `Add/Sub/Mul/Div/Min/Max/Sqrt/Abs/Neg/MulAdd/Floor/…`,
  comparisons → `Mask32x8`, `IfElse`, `IsNaN`. CPU checks: `archsimd.X86.AVX2()`.
- The portable `simd.Float32s` has a smaller op subset (no `Floor`/`Round`,
  `Reciprocal`, …). The compiler multi-versions any function that uses it,
  for widths {emulated, 128, 256, 512} on amd64 and {emulated, 128} on arm64,
  and picks one at run time.
- Ops are compiler intrinsics. Loads fold into memory operands
  (`VADDPS (mem)`). The compiler does **not** emit `VZEROUPPER`, so code must
  call `archsimd.ClearAVXUpperBits()` itself.
- The API broke between 1.26 and 1.27 (`LoadFloat32x8Slice` → `LoadFloat32x8`,
  `StoreSlice` → `Store`, …).
- Every file is `//go:build goexperiment.simd`. Without the experiment the
  packages have no files, so strata must gate its own files the same way.
- Open issues relevant to us:
  - golang/go#80835: SSE/AVX transition penalties, fix pending for 1.28.
  - golang/go#81405: `Float32x4.Abs/Neg` SIGILL on AVX-only CPUs.
  - golang/go#79781: SVE is only a proposal.

Sources: go.dev/doc/go1.26#simd, go.dev/doc/go1.27#simd, golang/go#73787,
golang/go#78902, `src/simd` in the go1.27.1 toolchain.

## Options

- **A. Hand-written Plan 9 assembly** (status quo, extended to NEON and terrain).
- **B. `simd/archsimd`**: fixed width per architecture, gated by `goexperiment.simd`.
- **C. Portable `simd`**: width-agnostic, written once for all architectures, gated likewise.

## Spike

The prototype was `internal/spike/simdbackend` (throwaway, never imported;
removed in STRATA-6, last present in ad531b4). It implements `Add`, `Clamp`, and a 3×3 Horn slope-magnitude row
kernel (8 neighbour loads, 17 ops, `sqrt`; the §20 shape). Variants:

- scalar
- AVX2 asm (existing `vec.Add`/`vec.Clamp`, plus a new slope-row `.s`)
- archsimd written naively
- archsimd written for bounds-check elimination ("BCE")
- portable `simd`

Every variant is checked bit-for-bit against scalar, over NaN, ±Inf, ±0 and
tail lengths 0–100.

Machine: AMD Ryzen 9 3900X (Zen 2, AVX2, no AVX-512), Windows, go1.27.1,
`GOEXPERIMENT=simd`, `GOAMD64=v1`, 4096 cells per call (fits in cache,
compute-bound). Values are medians of 5 runs, in ns/op; lower is better.

| Kernel | scalar | asm | archsimd BCE | archsimd naive | portable (default, 4 lanes) | portable `GODEBUG=simd=+256` |
|---|---:|---:|---:|---:|---:|---:|
| Add       |  1641 |  400 |  708 | 1030 |  2271 | 1116 |
| Clamp     |  5236 |  275 |  567 |  817 |  1835 |  993 |
| SlopeRow  | 11656 | 1449 | **1639** | 5087 | 10605 | 5400 |

Speedup over scalar:

| Kernel | asm | archsimd BCE |
|---|---:|---:|
| Add | 4.1× | 2.3× |
| Clamp | 19× | 9.2× |
| SlopeRow | 8.0× | 7.1× |

`GOAMD64=v3` did not change the SIMD numbers.

Observations:

1. **Heavy kernels: archsimd is within about 12% of asm** (0.40 vs 0.35 ns/cell
   for slope). The instructions are the same. What remains is Go's overhead
   per iteration.
2. **Trivial kernels: asm is about 2× faster**, because the loop body is only a
   few instructions and overhead dominates. In the BCE build, each iteration
   advances every slice with a branch-free masked pointer bump (to avoid
   pointers past the end); asm uses one shared index. At raster scale these
   kernels are memory-bandwidth-bound anyway (§28).
3. **How the code is written matters a lot.** Naive `Load…(s[i:])` code keeps a
   bounds check and sub-slice arithmetic on every load and is 3.5× slower than
   the BCE form on slope. The pattern that fixes it is array-pointer loads over
   shrinking slices, and it has to be taught in review.
4. **On this AVX2 CPU, portable `simd` picks 128-bit vectors by default.** It
   only uses 256-bit when the CPU also has VPCLMULQDQ, which Zen 2 lacks. That
   makes it slower than scalar on slope. `GODEBUG=simd=+256` fixes it, but only
   the main program can set that, not a library.
5. Reviewability for slope: about 60 lines of register-juggling `.s` plus a
   stub and wrapper, versus about 25 lines of Go that mirror the scalar
   formula. The asm had to be written, commented and verified by hand. The Go
   version was a direct transliteration of the scalar formula.

## Comparison

| Criterion | A. asm | B. archsimd | C. portable simd |
|---|---|---|---|
| Maintainability / review | Poor: register-level, no type checking, arity/offset bugs are silent | Good: typed Go, mirrors scalar formula; needs BCE idioms | Best: one source for all arches |
| Go version / experiment | Any Go; none | Go ≥1.27; users must build with `GOEXPERIMENT=simd` to get SIMD | same as B |
| Effect on library users | SIMD in default builds | Default builds get scalar; opt-in via env | same as B, plus width depends on GODEBUG and CPU features |
| ARM64 (STRATA-11) | Write every kernel again in NEON asm (1.27 assembler has `VFADD/VFMIN/VFSQRT…`) | Write again with `Float32x4` in Go, same shape as amd64 | Free, 128-bit |
| Perf, heavy kernels | Best | ~1.1× asm | 128-bit by default on many AVX2 CPUs |
| Perf, trivial kernels | Best | ~2× asm | worse |
| Multi-row terrain (§20) | Every stencil is new hand asm | Natural: loads at offsets, expression per lane | Natural, width-agnostic stencils |
| Operation fusion (§29) | A fusion generator would have to emit asm | A generator emits Go; the compiler fuses loads into operands | Same as B, in one version |
| API stability risk | None from Go | High: churned 1.26→1.27; experimental | Higher: newest package |
| SIMD out of public API | Trivially | Yes, if confined to `internal/` (build tags stay internal) | same |

## Decision

1. **All SIMD code is written with `simd/archsimd`; strata carries no Plan 9
   assembly.** This covers the existing `internal/vec` kernels, the terrain row
   kernels (§20), fused kernels (§29), and the ARM64 backend (STRATA-11).
2. **go.mod targets `go 1.27.0`** (bumped in this ticket) so the standard-library
   SIMD packages are available.
3. **SIMD files are gated by `//go:build goexperiment.simd && <arch>`**, and
   `archsimd` may only be imported under `internal/`. Scalar remains canonical
   and is the path for any build without the experiment. Dispatch stays as
   function variables swapped in `init` (§17), using `archsimd.X86.AVX2()` for
   detection in simd builds.
4. **The 11 AVX2 asm kernels are replaced now**, in this ticket, by archsimd
   versions in `internal/vec/simd_amd64.go`. The asm and hand-rolled CPUID
   code is removed. We accept the costs below (2× slower trivial kernels, no
   SIMD in default builds) in exchange for a single, reviewable,
   standard-library SIMD implementation. We will not revisit that unless
   the experiment is withdrawn or benchmarks regress badly on a new Go
   release.
5. **Fixed-width `archsimd` over portable `simd` for now.** Portable's default
   width selection costs 2× on common AVX2 hardware and a library cannot
   override it. Its op subset is also smaller. Kernels are written as
   load → expression → store so a move to `simd` later is mechanical. Revisit
   with Go 1.28.

## Port results

The ports keep the §17 dispatch model: function variables swapped in `init`
when `archsimd.X86.AVX2()` is true. They are checked bit-for-bit against
scalar (any NaN matches any NaN, but +0 ≠ −0), including signed-zero and NaN
clamp bounds.

That stricter test motivated exact `min`/`max` lanes. `VMINPS`/`VMAXPS`
return the second operand on NaN and on equal operands. The removed asm
patched only the NaN case, so it returned the wrong sign for, e.g.,
`Min(-0, +0)`; the old `==`-based tests could not see that. The archsimd
kernels OR (min) or AND (max) the bits of equal lanes and restore NaN lanes
from the first operand.

Results on the same machine, go1.27.1 with `GOEXPERIMENT=simd`, from the
`internal/vec` benchmarks (4096 cells, medians of 5, ns/op):

| Kernel | scalar | archsimd | speedup | removed asm (spike run) |
|---|---:|---:|---:|---:|
| Add   | 2067 | 711  | 2.9× | 400 |
| Min   | 3027 | 1005 | 3.0× | — |
| Clamp | 4094 | 934  | 4.4× | 275 (NaN-only fix, wrong ±0) |

The asm column comes from the spike's inputs, so compare ratios, not
absolute numbers. Exact signed-zero handling accounts for most of the Clamp
gap versus the spike's NaN-only archsimd Clamp (567).

## Consequences

- **Users** need Go ≥1.27. **Builds without `GOEXPERIMENT=simd` are now
  scalar-only on every architecture**, including amd64, which previously got
  asm without opting in. With the experiment (for example
  `go env -w GOEXPERIMENT=simd`) they get all SIMD kernels. This must be
  documented prominently once a public API exists. The experiment is a
  whole-build flag; it cannot be set from strata's go.mod. Benchmarks and
  published numbers must state which mode they ran in.
- **Toolchain churn:** expect `archsimd` renames on each Go release. The blast
  radius is `internal/` only. Upgrading Go is a deliberate change that re-runs
  the equivalence tests and benchmarks.
- **CI** must test three configurations:
  - default,
  - `GOEXPERIMENT=simd`,
  - `GOARCH=arm64` with `GOEXPERIMENT=simd` (build at minimum, run on arm64
    hardware or emulation for STRATA-11).
  Scalar-equivalence tests (§39) run in every configuration.
- **Kernel-writing rules** for review:
  - write loops in BCE form: array-pointer loads over shrinking slices;
  - call `archsimd.ClearAVXUpperBits()` before scalar tails on amd64;
  - between a lane loop's first 256-bit instruction and that call, emit no
    legacy (non-VEX) SSE: no struct copies or zeroing of vectors (`MOVUPS`)
    and no zero vectors from `Broadcast…(0)` (`XORPS`). Keep constant
    vectors in package variables set in `init`. Each such instruction cost
    about 65 ns on Zen 2, once per row (STRATA-9,
    benchmarks/engine/RESULTS.md; `BenchmarkRowWidth` in internal/stencil
    shows it as a per-call cost);
  - handle NaN explicitly with `IsNaN` / `IfElse`. `VMINPS`/`VMAXPS` return the
    second operand on NaN, and arm64 `FMIN` propagates it, so raw `Min`/`Max`
    semantics differ by arch;
  - prevent FMA fusion in scalar references with explicit `float32(...)`
    conversions so SIMD and scalar stay bit-identical;
  - gate amd64 SIMD on AVX2 (not just AVX), avoiding golang/go#81405.
- **STRATA-11 (ARM64):** implement with `archsimd.Float32x4` (NEON is baseline
  on arm64, so no runtime detection is needed), behind
  `goexperiment.simd && arm64`, reusing the amd64 file's loop shape and its
  exact `min`/`max` approach. Check lanewise NaN and ±0 behaviour of NEON
  `FMIN`/`FMAX` against scalar rather than assuming x86 semantics. Default
  arm64 builds use scalar until the experiment graduates. SVE is out of scope
  (golang/go#79781).
- **Fusion (§29)** becomes a Go code-generation problem rather than an
  assembly one, which keeps it viable.
- `internal/spike/simdbackend` kept the asm-vs-archsimd slope-row comparison
  reproducible until the first real terrain kernel landed. It was deleted in
  STRATA-6; check out ad531b4 to re-run it.
