# STRATA-3: NoData representation spike, results

## Recommendation

Carry a **validity bitmap** and **no `NoData` value** in the raster struct:

```go
type Raster[T Number] struct {
    Data []T

    Width  int
    Height int
    Stride int

    // Valid marks which cells hold data.
    //   nil  => every cell is valid (the fast path; costs nothing).
    //   else => bit i of Valid[i>>6] (LSB first) is set iff Data[i] is valid,
    //           len(Valid) == (len(Data)+63)/64, bits >= len(Data) are zero.
    // Data under a cleared bit is unspecified; kernels may overwrite it.
    Valid []uint64
}
```

`Float32Raster` gets the same `Valid []uint64` in place of `NoData float32`.

Rules that go with it:

1. **Validity is never inferred from Data.** A NaN (or -9999) in `Data`
   with its bit set is an ordinary value, and arithmetic handles it as
   IEEE says. Kernels compute every cell unconditionally, then derive the
   output mask with word-level operations: AND of the inputs for
   pointwise ops, and a 3×3 erosion for radius-1 stencils.
2. **Fill values belong to the IO adapters.** GeoTIFF/Zarr readers turn
   `GDAL_NODATA`, `_FillValue`, `missing_value`, `"NaN"` and so on into
   `Valid` on read (`MaskFromSentinel`, about 1.35 ns/cell scalar today).
   Writers put the format's fill value back under cleared bits
   (`FillInvalid`). The fill value travels as metadata next to the
   raster, not inside it.
3. **nil means all valid.** If every input is nil, the output is nil, so
   rasters with no NoData run exactly the NaN/plain code path.
4. For STRATA-4: the bit index is the **Data index**, so it follows
   `Stride`. A `Window` sharing its parent's mask needs a bit offset
   (for example `ValidOffset int`), because only word-aligned offsets can
   be sub-sliced. The engine should allocate tile and halo buffers with
   `Stride` rounded up to a multiple of 64. Then row masks start on word
   boundaries, and the mask pass skips the extract/deposit shifting
   (see "Caveats").

Why, in one paragraph: vectorized with `simd/archsimd`, the mask costs
about the same as NaN on Add (+0–6% at 1024²) and 0–20% more on the 3×3
slope. Sentinel costs 14–25% more on both. Only the mask has none of the
correctness hazards below. It works unchanged for integer rasters, which
have no NaN. It carries validity through fused pipelines (§29) as one
cheap side pass, not per-op logic. NaN is the fastest option but not by
enough to justify its hazards: it cannot tell missing apart from computed
NaN, it silently fails to propagate through stencils that skip a cell and
through comparisons, and it does not exist for `uint8`/`int16` sources.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200 |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.27.0 windows/amd64, `GOAMD64=v1`, **`GOEXPERIMENT=simd`** |
| SIMD | `simd/archsimd` (ADR 0001): `internal/vec` for Add, `simd_amd64.go` in this package for the rest. No assembly. |
| Run | `GOEXPERIMENT=simd go test -c`, then `nodata.test.exe -test.run '^$' -test.bench . -test.benchmem -test.count 6`, pinned to one logical CPU (affinity `0x10`), `GOMAXPROCS=1`, High priority |
| Stats | median of 6 runs, each ≥1 s (`b.Loop`) |

To reproduce the tables:

```
GOEXPERIMENT=simd go test ./benchmarks/nodata -run '^$' -bench . -benchmem -count 6 -timeout 3h > bench.txt
go run ./benchmarks/nodata/cmd/nodatatable < bench.txt
```

Without `GOEXPERIMENT=simd` the `*-avx2*` variants are skipped, and the
`*-vec*` variants run the scalar `internal/vec` kernels. Numbers from
such a build are not comparable with these.

This was a desktop with other applications open. Most cells vary 1–4%
between runs, with a few outliers up to 23%. Differences under about 10%
should not be read as real.

Workload details:

- **Add**: `dst = a + b`. `a` is a synthetic DEM, `b` a smooth field, and
  each has an *independent* NoData layout of the named pattern, so at
  30% the output is about 51% NoData.
- **Slope**: 3×3 Horn gradient magnitude. Any NoData among the nine
  cells, or the border, makes the output NoData.
- **Patterns**: 0%; 1% scattered (independent cells); 30% scattered;
  30% clustered (random discs, radius 4–63 cells, the same absolute size
  at both raster sizes).
- The **mask** variants read the same `-9999`-filled data arrays as the
  sentinel variants, as if the reader left the fill value in place.
- Go's gc compiler does not auto-vectorize, so "scalar" really is scalar.

Variant names:

| suffix | meaning |
|---|---|
| `scalar-branchy` | the obvious per-cell `if` |
| `scalar-select` | compute unconditionally, OR the compare flags, then one select |
| `vec+fixup` | `vec.Add`, then a scalar pass rewriting NoData cells |
| `avx2-blend` | compute + `Equal` against the sentinel, `Or` the masks, `IfElse` blend, 8 lanes |
| `nan-*` | plain arithmetic. For slope it includes a `z5·0` term, see Hazards |
| `mask-scalar` / `mask-vec` / `mask-avx2` | plain arithmetic + word-level mask pass |
| `mask-*+fill` | additionally writes a fill value under invalid cells (export cost) |

All variants have 0 allocations and 0 B/op. The only scratch is the
slope mask pass: 2·⌈width/64⌉ words (256 B at 1024, 1 KiB at 4096).

## Results

Cells show median **ns/cell (million cells/s)**. B/op and allocs/op are 0 in every row.

### Add, 1024×1024 (compute/cache-bound: 12 MiB working set)

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 0.985 (1015) | 1.15 (870) | 4.99 (200) | 1.07 (938) |
| sentinel-scalar-select | 1.76 (569) | 1.99 (503) | 6.99 (143) | 1.88 (532) |
| sentinel-vec+fixup | 0.929 (1076) | 1.11 (903) | 5.09 (197) | 1.21 (825) |
| sentinel-avx2-blend | 0.223 (4477) | 0.223 (4482) | 0.222 (4498) | 0.231 (4328) |
| nan-scalar | 0.495 (2018) | 0.499 (2003) | 0.496 (2015) | 0.524 (1909) |
| nan-vec | 0.189 (5297) | 0.179 (5585) | 0.179 (5587) | 0.187 (5352) |
| mask-scalar | 0.514 (1945) | 0.503 (1987) | 0.506 (1978) | 0.509 (1966) |
| mask-vec | 0.190 (5258) | 0.188 (5319) | 0.187 (5339) | 0.190 (5271) |
| mask-vec+fill | 0.195 (5116) | 0.308 (3248) | 0.948 (1054) | 0.877 (1140) |

### Add, 4096×4096 (memory-bandwidth-bound: 192 MiB working set)

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 1.31 (761) | 1.49 (670) | 5.57 (180) | 1.37 (728) |
| sentinel-scalar-select | 1.91 (525) | 2.15 (465) | 7.75 (129) | 1.98 (506) |
| sentinel-vec+fixup | 1.44 (696) | 1.85 (539) | 5.77 (173) | 2.05 (488) |
| sentinel-avx2-blend | 0.646 (1547) | 0.661 (1513) | 0.650 (1538) | 0.654 (1529) |
| nan-scalar | 0.626 (1598) | 0.631 (1586) | 0.630 (1587) | 0.619 (1617) |
| nan-vec | 0.550 (1818) | 0.547 (1828) | 0.549 (1821) | 0.545 (1834) |
| mask-scalar | 0.799 (1252) | 0.804 (1243) | 0.806 (1241) | 0.810 (1234) |
| mask-vec | 0.677 (1476) | 0.665 (1503) | 0.665 (1504) | 0.667 (1499) |
| mask-vec+fill | 0.674 (1484) | 0.884 (1131) | 1.43 (702) | 1.43 (701) |

### Slope (3×3 Horn), 1024×1024

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 5.96 (168) | 5.95 (168) | 6.86 (146) | 4.62 (216) |
| sentinel-scalar-select | 7.40 (135) | 7.90 (127) | 8.25 (121) | 7.50 (133) |
| sentinel-avx2-blend | 0.594 (1683) | 0.601 (1665) | 0.599 (1670) | 0.595 (1681) |
| nan-scalar | 3.30 (303) | 2.95 (339) | 2.95 (340) | 2.95 (339) |
| nan-avx2 | 0.522 (1915) | 0.482 (2075) | 0.482 (2074) | 0.483 (2069) |
| mask-scalar-branchy | 9.34 (107) | 9.26 (108) | 6.53 (153) | 8.36 (120) |
| mask-scalar | 2.68 (373) | 2.68 (373) | 2.68 (373) | 2.69 (372) |
| mask-avx2 | 0.526 (1902) | 0.525 (1906) | 0.527 (1897) | 0.524 (1908) |
| mask-avx2+fill | 0.543 (1843) | 0.775 (1291) | 1.68 (597) | 0.955 (1048) |

### Slope (3×3 Horn), 4096×4096

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 6.04 (166) | 6.05 (165) | 6.87 (145) | 4.70 (213) |
| sentinel-scalar-select | 7.47 (134) | 7.97 (125) | 8.07 (124) | 7.60 (132) |
| sentinel-avx2-blend | 0.655 (1528) | 0.659 (1518) | 0.661 (1512) | 0.657 (1522) |
| nan-scalar | 2.93 (341) | 2.97 (337) | 2.93 (341) | 2.96 (337) |
| nan-avx2 | 0.562 (1780) | 0.563 (1775) | 0.555 (1803) | 0.559 (1790) |
| mask-scalar-branchy | 10.0 (100) | 9.88 (101) | 6.88 (145) | 8.88 (113) |
| mask-scalar | 3.58 (279) | 3.56 (281) | 3.59 (278) | 3.59 (279) |
| mask-avx2 | 0.652 (1534) | 0.641 (1559) | 0.666 (1502) | 0.643 (1555) |
| mask-avx2+fill | 0.679 (1472) | 0.990 (1010) | 1.87 (534) | 1.20 (833) |

**Where the 4096² mask gap comes from.** A follow-up probe was run on
the earlier Go 1.25 build. That build used hand-written AVX2 kernels
with the same instructions, and the mask pass was the same scalar Go.
It separated two effects (30% clustered, ns/cell):

| measurement | 1024² | 4096² |
|---|---:|---:|
| `SlopeMask` pass alone | 0.078 | 0.072 |
| AVX2 rows without centre term, on the `-9999` array | 0.364 | 0.568 |
| AVX2 rows without centre term, on the NaN array | 0.380 | 0.509 |
| AVX2 rows with `z5·0` centre term, on the `-9999` array | 0.409 | 0.591 |
| AVX2 rows with `z5·0` centre term, on the NaN array | 0.427 | 0.529 |

At 4096² the *same kernel* ran about 11% slower over the `-9999` input
array than over the NaN one. That comes from where the array landed in
memory (TLB/prefetch behaviour of that allocation), not from the values.
The mask variants read that array, and the same gap appears in the
4096² mask-vec Add row above. So roughly half of mask's 4096² gap is an
artifact of this setup. The part that belongs to the representation is
the mask pass (about 0.07 ns/cell), partly offset by the mask not
needing NaN's centre term (0.02–0.05 ns/cell).

### Ingest (fill value → in-memory representation, scalar)

| variant | size | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---|---:|---:|---:|---:|
| to-nan | 1024² | 0.485 (2060) | 0.610 (1639) | 3.05 (328) | 0.613 (1632) |
| to-mask | 1024² | 1.35 (742) | 1.35 (740) | 1.35 (742) | 1.34 (744) |
| to-nan | 4096² | 0.778 (1286) | 0.851 (1176) | 3.74 (267) | 0.858 (1165) |
| to-mask | 4096² | 1.38 (726) | 1.38 (725) | 1.38 (726) | 1.38 (725) |

`to-mask` is branch-free scalar Go, so its cost does not depend on the
data. It is an obvious 8-lane `Equal` + `ToBits` target when IO
throughput matters. `to-nan` branches.

### Memory overhead

| raster | Data | sentinel | NaN | mask | mask as % of Data |
|---|---:|---:|---:|---:|---:|
| 1024² float32 | 4 MiB | 4 B (field) | 0 | 128 KiB | 3.125% |
| 4096² float32 | 64 MiB | 4 B | 0 | 2 MiB | 3.125% |
| 16384² float32 | 1 GiB | 4 B | 0 | 32 MiB | 3.125% |
| any uint8 / int16 raster | | 1–2 B | n/a | | 12.5% / 6.25% |

A nil mask (no NoData) costs nothing.

## What the numbers say

- **Vectorized pointwise ops: sentinel pays a visible compare+blend, the
  mask doesn't.** At 1024² Add is 0.18–0.19 ns/cell for NaN and mask, and
  0.22 for sentinel-blend (+18–25%). archsimd's `IfElse` lowers to a
  byte-wise `VPBLENDVB`. At 4096² all three are memory-bandwidth-bound
  (§28) at 0.55–0.68 ns/cell. There the mask row's extra 0.12 is mostly
  the input-array placement effect described above; the AND pass itself
  is 16 k words (about 6 µs, 0.005 ns/cell at 1024²).
- **For stencils the compare cost shows more.** Sentinel-blend slope
  costs +14–24% (1024²) and +17–19% (4096²) over NaN: nine compares, eight
  ORs and a blend per lane. The mask is +0–9% (1024²) and +14–20% (4096²,
  about half of it memory placement). The mask pass is a fixed per-cell
  cost (about 0.07 ns/cell, 64 cells per word), independent of stencil
  size, and it can be skipped entirely when all inputs are nil.
- **The straightforward scalar forms are where `if value == NoData`
  hurts.** Branchy sentinel Add falls from 1.0–1.3 to 5.0–5.6 ns/cell at
  30% scattered NoData: branch mispredictions, a 4–5× slowdown the
  vectorized forms don't show. The mask-branchy slope (per-cell bit
  tests, computing only valid cells) is the slowest variant of all,
  6.5–10 ns/cell, because skipping work does not pay for nine bit
  extractions and a mispredicted branch. Kernels must compute
  unconditionally and fix validity separately, whatever the
  representation. The mask makes that the natural way to write them.
- **"Compute then fix up" with a scalar fix pass doesn't help sentinel**
  (`vec+fixup`, 0.9–5.8 ns/cell): the fix pass is the branchy loop again.
- **Materialising a fill value is density-dependent** (`+fill` rows, up
  to +1.2 ns/cell at 30% scattered). It is paid once at export, not per
  op. Our `FillInvalid` walks set bits; a vectorized blend would flatten it.
- **Relative to the earlier hand-written-assembly run** (Go 1.25), the
  archsimd NaN and mask slope kernels are about 10–17% slower at 1024²
  and about the same at 4096². `vec.Add` is about 30% slower at 1024²
  and about the same at 4096². That matches ADR 0001. None of the
  rankings changed.

### archsimd pitfall found in this spike

The first archsimd sentinel slope kernel ran at **11–12 ns/cell**, slower
than scalar, although its instructions were the expected
`VCMPPS`/`VPOR`/`VPBLENDVB`. The disassembly showed the cause: the
float32 parameters were still needed by the scalar tail after the loop,
so the compiler spilled one to the stack. It then reloaded it with a
**legacy-SSE `MOVSS` inside the AVX loop** on every iteration, paying an
SSE/AVX transition each time (golang/go#80835, listed in ADR 0001). The
NaN and mask kernels escaped only because of different register
allocation.

The fix used in `simd_amd64.go` is to split each kernel. A `*Lanes`
function runs the vector loop, calls `ClearAVXUpperBits`, and returns
how many cells it wrote. The caller runs the scalar tail with the float
parameters. That brought the sentinel slope to 0.59 ns/cell. This is
worth adding to ADR 0001's kernel-writing rules, since `internal/vec`'s
single-function kernels are exposed to the same failure.

## Correctness hazards

`hazards_test.go` pins each of these down as an executable test.

**Sentinel**

- *Collisions from arithmetic:* `-10000 + 1 == -9999`, so a valid
  result becomes NoData. Every op that can land on the sentinel corrupts
  validity silently.
- *Zero sentinels also match `-0`*, because `-0 == +0`.
- *Metadata round-trip:* fill values arrive as text or float64
  (`GDAL_NODATA` is an ASCII TIFF tag; Zarr `fill_value` is JSON). A
  value inexact in float32 (e.g. `1e-9`) only matches if the adapter
  rounds exactly as the writer did.
- *A NaN fill value never matches `==`,* so sentinel code quietly treats
  every NoData cell as valid. The test shows a NaN centre cell producing
  a valid-looking slope.
- Every kernel must implement propagation; forgetting it produces
  plausible numbers, not errors.

**NaN**

- *Legitimate NaN:* `0/0`, `sqrt(-1)` and `Inf-Inf` create NaN during
  computation, and those become indistinguishable from missing input.
- *No propagation through cells that aren't read.* The Horn stencil
  never reads z5, so a NaN centre gives a valid output. The spike's own
  correctness test caught this. The NaN variants need an explicit
  `+ z5·0` term, which also turns a valid ±Inf centre into NaN.
  Aspect, curvature and any stencil with zero-weight taps have the same
  trap.
- *Comparisons swallow NaN:* `slope > 0.5` is `false`, so NoData is
  classified "not steep". Thresholds, reclassification, `Select` and
  `Compare` (§16 future ops) all need explicit NaN handling.
- *Integer rasters have no NaN* (uint8 land cover, int16 SRTM with
  `-32768`, uint16 counts). A NaN design needs a second mechanism for
  them anyway, and `int32(NaN)` on amd64 gives `-2147483648`, a
  valid-looking integer.
- `internal/vec` Min/Max/Clamp propagate NaN. That's consistent with
  NaN-as-NoData, but it fixes the `fmax`-style "ignore the missing
  operand" semantics out of the API.

**Mask**

- *None of the value-level hazards:* every bit pattern, including NaN,
  `-0` and `-9999`, is a valid value (`TestMaskAcceptsEveryValue`).
- *It can go stale:* code that writes `Data` without updating `Valid`,
  or leaves bits set past `len(Data)`, is wrong. The spike's tests
  poison the output mask to catch exactly that. STRATA-4 should add an
  invariant check helper.
- *Two slices to keep in sync* for views, copies and halos.

## Interoperability

- **GeoTIFF/GDAL:** a single per-dataset NoData value (`GDAL_NODATA`,
  ASCII) is typical: `-9999`, `-32768`, `-3.4028235e38`, `0` or `nan`.
  GDAL also has per-dataset 1-bit mask bands (internal `.msk`), which
  map directly onto `Valid`. On read, NoData value, mask band and
  alpha all become `Valid`. On write, `FillInvalid` with the dataset's
  NoData value, or emit a mask band.
- **Zarr / CF / xarray:** `fill_value` (`"NaN"` allowed for floats) plus
  CF `_FillValue`, `missing_value` and `valid_min`/`valid_max`/
  `valid_range`. Several markers at once, and range-based validity, fold
  naturally into one mask. With NaN they would require overwriting data,
  and with a sentinel they would be lossy.
- **Arrow:** validity bitmaps use the same convention (1 = valid,
  LSB-first bit order). Our little-endian `uint64` words are
  byte-compatible, which leaves a path to zero-copy exchange.
- NaN and sentinel both need the adapter to convert anyway, because
  source fill values vary. None of the representations avoids adapter
  work, but only the mask stays the same for every source dtype.

## Tiles, halos and operation fusion

- **Halos:** out-of-raster halo cells are just zero bits, so edges
  become NoData with no per-kernel border code. NaN padding gives the
  same for float types only; sentinel padding needs every kernel to
  check. The cost is that halo construction copies bit ranges too
  (`extractBits`/`depositBits` in `bits.go`, correctness-tested at
  arbitrary offsets). With `Stride % 64 == 0` those are plain word copies.
- **Skipping work:** a mask exposes sparsity cheaply. All-zero words
  (clustered voids, ocean) or all-zero tiles can skip compute, and nil
  marks all-valid tiles. NaN can't show that without scanning the data.
- **Fusion (§29):** with a mask, the fused data kernel is the same
  branch-free arithmetic NaN would use (the easiest thing to generate
  and vectorize). Validity is a separate stream computed *once* for the
  whole chain: AND of the leaf masks for pointwise chains, independent
  of chain length, and composite-footprint erosion for stencils.
  Sentinel pays compare+blend per leaf input in the fused kernel. NaN is
  free only while every op in the chain propagates NaN, so a fusion
  planner would need per-op NaN-propagation knowledge (comparisons,
  selects, integer casts and zero-tap stencils all break it).
- **Ops that should create NoData** (divide by zero, log of ≤0, out-of-
  domain) don't do so implicitly with a mask; IEEE results stay valid
  values. If needed, add an explicit opt-in op (`InvalidateNonFinite`)
  that costs one compare pass. That is a deliberate policy decision for
  STRATA-4 rather than an accident of representation.

## Caveats

- One machine (Zen 2, AVX2, 64 GB). Benchmarks with AVX-512 masks
  (`k` registers make blend-style sentinel handling cheaper) and on
  arm64/NEON are not covered.
- Single-threaded only. Mask handling is per-tile-local and should
  scale with workers exactly like the data path, but that was not
  measured.
- Both benchmark widths (1024, 4096) are multiples of 64, so mask row
  offsets were word-aligned. Odd widths are covered by the correctness
  tests only, and their extract/deposit shifting costs more. Hence the
  recommendation to pad `Stride` to 64.
- The archsimd slope and sentinel-blend kernels in `simd_amd64.go` are
  spike code, not `internal/vec` kernels, and `FillInvalid`/
  `MaskFromSentinel` are scalar. None of this is tuned production code;
  the comparisons are like-for-like within each form.
- The memory-placement probe was taken on the earlier assembly build,
  not re-run under archsimd. Memory placement moved single 4096²
  results by about 10%, so treat 4096² differences below that as noise.
