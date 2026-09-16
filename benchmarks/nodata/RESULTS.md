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
   `Valid` on read (`MaskFromSentinel`, about 1.25 ns/cell scalar today).
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

Why, in one paragraph: vectorized, the mask costs the same as NaN on
Add and 0–21% more on the 3×3 slope. Sentinel costs 25–35% more on the
slope. Only the mask has none of the correctness hazards below. It works
unchanged for integer rasters, which have no NaN. It carries validity
through fused pipelines (§20) as one cheap side pass, not per-op logic.
NaN is the fastest option but not by enough to justify its hazards: it
cannot tell missing apart from computed NaN, it silently fails to
propagate through stencils that skip a cell and through comparisons,
and it does not exist for `uint8`/`int16` sources.

## Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 9 3900X, 12C/24T, Zen 2, AVX2 (no AVX-512) |
| Memory | 64 GB DDR4-3200 |
| OS | Windows 11 Home 10.0.22631, power plan "AMD Ryzen High Performance" |
| Go | go1.25.3 windows/amd64, `GOAMD64=v1` |
| Run | `go test -c`, then `nodata.test.exe -test.run '^$' -test.bench . -test.benchmem -test.count 6`, pinned to one logical CPU (affinity `0x10`), `GOMAXPROCS=1`, High priority |
| Stats | median of 6 runs, each ≥1 s (`b.Loop`) |

To reproduce the tables:

```
go test ./benchmarks/nodata -run '^$' -bench . -benchmem -count 6 -timeout 3h > bench.txt
go run ./benchmarks/nodata/cmd/nodatatable < bench.txt
```

This was a desktop with other applications open. Most cells vary 2–8%
between runs, with a few outliers noted below. Differences under about
10% should not be read as real.

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
- Go's gc compiler does not auto-vectorize, so "scalar" really is
  scalar. The vectorized forms use `internal/vec` (AVX2) for Add and a
  fused AVX2 Horn kernel written for this spike (`avx2_amd64.s`).

Variant names:

| suffix | meaning |
|---|---|
| `scalar-branchy` | the obvious per-cell `if` |
| `scalar-select` | compute unconditionally, OR the compare flags, then one select |
| `vec+fixup` | `vec.Add`, then a scalar pass rewriting NoData cells |
| `avx2-blend` | compute + `VCMPPS` against the sentinel + `VBLENDVPS`, 8 lanes |
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
| sentinel-scalar-branchy | 1.07 (939) | 1.19 (844) | 5.05 (198) | 1.38 (723) |
| sentinel-scalar-select | 1.83 (547) | 2.02 (496) | 6.94 (144) | 1.99 (503) |
| sentinel-vec+fixup | 0.931 (1074) | 1.16 (859) | 5.39 (186) | 1.18 (847) |
| sentinel-avx2-blend | 0.134 (7443) | 0.146 (6861) | 0.136 (7348) | 0.135 (7391) |
| nan-scalar | 0.327 (3057) | 0.304 (3290) | 0.294 (3404) | 0.295 (3386) |
| nan-vec | 0.145 (6897) | 0.133 (7539) | 0.137 (7313) | 0.138 (7246) |
| mask-scalar | 0.597 (1674) | 0.525 (1906) | 0.555 (1803) | 0.514 (1945) |
| mask-vec | 0.148 (6773) | 0.154 (6491) | 0.243 (4123)¹ | 0.133 (7513) |
| mask-vec+fill | 0.160 (6244) | 0.281 (3553) | 1.28 (780) | 0.826 (1210) |

¹ This is the run's worst spread (75% between runs). The mask pass here
is 16 k ANDed words, measured separately at about 5.7 µs (0.005
ns/cell), so this cell is noise.

### Add, 4096×4096 (memory-bandwidth-bound: 192 MiB working set)

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 1.34 (746) | 1.48 (676) | 5.63 (178) | 1.45 (689) |
| sentinel-scalar-select | 1.92 (522) | 2.17 (461) | 7.72 (130) | 2.05 (487) |
| sentinel-vec+fixup | 1.34 (745) | 1.73 (578) | 5.73 (175) | 2.04 (490) |
| sentinel-avx2-blend | 0.572 (1747) | 0.556 (1798) | 0.585 (1711) | 0.601 (1663) |
| nan-scalar | 0.578 (1730) | 0.580 (1723) | 0.599 (1670) | 0.640 (1561) |
| nan-vec | 0.542 (1844) | 0.569 (1758) | 0.546 (1833) | 0.579 (1726) |
| mask-scalar | 0.808 (1237) | 0.826 (1210) | 0.814 (1228) | 0.906 (1104) |
| mask-vec | 0.567 (1765) | 0.571 (1752) | 0.585 (1709) | 0.635 (1574) |
| mask-vec+fill | 0.586 (1707) | 0.794 (1260) | 1.34 (745) | 1.42 (704) |

### Slope (3×3 Horn), 1024×1024

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 4.54 (221) | 4.56 (219) | 6.52 (153) | 3.63 (275) |
| sentinel-scalar-select | 7.69 (130) | 8.09 (124) | 8.26 (121) | 7.75 (129) |
| sentinel-avx2-blend | 0.601 (1665) | 0.589 (1699) | 0.586 (1707) | 0.575 (1738) |
| nan-scalar | 3.59 (278) | 3.02 (331) | 3.13 (320) | 2.99 (335) |
| nan-avx2 | 0.450 (2224) | 0.428 (2338) | 0.429 (2330) | 0.420 (2382) |
| mask-scalar-branchy | 9.83 (102) | 9.82 (102) | 6.90 (145) | 8.79 (114) |
| mask-scalar | 2.71 (369) | 2.71 (369) | 2.72 (368) | 2.69 (372) |
| mask-avx2 | 0.454 (2202) | 0.452 (2215) | 0.452 (2212) | 0.448 (2230) |
| mask-avx2+fill | 0.471 (2125) | 0.671 (1491) | 1.66 (601) | 0.896 (1116) |

### Slope (3×3 Horn), 4096×4096

| variant | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---:|---:|---:|---:|
| sentinel-scalar-branchy | 5.33 (188) | 5.08 (197) | 6.55 (153) | 4.05 (247) |
| sentinel-scalar-select | 7.70 (130) | 8.13 (123) | 8.15 (123) | 7.61 (131) |
| sentinel-avx2-blend | 0.674 (1484) | 0.659 (1517) | 0.661 (1513) | 0.648 (1543) |
| nan-scalar | 3.03 (330) | 3.01 (332) | 3.00 (333) | 3.00 (334) |
| nan-avx2 | 0.530 (1888) | 0.526 (1901) | 0.533 (1877) | 0.534 (1873) |
| mask-scalar-branchy | 10.6 (94.7) | 10.4 (96.4) | 7.13 (140) | 9.32 (107) |
| mask-scalar | 3.48 (287) | 3.44 (291) | 3.49 (286) | 3.46 (289) |
| mask-avx2 | 0.654 (1529) | 0.632 (1582) | 0.639 (1566) | 0.642 (1558) |
| mask-avx2+fill | 0.661 (1514) | 0.982 (1019) | 1.85 (540) | 1.18 (846) |

**Breaking down the 4096² slope gap** (follow-up probe, 30% clustered,
same pinning, 3 runs):

| measurement | 1024² | 4096² |
|---|---:|---:|
| `SlopeMask` pass alone | 0.078 | 0.072 |
| AVX2 rows without centre term, on the `-9999` array | 0.364 | 0.568 |
| AVX2 rows without centre term, on the NaN array | 0.380 | 0.509 |
| AVX2 rows with `z5·0` centre term, on the `-9999` array | 0.409 | 0.591 |
| AVX2 rows with `z5·0` centre term, on the NaN array | 0.427 | 0.529 |

At 4096² the *same kernel* runs about 11% slower over the `-9999` input
array than over the NaN one. That comes from where the array landed in
memory (TLB/prefetch behaviour of that allocation), not from the values.
At 1024² the order is reversed. So about half of the mask-vs-NaN slope
gap at 4096² is an artifact of this setup. The part that belongs to the
representation is the mask pass (0.07 ns/cell), partly offset by the
mask not needing NaN's centre term (0.02–0.05 ns/cell).

### Ingest (fill value → in-memory representation, scalar)

| variant | size | 0% | 1% scattered | 30% scattered | 30% clustered |
|---|---|---:|---:|---:|---:|
| to-nan | 1024² | 0.481 (2081) | 0.607 (1648) | 2.94 (340) | 0.604 (1655) |
| to-mask | 1024² | 1.23 (815) | 1.22 (817) | 1.22 (819) | 1.22 (818) |
| to-nan | 4096² | 0.726 (1377) | 0.874 (1144) | 3.64 (275) | 0.887 (1127) |
| to-mask | 4096² | 1.26 (796) | 1.28 (779) | 1.28 (781) | 1.27 (788) |

`to-mask` is branch-free scalar Go, so its cost does not depend on the
data. It is an obvious 8-lane `VCMPPS`+`VMOVMSKPS` target when IO
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

- **Vectorized, representation barely matters for pointwise ops.** Add
  runs at about 0.14 ns/cell at 1024² and about 0.55–0.6 ns/cell at
  4096² for sentinel-blend, NaN and mask alike. At 4096² all three are
  memory-bandwidth-bound (§19): the extra compare+blend or the 1/32-size
  mask stream is lost in the cost of moving 192 MiB. `VCMPPS`+`VBLENDVPS`
  on two inputs is nearly free next to that.
- **For stencils the compare cost shows.** Sentinel-blend slope costs
  +33% (1024²) and +25% (4096²) over NaN: nine compares, eight ORs and a
  blend per lane. The mask is +1% (1024²) and +21% (4096², about half of it memory placement, see the probe). It is a fixed per-cell cost
  (about 0.07 ns/cell, 64 cells per word), independent of stencil size,
  and it can be skipped entirely when all inputs are nil.
- **The straightforward scalar forms are where `if value == NoData`
  hurts.** Branchy sentinel Add falls from 1.1 to 5.1–5.6 ns/cell at 30%
  scattered NoData: branch mispredictions, a 4–5× slowdown the vectorized
  forms don't show. The mask-branchy slope (per-cell bit tests, computing
  only valid cells) is the slowest variant of all, 6.9–10.6 ns/cell,
  because skipping work does not pay for nine bit extractions and a
  mispredicted branch. Kernels must compute unconditionally and fix
  validity separately, whatever the representation. The mask makes that
  the natural way to write them.
- **"Compute then fix up" with a scalar fix pass doesn't help sentinel**
  (`vec+fixup`, 0.9–5.7 ns/cell): the fix pass is the branchy loop again.
- **Materialising a fill value is density-dependent** (`+fill` rows, up
  to +1.2 ns/cell at 30% scattered). It is paid once at export, not per
  op. Our `FillInvalid` walks set bits; a vectorized blend would flatten it.

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
  `Compare` (§9 future ops) all need explicit NaN handling.
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
- **Fusion (§20):** with a mask, the fused data kernel is the same
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
- The AVX2 Horn kernels are spike code in this package, not
  `internal/vec` kernels, and `FillInvalid`/`MaskFromSentinel` are
  scalar. None of this is tuned production code; the comparisons are
  like-for-like within each form.
- Memory placement moved single 4096² results by about 10% (see the
  probe table), so treat 4096² differences below that as noise.
