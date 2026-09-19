# Band shape: results (DESIGN.md §53)

The measurement that decided whether the engine should shape a band in two
dimensions or keep it a whole tile row — the lever §51 left open, that the
IO tile and the compute tile were one parameter.

`BenchmarkBandWidth` sweeps `Options.ComputeWidth` over one full-width
tile, so every case reads and writes the same cells through the same
sources with the same number of source calls, and only the rectangle one
kernel call covers changes.

## Not comparable with the other results in this directory

This ran on a different machine from every other figure in
`benchmarks/`, which are all from the 12-core Zen 2 box described in
[`RESULTS.md`](RESULTS.md). **No absolute number here — ns/op, cells/s,
GB/s — may be compared with a number from any other results file.**
Different CPU, different cache hierarchy, different OS, and the SIMD path
is `amd64` only (DESIGN.md §17), so this is the scalar backend where those
runs are AVX2.

What survives that, and is the reason a conclusion can be drawn at all:

- **The traffic figures are counted, not timed.** `engine.Stats` is
  deterministic arithmetic over the plan (§51), so the halo and B/cell
  columns below are reproducible on any machine and do reproduce §51's
  table. They are a property of the shape, not of the hardware.
- **The timing comparison is within one run.** Every row compares shapes
  against each other in the same binary on the same machine, interleaved,
  not against a figure recorded elsewhere. That is what makes the
  conclusion a measurement rather than a mismatch.

Anything that would need cross-machine comparison — whether shaping the
band is worth it *on the Zen 2 box*, or how these absolute rates relate to
§27's 770–845 M cells/s — this run does not answer and cannot.

## Machine and method

| | |
|---|---|
| CPU | Apple M4, 10 cores (4P + 6E), 128 KiB L1d per P core |
| OS | Darwin 25.6.0 |
| Go | go1.27.0 darwin/arm64. `GOEXPERIMENT=simd` set, but the build is **scalar**: the SIMD path is `simd_amd64.go` only (DESIGN.md §17) |
| Run | `GOEXPERIMENT=simd go test ./benchmarks/engine -run '^$' -bench BandWidth -count 6`, not pinned |
| Raw output | [`testdata/bandwidth.txt`](testdata/bandwidth.txt) |

## Traffic, by band width

Counted, so machine-independent. The shapes are bands of the given width
inside one full-width tile; `rows` is a band as wide as the tile, which is
what the engine planned before §53.

| | rows | 2048 | 1024 | 512 | 256 |
|---|---:|---:|---:|---:|---:|
| halo, 4096² | 12.4% | 6.3% | 3.2% | 1.9% | 1.5% |
| halo, 16384² | 50.0% | 6.3% | 3.3% | 1.9% | 1.5% |
| B/cell, 16384² | 10.00 | 8.25 | 8.13 | 8.08 | 8.06 |

A whole-row band of a 16384-wide tile is four rows tall, so half of what
its kernel reads is halo, and a fifth of everything the call moves.
Shaping the band to 1024×64 removes it. §51 does not count validity, so
these do not depend on the mask.

## Time, by band width

Slope, median of 6 runs (12 for 1024, which the sweep ran twice under two
names), per cent against whole-row bands **in the same run**:

| Slope, ms | rows | 2048 | 1024 | 512 | 256 |
|---|---:|---:|---:|---:|---:|
| 4096², 1 worker | 52.2 | +2% | +2% | +5% | +13% |
| 4096², 12 workers | 8.3 | +6% | +4% | +10% | +17% |
| 4096², masked, 1 worker | 51.9 | +3% | +3% | +6% | +15% |
| 4096², masked, 12 workers | 8.5 | +14% | +23% | +24% | +23% |
| 16384², 1 worker | 847.8 | −2% | −1% | +4% | +12% |
| 16384², 12 workers | 140.8 | +2% | +3% | +5% | +10% |
| 16384², masked, 1 worker | 911.4 | −7% | −6% | +0% | +17% |
| 16384², masked, 12 workers | 151.0 | −3% | +3% | +12% | +21% |

Run-to-run spread, measured the same way: the 1024×64 shape ran twice
under two names, and the two medians land 0.5–1.4% apart at 4096² and
0.6–4.8% apart at 16384². The masked 12-worker case at 4096² is not
readable at all — those two landed 30% apart.

## What the numbers say

- **The halo behaves exactly as the arithmetic predicts.** A band of
  `bandW × bandH` reads `(bandW+2r)(bandH+2r)`, so a thin band is nearly
  all perimeter. This is not a rounding error at 16384 wide: a fifth of
  everything the call moves is re-read halo.
- **Removing it buys no time.** At 16384² the two shapes run at 847.8 ms
  and 842.6 ms while moving 10.00 and 8.13 bytes per cell — a 19%
  difference in traffic worth nothing either way, inside the spread
  above. At 4096² every shaped case is slower. Only 16384² masked on one
  worker gains, and its 12-worker mirror does not.
- **Below 1024-cell rows the fixed per-row cost dominates**: 512 and 256
  are 4–24% worse across the board, monotonically, whatever halo they
  read. This is measured here, on this machine, and does not rest on the
  Zen 2 run's 256×256 tile figures — though those found the same shape of
  curve, which is worth noting and is not evidence.
- The conclusion is §51's own caveat holding for the change §51 proposed:
  these are bytes the engine moves between stages, not bytes that reach
  memory. Neighbouring full-width bands share halo *rows* and run
  back-to-back in plan order, so the re-read was already an L2 hit;
  shaping the band turns it into shared halo *columns*, also an L2 hit.

So bands stay whole tile rows. The mechanism, `ComputeWidth` and
`ComputeHeight` are in, and `minBandWidth` is the one constant to restore
to turn it on. **This should be rerun on the Zen 2 box before the result
is treated as settled** — a quarter of the L1d, a different prefetcher,
and the AVX2 kernels rather than the scalar ones all push on exactly the
two costs being traded. DESIGN.md §53 says what result would earn the
default.
