# The zarr adapter's chunk cache: results

What the [`zarr`](../../zarr/) module's decoded-chunk cache
([ADR 0003](../../docs/adr/0003-zarr-adapter.md)) buys when the engine
reads a Zarr array a tile and a halo at a time. Zarr decodes a chunk
whole, however little of it a read needs, and neighbouring tiles need the
same chunks, so without a cache each decodes them again.

## Headline

- **With the cache every chunk is decoded exactly once, in every case;
  without it, 2.875 times (256² chunks) or 3.75 times (512²).** The
  engine's 256-row strips, with their one-row halos, touch three rows of
  256² chunks each, and two strips share every row of 512² chunks.
- **On one core, slope is 2.5× faster with the cache on 256² chunks and
  3.0× on 512²**: 1049 → 425 ms and 1407 → 470 ms over a 4096² float32
  DEM, gzip level 5. The focal mean (11 × 11) gains the same: 2.4× and
  2.9×.
- **With the default read concurrency (8 chunks of a window at once),
  the cache gives 1.8–1.9× on one engine worker and 2.3–2.4× on four.**
  Slope on four workers: 307 → 126 ms (256²), 398 → 172 ms (512²).
- **A window the cache holds costs 0.1 µs for a pixel, 0.3 µs for a 3 × 3
  across four chunks, and 14 µs for 256² across four**, against
  1.2–12 ms decoded each time: the costs the library's own benchmarks
  report, which this cache exists to pay once.
- **Decoding still costs more than the computation.** Slope over the
  same cells in memory takes 107 ms on one worker and 39 ms on four;
  from Zarr with the cache, 224 ms and 126 ms. gzip is the rest.
- **Load chunks concurrently.** With one chunk at a time (`conc=1`) and
  512² chunks, four engine workers run slope only 1.25× faster than one
  (470 → 377 ms): consecutive strips need the same chunk row, so while
  one worker decodes a row serially, the others wait for it in the
  cache. Loading a window's chunks concurrently, the default, gets
  172 ms.

## Machine and method

| | |
|---|---|
| Machine | a cloud container: 4 vCPUs of an Intel Xeon at 2.10 GHz, 15 GiB, Linux |
| Go | go1.27.0 linux/amd64, default build (no `GOEXPERIMENT=simd`), `GOMAXPROCS=4` |
| Raster | a synthetic 4096 × 4096 float32 DEM, smooth with noise, about 1% NoData in holes, fill value −9999 |
| Store | `zarrv3.DirStore` in a temporary directory (in the page cache after the first read), gzip level 5, chunks of 256² or 512² |
| Engine | `engine.Options{TileHeight: 256}`, full-width strips; 1 and 4 workers; output to a `MemorySink` |
| Source | a new `zarr.Source` for every run, so every run starts with an empty cache; `CacheBytes` 0 (the default: its 64 MiB floor for 256² chunks, 8 rows or 66 MiB for 512²) or −1 (none); `ReadConcurrency` 1 or 8 (the default) |
| Runs | `-count 5`, each at least 1 s (`b.Loop`); medians |

These numbers come from a shared cloud machine, not the desktop the other
pages under `benchmarks/` use, and they are not comparable with them.
Run to run ((max − min) / median of the five), 36 of the 64 cases vary
by under 10% and the worst by 29%, so a difference under about 30% in
one pair of cells should not be read as real. The cache's effects, 1.8×
to 3.0× and the decode counts, are larger than that, and the decode
counts are exact.

```bash
go test -run '^$' -bench . -count 5 -timeout 2h > testdata/bench.txt
python summarize.py testdata/bench.txt   # the tables below
```

`decodes/chunk` is the chunk reads the store served, over the number of
chunks in the raster and the runs: 1.00 is every chunk read and decoded
once. `memory` is the same operation from a `MemorySource` of the same
cells: the floor under the format.

## The numbers

<!-- summarize.py output begin -->

#### Slope

| chunk | cache | conc | 1 worker: ms | 4 workers: ms | decodes/chunk |
| ---: | --- | ---: | ---: | ---: | ---: |
| memory | | | 107.1 | 38.6 | |
| 256 | on | 1 | 425.4 | 221.2 | 1.000 |
| 256 | on | 8 | 223.6 | 126.0 | 1.000 |
| 256 | off | 1 | 1049.0 | 327.9 | 2.875 |
| 256 | off | 8 | 419.1 | 307.2 | 2.875 |
| 512 | on | 1 | 470.4 | 376.8 | 1.000 |
| 512 | on | 8 | 270.8 | 172.4 | 1.000 |
| 512 | off | 1 | 1407.0 | 389.8 | 3.750 |
| 512 | off | 8 | 489.2 | 397.5 | 3.750 |

#### Mean

| chunk | cache | conc | 1 worker: ms | 4 workers: ms | decodes/chunk |
| ---: | --- | ---: | ---: | ---: | ---: |
| memory | | | 140.4 | 50.1 | |
| 256 | on | 1 | 469.3 | 251.2 | 1.000 |
| 256 | on | 8 | 254.6 | 142.7 | 1.000 |
| 256 | off | 1 | 1117.3 | 304.4 | 2.875 |
| 256 | off | 8 | 409.5 | 293.2 | 2.875 |
| 512 | on | 1 | 501.6 | 337.4 | 1.000 |
| 512 | on | 8 | 264.6 | 153.9 | 1.000 |
| 512 | off | 1 | 1465.1 | 401.6 | 3.750 |
| 512 | off | 8 | 532.6 | 397.4 | 3.750 |

#### Reading the whole raster in 256-row strips with 1-row halos

| chunk | cache | conc | 1 worker: ms | 4 workers: ms | decodes/chunk |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | on | 1 | 357.5 | 190.5 | 1.000 |
| 256 | on | 8 | 125.9 | 101.7 | 1.000 |
| 256 | off | 1 | 1071.0 | 303.9 | 2.875 |
| 256 | off | 8 | 382.3 | 284.2 | 2.875 |
| 512 | on | 1 | 391.1 | 311.0 | 1.000 |
| 512 | on | 8 | 136.4 | 109.4 | 1.000 |
| 512 | off | 1 | 1560.9 | 405.8 | 3.750 |
| 512 | off | 8 | 479.6 | 389.7 | 3.750 |

#### One window, read again and again

| chunk | window | cache off | cache on (warm) | off / on |
| ---: | --- | ---: | ---: | ---: |
| 256 | pixel | 1.22 ms | 0.10 µs | 12,040× |
| 256 | 3x3-corner | 3.43 ms | 0.31 µs | 11,079× |
| 256 | 256x256-4chunks | 3.50 ms | 14.09 µs | 248× |
| 512 | pixel | 6.07 ms | 0.11 µs | 57,357× |
| 512 | 3x3-corner | 11.96 ms | 0.28 µs | 42,467× |
| 512 | 256x256-4chunks | 11.38 ms | 13.97 µs | 815× |

<!-- summarize.py output end -->

## What this does not tell you

- **Other codecs.** Only gzip was timed. With zstd, or none, a decode is
  cheaper and the cache matters less on one core, though never less
  than the decodes/chunk ratio says it saves.
- **Object storage.** The store was a local directory in the page cache.
  Over HTTP each chunk read is a request, and the cache saves requests as
  well as decodes.
- **Sharded arrays**, and several sources over planes of one array, which
  share no cache (ADR 0003, "Known limits").
- **A cache smaller than a row of chunks.** As cog's page shows, an LRU
  smaller than the chunks one band of tiles needs evicts each chunk
  before the next tile comes back for it; the default is 8 rows.
