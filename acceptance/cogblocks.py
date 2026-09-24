#!/usr/bin/env python3
"""Set the HTTP requests strata made reading each file against the blocks
GDAL says the file holds.

Runs after `go run ./cog -url ...` (coghttpcheck.sh), in the GDAL
container. For every file in manifest.json it counts, through GDAL's
BLOCK_OFFSET_x_y metadata of every band of every level:

  - blocks:       the distinct blocks stored in the file (sparse ones,
                  offset 0, are not stored, and GDAL reports none);
  - block reads:  the blocks strata's sources read to read every band of
                  every level once: a Source reads one band, so a
                  pixel-interleaved block is read once per band;
  - fetches:      the blocks that end past the prefetched header, each of
                  which should cost one range request. The sources of a
                  pixel-interleaved file share its compressed blocks, so
                  a block is fetched once however many bands read it. A
                  COG stores its smallest overviews first, so their
                  blocks often lie inside the prefetch and cost none.

and prints them next to strata.json's requests and bytes. "other" is what
the requests spent beyond the prefetch and one request per fetch: IFD and
tag reads outside the prefetched header, and any retries or re-reads. It
fails if any file was not read over HTTP, so a strata.json left over from
a local read cannot pass for this one.

GDAL reads a single uncompressed strip as virtual strips of a few rows,
and reports those as its blocks (GDAL_ENABLE_TIFF_SPLIT=NO does not change
that in GDAL 3.14), so gtiff-Float32-onestrip, one strip of 1.4 MB, is
printed with a note and left out of the totals. cog fetches that strip in
1 MiB pieces (tiff.go, readChunk), so its reading takes 3 requests.

Usage:  python3 cogblocks.py [dir]
"""

import json
import os
import sys

from osgeo import gdal

gdal.UseExceptions()

# Files whose blocks GDAL does not report as they are stored.
VIRTUAL = {"gtiff-Float32-onestrip": "one strip, which GDAL reports as virtual strips"}

D = sys.argv[1] if len(sys.argv) > 1 else "out-cog"
gdal_side = json.load(open(os.path.join(D, "manifest.json")))
strata_side = {r["name"]: r for r in json.load(open(os.path.join(D, "strata.json")))}


def count_blocks(path, prefetch):
    ds = gdal.Open(path)
    reads, stored = 0, {}
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        for lv in [band] + [band.GetOverview(k) for k in range(band.GetOverviewCount())]:
            bw, bh = lv.GetBlockSize()
            for y in range((lv.YSize + bh - 1) // bh):
                for x in range((lv.XSize + bw - 1) // bw):
                    off = lv.GetMetadataItem(f"BLOCK_OFFSET_{x}_{y}", "TIFF")
                    if off:
                        size = int(lv.GetMetadataItem(f"BLOCK_SIZE_{x}_{y}", "TIFF"))
                        reads += 1
                        stored[int(off)] = size
    fetches = sum(off + size > prefetch for off, size in stored.items())
    return len(stored), reads, fetches


print(f"{'file':<46} {'KiB':>6} {'blocks':>6} {'reads':>6} {'fetches':>7} {'requests':>8} {'other':>5} {'fetched':>7}")
failed = 0
tot = dict(files=0, blocks=0, reads=0, fetches=0, requests=0, other=0, bytes=0, size=0)
for g in gdal_side["files"]:
    name = g["name"]
    s = strata_side.get(name, {})
    h = s.get("http")
    if not h:
        failed += 1
        print(f"{name:<46} not read over HTTP: {s.get('error', 'missing from strata.json')}")
        continue
    if name in VIRTUAL:
        print(f"{name:<46} {h['size'] / 1024:>6.0f} {VIRTUAL[name]}: {h['requests']} requests, not counted")
        continue
    blocks, reads, fetches = count_blocks(os.path.join(D, name + ".tif"), h["prefetch"])
    other = h["requests"] - 1 - fetches
    print(f"{name:<46} {h['size'] / 1024:>6.0f} {blocks:>6} {reads:>6} {fetches:>7} {h['requests']:>8} {other:>5} "
          f"{h['bytes'] / h['size']:>7.0%}")
    tot["files"] += 1
    tot["blocks"] += blocks
    tot["reads"] += reads
    tot["fetches"] += fetches
    tot["requests"] += h["requests"]
    tot["other"] += other
    tot["bytes"] += h["bytes"]
    tot["size"] += h["size"]

n = len(gdal_side["files"])
print(f"\n{n - failed}/{n} files read over HTTP. The {tot['files']} counted: "
      f"{tot['requests']:,} requests for {tot['blocks']:,} blocks "
      f"({tot['reads']:,} block reads, {tot['fetches']:,} blocks past the prefetch) = "
      f"{tot['files']:,} prefetches + {tot['fetches']:,} fetches + {tot['other']:,} other. "
      f"{tot['bytes'] / 2**20:.1f} MiB fetched of {tot['size'] / 2**20:.1f} MiB")
sys.exit(1 if failed else 0)
