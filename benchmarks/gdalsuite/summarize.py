#!/usr/bin/env python3
"""Turn gdalsuite.sh's timings into the RESULTS.md tables.

    python summarize.py timings.txt [--baseline older-timings.txt]

Reads the `tier=... op=... tool=... cfg=... threads=... run=...` lines
runsuite.sh prints, and the `agree` and `# identical` lines, and prints:

  1. the scoreboard: per operation and tier, how many times faster
     strata is than GDAL, on one thread and on N;
  2. where the time goes: how much of each tool's whole flow is compute;
  3. the medians behind every speedup, per tier;
  4. how far apart the two tools' answers are;
  5. with --baseline, what changed since that run.

Every number is a median over the timed runs, never the minimum. Every
speedup is against the *fastest* GDAL configuration for that operation
and tier (ENVI or GeoTIFF, GeoTIFF or ENVI output, gdal raster calc or
gdal_calc.py): on one thread against GDAL's fastest single-threaded
configuration, on N against its fastest at any thread count. So a
speedup is what strata gains over GDAL at its best on this machine.
"""

import argparse
import math
import re
import statistics
import sys
from collections import defaultdict

FAMILIES = [
    ("terrain", ["slope", "aspect", "hillshade", "tri", "tpi", "roughness"]),
    ("focal", ["mean3", "mean11", "min3", "max3", "gauss5", "conv5"]),
    ("algebra", ["add", "mul", "min", "max", "clamp"]),
    ("reduce", ["stats", "minmax"]),
    ("resample", ["near-half", "bilinear-half", "cubic-half", "lanczos-half",
                  "average-half", "cubic-double"]),
]
FAMILY = {op: fam for fam, ops in FAMILIES for op in ops}
TIERS = [("compute", "compute"), ("raw", "file to file"), ("cog", "whole flow from a COG")]
SPREAD_WARN = 0.15


class Run:
    def __init__(self, path):
        self.notes, self.agree, self.identical = [], {}, {}
        self.cases = defaultdict(list)  # key -> [(value s, cpu s | None)]
        self.w = self.h = None
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("#"):
                self.notes.append(line)
                m = re.match(r"# Size is (\d+), (\d+)", line)
                if m:
                    self.w, self.h = int(m[1]), int(m[2])
                m = re.match(r"# (identical|DIFFERENT): (\S+) ", line)
                if m:
                    self.identical[m[2]] = m[1] == "identical"
                continue
            if line.startswith("agree "):
                f = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                self.agree[f["op"]] = line
                continue
            if not line.startswith("tier="):
                continue
            f = dict(p.split("=", 1) for p in line.split() if "=" in p)
            key = (f["tier"], f["op"], f["tool"], f["cfg"], int(f["threads"]))
            if f["real"] != "-":
                try:
                    val = float(f["real"])
                except ValueError:  # bash's time now and then prints e.g. "6.:00"
                    self.notes.append(f"# dropped, unreadable time: {line}")
                    continue
                try:
                    cpu = float(f["user"]) + float(f["sys"])
                except ValueError:  # bash's time now and then prints e.g. "2.:00"
                    cpu = None
            else:
                val, cpu = float(f["ms"]) / 1000, None
            self.cases[key].append((val, cpu))
        self.cells = (self.w or 0) * (self.h or 0)
        self.n = max((k[4] for k in self.cases), default=1)
        self.ops = []
        for k in self.cases:
            if k[0] != "floor" and k[1] not in self.ops:
                self.ops.append(k[1])

    def med(self, key):
        v = self.cases.get(key)
        return statistics.median(x[0] for x in v) if v else None

    def cpu(self, key):
        v = [x[1] for x in self.cases.get(key, []) if x[1] is not None]
        return statistics.median(v) if v else None

    def spread(self, key):
        v = sorted(x[0] for x in self.cases.get(key, []))
        return (v[-1] - v[0]) / statistics.median(v) if len(v) > 1 else 0.0

    def gdal(self, tier, op, one_thread):
        keys = [k for k in self.cases if k[0] == tier and k[1] == op and k[2] == "gdal"
                and (k[4] == 1 or not one_thread)]
        if not keys:
            return None, None
        best = min(keys, key=self.med)
        return self.med(best), best

    def strata(self, tier, op, threads, cfg="simd"):
        return self.med((tier, op, "strata", cfg, threads))

    def speedup(self, tier, op, threads):
        g, _ = self.gdal(tier, op, threads == 1)
        s = self.strata(tier, op, threads)
        return g / s if g and s else None


def x(v):
    return "–" if v is None else (f"{v:.0f}×" if v >= 100 else f"{v:.1f}×")


def secs(v):
    if v is None:
        return "–"
    return f"{v * 1000:.1f} ms" if v < 1 else f"{v:.2f} s"


def geomean(vals):
    vals = [v for v in vals if v]
    return math.exp(sum(map(math.log, vals)) / len(vals)) if vals else None


def scoreboard(r):
    print("### Scoreboard: how many times faster strata is than GDAL\n")
    print(f"`1` is one thread each; `{r.n}` is strata on {r.n} workers against GDAL's "
          "fastest configuration at any thread count. Above 1× strata is faster.\n")
    head = "| op | " + " | ".join(f"{name}, 1 | {name}, {r.n}" for _, name in TIERS) + " |"
    print(head)
    print("| --- |" + " ---: |" * (2 * len(TIERS)))
    for fam, ops in FAMILIES:
        ops = [o for o in ops if o in r.ops]
        if not ops:
            continue
        for op in ops:
            cells = [x(r.speedup(t, op, n)) for t, _ in TIERS for n in (1, r.n)]
            print(f"| {op} | " + " | ".join(cells) + " |")
        cells = [x(geomean([r.speedup(t, o, n) for o in ops])) for t, _ in TIERS for n in (1, r.n)]
        print(f"| **{fam}** (geometric mean) | " + " | ".join(f"**{c}**" for c in cells) + " |")
    cells = [x(geomean([r.speedup(t, o, n) for o in r.ops])) for t, _ in TIERS for n in (1, r.n)]
    print(f"| **all {len(r.ops)} operations** | " + " | ".join(f"**{c}**" for c in cells) + " |")
    print()


def where_time_goes(r):
    print("### Where the time goes: compute against the whole flow, one thread\n")
    print("The compute tier times the call with the input already in memory; the "
          "whole flow times the process from a Deflate COG to a file. `compute share` is "
          "the first over the second: how much of what a user waits for is arithmetic. "
          "The last columns are the speedup at each end.\n")
    print("| op | strata compute | strata flow | compute share | GDAL compute | GDAL flow "
          "| compute share | × compute | × flow |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for op in r.ops:
        sc, sf = r.strata("compute", op, 1), r.strata("cog", op, 1)
        gc, _ = r.gdal("compute", op, True)
        gf, _ = r.gdal("cog", op, True)
        share = lambda a, b: f"{a / b * 100:.0f}%" if a and b else "–"
        print(f"| {op} | {secs(sc)} | {secs(sf)} | {share(sc, sf)} | {secs(gc)} | {secs(gf)} "
              f"| {share(gc, gf)} | {x(gc / sc if gc and sc else None)} "
              f"| {x(gf / sf if gf and sf else None)} |")
    print()


def cfgs(r, tier):
    seen = []
    for k in r.cases:
        if k[0] == tier and (k[2], k[3], k[4]) not in seen:
            seen.append((k[2], k[3], k[4]))
    order = {"gdal": 0, "strata": 1}
    return sorted(seen, key=lambda c: (order[c[0]], c[1] == "scalar", c[2], c[1]))


def detail(r, tier, name):
    cs = cfgs(r, tier)
    if not cs:
        return
    print(f"### {name}: medians\n")
    print("Throughput is source cells over the median time. `worst spread` is the largest "
          "(max − min)/median over the cases in the row.\n")
    labels = [f"{t} {c}, {n}" for t, c, n in cs]
    print("| op | " + " | ".join(labels) + " | strata 1, M cells/s | scalar/SIMD | worst spread |")
    print("| --- |" + " ---: |" * (len(cs) + 3))
    for op in r.ops:
        vals = [secs(r.med((tier, op) + c)) for c in cs]
        s1 = r.strata(tier, op, 1)
        sc = r.strata(tier, op, 1, "scalar")
        rate = f"{r.cells / s1 / 1e6:.0f}" if s1 and r.cells else "–"
        spread = max((r.spread((tier, op) + c) for c in cs if (tier, op) + c in r.cases), default=0)
        warn = " ⚠" if spread > SPREAD_WARN else ""
        print(f"| {op} | " + " | ".join(vals) +
              f" | {rate} | {x(sc / s1 if sc and s1 else None)} | {spread * 100:.0f}%{warn} |")
    print()


def cpu_table(r):
    print(f"### CPU spent: the whole flow from a COG, CPU-seconds (user + sys)\n")
    print("What each tool burns for the same result. A tool that returns sooner "
          "on more threads can still be spending more.\n")
    print(f"| op | GDAL, 1 | strata, 1 | GDAL, {r.n} | strata, {r.n} |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for op in r.ops:
        row = []
        for one in (True, False):
            _, gk = r.gdal("cog", op, one)
            n = 1 if one else r.n
            gk = gk if one else ("cog", op, "gdal", gk[3], r.n) if gk else None
            row += [r.cpu(gk) if gk else None, r.cpu(("cog", op, "strata", "simd", n))]
        print(f"| {op} | " + " | ".join("–" if v is None else f"{v:.2f}" for v in row) + " |")
    print()


def floors(r):
    keys = [k for k in r.cases if k[0] == "floor"]
    if not keys:
        return
    print("### Floors: each tool's cost before it computes anything\n")
    print("| case | median | spread |")
    print("| --- | ---: | ---: |")
    for k in keys:
        print(f"| {k[2]} {k[3]}, {k[4]} thread{'s' if k[4] > 1 else ''} | {secs(r.med(k))} "
              f"| {r.spread(k) * 100:.0f}% |")
    print()


def agreement(r):
    print("### Do they agree?\n")
    print("strata's output against GDAL's, from the same file, excluding each tool's "
          "differently defined border. `identical` is whether strata's whole flow from "
          "the COG on N workers wrote exactly the bytes of its raw path on one.\n")
    print("| op | cells compared | validity differs | max abs diff | mean abs diff "
          "| value range | COG == raw |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    for op in r.ops:
        line = r.agree.get(op)
        ident = {True: "yes", False: "**NO**"}.get(r.identical.get(op), "–")
        if not line:
            print(f"| {op} | – | – | – | – | – | {ident} |")
            continue
        f = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        if f.get("kind") == "reduce":
            worst = max((float(v.split(":rel=")[1]) for k, v in f.items()
                         if ":rel=" in v), default=0)
            print(f"| {op} | all | – | relative {worst:.1g} | – | – | {ident} |")
            continue
        n, of = int(f["compared"]), int(f["of"])
        mm = int(f["validity_mismatch"])
        print(f"| {op} | {n:,} ({n / of * 100:.1f}%) | {mm:,} | {f.get('maxdiff', '–')} "
              f"| {f.get('meandiff', '–')} | {f.get('range', '–')} | {ident} |")
    print()


def since(r, b, label):
    print(f"### Since the baseline ({label})\n")
    print("`now/then` below 1 is faster now. The speedup columns are this run's and the "
          "baseline's, each against the GDAL of its own run, so a change in either tool "
          "or the machine shows up there.\n")
    print("| op | tier | strata 1, then | strata 1, now | now/then | ×GDAL then | ×GDAL now |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for op in r.ops:
        for t, name in TIERS:
            a, c = b.strata(t, op, 1), r.strata(t, op, 1)
            if a is None or c is None:
                continue
            print(f"| {op} | {name} | {secs(a)} | {secs(c)} | {c / a:.2f} "
                  f"| {x(b.speedup(t, op, 1))} | {x(r.speedup(t, op, 1))} |")
    print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("timings")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    r = Run(a.timings)
    for n in r.notes:
        if not n.startswith(("# ---", "# identical", "# DIFFERENT")):
            print(n)
    reps = max((len(v) for v in r.cases.values()), default=0)
    print(f"\n{r.w} × {r.h} = {r.cells / 1e6:.1f}M cells, {reps} timed runs per case, "
          f"median reported. N = {r.n}.\n")
    scoreboard(r)
    where_time_goes(r)
    for t, name in TIERS:
        detail(r, t, name)
    cpu_table(r)
    floors(r)
    agreement(r)
    if a.baseline:
        b = Run(a.baseline)
        stamp = next((n[2:] for n in b.notes if n.startswith("# strata:")), a.baseline)
        since(r, b, stamp)


if __name__ == "__main__":
    main()
