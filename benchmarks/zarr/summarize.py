"""Turn `go test -bench` output from this directory into RESULTS.md's tables.

    python summarize.py testdata/bench.txt

Every figure is the median of the -count runs of its case.
"""

import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r"^Benchmark(\w+)/(\S+?)-\d+\s+\d+\s+([\d.]+) ns/op(.*)$")


def parse(path):
    runs = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            op, name, ns, rest = m.groups()
            key = (op, name)
            runs[key]["ns"].append(float(ns))
            for value, unit in re.findall(r"([\d.]+) (\S+)", rest):
                runs[key][unit].append(float(value))
    return {k: {u: statistics.median(v) for u, v in d.items()} for k, d in runs.items()}


def fields(name):
    return dict(part.split("=", 1) for part in name.split("/"))


def ms(ns):
    return f"{ns / 1e6:.1f}"


def engine_table(res, op):
    rows = [(fields(n), v) for (o, n), v in res.items() if o == op]
    mem = {f["workers"]: v for f, v in rows if f["src"] == "memory"}
    workers = sorted(mem, key=int)
    print(f"#### {op}\n")
    head = " | ".join(f"{w} worker{'s' if w != '1' else ''}: ms" for w in workers)
    print(f"| chunk | cache | conc | {head} | decodes/chunk |")
    print("| ---: | --- | ---: |" + " ---: |" * len(workers) + " ---: |")
    print("| memory | | | " + " | ".join(ms(mem[w]["ns"]) for w in workers) + " | |")
    zarr = defaultdict(dict)
    for f, v in rows:
        if f["src"] == "zarr":
            zarr[(int(f["chunk"]), f["cache"] == "off", int(f["conc"]))][f["workers"]] = v
    for (chunk, off, conc), by in sorted(zarr.items()):
        cells = " | ".join(ms(by[w]["ns"]) for w in workers)
        dec = by[workers[0]].get("decodes/chunk", 0)
        print(f"| {chunk} | {'off' if off else 'on'} | {conc} | {cells} | {dec:.3f} |")
    print()


def read_all_table(res):
    rows = [(fields(n), v) for (o, n), v in res.items() if o == "ReadAll"]
    workers = sorted({f["workers"] for f, _ in rows}, key=int)
    by = defaultdict(dict)
    for f, v in rows:
        by[(int(f["chunk"]), f["cache"] == "off", int(f["conc"]))][f["workers"]] = v
    print("#### Reading the whole raster in 256-row strips with 1-row halos\n")
    head = " | ".join(f"{w} worker{'s' if w != '1' else ''}: ms" for w in workers)
    print(f"| chunk | cache | conc | {head} | decodes/chunk |")
    print("| ---: | --- | ---: |" + " ---: |" * len(workers) + " ---: |")
    for (chunk, off, conc), b in sorted(by.items()):
        cells = " | ".join(ms(b[w]["ns"]) for w in workers)
        print(f"| {chunk} | {'off' if off else 'on'} | {conc} | {cells} | {b[workers[0]]['decodes/chunk']:.3f} |")
    print()


def window_table(res):
    rows = [(fields(n), v) for (o, n), v in res.items() if o == "Window"]
    by = defaultdict(dict)
    for f, v in rows:
        by[(int(f["chunk"]), f["window"])][f["cache"]] = v["ns"]
    order = {"pixel": 0, "3x3-corner": 1, "256x256-4chunks": 2}
    print("#### One window, read again and again\n")
    print("| chunk | window | cache off | cache on (warm) | off / on |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for (chunk, win), b in sorted(by.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        off, on = b["off"], b["on"]
        print(f"| {chunk} | {win} | {off / 1e6:.2f} ms | {on / 1e3:.2f} µs | {off / on:,.0f}× |")
    print()


def main():
    res = parse(sys.argv[1])
    print("<!-- summarize.py output begin -->\n")
    engine_table(res, "Slope")
    engine_table(res, "Mean")
    read_all_table(res)
    window_table(res)
    print("<!-- summarize.py output end -->")


if __name__ == "__main__":
    main()
