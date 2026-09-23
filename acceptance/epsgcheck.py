#!/usr/bin/env python3
"""Check cog/epsg_table.go against epsg.io.

epsg_gen.py builds the table from PROJ's proj.db, and GDAL, which the
cog comparisons treat as the truth, reads the same proj.db. A mistake in
it would be shared by both and pass every comparison. epsg.io is the
EPSG registry imported by MapTiler, without PROJ's database in between,
so it is a second opinion on the same registry version:

  python3 epsgcheck.py                 # judge the table
  python3 epsgcheck.py --sabotage      # check the checker: must go red

Responses are cached in out-epsg/cache, so a rerun is offline; delete the
directory to fetch again. No API key is needed: only the free per-code
pages (https://epsg.io/<code>.json, https://epsg.io/<code>) are used.

What is checked:

  1. units      Every code in epsgProjectedUnits and epsgGeographicUnits
                is a projected (or geographic 2D) CRS whose first axis is
                in the unit the table gives. Unit codes are named through
                epsg.io's own unit pages, and compared by name.
  2. replaced   Every code in epsgReplaced is deprecated on epsg.io, and
                its replacement is another EPSG CRS of the same kind.
                epsg.io does not publish which code replaces which, so
                the pairing itself is not checked, only that it is
                plausible. A replacement that is deprecated itself is
                noted, not failed: EPSG chains replacements, and GDAL,
                like the table, follows one step.
  3. defaults   Codes the table leaves out are in metres (projected) or
                degrees (geographic 2D). Sampled: every neighbour of a
                table code (zone series change unit at their edges) and
                SAMPLE random codes. Codes epsg.io answers with an ESRI
                definition are not EPSG's and are skipped. A deprecated
                code outside
                epsgReplaced is reported, not failed: it may have no
                replacement, or several.

The registry version epsg.io serves is printed next to the table's. If
they differ, disagreements may be real registry changes, not errors.
"""

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "..", "cog", "epsg_table.go")
CACHE = os.path.join(HERE, "out-epsg", "cache")
BASE = "https://epsg.io/"
SAMPLE = 400
SEED = 36
WORKERS = 8

METRE, DEGREE = "metre", "degree"


def table():
    """The three tables of epsg_table.go as dicts, and its EPSG version."""
    src = open(TABLE, encoding="utf-8").read()
    out = {}
    for name in ("epsgProjectedUnits", "epsgGeographicUnits", "epsgReplaced"):
        body = src.split(f"var {name} = [...]uint32{{", 1)[1].split("}", 1)[0]
        out[name] = {v >> 16: v & 0xFFFF for v in (int(x, 16) for x in re.findall(r"0x[0-9a-f]+", body))}
    version = re.search(r"EPSG registry v?([\d.]+)", src)
    return out, version.group(1) if version else "?"


def fetch(path):
    """(status, body) of https://epsg.io/<path>, cached on disk."""
    f = os.path.join(CACHE, hashlib.sha1(path.encode()).hexdigest())
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            status, body = fh.read().split("\n", 1)
            return int(status), body
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "strata-epsgcheck"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status, body = r.status, r.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            status, body = e.code, ""
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    os.makedirs(CACHE, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(f"{status}\n{body}")
    return status, body


def fetch_all(paths):
    paths = sorted(set(paths))
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        return dict(zip(paths, ex.map(fetch, paths)))


def unit_name(u):
    """A PROJJSON unit (a string, or an object with a name) as a name."""
    return u if isinstance(u, str) else u.get("name", "?")


class CRS:
    """What epsg.io says about one code: None if it has no CRS by it."""

    def __init__(self, doc):
        # A CRS with a TOWGS84 comes wrapped with it as a BoundCRS.
        if doc.get("type") == "BoundCRS":
            doc = doc.get("source_crs", {})
        self.kind = doc.get("type")
        axes = doc.get("coordinate_system", {}).get("axis", [])
        self.axes = len(axes)
        self.unit = unit_name(axes[0].get("unit", "?")) if axes else "?"
        self.name = doc.get("name", "?")
        # epsg.io serves an ESRI definition under a code EPSG does not have.
        self.authority = (doc.get("id") or {}).get("authority", "?")

    @property
    def family(self):
        """projected, geographic (2D only), or None for anything else,
        including a code that is not EPSG's."""
        if self.authority != "EPSG":
            return None
        if self.kind == "ProjectedCRS":
            return "projected"
        if self.kind == "GeographicCRS" and self.axes == 2:
            return "geographic"
        return None


def crs(pages, code):
    status, body = pages[f"{code}.json"]
    if status != 200:
        return None
    try:
        return CRS(json.loads(body))
    except (ValueError, AttributeError):
        return None


def deprecated(pages, code):
    status, body = pages[str(code)]
    return status == 200 and re.search(rf"<h1><b>EPSG:{code} DEPRECATED</b>", body) is not None


def registry_version():
    status, body = fetch("docs")
    m = re.search(r"EPSG (?:database|dataset|registry)[^0-9]{0,40}v?(\d+\.\d+)", body, re.I)
    return m.group(1) if status == 200 and m else "?"


def sabotage(t):
    """Plausible defects epsg_gen.py might make, and the codes each must
    be caught at."""
    proj, geog, repl = t["epsgProjectedUnits"], t["epsgGeographicUnits"], t["epsgReplaced"]
    listed = set(proj) | set(geog) | set(repl)
    broken = []
    # A unit mixed up: US survey foot where the table says international foot.
    code = next(c for c, u in sorted(proj.items()) if u == 9002)
    proj[code] = 9003
    broken.append(code)
    # A non-metre CRS missed, next to one the table still lists (so sampled).
    code = next(c for c in sorted(proj) if c + 1 in proj and c not in repl)
    del proj[code]
    broken.append(code)
    # A geographic unit dropped to degrees by a wrong filter.
    code = next(c for c in sorted(geog) if c not in repl and {c - 1, c + 1} & listed)
    del geog[code]
    broken.append(code)
    # A replacement pointing back at the deprecated code.
    code = min(repl)
    repl[code] = code
    broken.append(code)
    return t, broken


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sabotage", action="store_true", help="break the table in memory; the check must fail")
    args = ap.parse_args()

    t, version = table()
    if args.sabotage:
        t, broken = sabotage(t)
    proj, geog, repl = t["epsgProjectedUnits"], t["epsgGeographicUnits"], t["epsgReplaced"]
    listed = set(proj) | set(geog)

    rng = random.Random(SEED)
    sample = {c + d for c in listed | set(repl) for d in (-1, 1)} | set(rng.sample(range(2000, 32768), SAMPLE))
    sample = {c for c in sample if 0 < c < 65535} - listed - set(repl)

    units = sorted(set(proj.values()) | set(geog.values()))
    print(f"table: EPSG {version}, {len(proj)} projected, {len(geog)} geographic, {len(repl)} replaced")
    print(f"epsg.io: EPSG {registry_version()}")
    print(f"fetching {len(listed) + 2 * len(repl) + len(sample)} CRSs and {len(units)} units ...", flush=True)

    pages = fetch_all([f"{u}-units" for u in units]
                      + [f"{c}.json" for c in listed | set(repl) | set(repl.values()) | sample])
    names = {}
    for u in units:
        status, body = pages[f"{u}-units"]
        m = re.search(r'og:title" content="\s*(.*?) - EPSG:' + str(u), body, re.S)
        names[u] = html.unescape(m.group(1).strip()) if status == 200 and m else f"?{u}"
    existing = {c for c in sample if crs(pages, c) and crs(pages, c).family}
    pages.update(fetch_all([str(c) for c in set(repl) | set(repl.values()) | existing]))

    failures, notes = [], {}

    def note(kind, msg):
        notes.setdefault(kind, []).append(msg)

    def fail(check, code, msg):
        failures.append(f"{check:9} EPSG:{code:<5} {msg}")

    # 1. units
    for family, tab in (("projected", proj), ("geographic", geog)):
        for code, u in sorted(tab.items()):
            c = crs(pages, code)
            if c is None:
                fail("units", code, f"not on epsg.io (table: {family}, {names[u]})")
            elif c.family != family:
                fail("units", code, f"is {c.kind} with {c.axes} axes, table lists it as {family}")
            elif c.unit != names[u]:
                fail("units", code, f"{c.name}: axis unit is {c.unit!r}, table says {names[u]!r} (EPSG:{u})")

    # 2. replaced
    for old, new in sorted(repl.items()):
        a, b = crs(pages, old), crs(pages, new)
        if a is None or b is None:
            fail("replaced", old, f"-> {new}: {'old' if a is None else 'new'} code not on epsg.io")
            continue
        if not deprecated(pages, old):
            fail("replaced", old, f"{a.name}: not deprecated on epsg.io (table: replaced by {new})")
        if old == new or a.authority != "EPSG" or b.authority != "EPSG":
            fail("replaced", old, f"-> {new}: not a replacement by another EPSG code")
        elif a.kind != b.kind:
            fail("replaced", old, f"-> {new}: {a.kind} replaced by {b.kind}")
        elif deprecated(pages, new):
            # EPSG deprecates replacements too. GDAL follows one step, as the
            # table does, so the code reported is deprecated: not the
            # table's error.
            why = f"replaced in turn by {repl[new]}" if new in repl else "with no single replacement"
            note("chains", f"EPSG:{old} -> {new}, which is deprecated too, {why}")
        elif a.axes != b.axes:
            note("dimension", f"EPSG:{old} ({a.axes}D) -> {new} ({b.axes}D) {b.name}")

    # 3. defaults
    for code in sorted(existing):
        c = crs(pages, code)
        want = METRE if c.family == "projected" else DEGREE
        if c.unit != want:
            fail("defaults", code, f"{c.name}: {c.family} in {c.unit!r}, not in the table (so taken as {want})")
        if deprecated(pages, code):
            note("unlisted", f"EPSG:{code} {c.name} is deprecated, not in epsgReplaced")

    print(f"checked: {len(proj) + len(geog)} unit entries, {len(repl)} replacements, "
          f"{len(existing)} unlisted CRSs (of {len(sample)} codes sampled)")
    for kind, title in (("chains", "replacements that are deprecated themselves"),
                        ("dimension", "3D CRSs replaced by 2D ones, or the reverse"),
                        ("unlisted", "deprecated codes without a single replacement")):
        ns = notes.get(kind, [])
        if ns:
            print(f"\nnote: {len(ns)} {title} (not failures):")
            for n in ns[:8]:
                print("  " + n)
            if len(ns) > 8:
                print(f"  ... and {len(ns) - 8} more")
    if failures:
        print(f"\nFAIL: {len(failures)}")
        for f in failures:
            print("  " + f)
    else:
        print("\nOK: the table agrees with epsg.io")

    if args.sabotage:
        caught = {int(f.split()[1].removeprefix("EPSG:")) for f in failures}
        missed = [c for c in broken if c not in caught]
        print(f"\nsabotage: {len(broken) - len(missed)} of {len(broken)} defects caught"
              + (f"; NOT caught at EPSG:{', '.join(map(str, missed))}" if missed else ""))
        return 1 if missed else 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
