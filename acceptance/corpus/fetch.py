#!/usr/bin/env python3
"""Download the foreign-GeoTIFF corpus and check every file's SHA-256.

The corpus is GeoTIFFs that GDAL did not write for this test: other
writers' files, libtiff's and GDAL's deliberately odd test images, and
satellite and elevation products. sources.tsv lists each one with its
URL, SHA-256 and licence; nothing here is committed, so the repository
stays small and every file's provenance is a line of text.

  python3 fetch.py [--dir files] [--group g ...] [--pin]

Files land in <dir>/<group>/<name>. A file already there with the right
hash is not fetched again. A hash mismatch is an error: the corpus must
be the corpus the results in ../README.md were produced on. --pin
instead records the hashes of what it downloads, for adding sources.

URL forms besides plain https:
  <url>#<member>   a member of a .zip or .tar.gz archive
  pc://<account>/<container>/<path>
                   a Microsoft Planetary Computer blob; fetch.py gets a
                   free anonymous read token for the container first

Standard library only.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "sources.tsv")
COLUMNS = ["group", "name", "sha256", "url", "licence"]

_archives = {}
_tokens = {}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "strata-corpus/1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def resolve(url):
    if url.startswith("pc://"):
        account, container, path = url[5:].split("/", 2)
        key = (account, container)
        if key not in _tokens:
            t = json.loads(get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"))
            _tokens[key] = t["token"]
        return f"https://{account}.blob.core.windows.net/{container}/{path}?{_tokens[key]}"
    return url


def fetch(url):
    if "#" in url:
        archive, member = url.split("#", 1)
        if archive not in _archives:
            _archives[archive] = get(archive)
        data = _archives[archive]
        if archive.endswith(".zip"):
            return zipfile.ZipFile(io.BytesIO(data)).read(member)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            return t.extractfile(member).read()
    return get(resolve(url))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "files"))
    ap.add_argument("--group", action="append", help="fetch only these groups")
    ap.add_argument("--pin", action="store_true", help="record hashes instead of checking them")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SOURCES, newline=""), delimiter="\t"))
    bad = fetched = 0
    for row in rows:
        if a.group and row["group"] not in a.group:
            continue
        path = os.path.join(a.dir, row["group"], row["name"])
        if os.path.exists(path) and not a.pin:
            if hashlib.sha256(open(path, "rb").read()).hexdigest() == row["sha256"]:
                continue
        try:
            data = fetch(row["url"])
        except Exception as e:  # noqa: BLE001 -- report and go on
            print(f"FAILED   {row['group']}/{row['name']}: {e}")
            bad += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        if a.pin:
            row["sha256"] = digest
        elif digest != row["sha256"]:
            print(f"MISMATCH {row['group']}/{row['name']}: sha256 {digest}, want {row['sha256']}")
            bad += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        fetched += 1
        print(f"fetched  {row['group']}/{row['name']} ({len(data):,} bytes)")

    if a.pin:
        with open(SOURCES, "w", newline="") as f:
            w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    n = sum(1 for r in rows if not a.group or r["group"] in a.group)
    print(f"{n} files listed, {fetched} fetched, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
