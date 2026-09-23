# Foreign GeoTIFF corpus

GeoTIFFs that GDAL did not write for `../cogcheck.sh`, for judging
strata's `cog` reader against GDAL with `../cogcorpus.sh`. What they are,
and the results, are in `../README.md`, "Reading GeoTIFFs other software
wrote"; one row per file in [`RESULTS.md`](RESULTS.md).

| File | What |
|---|---|
| `sources.tsv` | every downloaded file: group, name, SHA-256, URL, licence |
| `fetch.py` | downloads them into `files/` (ignored by git) and checks each hash; `--pin` records hashes when adding sources |
| `generate.sh` | writes `files/generated-*` with tiffcp, rasterio/rio-cogeo and GDAL, each in a container (`gen_tiffcp.sh`, `gen_rasterio.py`, `gen_gdal.py`) |
| `findings.tsv` | the files that failed before the reader was fixed, and the finding each belongs to |
| `report.py` | writes `RESULTS.md` from `../out-corpus/results.json`, and prints the summary table |

```bash
python3 fetch.py && ./generate.sh && ../cogcorpus.sh && python3 report.py
```
