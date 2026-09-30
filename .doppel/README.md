# doppel labels

`labels.json` is a human-style review of the top 60 pairs `doppel analyze .` reported
for this repository on 2026-09-30 (default settings, tests excluded). Each pair is
`merge`, `refactor` or `false_positive`; a false positive also carries a `kind`
(mirror, entrypoint, separate-programs, skeleton, already-factored, accessor-family,
vocabulary, other) naming why it is not worth acting on. Contested pairs were left out.

It is ground truth for doppel's golden benchmark, not configuration: doppel's walk skips
dot-directories, so nothing here affects an analysis. Score it from a doppel checkout:

    DOPPEL_BENCH_CORPUS=<path to this repo> DOPPEL_BENCH_LABELS=<path to this repo>/.doppel/labels.json \
      go test ./internal/bench/ -run TestGoldenRanking -v -count=1

Baseline when recorded (mean rank per class): merge 42.0 (4/4 retrieved, 2 in top 50) · refactor 27.9 · false_positive 31.4 · 7 false positives in the top 20.
The benchmark's hard assertions fail at this baseline; that failure is what a
false-positive fix is measured against.

Pairs whose two sides share a qualified name (two `main.main`s, two `init`s, a helper
duplicated across two JS files) cannot be expressed in this format and are absent.
