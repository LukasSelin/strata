"""Medians of abrun.sh's lines: one row per case and variant, with the
spread (max-min over median) and the phases, and each variant's change
against the first variant listed.

    python summarize.py timings.txt [--first old]
"""
import collections
import re
import statistics
import sys

LINE = re.compile(r'dir=(\S+) op=(\S+) src=(\S+) n=(\S+) v=(\S+) round=\S+ ms=([0-9.]+) '
                  r'open=(\S+) run=(\S+) close=(\S+) mapped=(\S+)')


def main(path, first=None):
    runs = collections.defaultdict(list)
    order = []
    for line in open(path, encoding='utf-8'):
        m = LINE.match(line)
        if not m:
            continue
        d, op, src, n, v = m.groups()[:5]
        if v not in order:
            order.append(v)
        ph = [float(x) if x != '-' else float('nan') for x in m.groups()[6:9]]
        runs[(d, op, src, int(n), v)].append((float(m.group(6)), *ph, m.group(10)))
    base = first or order[0]
    print('| disk | op | input | workers | build | median ms | spread | open | run | close | mapped | vs %s |' % base)
    print('|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|')
    for key in sorted(runs, key=lambda k: (k[0], k[1], k[2], k[3], order.index(k[4]))):
        rs = runs[key]
        ms = sorted(r[0] for r in rs)
        med = statistics.median(ms)
        cols = [statistics.median(r[i] for r in rs) for i in (1, 2, 3)]
        b = runs.get(key[:4] + (base,))
        vs = '' if key[4] == base or not b else '%+.0f%%' % ((med / statistics.median(r[0] for r in b) - 1) * 100)
        print('| %s | %s | %s | %d | %s | %.0f | %.0f%% | %.0f | %.0f | %.0f | %s | %s |' % (
            key[0], key[1], key[2], key[3], key[4], med, (ms[-1] - ms[0]) / med * 100, *cols, rs[0][4], vs))


if __name__ == '__main__':
    args = sys.argv[1:]
    first = None
    if '--first' in args:
        i = args.index('--first')
        first = args[i + 1]
        del args[i:i + 2]
    main(args[0], first)
