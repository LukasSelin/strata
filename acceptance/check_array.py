#!/usr/bin/env python3
"""Judge the N-dimensional array files written by `go run .` (array.go)
without reading strata's source.

The references are written from the published definitions:

  * Broadcasting is numpy's: numpy itself broadcasts the inputs, after
    the view steps the manifest records (slice, select, transpose,
    expand) are applied with numpy's own indexing.
  * Elementwise arithmetic is IEEE float32, or int16 wrapping, which
    numpy computes too. Min and max follow Go's builtins, which numpy
    does not promise for signed zeros, so the reference sets them
    explicitly: min(-0, +0) is -0, max(-0, +0) is +0, NaN propagates.
  * A result is valid iff every broadcast input is valid there.
  * Sum and Mean are exact: every float32 is an integer times 2^-149,
    so the valid values are summed as Python integers and divided once,
    which Python rounds correctly. No floating-point sum is involved.
    NaN, or both infinities, make them NaN; one infinity makes them that
    infinity; a sum of zeros is -0 only if every value is -0. Mean, Min
    and Max are invalid with no valid values; Sum is 0 and Count 0.

Every check is exact: validity equal, values equal bit for bit on valid
cells, except that any NaN matches any NaN.

Usage:  python check_array.py [dir]             (default: out)
        python check_array.py out --sabotage   every injected defect
                                               must be caught
"""

import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

DTYPES = {"float32": "<f4", "float64": "<f8", "int16": "<i2", "int64": "<i8"}
SCALE = 149  # every float32 is an integer times 2^-149


def load(d, name, dtype, shape):
    return np.fromfile(os.path.join(d, name), dtype=DTYPES[dtype]).reshape(shape)


def load_mask(d, name, shape):
    if not name:
        return np.ones(shape, dtype=bool)
    return np.fromfile(os.path.join(d, name), dtype=np.uint8).reshape(shape) != 0


def view(a, steps):
    """Apply the manifest's view steps with numpy's own indexing."""
    for s in steps or []:
        kind, args = s[0], s[1:]
        if kind == "slice":
            ax, lo, hi = args
            idx = [slice(None)] * a.ndim
            idx[ax] = slice(lo, hi)
            a = a[tuple(idx)]
        elif kind == "select":
            a = np.take(a, args[1], axis=args[0])
        elif kind == "transpose":
            a = np.transpose(a, args)
        elif kind == "expand":
            a = np.expand_dims(a, args[0])
        elif kind == "broadcast":
            a = np.broadcast_to(a, args)
        else:
            raise ValueError(kind)
    return a


def go_min(a, b):
    out = np.minimum(a, b)
    if a.dtype.kind == "f":
        z = (a == 0) & (b == 0)
        out = np.where(z, np.where(np.signbit(a) | np.signbit(b), -0.0, 0.0).astype(a.dtype), out)
    return out


def go_max(a, b):
    out = np.maximum(a, b)
    if a.dtype.kind == "f":
        z = (a == 0) & (b == 0)
        out = np.where(z, np.where(np.signbit(a) & np.signbit(b), -0.0, 0.0).astype(a.dtype), out)
    return out


def exact_int(v):
    """v, a float32 or integer, as an integer multiple of 2^-149."""
    n, d = Fraction(float(v)).as_integer_ratio()
    return n * (2**SCALE // d)


def fold(vals, op):
    """The reference for one output element from its valid values."""
    if op == "count":
        return len(vals), True
    fvals = [float(v) for v in vals]
    nan = any(math.isnan(v) for v in fvals)
    pinf = any(v == math.inf for v in fvals)
    ninf = any(v == -math.inf for v in fvals)
    if op in ("min", "max"):
        if not vals:
            return None, False
        if nan:
            return math.nan, True
        if op == "min":
            return min(fvals, key=lambda v: (v, not math.copysign(1, v) < 0)), True
        return max(fvals, key=lambda v: (v, math.copysign(1, v) > 0)), True
    # sum and mean
    if op == "mean" and not vals:
        return None, False
    if nan or (pinf and ninf):
        return math.nan, True
    if pinf:
        return math.inf, True
    if ninf:
        return -math.inf, True
    if vals and all(v == 0 and math.copysign(1, v) < 0 for v in fvals):
        return -0.0, True
    s = sum(exact_int(v) for v in vals)
    if op == "sum":
        return s / 2**SCALE, True  # int / int: correctly rounded
    return s / (len(vals) * 2**SCALE), True


def reduce_ref(src, valid, axes, op):
    kept = [k for k in range(src.ndim) if k not in axes]
    s = np.transpose(src, kept + list(axes))
    m = np.transpose(valid, kept + list(axes))
    out_shape = s.shape[: len(kept)]
    n = int(np.prod(out_shape)) if out_shape else 1
    s = s.reshape(n, -1)
    m = m.reshape(n, -1)
    vals, ok = [], []
    for i in range(n):
        v, k = fold(list(s[i][m[i]]), op)
        vals.append(v)
        ok.append(k)
    return vals, np.array(ok).reshape(out_shape)


def same(got, want):
    if want is None:
        return True
    g, w = float(got), float(want)
    if math.isnan(w):
        return math.isnan(g)
    return g == w and math.copysign(1, g) == math.copysign(1, w)


def judge(case, out, outmask, inputs):
    """Return a list of failure strings for one case."""
    fails = []
    srcs = [(view(inputs[o["input"]][0], o.get("view")), view(inputs[o["input"]][1], o.get("view")))
            for o in case["inputs"]]
    op = case["op"]
    if not case.get("axes"):
        if op == "convert":
            want = np.broadcast_to(srcs[0][0], out.shape).astype(out.dtype)
            wvalid = np.broadcast_to(srcs[0][1], out.shape)
        else:
            (a, am), (b, bm) = srcs
            with np.errstate(all="ignore"):
                want = {"sub": np.subtract, "mul": np.multiply, "add": np.add,
                        "min": go_min, "max": go_max}[op](a, b)
            want = np.broadcast_to(want, out.shape)
            wvalid = np.broadcast_to(am & bm, out.shape)
        if not np.array_equal(outmask, wvalid):
            fails.append("validity differs at %d elements" % int(np.sum(outmask != wvalid)))
        v = wvalid
        g, w = out[v], want[v]
        if out.dtype.kind == "f":
            bad = ~((g.view(np.uint32 if g.dtype == np.float32 else np.uint64) ==
                     w.view(np.uint32 if w.dtype == np.float32 else np.uint64)) |
                    (np.isnan(g) & np.isnan(w)))
        else:
            bad = g != w
        if bad.any():
            fails.append("%d valid elements differ" % int(bad.sum()))
        return fails
    (src, valid), = srcs
    vals, ok = reduce_ref(src, valid, case["axes"], op)
    if not np.array_equal(outmask, ok):
        fails.append("validity differs at %d elements" % int(np.sum(outmask != ok)))
    flat = out.reshape(-1)
    bad = [i for i, w in enumerate(vals) if not same(flat[i], w)]
    if bad:
        i = bad[0]
        fails.append("%d of %d elements differ; first at %d: %r, want %r" % (len(bad), len(vals), i, flat[i], vals[i]))
    return fails


# --- checking the checker ------------------------------------------------

def sabotages(cases, outs, inputs):
    """Plausible defects, each as (description, case name, damaged output,
    damaged mask)."""
    by = {c["name"]: c for c in cases}
    found = []

    def src_of(name):
        o = by[name]["inputs"][0]
        return view(inputs[o["input"]][0], o.get("view")), view(inputs[o["input"]][1], o.get("view"))

    # A mean from the rounded sum, divided and rounded again. Only where
    # the data can tell: an int16 sum is exact in float64 already, and five
    # spatial means happen to round alike.
    for name in ("mean-time", "mean-view-time"):
        out, mask = outs[name]
        sums = outs["sum-" + name.split("-", 1)[1]][0].reshape(-1)
        src, valid = src_of(name)
        axes = by[name]["axes"]
        kept = [k for k in range(src.ndim) if k not in axes]
        m = np.transpose(valid, kept + list(axes)).reshape(sums.size, -1).sum(axis=1)
        with np.errstate(all="ignore"):
            bad = np.where(m > 0, sums / np.maximum(m, 1), out.reshape(-1)).reshape(out.shape)
        found.append(("mean by the rounded sum (%s)" % name, name, bad, mask))
    # A sum accumulated in float64, in index order. Along x, and over the
    # whole stack, the float32 values share a scale and a float64 running
    # sum happens to be exact; over time the scales differ by 10^13.
    for name in ("sum-time", "sum-view-time"):
        src, valid = src_of(name)
        axes = by[name]["axes"]
        kept = [k for k in range(src.ndim) if k not in axes]
        s = np.transpose(src, kept + list(axes)).reshape(outs[name][0].size, -1).astype(np.float64)
        m = np.transpose(valid, kept + list(axes)).reshape(s.shape)
        acc = np.zeros(s.shape[0])
        with np.errstate(all="ignore"):
            for j in range(s.shape[1]):
                acc = np.where(m[:, j], acc + s[:, j], acc)
        found.append(("sum in float64, in order (%s)" % name, name, acc.reshape(outs[name][0].shape), outs[name][1]))
    # Min over time reading NoData cells too.
    src, valid = src_of("min-time")
    with np.errstate(all="ignore"):
        found.append(("min over NoData too", "min-time", np.min(src, axis=0), np.ones_like(outs["min-time"][1])))
    # Broadcast aligned at the first axis: weights [T,1,1] applied along x.
    out, mask = outs["mul-weights"]
    w = inputs["weights"][0].reshape(-1)
    s = inputs["stack"][0]
    xw = np.resize(w, s.shape[2]).astype(np.float32)
    with np.errstate(all="ignore"):
        found.append(("weights broadcast along x", "mul-weights", s * xw, mask))
    # One NoData element leaking into a result's validity.
    out, mask = outs["sub-layer"]
    m2 = mask.copy()
    i = np.argwhere(~m2)[0]
    m2[tuple(i)] = True
    found.append(("one NoData element marked valid", "sub-layer", out, m2))
    # One element one ulp off.
    out, mask = outs["max-transposed"]
    o2 = out.copy()
    i = np.argwhere(mask & np.isfinite(out) & (out != 0))[len(np.argwhere(mask)) // 2]
    o2[tuple(i)] = np.nextafter(o2[tuple(i)], np.float32(np.inf))
    found.append(("one element one ulp off", "max-transposed", o2, mask))
    # int16 addition saturating instead of wrapping.
    out, mask = outs["add-int16"]
    a = inputs["labels"][0].astype(np.int32) + inputs["offsets"][0].astype(np.int32)
    found.append(("int16 add saturating", "add-int16", np.clip(a, -32768, 32767).astype(np.int16), mask))
    # Go's min(-0, +0) ignored: numpy's own answer.
    out, mask = outs["min-time"]
    o2 = out.copy()
    z = np.argwhere(mask & (out == 0) & np.signbit(out))
    if len(z):
        o2[tuple(z[0])] = np.float32(0)
        found.append(("min(-0, +0) as +0", "min-time", o2, mask))
    # A transposed view read as if compact.
    src, valid = src_of("sum-view-time")
    out, mask = outs["sum-view-time"]
    wrong = np.array(src).reshape(src.shape[1], src.shape[0], src.shape[2])
    mwrong = np.array(valid).reshape(wrong.shape)
    vals, _ = reduce_ref(wrong.transpose(1, 0, 2), mwrong.transpose(1, 0, 2), [1], "sum")
    found.append(("transposed view read as compact", "sum-view-time", np.array(vals).reshape(out.shape), mask))
    return found


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else "out"
    sabotage = "--sabotage" in sys.argv
    with open(os.path.join(d, "array.json")) as f:
        man = json.load(f)
    inputs = {}
    for i in man["inputs"]:
        inputs[i["name"]] = (load(d, i["file"], i["dtype"], i["shape"]), load_mask(d, i.get("mask"), i["shape"]))
    outs = {}
    for c in man["cases"]:
        outs[c["name"]] = (load(d, c["out"], c["dtype"], c["shape"]), load_mask(d, c.get("mask"), c["shape"]))

    if sabotage:
        caught = 0
        found = sabotages(man["cases"], outs, inputs)
        by = {c["name"]: c for c in man["cases"]}
        for desc, name, out, mask in found:
            fails = judge(by[name], np.asarray(out, dtype=outs[name][0].dtype), np.asarray(mask), inputs)
            print("%-7s %-45s %s" % ("caught" if fails else "MISSED", desc, "; ".join(fails)[:90]))
            caught += bool(fails)
        ok = caught == len(found)
        print("%s  sabotage: %d of %d injected defects caught" % ("ok  " if ok else "FAIL", caught, len(found)))
        return 0 if ok else 1

    failed = 0
    for c in man["cases"]:
        out, mask = outs[c["name"]]
        fails = judge(c, out, mask, inputs)
        valid = int(mask.sum())
        if fails:
            failed += 1
            print("FAIL  %-22s %s" % (c["name"], "; ".join(fails)))
        else:
            print("ok    %-22s %s %s, %d of %d valid, exact" % (c["name"], c["dtype"], c["shape"], valid, mask.size))
    print("%d of %d array checks passed" % (len(man["cases"]) - failed, len(man["cases"])))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
