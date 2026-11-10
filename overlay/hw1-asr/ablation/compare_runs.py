#!/usr/bin/env python3
"""Compare two sweep result folders (results.csv): how far apart are the per-case medians?

    python ablation/compare_runs.py ablation/results/<A> ablation/results/<B>

Prints the number of cases compared, the median and maximum relative difference, and every case
that differs by more than 10% (only cases with a median of at least 0.02 ms: shorter calls are
launch-bound and their 3-decimal rounding makes ratios meaningless).
"""
import csv, statistics, sys

def load(d):
    out = {}
    with open(f"{d.rstrip('/')}/results.csv") as f:
        for r in csv.DictReader(f):
            if r["median_ms"]:
                out[(r["variant"], r["case"])] = float(r["median_ms"])
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
big = [k for k in keys if min(a[k], b[k]) >= 0.02]
rel = {k: abs(a[k] - b[k]) / min(a[k], b[k]) for k in big}
print(f"{len(keys)} cases in both runs, {len(big)} with median >= 0.02 ms")
if rel:
    print(f"relative difference: median {statistics.median(rel.values()):.1%}, max {max(rel.values()):.1%}")
for k, v in sorted(rel.items(), key=lambda kv: -kv[1]):
    if v > 0.10:
        print(f"  {v:6.1%}  {k[0]:<18} {k[1]:<40} {a[k]:.3f} vs {b[k]:.3f} ms")
