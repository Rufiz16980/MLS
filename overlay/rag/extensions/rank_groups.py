"""Post hoc analyses of the judged answers (no ELM calls, no course assets needed).

Reads the per-query CSV written by answer_quality_summary.py --csv (55 queries x k = 1, 5, 20) and,
for the policy replays, adaptive_k.json. Reports, with paired bootstrap 95% intervals:
  1. score differences between depths within groups formed by the rank of the first relevant record
     (rank 1; ranks 2-20; none in the top 20), per split and pooled, and the difference between
     the k5-k1 effects of the ranks 2-20 and rank-1 groups;
  2. how many answers score 0 although a relevant record was in the prompt;
  3. power: the paired SD, interval half-width and the 80%-power minimum detectable effect;
  4. replays of the depth gate and of the oracle with the high arm k = 20 (the fixed protocol)
     or k = 5 (exploratory sensitivity check, not part of the protocol).

  python rag/extensions/rank_groups.py --csv rag/results/aq_pooled_practice_dev_summary.csv \
      --adaptive rag/results/adaptive_k.json
In the supplementary zip both files are under outputs/rag/.
"""
import argparse
import csv
import json
import numpy as np


def ci(values, n_boot, seed):
    v = np.asarray(values, dtype=np.float64)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_diff_of_means(a, b, n_boot, seed):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = ma - mb
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def group(q):
    if q[1]["rel"] > 0:
        return "rank 1"
    if q[20]["rel"] > 0:
        return "ranks 2-20"
    return "none in top 20"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--adaptive", default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B, seed = args.bootstrap, args.seed

    qs = {}
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        qs.setdefault((r["split"], r["query_id"]), {})[int(r["k"])] = {
            "score": float(r["judge_score"]), "rel": int(r["relevant_in_top_k"]), "tokens": float(r["prompt_tokens"])}
    print(f"queries: {len(qs)}; groups:", {g: sum(1 for q in qs.values() if group(q) == g)
                                           for g in ("rank 1", "ranks 2-20", "none in top 20")})

    print("\n1. Paired differences by group (mean, 95% CI)")
    print("| split | group | n | k5-k1 | k20-k1 | k20-k5 |\n| --- | --- | --- | --- | --- | --- |")
    for split in ("practice", "dev", "pooled"):
        for g in ("rank 1", "ranks 2-20"):
            sel = [q for (sp, _), q in qs.items() if group(q) == g and split in (sp, "pooled")]
            cells = []
            for a, b in ((5, 1), (20, 1), (20, 5)):
                m, lo, hi = ci([q[a]["score"] - q[b]["score"] for q in sel], B, seed)
                cells.append(f"{m:+.3f} ({lo:+.3f}, {hi:+.3f})")
            print(f"| {split} | {g} | {len(sel)} | " + " | ".join(cells) + " |")
    lo_grp = [q[5]["score"] - q[1]["score"] for q in qs.values() if group(q) == "ranks 2-20"]
    r1_grp = [q[5]["score"] - q[1]["score"] for q in qs.values() if group(q) == "rank 1"]
    m, lo, hi = ci_diff_of_means(lo_grp, r1_grp, B, seed)
    print(f"pooled k5-k1 effect, ranks 2-20 minus rank 1: {m:+.3f} ({lo:+.3f}, {hi:+.3f})")

    print("\n2. Answers scoring 0 although a relevant record was in the prompt")
    for k in (1, 5, 20):
        with_rel = [q for q in qs.values() if q[k]["rel"] > 0]
        zero = sum(1 for q in with_rel if q[k]["score"] == 0)
        print(f"k = {k}: {zero} of {len(with_rel)} ({zero / len(with_rel):.0%})")
    all_zero = [q for q in qs.values() if all(q[k]["score"] == 0 for k in (1, 5, 20))]
    print(f"zero at every k: {len(all_zero)}, of which {sum(1 for q in all_zero if group(q) == 'none in top 20')} "
          f"have no relevant record in the top 20")

    print("\n3. Power of the pooled paired comparisons")
    for a, b in ((5, 1), (20, 1), (20, 5)):
        d = np.array([q[a]["score"] - q[b]["score"] for q in qs.values()])
        sd, n = d.std(ddof=1), len(d)
        print(f"k{a}-k{b}: paired SD {sd:.3f}, 95% half-width {1.96 * sd / n ** 0.5:.3f}, "
              f"80%-power minimum detectable effect {2.80 * sd / n ** 0.5:.3f}")

    if args.adaptive:
        ad = json.load(open(args.adaptive, encoding="utf-8"))
        print("\n4. Policy replays against fixed k = 5 (gate = the frozen margin rule; oracle = k_lo iff rank 1 relevant)")
        print("| policy | split | n | diff vs k=5 (95% CI) | tokens vs k=5 |\n| --- | --- | --- | --- | --- |")
        for name, use_gate, k_hi in (("gate, k 1 or 20 (protocol)", True, 20), ("oracle, k 1 or 20", False, 20),
                                     ("gate, k 1 or 5 (exploratory)", True, 5), ("oracle, k 1 or 5", False, 5)):
            rows = {}
            for sp in ("practice", "dev"):
                pq = {p["query_id"]: p for p in ad["splits"][sp]["per_query"]}
                for qid in ad["replay"][sp]["queries"]:
                    q, p = qs[(sp, qid)], pq[qid]
                    low = (p["k"] == ad["k_lo"]) if use_gate else (p["first_rel_rank"] == 1)
                    k = 1 if low else k_hi
                    rows.setdefault(sp, []).append((q[k]["score"] - q[5]["score"], q[k]["tokens"], q[5]["tokens"]))
            rows["pooled"] = rows["practice"] + rows["dev"]
            for sp in ("practice", "dev", "pooled"):
                m, lo, hi = ci([r[0] for r in rows[sp]], B, seed)
                tok = sum(r[1] for r in rows[sp]) / sum(r[2] for r in rows[sp])
                print(f"| {name} | {sp} | {len(rows[sp])} | {m:+.3f} ({lo:+.3f}, {hi:+.3f}) | {tok:.2f}x |")


if __name__ == "__main__":
    main()
