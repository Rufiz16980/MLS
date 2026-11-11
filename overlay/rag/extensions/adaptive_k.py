"""Adaptive retrieval depth: small k when the top match is confident, large k otherwise (no ELM calls).

Motivation (E16): when the rank-1 record is relevant, k = 1 gives the best answers (extra records dilute);
when it is not, a larger k is needed to bring a relevant record in. So predict "is rank 1 relevant?"
from the retrieval scores alone and pick k per query:
    k = K_LO  if feature >= tau  else  K_HI
feature is the top-1 cosine score (s1) or the margin s1 - s2. Both are free: the retriever already returns them.

Protocol (fixed before looking at answer quality):
  1. On the PRACTICE split (all queries, retrieval labels only) pick the feature with the higher AUC and
     the tau that maximises Youden's J = TPR - FPR for predicting "rank-1 record is relevant".
  2. Freeze feature and tau, and report the same numbers on DEV (held out).
  3. Replay: where answers and judge scores already exist for K_LO and K_HI (aq_<split>_k<k>.jsonl and
     its _judge folder, from run_answer_quality.sh), the adaptive answer for a query IS the existing
     answer at the k the policy picks, so its judge score costs no new ELM call. Compared, paired on the same
     queries, with every fixed k that has been judged.

  cd <course repo root>
  python rag/extensions/adaptive_k.py                         # practice (tune) + dev (check) + replay
  python rag/extensions/adaptive_k.py --k-lo 1 --k-hi 5        # other depth pair
Writes rag/results/adaptive_k.json (policy, tables, per-query features) and prints markdown tables.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import numpy as np


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def boot(values, n, seed):
    v = np.asarray(values, dtype=np.float64)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auc(pos, neg):
    """P(feature of a positive > feature of a negative), ties count 1/2 (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def features(split, cache_path, kmax, data, encoder, retriever):
    qrels = data.load_qrels(split)
    cache = {r["query_id"]: r["transcript"] for r in jsonl(cache_path)}
    rows = []
    for item in data.audio_items(split):
        qid = item["query_id"]
        r = retriever.retrieve(encoder.encode(cache[qid]), k=kmax)
        ids, sc = r["record_ids"], r["scores"]
        rel = set(qrels[qid])
        first = next((i + 1 for i, x in enumerate(ids) if x in rel), None)
        rows.append({"query_id": qid, "s1": sc[0], "margin": sc[0] - sc[1], "first_rel_rank": first,
                     "top1_rel": first == 1, "ids": ids, "n_rel": len(rel)})
    return rows


def policy_stats(rows, feat, tau, k_lo, k_hi, data):
    from src.metrics import recall_at_k
    lo = [r for r in rows if r[feat] >= tau]
    ks = [k_lo if r[feat] >= tau else k_hi for r in rows]
    rec, hit, words = [], [], []
    for r, k in zip(rows, ks):
        top = r["ids"][:k]
        rel_hit = r["first_rel_rank"] is not None and r["first_rel_rank"] <= k
        hit.append(float(rel_hit))
        rec.append(r["recall_at"][k])
        words.append(sum(len(data.record_text(i).split()) for i in top))
    pos = [r for r in rows if r["top1_rel"]]
    return {"tau": tau, "share_k_lo": len(lo) / len(rows),
            "precision_k_lo": (sum(r["top1_rel"] for r in lo) / len(lo)) if lo else float("nan"),
            "tpr": (sum(r["top1_rel"] for r in lo) / len(pos)) if pos else float("nan"),
            "fpr": (sum(not r["top1_rel"] for r in lo) / max(1, len(rows) - len(pos))),
            "recall_mean": float(np.mean(rec)), "hit_mean": float(np.mean(hit)),
            "context_words_mean": float(np.mean(words)), "mean_k": float(np.mean(ks))}


def fixed_stats(rows, k, data):
    hit = [float(r["first_rel_rank"] is not None and r["first_rel_rank"] <= k) for r in rows]
    words = [sum(len(data.record_text(i).split()) for i in r["ids"][:k]) for r in rows]
    return {"recall_mean": float(np.mean([r["recall_at"][k] for r in rows])), "hit_mean": float(np.mean(hit)),
            "context_words_mean": float(np.mean(words))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-lo", type=int, default=1)
    ap.add_argument("--k-hi", type=int, default=20)
    ap.add_argument("--fixed", type=int, nargs="+", default=[1, 5, 20], help="fixed depths to compare against")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.metrics import recall_at_k
    from src.retriever import ExactDenseRetriever

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy",
                                    data_dir / "embeddings/corpus_enc_a_ids.json")
    all_k = sorted(set(args.fixed) | {args.k_lo, args.k_hi})
    kmax = max(all_k + [2])
    splits = {}
    for split in ("practice", "dev"):
        qrels = data.load_qrels(split)
        rows = features(split, ROOT / f".cache/{split}_transcripts.jsonl", kmax, data, encoder, retriever)
        for r in rows:
            r["recall_at"] = {k: recall_at_k(r["ids"], qrels[r["query_id"]], k) for k in all_k}
        splits[split] = rows

    # 1. tune on practice: feature by AUC, tau by Youden's J (candidate taus = observed feature values)
    tune = splits["practice"]
    aucs = {}
    for split, rows in splits.items():
        aucs[split] = {f: auc([r[f] for r in rows if r["top1_rel"]], [r[f] for r in rows if not r["top1_rel"]])
                       for f in ("s1", "margin")}
    feat = max(("s1", "margin"), key=lambda f: aucs["practice"][f])
    best = None
    for tau in sorted({r[feat] for r in tune}):
        s = policy_stats(tune, feat, tau, args.k_lo, args.k_hi, data)
        j = s["tpr"] - s["fpr"]
        if best is None or j > best[0]:
            best = (j, tau)
    tau = best[1]

    out = {"k_lo": args.k_lo, "k_hi": args.k_hi, "feature": feat, "tau": tau, "youden_j_practice": best[0],
           "auc": aucs, "splits": {}}
    print(f"Policy (tuned on practice, retrieval labels only): k = {args.k_lo} if {feat} >= {tau:.4f} else {args.k_hi}")
    print(f"AUC for 'rank-1 record is relevant':  " + "; ".join(
        f"{s}: s1 {a['s1']:.3f}, margin {a['margin']:.3f}" for s, a in aucs.items()) + "\n")
    print("| split | n | rank-1 relevant | policy | mean k | share at k_lo | precision at k_lo | Recall | hit | context words |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split, rows in splits.items():
        base = sum(r["top1_rel"] for r in rows) / len(rows)
        st = policy_stats(rows, feat, tau, args.k_lo, args.k_hi, data)
        # fixed depth with the same mean k as the policy: separates "adaptive" from "just more context"
        mk = int(round(st["mean_k"]))
        for r in rows:
            r["recall_at"].setdefault(mk, recall_at_k(r["ids"], data.load_qrels(split)[r["query_id"]], mk))
        fx = {k: fixed_stats(rows, k, data) for k in sorted(set(all_k) | {mk})}
        out["splits"][split] = {"n": len(rows), "top1_rel_rate": base, "adaptive": st, "fixed": fx, "matched_k": mk,
                                "per_query": [{k: r[k] for k in ("query_id", "s1", "margin", "first_rel_rank")} |
                                              {"k": args.k_lo if r[feat] >= tau else args.k_hi} for r in rows]}
        for k in sorted(set(args.fixed) | {mk}):
            f = fx[k]
            tag = " (same mean k)" if k == mk and k not in args.fixed else ""
            print(f"| {split} | {len(rows)} | {base:.3f} | fixed k={k}{tag} | {k} | - | - | {f['recall_mean']:.3f} | "
                  f"{f['hit_mean']:.3f} | {f['context_words_mean']:.0f} |")
        print(f"| {split} | {len(rows)} | {base:.3f} | **adaptive** | {st['mean_k']:.1f} | {st['share_k_lo']:.3f} | "
              f"{st['precision_k_lo']:.3f} | {st['recall_mean']:.3f} | {st['hit_mean']:.3f} | {st['context_words_mean']:.0f} |")

    # 3. replay on already-judged answers (no ELM calls)
    res = ROOT / "results"
    out["replay"] = {}
    for split, rows in splits.items():
        judged = {}
        for k in all_k:
            a, j = res / f"aq_{split}_k{k}.jsonl", res / f"aq_{split}_k{k}_judge" / "answer_judge_per_query.jsonl"
            if a.exists() and j.exists():
                judged[k] = ({r["query_id"]: r for r in jsonl(a)}, {r["query_id"]: r for r in jsonl(j)})
        if args.k_lo not in judged or args.k_hi not in judged:
            continue
        pick = {r["query_id"]: (args.k_lo if r[feat] >= tau else args.k_hi) for r in rows}
        common = [r["query_id"] for r in rows if all(r["query_id"] in a and r["query_id"] in j and
                  j[r["query_id"]].get("normalised_answer_score") is not None for a, j in judged.values())]
        if not common:
            continue

        def score(k, q):
            return judged[k][1][q]["normalised_answer_score"]

        def ptok(k, q):
            return (judged[k][0][q].get("usage") or {}).get("prompt_tokens")

        arms = {f"fixed k={k}": {q: k for q in common} for k in sorted(judged)}
        arms["adaptive"] = {q: pick[q] for q in common}
        arms["oracle (k_lo if rank 1 relevant)"] = {q: args.k_lo if next(r for r in rows if r["query_id"] == q)["top1_rel"]
                                                    else args.k_hi for q in common}
        ref = "fixed k=5" if "fixed k=5" in arms else next(iter(arms))
        table = []
        print(f"\nReplay on {split}: n = {len(common)} queries already judged at every k in {sorted(judged)} (no new ELM calls)")
        print(f"| policy | judge score (95% CI) | diff vs {ref} (95% CI) | better / worse / tied | mean k | prompt tokens |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name, ks in arms.items():
            s = np.array([score(ks[q], q) for q in common])
            d = s - np.array([score(arms[ref][q], q) for q in common])
            pt = [ptok(ks[q], q) for q in common]
            pt = [p for p in pt if p is not None]
            row = {"policy": name, "score": float(s.mean()), "score_ci95": boot(s, args.bootstrap, args.seed),
                   "diff": float(d.mean()), "diff_ci95": boot(d, args.bootstrap, args.seed),
                   "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum()),
                   "mean_k": float(np.mean([ks[q] for q in common])),
                   "prompt_tokens_mean": float(np.mean(pt)) if pt else None}
            table.append(row)
            ptxt = f"{row['prompt_tokens_mean']:.0f}" if pt else "-"
            print(f"| {name} | {row['score']:.3f} ({row['score_ci95'][0]:.3f}-{row['score_ci95'][1]:.3f}) | "
                  f"{row['diff']:+.3f} ({row['diff_ci95'][0]:+.3f} to {row['diff_ci95'][1]:+.3f}) | "
                  f"{row['better']} / {row['worse']} / {row['tied']} | {row['mean_k']:.1f} | {ptxt} |")
        out["replay"][split] = {"n": len(common), "reference": ref, "queries": common, "rows": table}

    path = res / "adaptive_k.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
