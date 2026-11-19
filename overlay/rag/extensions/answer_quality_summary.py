"""Summarise the answer-quality-vs-depth experiment (no ELM calls).

Reads, for each k, the answers  rag/results/aq_<split>_k<k>.jsonl  (generate_from_transcripts.py)
and the judge output  rag/results/aq_<split>_k<k>_judge/answer_judge_per_query.jsonl  (evaluate_answers.py),
keeps only the queries present at EVERY k (paired comparison), and reports per k:
judge score (mean + 95% bootstrap CI), the three sub-scores, Recall@k on the same queries,
prompt tokens, generation latency, and paired differences against the smallest k.

  cd <course repo root>
  python rag/extensions/answer_quality_summary.py --split practice --ks 1 5 20
  python rag/extensions/answer_quality_summary.py --split practice dev --bootstrap 5000 --csv   # pooled, n = 55
Writes rag/results/aq_<split>_summary.json (pooled: aq_pooled_practice_dev_summary.json) and prints markdown
tables. --csv also writes the per-query scores (no answer or record text) to the same name with .csv.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import statistics as st
import numpy as np


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def boot(values, n, seed):
    v = np.asarray(values, dtype=np.float64)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], nargs="+", default=["practice"],
                    help="one split, or several to pool their queries (each query paired across k)")
    ap.add_argument("--csv", action="store_true", help="also write per-query scores as a CSV")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
                    help="only queries FIRST..LAST (1-based, manifest order), e.g. 16 30 for the second batch")
    args = ap.parse_args()
    from src.data import CourseData
    from src.metrics import recall_at_k
    data = CourseData(ROOT / "data/rag_corpus")
    res = ROOT / "results"
    if args.range and len(args.split) > 1:
        sys.exit("--range works with one split only")
    # Queries are keyed (split, query_id); a query counts only if it is answered and judged at every k.
    qrels, per_k, common = {}, {k: ({}, {}) for k in args.ks}, []
    for sp in args.split:
        qrels[sp] = data.load_qrels(sp)
        for k in args.ks:
            for r in jsonl(res / f"aq_{sp}_k{k}.jsonl"):
                per_k[k][0][(sp, r["query_id"])] = r
            for r in jsonl(res / f"aq_{sp}_k{k}_judge" / "answer_judge_per_query.jsonl"):
                per_k[k][1][(sp, r["query_id"])] = r
        order = [it["query_id"] for it in data.audio_items(sp)]  # manifest order = the order queries were run
        if args.range:
            order = order[args.range[0] - 1:args.range[1]]
        common += [(sp, q) for q in order
                   if all((sp, q) in a and (sp, q) in j and j[(sp, q)].get("normalised_answer_score") is not None
                          for a, j in per_k.values())]
    if not common:
        sys.exit("no query has both an answer and a judge score at every k")
    rows, scores = [], {}
    for k in args.ks:
        ans, jud = per_k[k]
        s = [jud[q]["normalised_answer_score"] for q in common]
        scores[k] = s
        rec = [recall_at_k(ans[q]["retrieved_record_ids"], qrels[q[0]][q[1]], k) for q in common]
        gen = [ans[q]["timing"]["generation_ms"] for q in common]
        ptok = [ans[q].get("usage", {}).get("prompt_tokens") for q in common if ans[q].get("usage")]
        ptok = [p for p in ptok if p is not None]
        words = [sum(len(data.record_text(i).split()) for i in ans[q]["retrieved_record_ids"]) for q in common]
        lo, hi = boot(s, args.bootstrap, args.seed)
        rows.append({"k": k, "n": len(common), "score": float(np.mean(s)), "score_ci95": [lo, hi],
                     **{d: float(np.mean([jud[q][d] for q in common])) for d in ("correctness", "grounding", "completeness")},
                     "recall": float(np.mean(rec)), "context_words": float(np.mean(words)),
                     "prompt_tokens_mean": float(np.mean(ptok)) if ptok else None,
                     "generation_ms_median": st.median(gen), "generation_ms_mean": st.mean(gen)})
    base = args.ks[0]
    diffs = []
    for i, kb in enumerate(args.ks):  # every pair of depths, paired over the same queries
        for k in args.ks[i + 1:]:
            d = np.asarray(scores[k]) - np.asarray(scores[kb])
            lo, hi = boot(d, args.bootstrap, args.seed)
            diffs.append({"k": k, "vs": kb, "mean_diff": float(d.mean()), "ci95": [lo, hi],
                          "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum())})
    name = args.split[0] if len(args.split) == 1 else "pooled_" + "_".join(args.split)
    out = {"split": args.split, "range": args.range, "queries": [list(q) for q in common], "rows": rows,
           "paired_vs_smallest_k": [d for d in diffs if d["vs"] == base], "paired_all": diffs,
           "zero_scores": {k: int(sum(s == 0 for s in scores[k])) for k in args.ks},
           "bootstrap": args.bootstrap, "seed": args.seed}
    tag = f"_q{args.range[0]}-{args.range[1]}" if args.range else ""
    path = res / f"aq_{name}_summary{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    if args.csv:
        import csv
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "query_id", "k", "judge_score", "correctness", "grounding", "completeness",
                        "relevant_in_top_k", "prompt_tokens", "generation_ms"])
            for k in args.ks:
                ans, jud = per_k[k]
                for q in common:
                    rel = sum(1 for i in ans[q]["retrieved_record_ids"] if i in qrels[q[0]][q[1]])
                    w.writerow([q[0], q[1], k, jud[q]["normalised_answer_score"], jud[q]["correctness"],
                                jud[q]["grounding"], jud[q]["completeness"], rel,
                                ans[q].get("usage", {}).get("prompt_tokens"), ans[q]["timing"]["generation_ms"]])
        print(f"-> {path.with_suffix('.csv')}")
    print(f"n = {len(common)} queries ({', '.join(args.split)}) answered and judged at every k in {args.ks}\n")
    print("| k | judge score (95% CI) | correct | grounded | complete | Recall@k | context words | prompt tokens | gen ms (median) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        pt = f"{r['prompt_tokens_mean']:.0f}" if r["prompt_tokens_mean"] else "-"
        print(f"| {r['k']} | {r['score']:.3f} ({r['score_ci95'][0]:.3f}-{r['score_ci95'][1]:.3f}) | {r['correctness']:.2f} | "
              f"{r['grounding']:.2f} | {r['completeness']:.2f} | {r['recall']:.3f} | {r['context_words']:.0f} | {pt} | "
              f"{r['generation_ms_median']:.0f} |")
    print("\nPaired differences in judge score (same queries at both depths)")
    print("| k vs k | mean diff (95% CI) | better / worse / tied |")
    print("| --- | --- | --- |")
    for d in diffs:
        print(f"| {d['k']} vs {d['vs']} | {d['mean_diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) | {d['better']} / {d['worse']} / {d['tied']} |")
    print("zero-score answers:", ", ".join(f"k = {k}: {out['zero_scores'][k]} of {len(common)}" for k in args.ks))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
