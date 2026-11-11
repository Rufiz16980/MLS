"""Summarise the answer-quality-vs-depth experiment (no ELM calls).

Reads, for each k, the answers  rag/results/aq_<split>_k<k>.jsonl  (generate_from_transcripts.py)
and the judge output  rag/results/aq_<split>_k<k>_judge/answer_judge_per_query.jsonl  (evaluate_answers.py),
keeps only the queries present at EVERY k (paired comparison), and reports per k:
judge score (mean + 95% bootstrap CI), the three sub-scores, Recall@k on the same queries,
prompt tokens, generation latency, and paired differences against the smallest k.

  cd <course repo root>
  python rag/extensions/answer_quality_summary.py --split practice --ks 1 5 20
Writes rag/results/aq_<split>_summary.json and prints markdown tables.
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
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
                    help="only queries FIRST..LAST (1-based, manifest order), e.g. 16 30 for the second batch")
    args = ap.parse_args()
    from src.data import CourseData
    from src.metrics import recall_at_k
    data = CourseData(ROOT / "data/rag_corpus")
    qrels = data.load_qrels(args.split)
    res = ROOT / "results"
    per_k = {}
    for k in args.ks:
        ans = {r["query_id"]: r for r in jsonl(res / f"aq_{args.split}_k{k}.jsonl")}
        jud = {r["query_id"]: r for r in jsonl(res / f"aq_{args.split}_k{k}_judge" / "answer_judge_per_query.jsonl")}
        per_k[k] = (ans, jud)
    both = set.intersection(*[set(a) & set(j) for a, j in per_k.values()])
    order = [it["query_id"] for it in data.audio_items(args.split)]  # manifest order = the order queries were run
    if args.range:
        order = order[args.range[0] - 1:args.range[1]]
    common = [q for q in order if q in both]
    if not common:
        sys.exit("no query has both an answer and a judge score at every k")
    rows, scores = [], {}
    for k in args.ks:
        ans, jud = per_k[k]
        s = [jud[q]["normalised_answer_score"] for q in common]
        scores[k] = s
        rec = [recall_at_k(ans[q]["retrieved_record_ids"], qrels[q], k) for q in common]
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
    for k in args.ks[1:]:
        d = np.asarray(scores[k]) - np.asarray(scores[base])
        lo, hi = boot(d, args.bootstrap, args.seed)
        diffs.append({"k": k, "vs": base, "mean_diff": float(d.mean()), "ci95": [lo, hi],
                      "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum())})
    out = {"split": args.split, "range": args.range, "queries": common, "rows": rows, "paired_vs_smallest_k": diffs,
           "bootstrap": args.bootstrap, "seed": args.seed}
    tag = f"_q{args.range[0]}-{args.range[1]}" if args.range else ""
    path = res / f"aq_{args.split}_summary{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"n = {len(common)} {args.split} queries answered and judged at every k in {args.ks}\n")
    print("| k | judge score (95% CI) | correct | grounded | complete | Recall@k | context words | prompt tokens | gen ms (median) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        pt = f"{r['prompt_tokens_mean']:.0f}" if r["prompt_tokens_mean"] else "-"
        print(f"| {r['k']} | {r['score']:.3f} ({r['score_ci95'][0]:.3f}-{r['score_ci95'][1]:.3f}) | {r['correctness']:.2f} | "
              f"{r['grounding']:.2f} | {r['completeness']:.2f} | {r['recall']:.3f} | {r['context_words']:.0f} | {pt} | "
              f"{r['generation_ms_median']:.0f} |")
    print("\nPaired difference in judge score against k =", base)
    print("| k | mean diff (95% CI) | better / worse / tied |")
    print("| --- | --- | --- |")
    for d in diffs:
        print(f"| {d['k']} | {d['mean_diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) | {d['better']} / {d['worse']} / {d['tied']} |")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
