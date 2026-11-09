"""Retrieval-depth sweep from CACHED transcripts (no ASR, no GPU, no ELM calls).

For one split it encodes every transcript once, retrieves the top k_max=100 once, and reports for each
depth k: Recall@k (mean + 95% bootstrap CI over queries), retrieval latency for retrieving k records
(median/mean ms, repeated), and the size of the context that depth would create (records, words, chars
of the concatenated record texts). MRR@10 and NDCG@10 do not depend on k (fixed cutoff) and are reported once.

  cd <course repo root>
  python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl
Writes rag/results/retrieval_sweep_<split>.json and prints a markdown table.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import statistics as st
import time
import numpy as np


def load_transcripts(path):
    cache = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cache[row["query_id"]] = row["transcript"]
    return cache


def boot_ci(values, n_boot, seed):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=5, help="latency repeats per query and depth")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
    from src.retriever import ExactDenseRetriever

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)
    cache = load_transcripts(args.transcripts)
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy",
                                    data_dir / "embeddings/corpus_enc_a_ids.json")
    ks = sorted(set(args.ks))
    kmax = max(max(ks), 100)
    vecs, ranked_all, rels = [], [], []
    for item in items:
        qid = item["query_id"]
        v = encoder.encode(cache[qid])
        vecs.append(v)
        ranked_all.append(retriever.retrieve(v, k=kmax)["record_ids"])
        rels.append(qrels[qid])
    rec = {k: [recall_at_k(r, rel, k) for r, rel in zip(ranked_all, rels)] for k in ks}
    mrr = [mrr_at_k(r, rel, 10) for r, rel in zip(ranked_all, rels)]
    ndcg = [ndcg_at_k(r, rel, 10) for r, rel in zip(ranked_all, rels)]
    rows = []
    for k in ks:
        lat = []
        for v in vecs:
            for _ in range(args.repeats):
                lat.append(retriever.retrieve(v, k=k)["latency_ms"])
        words = [sum(len(data.record_text(i).split()) for i in r[:k]) for r in ranked_all]
        chars = [sum(len(data.record_text(i)) for i in r[:k]) for r in ranked_all]
        lo, hi = boot_ci(rec[k], args.bootstrap, args.seed)
        rows.append({"k": k, "recall": float(np.mean(rec[k])), "recall_ci95": [lo, hi],
                     "latency_ms_median": st.median(lat), "latency_ms_mean": st.mean(lat),
                     "context_words_mean": float(np.mean(words)), "context_chars_mean": float(np.mean(chars))})
    out = {"split": args.split, "n_queries": len(items), "transcripts": str(args.transcripts.name),
           "mrr@10": float(np.mean(mrr)), "ndcg@10": float(np.mean(ndcg)),
           "mrr@10_ci95": list(boot_ci(mrr, args.bootstrap, args.seed)),
           "ndcg@10_ci95": list(boot_ci(ndcg, args.bootstrap, args.seed)),
           "bootstrap": args.bootstrap, "seed": args.seed, "latency_repeats": args.repeats, "rows": rows}
    path = args.out or ROOT / "results" / f"retrieval_sweep_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"| k | Recall@k (95% CI) | retrieval ms (median) | context words | context chars |")
    print("| --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['k']} | {r['recall']:.3f} ({r['recall_ci95'][0]:.3f}-{r['recall_ci95'][1]:.3f}) | "
              f"{r['latency_ms_median']:.2f} | {r['context_words_mean']:.0f} | {r['context_chars_mean']:.0f} |")
    print(f"MRR@10 {out['mrr@10']:.3f}  NDCG@10 {out['ndcg@10']:.3f}  (n={len(items)}) -> {path}")


if __name__ == "__main__":
    main()
