"""Recall of fixed depths k = 10, 14, 15, 16, 20 and their mean context words, to compare with an adaptive policy at equal context.
Run from the course repo root with the venv python:  ~/mls-rag-env/bin/python docs/analysis/fixed_k_matched_context.py
"""
import json, sys
from pathlib import Path
ROOT = Path("rag").resolve(); sys.path.insert(0, str(ROOT))
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.metrics import recall_at_k
import numpy as np
d = ROOT / "data/rag_corpus"; data = CourseData(d)
enc = LocalEncoder(ROOT / "models/enc_a", device="cpu")
ret = ExactDenseRetriever(d / "embeddings/corpus_enc_a.npy", d / "embeddings/corpus_enc_a_ids.json")
for split in ("practice", "dev"):
    cache = {json.loads(l)["query_id"]: json.loads(l)["transcript"] for l in open(f"rag/.cache/{split}_transcripts.jsonl")}
    q = data.load_qrels(split); items = data.audio_items(split)
    ranked = [ret.retrieve(enc.encode(cache[i["query_id"]]), k=25)["record_ids"] for i in items]
    for k in (10, 14, 15, 16, 20):
        rec = np.mean([recall_at_k(r, q[i["query_id"]], k) for r, i in zip(ranked, items)])
        words = np.mean([sum(len(data.record_text(x).split()) for x in r[:k]) for r in ranked])
        print(split, "fixed k=%d recall=%.3f context words=%.0f" % (k, rec, words))
