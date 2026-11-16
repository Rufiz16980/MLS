import sys, json
from pathlib import Path
ROOT=Path("rag").resolve(); sys.path.insert(0,str(ROOT))
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
import numpy as np
d=ROOT/"data/rag_corpus"; data=CourseData(d)
enc=LocalEncoder(ROOT/"models/enc_a",device="cpu")
ret=ExactDenseRetriever(d/"embeddings/corpus_enc_a.npy", d/"embeddings/corpus_enc_a_ids.json")
for split in ("practice","dev"):
    cache={}
    for l in open(f"rag/.cache/{split}_transcripts.jsonl"):
        r=json.loads(l); cache[r["query_id"]]=r["transcript"]
    qrels=data.load_qrels(split); items=data.audio_items(split)
    first=[]
    for it in items:
        q=it["query_id"]; ids=ret.retrieve(enc.encode(cache[q]),k=100)["record_ids"]
        rel=qrels[q]; ranks=[i+1 for i,x in enumerate(ids) if x in rel]
        first.append(ranks[0] if ranks else None)
    n=len(first)
    print(split,n,"queries; share with >=1 relevant record in top k (hit@k):",
      {k: round(sum(1 for r in first if r and r<=k)/n,3) for k in (1,5,10,20,50,100)},
      "| never in top-100:",round(sum(1 for r in first if r is None)/n,3),
      "| median first-relevant rank (when found):", int(np.median([r for r in first if r])))
