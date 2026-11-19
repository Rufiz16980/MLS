"""Demo of the whole system on ONE spoken query, in two stages (for the video).

Stage 1, `asr` (GPU node, no ELM key): our CUDA kernels transcribe the query's audio file.
Stage 2, `answer` (laptop, ELM key in rag/.env): encode the transcript, retrieve, build the context,
generate the answer (one ELM call), and print every stage's time.
The ELM key never leaves the laptop, so the transcript is carried over as a small JSON file.

  # 1. cluster, GPU node, from the course repo root (conda env mls-asr):
  srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --mem=16G --time=00:20:00 \
      python rag/extensions/demo_query.py asr --query-id q_7e9a0f2716c017d2
  #    -> rag/results/demo_transcript.json
  # 2. laptop, course repo root (venv with the RAG requirements), after copying that file over:
  python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k 5
  python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k adaptive
Without --query-id the first query of --split is used. `answer --transcript "..." --query-id ID`
works without stage 1 (e.g. for a rehearsal).
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import os
import time

DATA_DIR = ROOT / "data/rag_corpus"


def pick_item(data, split, query_id):
    items = data.audio_items(split)
    if query_id is None:
        return items[0]
    for it in items:
        if it["query_id"] == query_id:
            return it
    sys.exit(f"{query_id} is not a {split} query")


def cmd_asr(args):
    from src.asr_adapter import ASRAdapter
    from src.data import CourseData
    data = CourseData(DATA_DIR)
    item = pick_item(data, args.split, args.query_id)
    audio = DATA_DIR / item["audio_file"]
    print(f"query {item['query_id']} ({args.split}), audio {item['audio_file']}")
    print(f"ASR kernels: ASR_CUDA={args.cuda}  (all = our four CUDA kernels, none = PyTorch only)")
    asr = ASRAdapter(python=sys.executable, cuda_groups=args.cuda)
    t0 = time.perf_counter()
    asr.prepare()
    print(f"model loaded in {time.perf_counter() - t0:.1f} s (not part of the per-query time)")
    t0 = time.perf_counter()
    text = asr.transcribe(audio)
    ms = (time.perf_counter() - t0) * 1000.0
    asr.close()
    print(f"\nTRANSCRIPT: {text}\nASR time: {ms:.0f} ms")
    cache = ROOT / f".cache/{args.split}_transcripts.jsonl"
    if cache.exists():
        ref = {json.loads(l)["query_id"]: json.loads(l)["transcript"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
        if item["query_id"] in ref:
            same = ref[item["query_id"]].strip() == text.strip()
            print(f"cached PyTorch transcript: {ref[item['query_id']]}\nidentical: {same}")
    out = ROOT / "results/demo_transcript.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"query_id": item["query_id"], "split": args.split, "transcript": text,
                               "asr_ms": round(ms, 1), "asr_cuda": args.cuda}, ensure_ascii=False, indent=1))
    print(f"-> {out.relative_to(ROOT.parent)}")


def adaptive_k(scores):
    cfg = json.loads((ROOT / "results/adaptive_k.json").read_text())
    feat = scores[0] if cfg["feature"] == "s1" else scores[0] - scores[1]
    k = cfg["k_lo"] if feat >= cfg["tau"] else cfg["k_hi"]
    return k, f"{cfg['feature']} = {feat:.4f} {'>=' if feat >= cfg['tau'] else '<'} tau {cfg['tau']:.4f}"


def cmd_answer(args):
    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    from generate_from_transcripts import usage_generator
    if args.source:
        src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    elif args.transcript and args.query_id:
        src = {"query_id": args.query_id, "split": args.split, "transcript": args.transcript, "asr_ms": None}
    else:
        sys.exit("give --from FILE, or --transcript TEXT with --query-id ID")
    data = CourseData(DATA_DIR)
    rel = data.load_qrels(src["split"]).get(src["query_id"], {})
    print(f"query {src['query_id']} ({src['split']}), relevant records in qrels: {len(rel)}")
    print(f"TRANSCRIPT: {src['transcript']}")
    encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(DATA_DIR / "embeddings/corpus_enc_a.npy", DATA_DIR / "embeddings/corpus_enc_a_ids.json")
    t0 = time.perf_counter()
    vec = encoder.encode(src["transcript"])
    emb_ms = (time.perf_counter() - t0) * 1000.0
    if args.k == "adaptive":
        top2 = retriever.retrieve(vec, k=2)
        k, why = adaptive_k(top2["scores"])
        print(f"adaptive k: {why} -> k = {k}")
    else:
        k = int(args.k)
    r = retriever.retrieve(vec, k=k)
    print(f"\nTop {k} records (exact cosine over {len(retriever.record_ids)} records, {r['latency_ms']:.2f} ms):")
    for i, (rid, s) in enumerate(zip(r["record_ids"], r["scores"]), 1):
        text = " ".join(data.record_text(rid).split())
        print(f"  {i:2d}. {s:.3f}  {'RELEVANT ' if rid in rel else '         '}{rid}  {text[:90]}")
    built = SimpleContextBuilder().build(src["transcript"], r["record_ids"], data)
    gen = usage_generator()
    out = gen.generate(built["prompt"])
    usage = getattr(gen, "last_usage", {}) or {}
    print(f"\nANSWER ({gen.model}):\n{out['answer']}\n")
    asr_ms = src.get("asr_ms")
    total = (asr_ms or 0) + emb_ms + r["latency_ms"] + out["latency_ms"]
    print("| stage | ms |\n| --- | --- |")
    print(f"| ASR ({src.get('asr_cuda', 'n/a')} kernels, GPU) | {asr_ms if asr_ms is not None else 'not run'} |")
    print(f"| query embedding (CPU) | {emb_ms:.0f} |\n| retrieval | {r['latency_ms']:.2f} |")
    print(f"| generation (ELM, k = {k}) | {out['latency_ms']:.0f} |\n| total | {total:.0f} |")
    print(f"prompt tokens {usage.get('prompt_tokens')}, completion tokens {usage.get('completion_tokens')}")
    res = ROOT / "results/demo_answer.json"
    res.write_text(json.dumps({**src, "k": k, "retrieved": r["record_ids"], "scores": r["scores"],
                               "relevant_retrieved": [x for x in r["record_ids"] if x in rel], "answer": out["answer"],
                               "timing": {"embedding_ms": emb_ms, "retrieval_ms": r["latency_ms"],
                                          "generation_ms": out["latency_ms"]}, "usage": usage, "model": gen.model},
                              ensure_ascii=False, indent=1))
    print(f"-> {res.relative_to(ROOT.parent)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("asr", help="stage 1, GPU: transcribe one query with our kernels")
    a.add_argument("--split", choices=["practice", "dev"], default="practice")
    a.add_argument("--query-id", default=None)
    a.add_argument("--cuda", default=os.environ.get("ASR_CUDA", "all"))
    b = sub.add_parser("answer", help="stage 2, laptop: retrieve and answer (one ELM call)")
    b.add_argument("--from", dest="source", default=None, help="demo_transcript.json from stage 1")
    b.add_argument("--transcript", default=None)
    b.add_argument("--query-id", default=None)
    b.add_argument("--split", choices=["practice", "dev"], default="practice")
    b.add_argument("--k", default="5", help="a number, or 'adaptive' (uses rag/results/adaptive_k.json)")
    args = ap.parse_args()
    cmd_asr(args) if args.cmd == "asr" else cmd_answer(args)


if __name__ == "__main__":
    main()
