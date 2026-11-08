"""Answer generation from CACHED ASR transcripts (no ASR / no GPU needed).

Same pipeline as rag/scripts/generate_answers.py, but the ASR stage is replaced by a lookup in a
transcript cache written by evaluate_rag.py (--transcript-cache) or by the cluster ASR run.
Run from the course repo root:
  python rag/extensions/generate_from_transcripts.py --split practice \
      --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/baseline_k5.jsonl --limit 5
Timing note: timing.asr_ms is the cache lookup, NOT real ASR time; report real ASR time separately.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json


class CachedASR:
    """Drop-in for ASRAdapter.transcribe(): audio file name (query id) -> cached transcript."""

    def __init__(self, cache):
        self.cache = cache

    def transcribe(self, audio_path):
        qid = Path(audio_path).stem
        if qid not in self.cache:
            raise KeyError(f"no cached transcript for {qid}")
        return self.cache[qid]


def load_transcripts(path):
    cache = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cache[row["query_id"]] = row["transcript"]
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    args = ap.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.generator import ELMGenerator
    from src.rag import RAGPipeline
    from src.retriever import ExactDenseRetriever
    cache = load_transcripts(args.transcripts)
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    pipeline = RAGPipeline(CachedASR(cache), LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, SimpleContextBuilder(), ELMGenerator())
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k)
            row = {"query_id": item["query_id"], "split": args.split, "audio_file": item["audio_file"],
                   "transcript": result["transcript"], "answer": result["answer"],
                   "retrieved_record_ids": result["retrieval"]["record_ids"], "timing": result["timing"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])


if __name__ == "__main__":
    main()
