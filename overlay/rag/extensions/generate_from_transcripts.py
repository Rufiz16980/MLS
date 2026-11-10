"""Answer generation from CACHED ASR transcripts (no ASR / no GPU needed).

Same pipeline as rag/scripts/generate_answers.py, but the ASR stage is replaced by a lookup in a
transcript cache written by evaluate_rag.py (--transcript-cache) or by the cluster ASR run.
Run from the course repo root:
  python rag/extensions/generate_from_transcripts.py --split practice \
      --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/baseline_k5.jsonl --limit 5
Timing note: timing.asr_ms is the cache lookup, NOT real ASR time; report real ASR time separately.
--resume appends to an existing --output and skips query ids already in it, so growing a run
(e.g. --limit 15, later --limit 30 --resume) makes ELM calls only for the new queries.
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


def usage_generator():
    """ELMGenerator with the same request, plus the token counts the API returns (response.usage).
    The supplied generator is frozen, so this subclass lives here; the prompt and settings are identical."""
    import time

    class UsageELMGenerator(ELMGenerator):
        def generate(self, prompt):
            start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Use the supplied evidence to answer the user's question. "
                        "Do not invent facts that are unsupported by the evidence.")},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            u = getattr(response, "usage", None)
            self.last_usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                               "completion_tokens": getattr(u, "completion_tokens", None)}
            return {"answer": response.choices[0].message.content.strip(), "latency_ms": elapsed_ms}

    return UsageELMGenerator()


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
    ap.add_argument("--resume", action="store_true", help="Append to --output, skipping query ids already there")
    args = ap.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    done = set()
    if args.output.exists():
        if not args.resume:
            ap.error("Output exists; choose a new filename, or pass --resume to extend it")
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("split") != args.split:
                        ap.error("--resume: existing output is from another split")
                    done.add(row["query_id"])
    from src.context_builder import SimpleContextBuilder
    from src.data import CourseData
    from src.encoder import LocalEncoder
    from src.rag import RAGPipeline
    from src.retriever import ExactDenseRetriever
    cache = load_transcripts(args.transcripts)
    generator = usage_generator()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    pipeline = RAGPipeline(CachedASR(cache), LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, SimpleContextBuilder(), generator)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if it["query_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to generate (one ELM call each)")
    with args.output.open("a" if args.output.exists() else "x", encoding="utf-8") as f:
        for item in todo:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k)
            row = {"query_id": item["query_id"], "split": args.split, "audio_file": item["audio_file"],
                   "transcript": result["transcript"], "answer": result["answer"],
                   "k": args.k, "retrieved_record_ids": result["retrieval"]["record_ids"], "timing": result["timing"],
                   "usage": getattr(generator, "last_usage", None), "model": generator.model}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])


if __name__ == "__main__":
    main()
