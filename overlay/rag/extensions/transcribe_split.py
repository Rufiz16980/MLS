"""One-off ASR transcription of the course audio queries (run on a GPU node, NO ELM key needed).

Writes the same cache format evaluate_rag.py uses ({"query_id", "transcript"}) plus "asr_ms"
(per-query ASR time, model loading excluded). Resumable: queries already in the output are skipped.

  cd <course repo root>   # after scripts/sync_overlay.sh and install_assets.py
  ASR_CUDA=none python rag/extensions/transcribe_split.py --split all
Output: rag/.cache/<split>_transcripts.jsonl (git-ignored). Use ASR_CUDA=all for our own kernels
(write to a different --out-dir so the PyTorch and our-kernel transcripts are never mixed).
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=ROOT / ".cache")
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    args = ap.parse_args()
    from src.asr_adapter import ASRAdapter
    from src.data import CourseData
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    asr = ASRAdapter(python=sys.executable, cuda_groups=os.environ.get("ASR_CUDA", "none"))
    asr.prepare()  # model loading happens here, outside every per-query timing
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in (["practice", "dev"] if args.split == "all" else [args.split]):
        out = args.out_dir / f"{split}_transcripts.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(l)["query_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        items = data.audio_items(split)[: args.limit]
        todo = [i for i in items if i["query_id"] not in done]
        print(f"{split}: {len(items)} queries, {len(done)} cached, {len(todo)} to do", flush=True)
        with out.open("a", encoding="utf-8") as f:
            for n, item in enumerate(todo, 1):
                t0 = time.perf_counter()
                text = asr.transcribe(data_dir / item["audio_file"])
                ms = (time.perf_counter() - t0) * 1000.0
                f.write(json.dumps({"query_id": item["query_id"], "transcript": text, "asr_ms": round(ms, 1)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {split} {n}/{len(todo)}", flush=True)
    asr.close()


if __name__ == "__main__":
    main()
