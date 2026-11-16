"""Pretty-print one result line of generate_from_transcripts.py for the video demo (no API calls, no personal data).
Run from the course repo root with the RAG venv:
    python docs/video/demo_show.py rag/results/demo_k5.jsonl [LINE_NUMBER]
"""
import json
import sys
from pathlib import Path

ROOT = Path("rag").resolve()
sys.path.insert(0, str(ROOT))
from src.data import CourseData  # noqa: E402

path = Path(sys.argv[1])
line = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
r = rows[line - 1]
data = CourseData(ROOT / "data/rag_corpus")
print("query id   :", r["query_id"])
print("transcript :", r["transcript"], "   <- ASR output (cached, see video)")
print("retrieved  :", len(r["retrieved_record_ids"]), "records")
for i, rid in enumerate(r["retrieved_record_ids"][:3], 1):
    print(f"  evidence {i}: {data.record_text(rid)[:150].strip()!r}...")
print("answer     :", r["answer"][:600].replace("\n", " "))
t = r["timing"]
print("timing ms  :", {k: round(v) for k, v in t.items()})
