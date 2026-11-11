#!/usr/bin/env bash
# Answer quality vs retrieval depth (Plan A, part 2). Run from the course repo root on the LAPTOP
# (needs the ELM key in rag/.env). Costs N x len(KS) x 2 ELM calls (one generate + one judge each).
# Resumable: re-running with a larger N only generates and judges the new queries
# (generation skips done query ids; the judge has its own cache).
#   bash rag/extensions/run_answer_quality.sh 15          # 15 queries x 3 depths x 2 = 90 calls
#   bash rag/extensions/run_answer_quality.sh 30          # adds 15 more: 90 more calls, 180 in total
#   EXCLUDE="q_xxx" bash rag/extensions/run_answer_quality.sh 30   # skip queries the judge cannot parse
set -euo pipefail
N="${1:?usage: run_answer_quality.sh N_QUERIES [SPLIT]}"
SPLIT="${2:-practice}"
KS="${KS:-1 5 20}"
# EXCLUDE: space-separated query ids the course judge cannot score (its JSON reply is malformed at
# temperature 0, so retrying fails identically). They are dropped from judging at EVERY k, so all
# depths stay paired; answers are still generated and kept. Record the ids in the experiment log.
EXCLUDE="${EXCLUDE:-}"
T="rag/.cache/${SPLIT}_transcripts.jsonl"
for k in $KS; do
  echo "== k=$k: generate =="
  python rag/extensions/generate_from_transcripts.py --split "$SPLIT" --transcripts "$T" \
      --output "rag/results/aq_${SPLIT}_k${k}.jsonl" --k "$k" --limit "$N" --resume
  echo "== k=$k: judge =="
  CAND="rag/results/aq_${SPLIT}_k${k}.jsonl"
  if [ -n "$EXCLUDE" ]; then
    CAND="rag/results/aq_${SPLIT}_k${k}.judged.jsonl"
    EXCLUDE="$EXCLUDE" python - "rag/results/aq_${SPLIT}_k${k}.jsonl" "$CAND" <<'PY'
import json, os, sys
skip = set(os.environ["EXCLUDE"].split())
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip() and json.loads(l)["query_id"] not in skip]
open(sys.argv[2], "w", encoding="utf-8").writelines(rows)
print(f"judging {len(rows)} answers ({len(skip)} excluded)")
PY
  fi
  python rag/scripts/evaluate_answers.py --split "$SPLIT" --candidates "$CAND" \
      --allow-subset --out-dir "rag/results/aq_${SPLIT}_k${k}_judge"
done
python rag/extensions/answer_quality_summary.py --split "$SPLIT" --ks $KS
if [ "$N" -gt 15 ]; then   # batch check: the first 15 (earlier run) and the new queries, same settings
  echo; echo "== batch check: queries 1-15 =="; python rag/extensions/answer_quality_summary.py --split "$SPLIT" --ks $KS --range 1 15
  echo; echo "== batch check: queries 16-$N =="; python rag/extensions/answer_quality_summary.py --split "$SPLIT" --ks $KS --range 16 "$N"
fi
