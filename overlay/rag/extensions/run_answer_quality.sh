#!/usr/bin/env bash
# Answer quality vs retrieval depth (Plan A, part 2). Run from the course repo root on the LAPTOP
# (needs the ELM key in rag/.env). Costs N x len(KS) x 2 ELM calls (one generate + one judge each).
# Resumable: re-running with a larger N only generates and judges the new queries
# (generation skips done query ids; the judge has its own cache).
#   bash rag/extensions/run_answer_quality.sh 15          # 15 queries x 3 depths x 2 = 90 calls
#   bash rag/extensions/run_answer_quality.sh 30          # adds 15 more: 90 more calls, 180 in total
set -euo pipefail
N="${1:?usage: run_answer_quality.sh N_QUERIES [SPLIT]}"
SPLIT="${2:-practice}"
KS="${KS:-1 5 20}"
T="rag/.cache/${SPLIT}_transcripts.jsonl"
for k in $KS; do
  echo "== k=$k: generate =="
  python rag/extensions/generate_from_transcripts.py --split "$SPLIT" --transcripts "$T" \
      --output "rag/results/aq_${SPLIT}_k${k}.jsonl" --k "$k" --limit "$N" --resume
  echo "== k=$k: judge =="
  python rag/scripts/evaluate_answers.py --split "$SPLIT" --candidates "rag/results/aq_${SPLIT}_k${k}.jsonl" \
      --allow-subset --out-dir "rag/results/aq_${SPLIT}_k${k}_judge"
done
python rag/extensions/answer_quality_summary.py --split "$SPLIT" --ks $KS
