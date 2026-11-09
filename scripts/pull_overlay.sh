#!/usr/bin/env bash
# Copy edited task files FROM a course checkout back into overlay/ (so they can be committed).
# Usage: bash scripts/pull_overlay.sh [COURSE_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COURSE="${1:-${MLS_COURSE_DIR:-edin-mls-26-autumn}}"
COURSE="$(cd "$COURSE" && pwd)"
for f in hw1-asr/csrc/task1_rope.cu hw1-asr/csrc/task2_reduce.cuh hw1-asr/csrc/task2_reduce.cu \
         hw1-asr/csrc/task3_gemm.cu hw1-asr/csrc/task4_attention.cu; do
  cp "$COURSE/$f" "$HERE/overlay/$f"
done
python3 "$HERE/tools/guard/guard.py" selftest
