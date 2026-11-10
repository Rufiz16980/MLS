#!/usr/bin/env bash
# Submit ablation stages as batch jobs that run one after the other (they share build folders).
#   bash ~/MLS/scripts/cluster_submit.sh best            # one stage
#   bash ~/MLS/scripts/cluster_submit.sh sweep e2e best  # chained: each starts when the previous ends
# Stages: check sweep e2e best. Runs on the RTX 2080 Ti (see hw1-asr/ablation/ablate.sbatch).
set -euo pipefail
COURSE="${MLS_COURSE_DIR:-$HOME/edin-mls-26-autumn}"
[ $# -ge 1 ] || { echo "usage: cluster_submit.sh STAGE [STAGE...]" >&2; exit 2; }
git -C "$HOME/MLS" pull -q
bash "$HOME/MLS/scripts/sync_overlay.sh" "$COURSE" | tail -1
cd "$COURSE/hw1-asr"
prev=""
for stage in "$@"; do
  case "$stage" in check|sweep|e2e|best) ;; *) echo "unknown stage: $stage" >&2; exit 2;; esac
  if [ -z "$prev" ]; then prev=$(sbatch --parsable ablation/ablate.sbatch --stage "$stage")
  else prev=$(sbatch --parsable --dependency=afterany:"$prev" ablation/ablate.sbatch --stage "$stage"); fi
  echo "submitted $stage as job $prev"
done
squeue -u "$USER"
