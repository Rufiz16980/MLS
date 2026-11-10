#!/usr/bin/env bash
# One-paste status check for the cluster head node. Read-only apart from `git pull` of this repo
# and copying overlay/ into the course checkout (then the guard runs).
#   bash ~/MLS/scripts/cluster_status.sh
set -uo pipefail
COURSE="${MLS_COURSE_DIR:-$HOME/edin-mls-26-autumn}"
echo "== update repo + sync overlay"; git -C "$HOME/MLS" pull -q && bash "$HOME/MLS/scripts/sync_overlay.sh" "$COURSE" 2>&1 | tail -2
echo "== jobs now (empty = nothing queued or running)"; squeue -u "$USER"
echo "== recent jobs"; sacct -u "$USER" -S "$(date -d '3 days ago' +%F)" --format=JobID,JobName%12,State,Elapsed,NodeList -X 2>/dev/null | tail -15
cd "$COURSE/hw1-asr" || exit 1
echo "== result folders"
for d in ablation/results/*/; do echo "$d summary=$([ -f "$d/summary.md" ] && echo yes || echo no)"; done | tail -8
newest="$(ls -t ablation/results | head -1)"
echo "== newest summary ($newest)"; [ -f "ablation/results/$newest/summary.md" ] && cat "ablation/results/$newest/summary.md" || echo "(none yet)"
