#!/usr/bin/env bash
# Copy OUR files (overlay/) into a course checkout, then run the guard.
# Usage: bash scripts/sync_overlay.sh [COURSE_DIR]     (default: $MLS_COURSE_DIR or ./edin-mls-26-autumn)
# Direction is always repo -> course checkout. Edit overlay/ here, never the frozen course files.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COURSE="${1:-${MLS_COURSE_DIR:-edin-mls-26-autumn}}"
[ -d "$COURSE/hw1-asr" ] || { echo "not a course checkout: $COURSE" >&2; exit 2; }
( cd "$HERE/overlay" && find . -type f | while read -r f; do
    mkdir -p "$COURSE/$(dirname "$f")"; cp "$f" "$COURSE/$f"; done )
python3 "$HERE/tools/guard/guard.py" check --course "$COURSE"
