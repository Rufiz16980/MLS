#!/usr/bin/env bash
# Build supplementary_code.zip from this repo: our files only, no Git history, no personal paths.
#   bash scripts/make_supplementary_zip.sh OUT_DIR
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: make_supplementary_zip.sh OUT_DIR}"
STAGE="$(mktemp -d)/supplementary_code"
mkdir -p "$STAGE/hw1-asr" "$STAGE/rag" "$STAGE/outputs"
cp -r "$REPO/overlay/hw1-asr/csrc" "$REPO/overlay/hw1-asr/ablation" "$STAGE/hw1-asr/"
cp -r "$REPO/overlay/rag/extensions" "$STAGE/rag/"
cp "$REPO"/docs/outputs/*.txt "$STAGE/outputs/"
if [ -d "$REPO/docs/outputs/rag" ]; then cp -r "$REPO/docs/outputs/rag" "$STAGE/outputs/"; fi
cp "$REPO/submission/README.md" "$STAGE/README.md"
cp "$REPO/docs/report/experiment-log.md" "$STAGE/EXPERIMENT_LOG.md"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
if grep -r -l -i -E "s[0-9]{7}|rufiz|/home/[a-z]|@gmail|@ed\.ac\.uk" "$STAGE"; then echo "personal data found, not zipping" >&2; exit 1; fi
mkdir -p "$OUT"; rm -f "$OUT/supplementary_code.zip"
(cd "$(dirname "$STAGE")" && zip -q -r -X "$OUT/supplementary_code.zip" supplementary_code)
unzip -l "$OUT/supplementary_code.zip" | tail -n +1
