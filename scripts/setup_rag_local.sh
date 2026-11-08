#!/usr/bin/env bash
# One-time RAG setup for a laptop/desktop WITHOUT an NVIDIA GPU (CPU-only torch).
# Usage:  bash scripts/setup_rag_local.sh [COURSE_REPO_DIR]
#   COURSE_REPO_DIR defaults to ./edin-mls-26-autumn (cloned if missing).
# It never reads, prints or writes your ELM key. Put the key in rag/.env yourself.
set -euo pipefail

COURSE_URL="https://github.com/ed-aisys/edin-mls-26-autumn.git"
REPO="${1:-edin-mls-26-autumn}"
VENV="${MLS_RAG_VENV:-$HOME/mls-rag-env}"
BUNDLE="${MLS_RAG_ASSETS:-$HOME/mls-rag-assets}"   # keep OUTSIDE the repo and any cloud-synced folder
ZIP_SHA256="3b830368271e935b60791dc0955af2b0d50692d575150189475c6306e6c9e3b8"
ZIP_SIZE=196521100

say() { printf '\n== %s\n' "$*"; }

say "1/5 course repo"
if [ ! -d "$REPO/.git" ]; then git clone "$COURSE_URL" "$REPO"; fi
cd "$REPO"

say "2/5 rag-assets.zip (Git LFS, about 187 MiB)"
size=$(stat -c %s rag-assets.zip 2>/dev/null || echo 0)
if [ "$size" -ne "$ZIP_SIZE" ]; then
  if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is missing. Install it (Ubuntu/Debian: sudo apt install git-lfs), then re-run." >&2
    exit 1
  fi
  git lfs install --local
  git lfs pull --include="rag-assets.zip"
fi
echo "$ZIP_SHA256  rag-assets.zip" | sha256sum -c -

say "3/5 python environment ($VENV, CPU torch)"
if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -r rag/requirements.txt

say "4/5 extract and install course data"
mkdir -p "$BUNDLE"
unzip -n -q rag-assets.zip -d "$BUNDLE"
python rag/scripts/install_assets.py --from-dir "$BUNDLE"

say "5/5 checks"
[ -f rag/.env ] || cp rag/.env.example rag/.env
chmod 600 rag/.env
git check-ignore -q rag/.env && echo "rag/.env is git-ignored: OK"
python rag/scripts/check_setup.py || true
cat <<MSG

Done. Remaining by hand:
  * edit rag/.env and set ELM_API_KEY=<your own key>  (never commit or share it)
  * 'git status' will show rag-assets.zip as modified: that is expected. Never commit it.
  * ASR checks need the GPU cluster (see docs/setup/gpu-access.md).
MSG
