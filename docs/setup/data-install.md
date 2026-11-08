# Installing the course data and RAG environment (laptop, no GPU needed)

The RAG side (retrieval, answer generation, judging) runs on a normal laptop with CPU-only PyTorch. Only the ASR CUDA kernels need the cluster GPU.

## One command

From the folder where you want the course code (outside any cloud-synced folder such as Google Drive or Dropbox):

```bash
git clone https://github.com/Rufiz16980/MLS.git
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
bash MLS/scripts/setup_rag_local.sh edin-mls-26-autumn
```

The script clones the course repo if missing, fetches `rag-assets.zip` through Git LFS (about 187 MiB; needs `git-lfs`, e.g. `sudo apt install git-lfs`), verifies its checksum, builds a venv at `~/mls-rag-env` with CPU torch (about 1 GB; do NOT install plain `pip install torch`, it pulls about 5 GB of CUDA packages), extracts the data to `~/mls-rag-assets`, installs it into `rag/`, creates `rag/.env` from the template and runs `check_setup.py`.

## Then, by hand

1. Put your own ELM key in `rag/.env` as `ELM_API_KEY=...` (see [elm-api-key.md](elm-api-key.md)). Never commit or share it.
2. `git status` in the course repo will list `rag-assets.zip` as modified. This is expected (the LFS pointer is replaced by the real file). Never commit it.
3. Each session: `source ~/mls-rag-env/bin/activate`.

## Expected check_setup.py result

PASS for corpus, audio manifest, embeddings, enc_a model and ELM configuration. The ASR source check only confirms the path; real ASR inference runs on the cluster GPU.

## Facts

- Corpus: 24,000 records; 475 audio queries (350 practice, 125 dev); enc_a and enc_b corpus vectors (24,000 x 384).
- `rag-assets.zip`: 196,521,100 bytes, sha256 `3b830368271e935b60791dc0955af2b0d50692d575150189475c6306e6c9e3b8`.
