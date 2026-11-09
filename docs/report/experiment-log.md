# Experiment log

Format per entry: **Need** (why), **Command**, **Output**, **What it means**. Cluster login shown as `<UUN>`.
Reference hardware: Informatics Teaching partition, one NVIDIA GeForce RTX 2080 Ti (11 GB, sm_75), CUDA toolkit 12.8 (`/opt/cuda-12.8.0`).

## E01: Course code and data pinned and verified

**Need.** Everyone must work from the same pristine course code and data, so results are comparable and frozen files are provably untouched.

```bash
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git ~/edin-mls-26-autumn
cd ~/edin-mls-26-autumn && git log -1 --oneline
echo "3b830368271e935b60791dc0955af2b0d50692d575150189475c6306e6c9e3b8  rag-assets.zip" | sha256sum -c -
python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
```

```text
6bbdfbc (HEAD -> main, origin/main, origin/HEAD) Update final report instructions in README
rag-assets.zip: OK
Course assets verified: 492 files, 492 installed
```

**What it means.** The cluster checkout is course commit `6bbdfbc`, the same commit our frozen-file manifest is built from. The 196,521,100-byte asset archive matches its published SHA-256, and the installer's own checks passed for all 492 files (24,000 corpus records, 475 audio queries, enc_a and enc_b corpus vectors, enc_a model).

## E02: Frozen-file guard

**Need.** The coursework forbids changing the supplied model code, tests and benchmark. We guard them from outside the course repo.

```bash
bash scripts/sync_overlay.sh ~/edin-mls-26-autumn      # copies our files in, then checks
python3 tools/guard/guard.py check --course ~/edin-mls-26-autumn
```

```text
guard OK: 55 frozen files intact, 5 task files conform
```

**What it means.** 55 supplied files are byte-identical to commit `6bbdfbc`, and the five task files still expose every original function signature (including the `launch_*` entry points) and include no cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch headers. A tamper test (modified `ops.py`, renamed `launch_rope`, added cuBLAS include) is caught by the guard and by CI.

## E03: ASR baseline on PyTorch (all four task slots on PyTorch)

**Need.** The reference our own CUDA kernels will be compared against: time per transcription and per token, peak GPU memory and correctness.

```bash
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=00:45:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH && conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr && make && ./run.sh --cuda none
```

```text
GPU: NVIDIA GeForce RTX 2080 Ti (sm_75)
Kernels: rope=torch  reduce=torch  gemm=torch  attention=torch  provided=torch
Loaded in 153.5 s
Audio: 3.50 s, mel features (1, 128, 3000), prompt 59 tokens
  run 1:   1213.6 ms  (13 tokens)
  run 2:   1217.4 ms  (13 tokens)
  run 3:   1215.3 ms  (13 tokens)
Time: 1215.4 ms per transcription (+/- 1.5 ms), 93.5 ms per token
Peak GPU memory: 8.88 GiB
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

Node: `opencast` (single allocation, 2026-11-09).

**What it means.**
- The model produces the right transcript, so the setup is correct (PASS above 80%).
- 1215.4 ms is the mean of 3 timed runs after 1 warm-up (std 1.5 ms, i.e. very stable within a run). Per token = 1215.4 / 13 tokens = 93.5 ms. There is no KV cache, so every one of the 13 decoder passes recomputes all positions (59 growing to 71).
- Peak memory 8.88 GiB is consistent with 2.3 B float32 parameters (about 8.6 GiB of weights) plus activations; it fits the 11 GB card with little headroom, which constrains any experiment that adds GPU memory.
- The handout's reference shows 992.9 ms on its own node; ours is 1.22 times slower. The handout says times differ a lot between nodes, so this is not an error, but it means we only compare against numbers measured on the same GPU type, and we should repeat the baseline on a second allocation to report between-node variability. Model loading (153.5 s, cold network filesystem) is excluded from the per-transcription time.

PENDING: repeat on a second allocation; record `nvidia-smi -L` and driver version in the same session.

## E04: One-off transcription of all course audio queries (PyTorch ASR)

**Need.** ASR needs the GPU, but all later RAG experiments should reuse identical transcripts (the handout requires recording this choice). So we transcribe the 350 practice and 125 dev queries once and cache them.

```bash
cd ~/edin-mls-26-autumn
ASR_CUDA=none python rag/extensions/transcribe_split.py --split all
```

```text
practice: 350 queries, 0 cached, 350 to do
  practice 25/350 ... practice 350/350
dev: 125 queries, 0 cached, 125 to do
  dev 25/125 ... dev 125/125
```

**What it means.** All 475 queries completed without error. Output: `rag/.cache/practice_transcripts.jsonl` and `dev_transcripts.jsonl` (git-ignored; they hold query id, transcript and per-query ASR time with model loading excluded). Everything downstream (retrieval metrics, the k-sweep, answer generation) runs from this cache, so retrieval differences between conditions cannot be caused by ASR noise.

PENDING (not yet seen): line counts (expect 350 and 125), a sample of transcripts, mean per-query ASR time, and a manual check for empty or obviously wrong transcripts. Commands: `wc -l rag/.cache/*.jsonl`, `head -c 600 rag/.cache/practice_transcripts.jsonl`.

## E05: RAG setup on the laptop (CPU only)

**Need.** Everything after ASR runs on a laptop without a GPU, with the ELM key kept off shared machines.

```bash
python3 -m venv ~/mls-rag-env && source ~/mls-rag-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r rag/requirements.txt
python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
python rag/scripts/check_setup.py
python -m unittest discover -s rag/tests -v
```

```text
Course assets verified: 492 files, 492 installed
[PASS] corpus.jsonl: 24000 records
[PASS] student audio manifest: 475 audio items
[PASS] corpus embeddings: shape=(24000, 384)
[PASS] enc_a model directory: model payload present
[PASS] ELM configuration: API key configured
[PASS] ASR source path
Ran 3 tests ... OK
```

**What it means.** The retrieval side (24,000 records, 384-dimensional enc_a vectors) is ready on the laptop. Using CPU-only torch keeps the environment at about 1.5 GB instead of 5.8 GB. The ASR line only confirms the source path; real ASR inference was verified separately in E03/E04.
