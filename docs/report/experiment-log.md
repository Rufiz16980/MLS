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

**Verification (cluster, head node, read-only).**

```bash
cd ~/edin-mls-26-autumn && wc -l rag/.cache/*.jsonl && head -c 600 rag/.cache/practice_transcripts.jsonl
python3 - <<'EOF'
import json,statistics as s
for sp in ("practice","dev"):
    r=[json.loads(l) for l in open(f"rag/.cache/{sp}_transcripts.jsonl")]
    t=[x["asr_ms"] for x in r]
    print(sp,len(r),"empty:",sum(not x["transcript"].strip() for x in r),"asr_ms mean/median/max:",round(s.mean(t)),round(s.median(t)),round(max(t)))
EOF
```

```text
  125 rag/.cache/dev_transcripts.jsonl
  350 rag/.cache/practice_transcripts.jsonl
  475 total
{"query_id": "q_7e9a0f2716c017d2", "transcript": "Understanding sales tax implications when purchasing items in a retail store.", "asr_ms": 2060.8}
{"query_id": "q_ea0be00eda6ea088", "transcript": "Factors influencing the initial stock price at market opening.", "asr_ms": 1302.2}
{"query_id": "q_9cd791a970ab648f", "transcript": "Timeline for fund availability when depositing credit card payments.", "asr_ms": 1293.1}
practice 350 empty: 0 asr_ms mean/median/max: 1555 1467 3845
dev 125 empty: 0 asr_ms mean/median/max: 1814 1647 13312
```

**What it means.** All 350 practice and 125 dev queries have a transcript and none is empty. The queries are short finance/consumer questions (about 10 words). Per-query ASR time (model loading excluded, PyTorch kernels, RTX 2080 Ti, node `opencast`): practice mean 1555 ms (median 1467, max 3845); dev mean 1814 ms (median 1647, max 13312). The mean is above the 1215 ms of the 13-token test clip because these transcripts are longer and decoding costs about 94 ms per token (E03). The dev maximum of 13.3 s is a clear outlier (7 times the median): PENDING, check that query's transcript length (a long or runaway generation) versus a transient node effect before using dev timings.

PENDING: transcript quality. The course data has no reference transcripts, so word error rate cannot be computed; quality is only visible indirectly through retrieval (E07).

## E06: Transcripts copied to the laptop

**Need.** RAG experiments run on the laptop (no GPU, ELM key stays off the cluster), so the shared transcript cache must be copied there unchanged.

```bash
cd <course repo>
mkdir -p rag/.cache && scp -o ProxyJump=<UUN>@ssh.inf.ed.ac.uk '<UUN>@mlp.inf.ed.ac.uk:edin-mls-26-autumn/rag/.cache/*_transcripts.jsonl' rag/.cache/
wc -l rag/.cache/*.jsonl
```

```text
dev_transcripts.jsonl        100%   18KB
practice_transcripts.jsonl   100%   49KB
  125 rag/.cache/dev_transcripts.jsonl
  350 rag/.cache/practice_transcripts.jsonl
  475 total
```

**What it means.** The laptop holds the identical 350 + 125 transcripts (same line counts). `rag/.cache/` is git-ignored, so the cache is not in the repo; PENDING: record SHA-256 of both files in this log so any later run can prove it used the same inputs (`sha256sum rag/.cache/*.jsonl`).

## E07: Baseline retrieval quality (exact dense retrieval, enc_a, practice split)

**Need.** The reference every variant (retrieval depth, evidence budget, reranking, ...) is compared with: retrieval quality on the 350 practice queries from the cached PyTorch-ASR transcripts, using the supplied, unmodified pipeline.

```bash
source ~/mls-rag-env/bin/activate
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl
```

```text
Aggregate:
  Recall@5   0.421337
  Recall@10  0.479594
  Recall@20  0.578524
  Recall@100 0.769856
  MRR@10     0.488746
  NDCG@10    0.420461
```

Conditions: 350 practice queries, 24,000-record corpus, enc_a encoder on CPU (the library warned the model loaded in float16 on CPU), ASR transcripts from E04, no ELM calls, k_max 100.

**What it means.**
- With the baseline depth k = 5 the system finds on average 42% of a query's relevant records; doubling to k = 10 adds only 5.8 points (0.480), k = 20 adds 9.9 more (0.579), and k = 100 reaches 0.770. Gains shrink per extra record retrieved, which is exactly the trade-off Plan A studies: each extra retrieved record costs prompt length and generation time.
- Recall@100 = 0.770 means 23% of relevant records are missed even at depth 100. Anything that re-orders or truncates a top-100 candidate list (reranking, evidence budget) cannot recover those, so 0.770 is the ceiling for such methods. Beating it needs a better query or encoder, not a reranker.
- MRR@10 = 0.489: the first relevant record sits on average near rank 2 when it is found. NDCG@10 (0.420) is lower than MRR because queries have several relevant records and NDCG also rewards ranking all of them high.
- The per-query log shows a bimodal pattern: many queries score Recall@10 of exactly 0 or exactly 1 and few sit in between, so the aggregate hides a group of queries that retrieval fails on entirely. PENDING: count them and look at their transcripts to separate ASR errors from encoder or corpus limitations.
- Limits: this mixes ASR error and retrieval error, since no clean-text control exists (no reference transcripts). The encoder ran in float16 on CPU; PENDING: confirm the numbers match a float32 run so the baseline is reproducible on other machines.

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

## E08: Baseline retrieval quality on the dev split (final-evaluation split)

**Need.** Dev (125 queries) is kept for final numbers; practice (350) is for tuning. Having the baseline on both shows how much it moves between query sets before we compare variants.

```bash
python rag/scripts/evaluate_rag.py --device cpu --split dev --transcript-cache rag/.cache/dev_transcripts.jsonl 2>&1 | tail -8
```

```text
Aggregate:
  Recall@5   0.379497
  Recall@10  0.474595
  Recall@20  0.579521
  Recall@100 0.724426
  MRR@10     0.459708
  NDCG@10    0.398342
```

Same conditions as E07 (exact dense retrieval, enc_a on CPU in float16, PyTorch-ASR transcripts from E04, no ELM calls).

| Metric | Practice (350) | Dev (125) | Dev minus practice |
| --- | --- | --- | --- |
| Recall@5 | 0.421 | 0.379 | -0.042 |
| Recall@10 | 0.480 | 0.475 | -0.005 |
| Recall@20 | 0.579 | 0.580 | +0.001 |
| Recall@100 | 0.770 | 0.724 | -0.045 |
| MRR@10 | 0.489 | 0.460 | -0.029 |
| NDCG@10 | 0.420 | 0.398 | -0.022 |

**What it means.** The two splits agree on the shape: recall rises with depth with diminishing returns, and Recall@100 stays well below 1 (0.72 on dev, so about 28% of relevant records are never retrieved, a ceiling for any reranking or budgeting method). Dev is slightly lower at the top of the ranking (Recall@5 -4.2 points, MRR@10 -2.9) and at k = 100 (-4.5), and equal at k = 10 and 20. With only 125 dev queries and per-query scores that are mostly 0 or 1, sampling noise of a few points is expected, so we do not read these differences as real; PENDING: bootstrap confidence intervals for every reported metric, and state them whenever a variant's gain is of this size. Rule for the report: choose settings on practice, report dev once at the end, and say which split each number comes from.
