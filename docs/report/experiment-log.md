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

## E09: Retrieval-depth sweep (practice, n=350)

**Need.** The investigation (Plan A) is about how much retrieved evidence to give the answer model. Before spending any ELM calls, measure what each extra retrieved record buys in recall and costs in latency and context size.

```bash
python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl
```

```text
k    Recall@k (95% CI)       retrieval ms (median)   context words   context chars
1    0.229 (0.196-0.265)     0.44                    175             987
2    0.319 (0.281-0.359)     0.45                    354             1999
5    0.421 (0.378-0.464)     0.45                    897             5054
10   0.480 (0.436-0.521)     0.45                    1814            10193
20   0.579 (0.534-0.621)     0.44                    3640            20427
50   0.684 (0.644-0.723)     0.45                    9076            51003
100  0.770 (0.732-0.806)     0.45                    17957           100905
MRR@10 0.489   NDCG@10 0.420
```

Conditions: exact dense retrieval over 24,000 records, enc_a on CPU in float16, PyTorch-ASR transcripts from E04, bootstrap CIs (2000 resamples, seed 0) over queries, no ELM calls.

**What it means.**
- Retrieval time is flat at about 0.45 ms for every k, because exact search always compares against all 24,000 vectors. The cost that grows with k is context size: about 180 words per record, from 175 words at k=1 to about 18,000 at k=100. That text is what the answer model must read, so prompt tokens and generation time are the real price of depth.
- Recall gains diminish per extra word: k=1 to 5 adds 19 points for about 720 extra words, while k=20 to 100 adds 19 points for about 14,000 extra words. Between neighbouring depths the intervals overlap (for example k=5 vs k=10, +5.9 points), so small steps are not distinguishable at n=350.
- Recall@100 = 0.770 is the ceiling for any reranking or evidence-budget method that only reorders or trims the top 100.
- MRR@10 and NDCG@10 equal E07, which checks that the sweep script is consistent with the supplied evaluator.
- Caveats. PENDING: repeat on dev (final numbers only), and confirm float16 vs float32 encoder agreement. Recall is not answer quality; that needs E11 or later.

## E10: Five-query end-to-end smoke test (uses ELM, about 10 calls)

**Need.** Check that the whole path (cached transcript, retrieval, ELM generation, ELM judge) works and see where the time goes, before designing a larger answer-quality experiment. Five queries is a plumbing test, not a result.

```bash
python rag/extensions/generate_from_transcripts.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/smoke_k5.jsonl --limit 5
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/smoke_k5.jsonl --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/smoke_k5.jsonl --allow-subset --out-dir rag/results/smoke_judge
```

```text
Validated 5/350 practice candidates; no API calls
judge combined score (0-1) per query:
q_7e9a0f2716c017d2  0.333
q_ea0be00eda6ea088  0.583
q_9cd791a970ab648f  0.000
q_20e98c45013f6cc6  0.417
q_e3f1cef46917a2a5  0.417
per-query timing: asr_ms 0 (cache lookup), embedding 21-54 ms, retrieval 4-10 ms,
                  generation 2330-6197 ms, end to end 2357-6245 ms
```

Settings: k=5 (the generator's default), default generation model (`utter-project/EuroLLM-22B-Instruct-2512`, temperature 0.1), judge rubric as supplied, one judge repeat.

**What it means.**
- The path works. Mean judge score is about 0.35, which is not a finding at n=5.
- Generation takes 2.3 to 6.2 s per query, over 95% of end-to-end time. Embedding and retrieval are tens of milliseconds at most. So if the investigation changes how much evidence is sent to the answer model, generation time is where an effect would show. The `asr_ms` field is 0 because transcripts come from the cache; real ASR time is in E03/E04.
- The sales-tax answer drifted to business/LLC tax, suggesting off-topic evidence; the 0.000 query (q_9cd791a970ab648f) is the next one to inspect, retrieved records against the judge's reasons.
- Caveats. The judge scores the ASR transcript as the question, so it cannot detect ASR errors. The 4-10 ms retrieval times here are cold-start and are not comparable with E09's warm 0.45 ms. PENDING: inspect the retrieved records for the low-scoring queries.

## E11: Retrieval-depth sweep on the dev split (n=125)

**Need.** Same sweep as E09 on the final-evaluation split, to check that the practice conclusions (flat retrieval time, growing context, diminishing recall) hold on unseen queries. This is a confirmation run, not for tuning.

```bash
python rag/extensions/retrieval_sweep.py --split dev --transcripts rag/.cache/dev_transcripts.jsonl
```

```text
| k | Recall@k (95% CI) | retrieval ms (median) | context words | context chars |
| --- | --- | --- | --- | --- |
| 1 | 0.209 (0.153-0.272) | 0.46 | 183 | 1038 |
| 2 | 0.279 (0.214-0.351) | 0.45 | 330 | 1875 |
| 5 | 0.379 (0.311-0.454) | 0.54 | 830 | 4708 |
| 10 | 0.475 (0.404-0.553) | 0.52 | 1748 | 9908 |
| 20 | 0.580 (0.515-0.650) | 0.46 | 3598 | 20400 |
| 50 | 0.677 (0.613-0.742) | 0.45 | 9057 | 51336 |
| 100 | 0.724 (0.664-0.785) | 0.52 | 18101 | 102473 |
MRR@10 0.460  NDCG@10 0.398  (n=125)
```

**What it means.**
- The shape matches practice (E09): retrieval time stays at about 0.45-0.54 ms for every k, context grows from about 183 to about 18,100 words, and recall rises with diminishing returns.
- Dev recall is within a few points of practice at every k (for example k=5: 0.379 vs 0.421; k=100: 0.724 vs 0.770). The dev intervals are about twice as wide (about ±7 points, n=125), and every practice value lies inside the corresponding dev interval, so we see no evidence of a real difference between the splits.
- Recall@100 = 0.724 is the ceiling for any reranking or evidence-budget method on dev. MRR@10 and NDCG@10 equal E08 exactly, a second consistency check on the sweep script.
- Rule kept: choose settings on practice; report dev once at the end.
- Caveat. PENDING: the encoder ran in float16 on CPU; confirm float32 agreement.

## K01: Kernel correctness check on the RTX 2080 Ti (Tasks 1-4)

**Need.** Before timing any variant, confirm the four implemented kernels are correct: course tests, memory and race checks, and the full model on our kernels.

```bash
cd ~/edin-mls-26-autumn/hw1-asr
sbatch ablation/ablate.sbatch --stage check     # make test, compute-sanitizer memcheck + racecheck (small cases), ./run.sh
```

```text
GPU: NVIDIA GeForce RTX 2080 Ti (sm_75), 68 SMs, peak memory bandwidth about 616 GB/s
Task 1 (RoPE): ALL PASS
Task 2 (reductions): ALL PASS
Task 3 (tiled GEMM): ALL PASS
Task 4 (attention): ALL PASS
task1..task4 memcheck: ERROR SUMMARY: 0 errors
task1..task4 racecheck: RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
Time: 3285.7 ms per transcription (+/- 6.1 ms), 252.7 ms per token
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

**What it means.** All four kernels pass the course's CPU/cuBLAS-referenced tests, have no out-of-bounds accesses or shared-memory races on the small cases, and transcribe the test recording correctly when the whole model runs on them. Speed is 3285.7 ms per transcription. CORRECTION: an earlier version of this entry compared that with the 1215.4 ms PyTorch run of E03 (2.7x). That is a cross-job comparison and is not valid: the PyTorch-only time depends on the node's CPU and was 806.3 ms in the same-GPU-type run of K05. The valid slowdown is the within-job ratio, 4.1x (K05). Rule for the report: compare variants only inside one job.

## K02: Tasks 1 and 2 ablations (RoPE and reductions) on the RTX 2080 Ti

**Need.** The guides' "Try this" items for Tasks 1 and 2: block size, fast maths, atomic instead of shuffle reduction, one-pass versus textbook softmax. Each variant is the same file built with one `-D` flag (listed in `hw1-asr/ablation/README.md`); correctness is checked by the same tests, and every number is the median of 3 runs of the course test, in ms per call.

```bash
sbatch ablation/ablate.sbatch --stage sweep      # all variants; summary in ablation/results/<time>/summary.md
```

Task 1, RoPE (`ROPE_THREADS`; default 256):

| case | 32 | 64 | 128 | **256** | 512 | 1024 | fast_math |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio encoder (partial, 32/64) | 0.038 | 0.031 | 0.031 | **0.031** | 0.031 | 0.033 | FAIL |
| text decoder (GQA 16/4, full) | 0.005 | 0.004 | 0.004 | **0.004** | 0.004 | 0.005 | 0.004 |

Default speed: audio encoder 493.6 GB/s (80% of the 616 GB/s peak), text decoder 319.5 GB/s (52%). The tiny cases (2 x 3 tokens) take 0.003 ms in every variant.

Task 2, reductions (`REDUCE_THREADS`; default 256), selected cases:

| case | 32 | 64 | 128 | **256** | 512 | 1024 | atomic RMS | 3-pass softmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rmsnorm, text decoder | 0.028 | 0.015 | 0.010 | **0.006** | 0.005 | 0.005 | 0.022 | 0.006 |
| layernorm, audio encoder | 0.096 | 0.059 | 0.042 | **0.040** | 0.048 | 0.076 | 0.040 | 0.040 |
| softmax, 64 x 4096 | 0.047 | 0.025 | 0.014 | **0.009** | 0.006 | 0.006 | 0.009 | 0.009 |
| softmax, vocabulary row (59264) | 0.886 | 0.440 | 0.223 | **0.118** | 0.063 | 0.041 | 0.118 | 0.150 |

Default speed: layernorm encoder 383 GB/s (62% of peak), rmsnorm text decoder 166 GB/s (27%), softmax 64 x 4096 243 GB/s (39%), vocabulary row 4.0 GB/s (1%).

**What it means.**
- RoPE is memory-bound and already at 80% of peak on the large case, so block size hardly matters between 64 and 512. 32 threads per block is 23% slower, 1024 is 6% slower. The tiny cases take 3 microseconds in every variant, which is launch overhead, not work. `--use_fast_math` makes exactly one case fail, the audio encoder, which uses the largest angles: the fast sin/cos approximations lose accuracy as the angle grows.
- The best reduction block size depends on the shape. The vocabulary row is one block on one of the GPU's 68 SMs (1% of peak bandwidth), so adding threads speeds it up almost linearly: 1024 threads are 2.9x faster than 256. A 1280-column layernorm row has too little work for 1024 threads and is 1.9x slower there, and its best size is 128 to 256. One constant cannot be best for all three kernels; per-kernel sizes, or splitting a long row over many blocks, would be the real fix for the vocabulary row.
- Replacing the shuffle reduction in RMSNorm with `atomicAdd` on one shared float is 3.7x slower on the text-decoder case (0.022 vs 0.006 ms): all threads queue on one address. The layernorm columns are unchanged by this flag, as expected, since only RMSNorm uses it.
- The textbook softmax (separate max and sum passes) is 27% slower than our one-read version only on the long vocabulary row (0.150 vs 0.118 ms). On shorter rows the two are equal at the 0.001 ms resolution, so we do not claim a difference there. The row (237 KB) probably sits in the GPU cache, so the extra reads are cheaper than DRAM reads would be.
- Limits: differences below about 0.002 ms are noise at this resolution. Single GPU type (2080 Ti) and one node.

## K03: Task 3 ablations (tiled GEMM) on the RTX 2080 Ti

**Need.** The Task 3 "Try this" items: tile size, shared-memory padding, and the way the W tile is loaded. Same method as K02.

```bash
sbatch ablation/ablate.sbatch --stage sweep
```

| case (M x N x K) | tile 8 | **tile 16** | tile 32 | no pad | W strided | tile 32 + no pad |
| --- | --- | --- | --- | --- | --- | --- |
| decoder q_proj (64 x 2048 x 2048) | 0.470 | **0.315** | 0.267 | 0.704 | 0.444 | 1.217 |
| decoder gate_proj (64 x 6144 x 2048) | 1.352 | **0.869** | 0.791 | 2.004 | 1.243 | 3.638 |
| decoder down_proj (64 x 2048 x 6144) | 2.142 | **1.174** | 0.792 | 2.099 | 1.353 | 3.638 |
| lm_head, last token (1 x 59264 x 2048) | 1.537 | **2.723** | 4.842 | 4.724 | 2.861 | 16.989 |
| encoder q_proj (1500 x 1280 x 1280) | 4.166 | **2.595** | 2.300 | 6.015 | 3.716 | 10.608 |
| encoder fc1 + GELU (1500 x 5120 x 1280) | 24.845 | **14.207** | 9.302 | 23.999 | 15.221 | 42.041 |
| conv2 via im2col (1500 x 1280 x 3840) | 17.609 | **10.116** | 6.917 | 18.043 | 11.388 | 31.805 |

Times in ms. Default (tile 16) throughput: 1.4 to 1.9 TFLOP/s on the large cases (for example encoder q_proj 1897, fc1 1384, conv2 1458 GFLOP/s) and 89 GFLOP/s on lm_head. lm_head reads the 485 MB weight matrix: 1.537 ms (tile 8) is 316 GB/s, the default 2.723 ms is 178 GB/s, tile 32 at 4.842 ms is 100 GB/s.

**What it means.**
- Bigger tiles reuse each loaded value more often, so tile 32 is fastest on every large case (fc1: 9.3 ms vs 14.2 ms at tile 16, 35% less; conv2 32% less) and tile 8 is slowest (fc1 +75%).
- lm_head is the opposite: with M = 1 only one of a block's TILE rows of threads has a real output row, so only 1/8, 1/16 or 1/32 of the threads do useful work, and the smallest tile wins (tile 8 is 1.8x faster than 16, tile 32 is 1.8x slower). No single tile size is best for all shapes; a shape-dependent choice would beat every fixed one.
- Removing the padding column of the W tile makes the large cases 1.7 to 2.3x slower (encoder q_proj 6.0 vs 2.6 ms): with a row length of 16 floats, the 16 threads that read `ws[tx][k]` fall on only two of the 32 shared-memory banks, so the reads queue up (bank conflicts). With tile 32 and no padding all 32 lanes hit the same bank, and the kernel is 4.5x slower than with padding (fc1 42.0 vs 9.3 ms). Padding matters more as the tile grows.
- Loading the W tile with consecutive threads K floats apart (the lecture's way for B) is 5 to 43% slower: largest on decoder q_proj, decoder gate_proj and encoder q_proj (+41 to 43%), middling on decoder down_proj and conv2 (+13 to 15%), small on lm_head (+5%) and fc1 (+7%). The caches soften the penalty; we did not measure why lm_head is less affected than the guide suggests.
- Limits: 2080 Ti only (the guide says the best tile differs per GPU). Not yet compared with cuBLAS in this log; the course test prints the cuBLAS and naive throughput per case, which is PENDING.

## K04: Task 4 ablations (attention) on the RTX 2080 Ti

**Need.** The Task 4 "Try this" items: queries per block (`ATTN_WARPS`) and the +1 padding of the key tile.

```bash
sbatch ablation/ablate.sbatch --stage sweep
```

| case | 1 warp | 2 | **4** | 8 | 16 | no pad (4 warps) |
| --- | --- | --- | --- | --- | --- | --- |
| audio encoder (20 heads, 1500) | 103.585 | 30.061 | **11.498** | 8.344 | 7.576 | 54.236 |
| text decoder (GQA 16/4, causal) | 0.640 | 0.185 | **0.067** | 0.034 | 0.022 | 0.147 |
| causal, seq_q = seq_k | 0.127 | 0.053 | **0.025** | 0.021 | 0.022 | 0.043 |

Times in ms. Default throughput: audio encoder 1002 GFLOP/s, text decoder 252 GFLOP/s.

**What it means.**
- Each block loads the whole key/value sequence from global memory once, so the number of times a key is loaded is the number of blocks per head, seq_q / ATTN_WARPS. Going from 1 to 4 warps cuts those loads 4x and the encoder time 9x (103.6 to 11.5 ms), going to 16 cuts them 16x and the time 13.7x (7.6 ms). Gains shrink as the load traffic stops being the bottleneck (4 to 8 warps: 27%; 8 to 16: 9%).
- The text decoder case benefits even more (0.067 to 0.022 ms at 16 warps), but at that size the kernel takes microseconds, so we expect launch and tail effects; the end-to-end effect is in K05.
- Without the +1 padding of the key tile, the encoder case is 4.7x slower (54.2 vs 11.5 ms): the 32 lanes read `k_tile[lane * 64 + d]`, a stride of 64 floats, so all lanes hit the same shared-memory bank and the 32 reads are serialised. With the +1 the strides are 65 floats, so lanes spread over all banks.
- Limits: ATTN_WARPS = 16 uses 512 threads per block and about 20 KB of shared memory; larger values were not tried. 2080 Ti only.

## K05: End-to-end timing with the kernels switched on one task at a time

**Need.** Micro-benchmarks do not say what matters for a whole transcription. Run the full model with PyTorch only, then add our kernels cumulatively, then three configuration variants. All rows come from one job (one allocation), so they are comparable with each other.

```bash
sbatch ablation/ablate.sbatch --stage e2e       # benchmark.py --cuda <groups> --runs 3
```

| run | kernels on CUDA | ms per transcription (+/- ) | ms per token | vs PyTorch only | accuracy |
| --- | --- | --- | --- | --- | --- |
| torch_only | none | 806.3 (2.4) | 62.0 | 1.00x | 100% |
| task1 | RoPE | 727.1 (2.0) | 55.9 | 0.90x | 100% |
| task1-2 | + reductions | 705.2 (0.7) | 54.2 | 0.87x | 100% |
| task1-3 | + GEMM | 2998.2 (6.7) | 230.6 | 3.72x | 100% |
| all_default | + attention | 3288.3 (7.8) | 252.9 | 4.08x | 100% |
| all_tile8 | all, GEMM tile 8 | 4992.2 (9.0) | 384.0 | 6.19x | 100% |
| all_tile32 | all, GEMM tile 32 | 2865.2 (6.6) | 220.4 | 3.55x | 100% |
| all_attn_w8 | all, 8 attention warps | 3184.0 (7.0) | 244.9 | 3.95x | 100% |

**What it means.**
- Our RoPE and reduction kernels give a faster run than PyTorch's (0.87x together). The profile (K06) shows why: PyTorch computes RoPE and RMSNorm as chains of several small unfused operations, while our kernels do each in one pass over the data. Where PyTorch has a single fused kernel (LayerNorm) the two are equal.
- Our GEMM is the problem: swapping it in adds 2293 ms (3.72x total), an amount equal to 70% of the final all-kernels run time. (The share of kernel time inside the GEMM is in K06.) It reaches 1.4 to 1.9 TFLOP/s in K03; PyTorch uses cuBLAS (tuned library GEMM). Attention adds a further 290 ms over PyTorch's. The profile (K06) puts GEMM at 83.6% and attention at 15.9% of the profiled kernel time in the final configuration.
- GEMM tile size is the best single change: tile 32 saves 423 ms (13%), tile 8 costs 1704 ms (+52%). 8 attention warps save 104 ms (3.2%). That matches the micro-benchmark (3.15 ms saved per call on the encoder case times 32 encoder layers = 101 ms). So attention is worth much less than the GEMM although its micro-benchmark gain looks bigger.
- The all-default time here (3288.3 ms) matches the K01 run (3285.7 ms) within 0.1%, so our kernels' timing is reproducible across jobs. The PyTorch-only time is not: 806.3 ms here, 1215.4 ms in E03 on the same GPU type. We treat PyTorch's time as node-dependent (it includes host-side work) and quote only within-job ratios.
- Accuracy is 100% for every configuration, so none of these variants changes the transcription on the test recording.
- PENDING: the combined best configuration (tile 32 + 16 attention warps, not yet run); a repeat on a second allocation (two duplicate jobs were queued by mistake and can serve as this repeat).

## K06: Where the time goes, per operation (profiles from the K05 runs)

**Need.** K05 gives whole-run times; the handout also asks for the `--profile` output. This shows which operation takes the time, with PyTorch only and with all four tasks on our kernels.

```bash
cat ablation/results/<e2e run>/e2e_torch_only.txt ablation/results/<e2e run>/e2e_all_default.txt     # produced by: benchmark.py --profile
```

| operation | calls | PyTorch only: ms (share) | our kernels: ms (share) |
| --- | --- | --- | --- |
| linear (GEMM, Task 3) | 2757 | 568.0 (47.6%) | 2871.9 (83.6%) |
| attention (Task 4) | 396 | 267.8 (22.5%) | 544.9 (15.9%) |
| rope (Task 1) | 396 | 298.7 (25.0%) | 3.0 (0.1%) |
| rmsnorm (Task 2) | 741 | 50.0 (4.2%) | 6.6 (0.2%) |
| layernorm (Task 2) | 65 | 2.6 (0.2%) | 2.4 (0.1%) |
| softmax (Task 2) | 13 | 0.3 (0.0%) | 1.5 (0.0%) |
| silu_mul, embedding, im2col (provided) | 364 / 13 / 2 | 5.2 (0.4%) | 5.2 (0.2%) |
| **sum of profiled kernel times** | | 1192.6 | 3435.7 |
| wall-clock time per transcription (K05) | | 806.3 | 3288.3 |

**What it means.**
- With our kernels the run is almost entirely GEMM (83.6%) and attention (15.9%). RoPE, RMSNorm, LayerNorm and softmax together are 0.5% of the time, so tuning them (K02) cannot change the end-to-end time; their value is in the report's explanations, not in speed. The lever for speed is the GEMM.
- The profile times are not additive with the wall-clock times: profiling times each operation separately, and the sum (1192.6 ms) is 48% above the wall-clock time (806.3 ms) for PyTorch, which runs many tiny operations (RoPE alone costs 0.75 ms per call in PyTorch). For our kernels the two agree within 4.5% (3435.7 vs 3288.3 ms). We therefore use the profile for shares and ordering and the wall clock for totals. Entering both here is deliberate: the handout asks for the profile output.
- 396 attention calls = 32 encoder layers + 28 decoder layers x 13 generated tokens (364). 13 tokens x 28 layers is also why the linear count is large: the model re-runs the decoder over the whole sequence for every token (no cache of past keys and values).
- Cross-check, PENDING: ours attention 544.9 ms is larger than 32 x 11.5 ms (encoder, micro-benchmark K04) = 368 ms plus 364 decoder calls at 0.067 ms = 24 ms. About 150 ms is not explained by the micro-benchmarks (the decoder sequences may be longer or have other shapes than the test case, or the profiler adds overhead). We do not use the unexplained part in any claim.
- PyTorch's softmax (0.3 ms) is faster than ours (1.5 ms) on the vocabulary row, as K02 predicts for a one-block-per-row kernel; irrelevant for the total (13 calls).
