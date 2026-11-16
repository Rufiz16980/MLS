# Experiment log (source material for the report)

How to read: each entry has **Need** (why), **Command**, **Output** and **Explanation**.
Not the report itself: the report (`report/latex_template/report.tex`, 6-page body limit)
should take its numbers and wording from here. Update this file after every run.

Privacy rules for this file: no API keys, no student IDs, no usernames, no real hostnames,
no home paths. Use `<user>`, `<gpu-node>`, `<team-repo>`. Outputs below are trimmed.
"Reported" means the owner pasted the output and it was not re-run by Claude.

---

## 1. Course assets (laptop and cluster)

**Need.** The corpus (24,000 records), 475 query audio files, embeddings and the `enc_a`
encoder are shipped as one Git-LFS archive, `rag-assets.zip`.

```bash
sha256sum rag-assets.zip          # expected 3b830368271e935b60791dc0955af2b0d50692d575150189475c6306e6c9e3b8
mkdir -p ~/mls-rag-assets && unzip -q rag-assets.zip -d ~/mls-rag-assets
python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
```
```text
rag-assets.zip: OK                       (196,521,100 bytes)
Course assets verified: 492 files, 492 installed
```
**Explanation.** The hash matches, so the archive is the genuine course file. 492 files
installed into `rag/data` and `rag/models` (about 233 MB; git-ignored, never committed).
Git shows the zip as "modified" only when Git LFS is missing locally (the committed blob is a
134-byte pointer): do not commit it.

## 2. Setup check (laptop)

**Need.** Confirm data, embeddings, model, API configuration and ASR path before any experiment.

```bash
python rag/scripts/check_setup.py
python -m unittest discover -s rag/tests -v
```
```text
[PASS] corpus.jsonl: 24000 records
[PASS] student audio manifest: 475 audio items
[PASS] corpus embeddings: shape=(24000, 384)
[PASS] enc_a model directory: model payload present
[PASS] ELM configuration: API key configured
[PASS] ASR source path: source found ...
Ran 3 tests ... OK
```
**Explanation.** All six checks and the 3 unit tests pass. 475 audio items = 350 practice +
125 dev queries. The key is read from a git-ignored `rag/.env` (mode 600); the checker does not
print it and does not call the API.

## 3. ASR environment (cluster head node)

**Need.** GLM-ASR-Nano (about 2.3 B parameters, float32, about 9 GB on the GPU) and its conda
environment for the ASR part and for the RAG audio path.

```bash
bash hw1-asr/setup_env.sh        # run once on the head node, in the background
```
```text
... Successfully installed ... torch ... transformers-5.0.0 ...
Fetching 10 files: 100%
>>> Setup finished. In every new shell, before working on hw1-asr, run: conda activate mls-asr
```
**Explanation.** Conda env `mls-asr` (Python 3.11, CUDA 12.8 PyTorch, transformers 5.0) and
the model snapshot are in the cluster home cache. Home directory permissions are 700.

## 4. ASR baseline on a GPU (all PyTorch kernels)

**Need.** A reference speed and correctness number before any CUDA kernel is used
(`--cuda none` = every operator runs through PyTorch).

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:45:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH && conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr && make && ./run.sh --cuda none
```
```text
GPU: NVIDIA GeForce RTX 2080 Ti (sm_75)
Kernels: rope=torch reduce=torch gemm=torch attention=torch provided=torch
Loaded in 153.5 s
Audio: 3.50 s, mel features (1, 128, 3000), prompt 59 tokens
Time: 1215.4 ms per transcription (+/- 1.5 ms), 93.5 ms per token
Peak GPU memory: 8.88 GiB
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%   Status: PASS
```
**Explanation.** One 3.5 s clip takes about 1.2 s (13 decoder runs, about 93.5 ms each), with
very small run-to-run spread (+/- 1.5 ms over 3 runs, so a 1-node, 1-clip measurement). The
decoder re-runs the whole sequence each token (no KV cache), which is why per-token cost is
high. Peak memory 8.88 GiB is almost all weights, so an 11 GB card is nearly full. `make` also
built the CUDA library and the four task tests (they compiled; test results were not shown).
Hardware to report: RTX 2080 Ti, sm_75, CUDA 12.8, 2 CPU cores, 16 GB RAM allocation.
**Limits.** One clip, 3 runs, one GPU node. First-run model load (153.5 s) is a cold file-system
read and should not be mixed with inference time.

## 5. Transcribing all queries once (cluster)

**Need.** The RAG pipeline starts from ASR text. Transcribing the 475 query audio files once and
caching the text lets every later retrieval variant reuse identical transcripts (a controlled
comparison, as the README requires) without re-running ASR.

```bash
cd ~/<team-repo> && git pull
bash scripts/sync_overlay.sh ~/edin-mls-26-autumn
cd ~/edin-mls-26-autumn
python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets
ASR_CUDA=none python rag/extensions/transcribe_split.py --split all
```
```text
guard OK: 55 frozen files intact, 5 task files conform
Course assets verified: 492 files, 492 installed
practice: 350 queries, 0 cached, 350 to do    ... practice 350/350
dev: 125 queries, 0 cached, 125 to do         ... dev 125/125
```
**Explanation.** `sync_overlay.sh` copies the team's added files (here
`rag/extensions/transcribe_split.py`, 56 lines) into the course checkout; the "guard" line
reports that the 55 course files that must not change are intact and the 5 task files conform
(this reading of the guard is an inference from its message; check `scripts/sync_overlay.sh`).
Both splits were fully transcribed (350 practice + 125 dev = 475) with the PyTorch reference ASR.
**Result of the run (reported, checked with a short script on the cluster).**
```bash
wc -l rag/.cache/*.jsonl          # 125 dev, 350 practice
head -c 600 rag/.cache/practice_transcripts.jsonl
```
```text
125 rag/.cache/dev_transcripts.jsonl     350 rag/.cache/practice_transcripts.jsonl
{"query_id": "q_7e9a...", "transcript": "Understanding sales tax implications when purchasing items in a retail store.", "asr_ms": 2060.8}
practice 350 empty: 0 asr_ms mean/median/max: 1555 1467 3845
dev      125 empty: 0 asr_ms mean/median/max: 1814 1647 13312
```
**Explanation.** The cache is one JSON line per query (`query_id`, `transcript`, `asr_ms`) in
`rag/.cache/{practice,dev}_transcripts.jsonl` (git-ignored). All 475 queries have a non-empty
transcript. ASR cost per query: about 1.5 s on average (median 1.47 s, practice), i.e. the
transcription of a short spoken question costs more than any retrieval step later in the
pipeline. The dev maximum of 13.3 s is an outlier (cause not investigated; possibly a cold start
or a shared-node effect), so report medians. The sample transcripts are fluent finance
questions, consistent with a working ASR.
**Not yet measured.** Word error rate against the reference text (not supplied here), and total
wall-clock time of the run.

## 6. Copying the transcripts to the laptop

**Need.** Reuse identical transcripts on the laptop for CPU-only retrieval experiments.

```bash
mkdir -p rag/.cache && scp -o ProxyJump=<user>@<gateway> '<user>@<cluster-head>:edin-mls-26-autumn/rag/.cache/*_transcripts.jsonl' rag/.cache/
wc -l rag/.cache/*.jsonl
```
```text
dev_transcripts.jsonl 100% 18KB    practice_transcripts.jsonl 100% 49KB
125 dev   350 practice   475 total
```
**Explanation.** The files are small (67 KB in total). The copy goes through the gateway as a jump
host because the cluster head node is not reachable directly. The line counts match the
cluster (125 + 350), so the copy is complete. The transcripts are course data, not personal
data, and the folder is git-ignored.

## 7. Baseline retrieval (README Step 5, laptop CPU, enc_a)

**Need.** The baseline every later variant is compared with: query = ASR transcript, encoder
`enc_a` on both sides, exact search over the 24,000 corpus vectors, practice split (350 queries).

```bash
source ~/mls-rag-env/bin/activate
python rag/scripts/evaluate_rag.py --device cpu --split practice \
  --transcript-cache rag/.cache/practice_transcripts.jsonl
```
```text
[  1/350] q_7e9a... R@10=0.500   ... (350 per-query lines, R@10 = fraction of relevant records in the top 10)
Aggregate:
  Recall@5   0.421337
  Recall@10  0.479594
  Recall@20  0.578524
  Recall@100 0.769856
  MRR@10     0.488746
  NDCG@10    0.420461
```
**Explanation.**
- Recall rises with retrieval depth k: 0.42 (k=5), 0.48 (10), 0.58 (20), 0.77 (100). Doubling
  k from 10 to 20 gains about 10 points, going to 100 gains about 29. Deeper retrieval finds more
  evidence but means more text to hand to the answer model (the cost side of the trade-off).
- About 23% of relevant records are not found even at k=100, so a ceiling remains from the
  encoder and from ASR errors in the query. The per-query lines show many queries with
  R@10 = 0 and many with 1.0, so the average hides a bimodal distribution; per-query analysis
  (which queries fail, and are their transcripts wrong?) is a natural follow-up.
- MRR@10 = 0.49: on average the first relevant record sits around rank 2. NDCG@10 = 0.42.
- Fractional per-query values such as 0.333 and 0.5 show that queries have several relevant
  records, so Recall@k is the fraction of them retrieved.
**Conditions to report.** One run, practice split only, transcripts from the PyTorch reference
ASR on a GPU node, enc_a on laptop CPU (the library warned that it loaded the model in float16 on
CPU; this may differ slightly from float32 results, so keep the same setting for every
comparison). No dev-split number yet, and no timings were recorded for retrieval.

---

## 8. Baseline retrieval on the dev split

**Need.** A second split to check that the practice baseline is not a fluke, and the reference
for held-out comparisons.

```bash
python rag/scripts/evaluate_rag.py --device cpu --split dev \
  --transcript-cache rag/.cache/dev_transcripts.jsonl 2>&1 | tail -8
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
**Explanation.** Dev (125 queries) is close to practice (350): Recall@10 0.475 vs 0.480 and
Recall@20 0.580 vs 0.579, so the baseline is stable across splits. The gaps are at the extremes:
Recall@5 is lower (0.379 vs 0.421), Recall@100 is lower (0.724 vs 0.770), MRR@10 0.460 vs 0.489
and NDCG@10 0.398 vs 0.420. With only 125 queries a difference of a few points can be noise, so
treat it as "similar, slightly harder", not as a real shift, unless a confidence interval or a
paired comparison says otherwise. Same conditions as entry 7 (PyTorch-ASR transcripts, enc_a on
CPU in float16, single run).

| Split (queries) | R@5 | R@10 | R@20 | R@100 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| practice (350) | 0.421 | 0.480 | 0.579 | 0.770 | 0.489 | 0.420 |
| dev (125) | 0.379 | 0.475 | 0.580 | 0.724 | 0.460 | 0.398 |

## 9. Team code added to the course checkout (overlay)

**Need.** Our own scripts live in the team repo (`<team-repo>/overlay/`) and are copied over a clean
course checkout, so course files are never edited. A guard checks that the course files are intact.

```bash
git -C ~/<team-repo> pull
bash ~/<team-repo>/scripts/sync_overlay.sh "$PWD"      # run from the course repo root
ls rag/extensions
```
```text
guard: note: course HEAD 9e8fc59 != pinned 6bbdfbc (frozen hashes are from the pinned commit)
guard OK: 55 frozen files intact, 5 task files conform
README.md  generate_from_transcripts.py  retrieval_sweep.py  transcribe_split.py
```
**Explanation.** Three added scripts: ASR transcription of a split (entry 5), answer generation
from cached transcripts, and the retrieval-depth sweep. The "note" only says this checkout has
extra local commits (our docs commits on branch `handoff-docs`), so its HEAD differs from the
commit the frozen-file hashes were taken from; the 55 course files still match. An earlier
attempt with a relative path failed (`cp: ... are the same file`); the script was fixed
upstream to cope with it, and passing an absolute path avoids it anyway.

## 10. Retrieval-depth sweep (practice split, no ELM calls)

**Need.** The planned investigation: how deep should retrieval go? Measure, for each depth k,
the evidence found (Recall@k with a bootstrap confidence interval), the retrieval time, and
the size of the context that depth would pass to the answer model.

```bash
python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl
```
```text
| k | Recall@k (95% CI) | retrieval ms (median) | context words | context chars |
| 1 | 0.229 (0.196-0.265) | 0.44 | 175 | 987 |
| 2 | 0.319 (0.281-0.359) | 0.45 | 354 | 1999 |
| 5 | 0.421 (0.378-0.464) | 0.45 | 897 | 5054 |
| 10 | 0.480 (0.436-0.521) | 0.45 | 1814 | 10193 |
| 20 | 0.579 (0.534-0.621) | 0.44 | 3640 | 20427 |
| 50 | 0.684 (0.644-0.723) | 0.45 | 9076 | 51003 |
| 100 | 0.770 (0.732-0.806) | 0.45 | 17957 | 100905 |
MRR@10 0.489  NDCG@10 0.420  (n=350)
```
**Explanation.**
- Retrieval time is flat (about 0.45 ms median at every k): exact search computes a similarity
  against all 24,000 vectors regardless of k, and only the cheap top-k selection changes. So
  retrieval latency is not the cost of going deeper.
- The cost is context: it grows almost linearly with k (about 180 words per record; 897 words at
  k=5, about 18,000 words at k=100, i.e. 20x more text for the answer model to read and pay for).
- Recall gains shrink per extra word: k=1 to 5 adds 19 points for 720 more words; k=20 to 100
  adds 19 points for 14,000 more words. At k=100 about 23% of relevant records are still missed.
- Confidence intervals are about +/- 4 points, so neighbouring depths (k=10 vs 20) are only
  barely separable and k=1/2 vs 5 clearly are.
- MRR@10 and NDCG@10 use a fixed cut-off at 10, so they do not change with k (0.489 / 0.420, same as
  entry 7, which is a consistency check for this script).
**Limits.** Recall is a proxy for answer quality. Whether more evidence improves the final
answer, or whether long contexts dilute it, still has to be measured with the answer scores
(entry 11 is only a 5-query smoke test). Latency here is warm-cache, CPU, one machine.

## 11. Smoke test: generation and judging on 5 queries (uses ELM calls)

**Need.** Check that the whole path works end to end (cached transcript -> retrieve k=5 ->
context -> ELM answer -> ELM judge) before spending calls on a bigger run.

```bash
python rag/extensions/generate_from_transcripts.py --split practice \
  --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/smoke_k5.jsonl --limit 5
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/smoke_k5.jsonl --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/smoke_k5.jsonl --allow-subset --out-dir rag/results/smoke_judge
```
```text
Validated 5/350 practice candidates; no API calls
q_7e9a...: 0.333   q_ea0b...: 0.583   q_9cd7...: 0.000   q_20e9...: 0.417   q_e3f1...: 0.417
```
Per-query timing in ms (embed / retrieve / generate / end-to-end): 54/7/3444/3506,
37/10/6197/6245, 25/4/5806/5835, 21/6/2330/2357, 34/4/5998/6037.

**Explanation.**
- The pipeline works: 5 answers generated, dry run validated them, and the judge scored them
  (normalised score = (correctness + grounding + completeness) / 12).
- Mean judge score is 0.35, but n = 5 and this is a pipeline test, not a result.
- The one 0.000 is q_9cd7..., a query whose Recall@10 was also 0 in entry 7; the answer hedges
  ("can vary") instead of giving a grounded timeline. That fits retrieval failure producing a poor
  answer, which supports using retrieval depth as the lever. One case is an anecdote only.
- The first answer talks about a single-owner LLC for a generic sales-tax question, which
  suggests retrieved evidence that drifted off-topic; to be checked against the evidence.
- Latency: generation (2.3-6.2 s) is 95-99% of end-to-end time; embedding is 21-54 ms and
  retrieval 4-10 ms. This agrees with entry 10: retrieval is cheap, the answer model is the cost
  and it will scale with context size.
- `asr_ms` is 0 because transcripts came from the cache (a lookup); real ASR cost is in entry 5.
- Retrieval here (4-10 ms) is slower than the sweep's 0.45 ms median because these are the
  first calls after loading (cold) and a single call includes overhead; compare like with like.
**ELM budget used so far:** 5 generation calls + 5 judge calls (1 repeat per answer; judge cache
has 5 entries) = 10 calls. Judge settings from the summary file: rubric `rag-answer-rubric-v0.1`,
judge model `utter-project/EuroLLM-22B-Instruct-2512`, question = ASR transcript, repeats 1.
Mean sub-scores (0-4): correctness 1.6, grounding 1.6, completeness 1.0 -> normalised 0.35.
Report the judge model and these settings with any answer-quality number.

## 12. Retrieval-depth sweep on the dev split (no ELM calls)

**Need.** Check that the depth trend from entry 10 also holds on the held-out dev queries.

```bash
python rag/extensions/retrieval_sweep.py --split dev --transcripts rag/.cache/dev_transcripts.jsonl
```
```text
| k | Recall@k (95% CI) | retrieval ms (median) | context words | context chars |
| 1 | 0.209 (0.153-0.272) | 0.46 | 183 | 1038 |
| 2 | 0.279 (0.214-0.351) | 0.45 | 330 | 1875 |
| 5 | 0.379 (0.311-0.454) | 0.54 | 830 | 4708 |
| 10 | 0.475 (0.404-0.553) | 0.52 | 1748 | 9908 |
| 20 | 0.580 (0.515-0.650) | 0.46 | 3598 | 20400 |
| 50 | 0.677 (0.613-0.742) | 0.45 | 9057 | 51336 |
| 100 | 0.724 (0.664-0.785) | 0.52 | 18101 | 102473 |
MRR@10 0.460  NDCG@10 0.398  (n=125)
```
**Explanation.** Same shape as practice: recall rises with k with diminishing returns, retrieval
time stays about 0.5 ms, and context grows linearly (about 180 words per record). Dev intervals are
wider (about +/- 7 points) because there are only 125 queries, so every dev/practice difference
is inside the noise (for example Recall@5 0.379 vs 0.421, Recall@100 0.724 vs 0.770: the
intervals overlap). Conclusion for the report: the depth trend replicates, and practice
(350 queries) is the better split for tuning, dev for a final check.

## 13. Custom CUDA kernels: correctness check and first end-to-end timing (cluster GPU)

**Need.** Before timing or ablating the kernels (RoPE, reductions/norms, tiled GEMM, attention),
prove they are correct and race-free, and see how the full ASR model behaves on them.
The kernels were written by a Claude session in the team repo (declare AI use in the report).

```bash
cd ~/<course-checkout>/hw1-asr
sbatch ablation/ablate.sbatch --stage check      # 1 x RTX 2080 Ti, batch job (job 3708631)
tail -n 40 ablate-3708631.out
```
```text
Task 1 (RoPE): ALL PASS      Task 2 (reductions): ALL PASS
Task 3 (tiled GEMM): ALL PASS   Task 4 (attention): ALL PASS
compute-sanitizer memcheck: 0 errors and racecheck: 0 hazards, for tasks 1-4
Time: 3285.7 ms per transcription (+/- 6.1 ms), 252.7 ms per token
Transcription: Concord returned to its place amidst the tents.   Accuracy: 100.0%   Status: PASS
```
**Explanation.** All four course test suites pass, and the sanitizer found no out-of-bounds access
and no shared-memory race on the small cases, so the kernels are correct on those inputs
(sanitizer coverage is only the small cases). The full model on our kernels produces the exact
reference transcript. Speed: 3285.7 ms per transcription versus 1215.4 ms for the all-PyTorch
baseline (entry 4), about 2.7x slower (252.7 vs 93.5 ms per token); see entry 15, which measures the torch baseline inside the same job and shows 4.1x. That is expected for simple
hand-written kernels against vendor-tuned library code, and it is the main motivation for the
ablations (which operation is the bottleneck, and which design choice costs what).
Same GPU and clip as entry 4 (RTX 2080 Ti, one 3.5 s clip), so the two numbers are comparable.
**Limits.** One clip, one node, one job; the +/- is run-to-run spread within the job.

## 14. Kernel ablation sweep (cluster, RTX 2080 Ti, batch job; ms per call, median of repeats)

**Need.** Find out how each design choice in the four kernels changes speed, using the course's
own test cases (shapes taken from the real model). Each variant is a separate build with one
compile-time knob changed; default = thread/tile values shipped with the handout.

```bash
cd ~/<course-checkout>/hw1-asr
sbatch ablation/ablate.sbatch --stage sweep          # results in ablation/results/<UTC time>/summary.md
cat ablation/results/*/summary.md
```
Selected rows (full tables are in the result folder; "default" = 16x16 tile, 256 threads, 4 warps):

| Task | case | default | variant | variant ms |
|---|---|---|---|---|
| 3 GEMM | encoder fc1+GELU 1500x5120x1280 | 14.21 | tile 32 / tile 8 / no pad | 9.30 / 24.85 / 24.00 |
| 3 GEMM | conv2 im2col 1500x1280x3840 | 10.12 | tile 32 / tile 8 | 6.92 / 17.61 |
| 3 GEMM | decoder q_proj 64x2048x2048 | 0.315 | tile 32 / no pad / W strided | 0.267 / 0.704 / 0.444 |
| 3 GEMM | lm_head 1x59264x2048 | 2.72 | tile 8 / tile 32 / no pad | 1.54 / 4.84 / 4.72 |
| 4 Attention | audio encoder (20 heads, 1500) | 11.50 | 1 / 2 / 8 / 16 warps / no pad | 103.6 / 30.1 / 8.34 / 7.58 / 54.2 |
| 4 Attention | text decoder (GQA 16/4, causal) | 0.067 | 8 / 16 warps | 0.034 / 0.022 |
| 2 Reduce | softmax, vocabulary row (59264) | 0.118 | 32 / 128 / 1024 threads, 3-pass | 0.886 / 0.223 / 0.041, 0.150 |
| 2 Reduce | rmsnorm, text decoder | 0.006 | atomic version / 32 threads | 0.022 / 0.028 |
| 2 Reduce | layernorm, audio encoder | 0.040 | 128 / 512 / 1024 threads | 0.042 / 0.048 / 0.076 |
| 1 RoPE | audio encoder (partial) | 0.031 | 32 / 64..512 threads | 0.038 / 0.031 |

**Explanation.**
- **GEMM tile size depends on the shape.** Tile 32 is 35% faster on the big encoder GEMMs
  (more reuse of each loaded value, fewer global loads) but 78% slower on lm_head (one row only,
  so a 32-wide tile mostly sits idle), where tile 8 wins. One fixed tile cannot be best for the
  whole model; the large encoder GEMMs dominate time, so tile 32 is the better default.
- **Shared-memory padding matters a lot.** Removing the +1 padding makes the 16-tile GEMM
  2.2x slower (q_proj 0.315 to 0.704) and the attention kernel 4.7x slower (11.5 to 54.2 ms):
  without padding, threads of a warp hit the same shared-memory bank and the accesses serialise
  (bank conflicts). Strided weight loads cost 41% in GEMM (uncoalesced global reads).
- **Attention parallelism:** 1 warp is 9x slower than 4; 8 to 16 warps are 27 to 34% faster on the
  encoder and up to 3x faster in the small decoder case (more warps hide memory latency).
- **Reduction kernels:** more threads help a long row (vocabulary softmax: 1024 threads are 2.9x faster
  than 256, 32 threads 7.5x slower) but not short rows (layernorm over 1280 columns is best near
  128 to 256 threads; 1024 is 1.9x slower because most threads idle). The single-pass softmax beats the
  3-pass version by 27% on the long row (one fewer read of the data). An atomic RMSNorm is 3.7x slower
  than the block reduction (contention).
- **RoPE is memory bound:** thread count barely matters; the audio-encoder case already reaches about
  80% of the card's peak bandwidth (about 494 of 616 GB/s). `--use_fast_math` failed the course
  accuracy test on that case, so it is not an option.
- **Efficiency context:** default GEMMs reach about 1.4 to 1.9 TFLOP/s (roughly 10 to 14% of the
  card's fp32 peak, quoted from memory, about 13 TFLOP/s), so a library GEMM will be much faster
  (entry 15). lm_head reaches only 89 GFLOP/s because one row has no reuse (memory bound).
**Limits.** Per-call times of tiny cases (0.003 ms) are launch overhead, not kernel speed. One GPU
model, one job, medians of 3 repeats from the test harness. The deliberately racy variants
(race stage) were not run.

## 15. Full-model ablation (end to end, same job)

**Need.** Convert the per-kernel numbers into what matters: whole-transcription time, with the
torch baseline measured in the same job (GPU clocks and node load change between jobs).

```bash
sbatch ablation/ablate.sbatch --stage e2e            # cumulative task groups + best variants
```
| run | --cuda | ms/transcription | ms/token | vs torch only | accuracy |
|---|---|---|---|---|---|
| torch_only | none | 806.3 | 62.0 | 1.00x | 100% |
| task1 | task1 | 727.1 | 55.9 | 0.90x | 100% |
| task1-2 | task1,task2 | 705.2 | 54.2 | 0.87x | 100% |
| task1-3 | task1,task2,task3 | 2998.2 | 230.6 | 3.72x | 100% |
| all (default) | all | 3288.3 | 252.9 | 4.08x | 100% |
| all, tile 8 | all | 4992.2 | 384.0 | 6.19x | 100% |
| all, tile 32 | all | 2865.2 | 220.4 | 3.55x | 100% |
| all, 8 attention warps | all | 3184.0 | 244.9 | 3.95x | 100% |

**Explanation.**
- RoPE and the norm/softmax kernels slightly beat PyTorch (13% together), because they are
  memory-bound and simple.
- Switching the linear layers to our GEMM makes the model 4.3x slower than the previous step
  (705 to 2998 ms): the GEMM is the bottleneck, as the sweep predicted (about 1.5 TFLOP/s against
  a tuned library). Attention adds about 290 ms more (3288 vs 2998).
- Best single change: GEMM tile 32 (-13% on the all-kernel run); tile 8 is 52% worse. 8 attention
  warps help only 3%, because attention is a small share of the total (Amdahl's law). Ranking by
  end-to-end gain is therefore different from ranking by micro-benchmark gain.
- Accuracy stays 100% in every run, so none of the variants changes the output.
**Important caveat.** The torch baseline here (806 ms) is much faster than the one in entry 4
(1215 ms), measured in a different job (see entry 16 for the likely cause: a different node).
Compare only numbers from the same job and node; always report the within-job ratio. Per-operation profiles for
torch_only, task1-3 and all_default were saved (`e2e_*.txt`) but not yet read.

## 16. Per-operation profile, combined best configuration, and a node effect

**Need.** Explain where the time goes, test the best single changes combined, and find out why the
PyTorch baseline differs between jobs (806 ms in entry 15, 1215 ms in entry 4).

```bash
sbatch ablation/ablate.sbatch --stage best        # full model, 4 runs; profiles are in the e2e run folder
sacct -u $USER -S 2026-11-09 --format=JobID,JobName%12,State,Elapsed,NodeList -X
```
GPU time per operation from the profile pass (shares are reliable; the profile pass synchronises
each operation, so its absolute milliseconds are inflated, e.g. 1192.6 ms of kernel time against an
806 ms wall clock):

| operation | PyTorch only: share | our kernels: share | ms (profile): torch -> ours |
|---|---|---|---|
| linear (2757 calls) | 47.6% | 83.6% | 568.0 -> 2871.9 (5.1x more) |
| attention (396) | 22.5% | 15.9% | 267.8 -> 544.9 (2.0x more) |
| RoPE (396) | 25.0% | 0.1% | 298.7 -> 3.0 (about 100x less) |
| RMSNorm (741) | 4.2% | 0.2% | 50.0 -> 6.6 (7.6x less) |

Combined configuration (full model, all four tasks on our kernels, all runs PASS at 100%):

| run | ms/transcription | ms/token | vs default |
|---|---|---|---|
| default (first measurement, entry 15) | 3288.3 | 252.9 | - |
| default (repeat) | 3294.0 | 253.4 | +0.2% |
| attention 16 warps | 3148.9 | 242.2 | -4.4% |
| GEMM tile 32 (first, entry 15 / again) | 2865.2 / 2876.6 | 220.4 / 221.3 | -13.0% / -12.7% |
| GEMM tile 32 + attention 16 warps | 2728.1 | 209.9 | -17.2% |

**Explanation.**
- Where the time goes: with PyTorch the work is split between linear layers, RoPE (many small
  elementwise operations) and attention. With our kernels the linear layers take 84% of the time,
  so the tiled GEMM is the only kernel that still matters much; optimising anything else is capped
  by Amdahl's law.
- The two best single changes (tile 32: -13%, 16 warps: -4.4%) combine to -17.2%, roughly additive
  (they affect different operations). Repeats agree to within 0.2% to 0.4%, so differences of
  a few percent between variants measured in one job are real.
- Node effect (new finding). Jobs ran on two nodes with the same GPU model: the PyTorch-only
  run (806 ms) and the cumulative runs were on one node (e2e job), while the correctness check
  and the 1215 ms baseline (entries 4, 13) and the `best` job ran on the other. Our kernels give the
  same time on both nodes (3285.7, 3288.3, 3294.0 ms), but the PyTorch-only path differs by 1.5x
  (1215 vs 806 ms). Hypothesis, not yet tested: the PyTorch path launches thousands of tiny GPU
  operations from Python and is limited by CPU speed, whereas our kernels are limited by the GPU,
  so only the PyTorch time depends on the node. Test: run `--cuda none` on both nodes.
- Consequence for the report: quote the node with every timing, quote ratios only within one
  node, and say that the slowdown of our kernels against PyTorch is 4.1x on the faster-CPU node
  and 2.7x on the other. A third value, 992.9 ms, quoted in the course plan, is consistent with
  node-to-node variation.
**Limits.** One 3.5 s clip. The node attribution comes from the scheduler records and has not
been re-measured; the CPU-bound explanation is a hypothesis.

## 17. Answer quality versus retrieval depth (Plan A, first 15 queries, uses ELM)

**Need.** Plan A asks how far retrieval depth k can be cut before answer quality drops, and what the
cut saves in prompt size and time. Entry 10 measured recall and context size; this adds the answer
quality, measured by the ELM judge, on the same cached transcripts.

```bash
bash rag/extensions/run_answer_quality.sh 15     # 15 practice queries x k in {1, 5, 20}
# per k: generate_from_transcripts.py --k K --limit 15 --resume ; evaluate_answers.py (judge, 1 repeat)
```
Budget used: 45 generation + 45 judge = 90 ELM calls (free locally hosted models). The judge, rubric
and prompt are the course ones (entry 11); the first 15 practice queries are fixed, so every depth
answers the same queries (paired comparison).
```text
| k | judge score (95% CI) | correct | grounded | complete | Recall@k | context words | prompt tokens | gen ms (median) |
| 1 | 0.178 (0.050-0.339) | 0.80 | 0.80 | 0.53 | 0.211 | 163 | 305 | 1804 |
| 5 | 0.317 (0.150-0.500) | 1.33 | 1.40 | 1.07 | 0.533 | 864 | 1246 | 4463 |
| 20 | 0.361 (0.167-0.567) | 1.40 | 1.60 | 1.33 | 0.722 | 3517 | 4908 | 6590 |
paired vs k=1: k=5 +0.139 (-0.006 to +0.311), 6 better / 2 worse / 7 tied
               k=20 +0.183 (+0.006 to +0.372), 6 better / 2 worse / 7 tied
```
Scores are (correctness + grounding + completeness) / 12; sub-scores are on a 0 to 4 scale.

**Explanation.**
- Going from one record to five clearly helps (0.18 to 0.32; recall 0.21 to 0.53), and 20 is a little
  better still (0.36). Between 5 and 20 the gain is small (+0.044) and well inside the noise
  (intervals of about +/- 0.2 with n = 15), while the cost is large: 4.1x the context words
  (864 to 3517), 3.9x the prompt tokens (1246 to 4908) and 1.5x the generation time (4463 to 6590 ms).
  Tentative reading: most of the benefit comes early; deeper retrieval pays an increasing price
  for little extra quality. This is a first look, not a conclusion.
- Generation time grows more slowly than prompt size (1.5x for 3.9x tokens), so most of the
  delay is producing the answer, not reading the context; this should be checked against output length.
- Completeness improves most (0.53 to 1.33): more evidence mainly makes answers more complete.
- Many scores are zero (10 of 15 at k=1, 7 of 15 at k=5 and at k=20). Four queries score zero at both
  k=5 and k=20 (three at all three depths), so extra evidence does not rescue them; likely causes are
  retrieval misses, ASR errors in the question, or a strict judge. The next step is to read those
  queries and their judge explanations.
- This 15-query subset is easier for retrieval than the whole practice split (Recall@5 0.533 here vs
  0.421 on 350 queries; Recall@20 0.722 vs 0.579), so absolute numbers should not be generalised.
**Limits.** n = 15, one judge repeat, one judge model (judge noise not measured), one generator,
single run; confidence intervals are bootstrap intervals over queries.
**Superseded by entry 19:** with 28 queries the trend largely disappears; this 15-query subset was favourable.

### 17b. The three queries that score zero at every depth (no ELM calls; read from saved results)

**Need.** Understand why extra evidence does not rescue some queries, before trusting the average.

```bash
# per-query files: rag/results/aq_practice_k{1,5,20}.jsonl and aq_practice_k{k}_judge/answer_judge_per_query.jsonl
# compared with rag/data/rag_corpus/qrels_practice.tsv (ranks of the relevant records in the top 20)
```
| query | relevant records | ranks of relevant records in top-20 | why it scores zero |
|---|---|---|---|
| sales-tax implications (vague question) | 2 | 4 and 13 | Evidence was retrieved (rank 4 is inside k=5) but the answer stayed generic; the judge wants the specific facts in the gold evidence (a state-specific grocery tax rule) |
| ROI and verifying financial projections | 2 | none | Retrieval miss: the relevant records never enter the top 20 for this vague transcript |
| historical price/earnings ratios for US indices | 1 | none | Retrieval miss, and the gold evidence is a vague record ("Internet sites, books, academic"); a concrete, plausible list of platforms is judged unsupported |

**Explanation.** Three different failure causes: (1) evidence present but not used well (generation
side, or vague question), (2) retrieval misses where depth up to 20 does not help (representation
or ASR/question wording issue, motivating plans B and C), (3) the judge scores against the gold
evidence reconstructed from the relevance labels, so a good answer that is not backed by that
specific record gets zero. Consequence: judge scores underestimate quality for some correct answers,
and the zero-score floor is one reason the curve flattens after k=5. Report this limit with every
judge number. Sample size is 3 queries, so these are illustrations, not frequencies.

## 18. How often does retrieval find any relevant record? (hit@k, all queries, no ELM calls)

**Need.** Recall@k averages over relevant records; the answer needs at least one good record. hit@k
(share of queries with at least one relevant record in the top k) separates "retrieval found
something" from "the model used it", and gives the retrieval ceiling for any depth.

```bash
# from the course repo root, venv active (script saved as docs/analysis/hit_at_k.py in this repo)
python docs/analysis/hit_at_k.py
```
```text
practice (350): hit@1 0.411, @5 0.591, @10 0.651, @20 0.737, @50 0.820, @100 0.880
dev (125):      hit@1 0.376, @5 0.584, @10 0.656, @20 0.768, @50 0.840, @100 0.880
never found in top 100: 12% of queries (both splits); median rank of the first relevant record: 2
```
**Explanation.**
- 41% of practice queries already have a relevant record at rank 1, 59% within 5, 74% within 20.
  Going from k=5 to k=20 raises hit@k by 15 points (0.591 to 0.737) for 4x the context: the same
  diminishing returns seen in answer quality (entry 17), now measured on all 350 queries.
- 12% of queries (both splits, identical) have no relevant record in the top 100, so no choice of k
  helps them; this is a hard ceiling set by the encoder and the query text (plans B and C target it).
- Dev and practice agree within a few points, so the finding is stable across splits.
- Combined with entry 17b: part of the zero answer scores come from this 12% (retrieval miss),
  part from the judge/generation side.
**Limits.** Uses the cached PyTorch-ASR transcripts and enc_a; relevance labels are the supplied qrels.

## 19. Answer quality versus retrieval depth, extended to 30 queries (28 judged; replaces entry 17)

**Need.** Entry 17 used 15 queries and gave a promising trend with wide intervals. To compare fairly,
the same experiment was extended with the next 15 practice queries under identical settings
(same cached transcripts, same prompt, same judge and rubric, one repeat, k in {1, 5, 20}); the
first 15 queries and their results were reused, not re-run. Only new queries cost calls.

```bash
EXCLUDE="q_d442a6395dbca951 q_4e29b0ce6fa96d1a" bash rag/extensions/run_answer_quality.sh 30
# EXCLUDE: queries whose judge reply is malformed JSON (see below); dropped at every depth to keep pairing
```
```text
n = 28 practice queries answered and judged at every k in [1, 5, 20]
| k | judge score (95% CI) | correct | grounded | complete | Recall@k | context words | prompt tokens | gen ms (median) |
| 1 | 0.250 (0.131-0.381) | 1.07 | 1.11 | 0.82 | 0.232 | 162 | 305 | 1803 |
| 5 | 0.268 (0.155-0.387) | 1.14 | 1.18 | 0.89 | 0.472 | 864 | 1249 | 3782 |
| 20 | 0.292 (0.173-0.411) | 1.18 | 1.32 | 1.00 | 0.639 | 3397 | 4730 | 4303 |
paired vs k=1:  k=5 +0.018 (-0.122 to +0.161), 8 better / 7 worse / 13 tied
                k=20 +0.042 (-0.101 to +0.182), 8 better / 9 worse / 11 tied
```
Mean judge score by query group (n = 15 first / 13 new): k=1 0.178 / 0.333; k=5 0.317 / 0.212;
k=20 0.361 / 0.212. Zero scores: 16 of 28 at k=1, 14 of 28 at k=5 and at k=20.

**Explanation.**
- With 28 queries there is no measurable answer-quality gain from deeper retrieval: all paired
  differences are small and their intervals include zero (k=5 is better on 8 queries and worse on 7).
  The trend seen in entry 17 came mostly from the first 15 queries; the 13 new queries show the
  opposite direction (0.333 at k=1, 0.212 at k=5 and k=20). That is what sampling noise looks like
  with a noisy judge, so entry 17's tentative conclusion is withdrawn.
- Costs are real and clear: prompt tokens grow 305, 1249, 4730 (4.1x then 3.8x) and the median generation
  time 1.8, 3.8, 4.3 s. Recall@k and hit@k (entry 18) rise strongly with k, yet judged answer quality
  does not follow: retrieving more relevant records is not turning into better judged answers here.
- Likely reasons (hypotheses): (i) the judge scores against the gold evidence only and gives zero to many
  answers (14 to 16 of 28), a hard floor (entry 17b); (ii) retrieval misses affect 12% of queries at any k
  (entry 18); (iii) the generator may not use extra context well, or longer context dilutes it.
- Working conclusion for plan A: under this judge, k about 1 to 5 gives about the same judged quality as k=20
  at a fraction of prompt size and latency; the evidence for any benefit of k=20 is absent, not negative.
**Data handling.** For 2 of 30 queries the course judge returned malformed JSON at k=5 (temperature 0, so
retries repeat the same failure); these queries (`q_d442a6395dbca951`, `q_4e29b0ce6fa96d1a`) were excluded at
every depth, leaving n = 28. Excluding judge failures could bias the sample if failures correlate with
answer length, so report n and the excluded ids. The judge itself was not modified.
**ELM budget.** About 90 + roughly 96 calls for this experiment (45 + 39 new generate/judge calls plus at most
a dozen repeats of the failing judge requests), plus the 10 smoke-test calls: about 190 in all.
**Limits.** n = 28 from one split, one judge repeat and one judge model (judge noise unmeasured), one generator,
single run; intervals are bootstrap over queries. A +/- 0.14 interval means differences below roughly 0.1 are not detectable.

## 20. Why does depth not raise judged quality? Evidence-in-context analysis (no ELM calls)

**Need.** Entry 19 shows flat judged quality while recall and hit rate rise with k. Split the 28 judged
queries by whether (and at which rank) a relevant record is in the context, and follow each group across depths.
Computed from saved answers, judge scores and the relevance labels (exploratory, after seeing the data).

```bash
# in the course repo root: rag/results/aq_practice_k{1,5,20}.jsonl + aq_practice_k{k}_judge/answer_judge_per_query.jsonl
# + rag/data/rag_corpus/qrels_practice.tsv ; a relevant record is "in context" if it is in the top-k retrieved ids
```
Mean judge score when a relevant record is in the context versus not:

| k | relevant record in context | mean score (zeros) | not in context | mean score (zeros) |
|---|---|---|---|---|
| 1 | 11 queries | 0.447 (4) | 17 queries | 0.123 (12) |
| 5 | 19 | 0.333 (8) | 9 | 0.130 (6) |
| 20 | 23 | 0.355 (9) | 5 | 0.000 (5) |

The same queries followed across depths, grouped by rank of the first relevant record:

| group (rank of first relevant record) | n | score at k=1 / 5 / 20 |
|---|---|---|
| rank 1 | 11 | 0.447 / 0.394 / 0.386 |
| ranks 2 to 5 | 8 | 0.208 / 0.250 / 0.344 |
| ranks 6 to 20 | 4 | 0.104 / 0.292 / 0.292 |
| not in top 20 | 5 | 0.000 / 0.000 / 0.000 |

**Explanation.**
- Having a relevant record in the context matters: answers score about 0.33 to 0.45 with one and about 0 to 0.13 without.
  More depth puts a relevant record in front of more queries (11, 19, 23 of 28), so it does help the queries it reaches.
- But two effects cancel in the average: queries whose relevant record is already at rank 1 get slightly worse
  with more context (0.447 to 0.386, the extra records act as distractors or dilute the answer), while
  queries whose evidence sits at ranks 2 to 20 improve (0.208 to 0.344 and 0.104 to 0.292). The net average is
  nearly flat (0.250, 0.268, 0.292).
- The five queries with no relevant record in the top 20 stay at 0 at every depth; no k fixes them (entry 18, plans B and C).
- Mechanism for the report: deeper retrieval is a trade between coverage (finding evidence that ranks lower) and
  dilution (more prompt text per answer, higher cost). A re-ranker, or adaptive k (small k when the top record is a
  confident match), could keep the gain and avoid the loss; this is a hypothesis to test.
**Limits.** Very small groups (n = 11, 8, 4, 5), groups defined after seeing the data, single judge repeat. Treat as an
explanation worth testing with more queries, not as a measured effect.

## 21. Hand-in run, part 1: `make test` on the submitted configuration, with GEMM naive vs tiled vs cuBLAS

**Need.** The coursework requires every case PASS on one GPU node and a GEMM comparison of our tiled kernel against the naive
kernel and cuBLAS. Configuration submitted: GEMM tile 32, 16 attention warps, 256 threads for RoPE and reductions (handout
defaults are 16 and 4). Node: a third node of the same GPU model (RTX 2080 Ti, sm_75, 68 SMs, about 616 GB/s peak bandwidth).

```bash
cd ~/<course-checkout>/hw1-asr
sbatch ablation/handin.sbatch        # nvidia-smi -L, kernel settings, make clean, make test, ./run.sh --profile
tail -n 90 handin-<JOBID>.out
```
All four test suites printed ALL PASS (largest error 3.6e-05, on the GEMM that sums 6144 products).

GEMM, GFLOP/s per case (our tiled kernel, tile 32, versus the naive kernel and cuBLAS, which the test also times):

| case (M x N x K) | tiled (ours) | naive | cuBLAS | cuBLAS / tiled |
|---|---|---|---|---|
| decoder q_proj 64x2048x2048 | 1462 | 153 | 4647 | 3.2x |
| decoder gate_proj 64x6144x2048 | 1497 | 162 | 6825 | 4.6x |
| decoder down_proj 64x2048x6144 | 1825 | 193 | 8143 | 4.5x |
| lm_head 1x59264x2048 | 63 | 111 | 291 | 4.6x |
| encoder q_proj 1500x1280x1280 | 2160 | 231 | 13201 | 6.1x |
| encoder fc1 + GELU 1500x5120x1280 | 2107 | 233 | 13363 | 6.3x |
| conv2 via im2col 1500x1280x3840 | 2130 | 232 | 13340 | 6.3x |

**Explanation.**
- Tiling is worth about 9x on the large encoder GEMMs (about 2.1 vs 0.23 TFLOP/s): staging 32x32 tiles in shared memory cuts the global loads per
  output by a factor equal to the tile width, which the naive kernel (every operand from global memory) cannot do.
- cuBLAS is another 6x faster on the large cases: about 13.3 TFLOP/s, close to the card's fp32 peak (about 13 TFLOP/s, from memory), while
  ours reaches roughly 16% of it. Likely reasons (from the course question; not measured here): cuBLAS computes several outputs per thread in
  registers (register tiling), uses vectorised loads and double-buffering, and is tuned per architecture.
- lm_head (one row, M = 1) is the exception: our tiled kernel (63 GFLOP/s) is slower than the naive one (111), because a 32-row tile has only 1 useful row
  and most threads idle; this is the shape where tile 8 won in entry 14. A shape-dependent tile would fix it.
- Memory-bound kernels: RoPE reaches 80% of peak bandwidth on the audio-encoder case (about 492 of 616 GB/s), layernorm 62%, softmax over 64 x 4096 39%.
  The vocabulary-row softmax reaches only 1% of peak (one block for one long row; entry 14), and the small cases are launch-bound.
**Limits.** Single run of each case on one node; small cases are checked against a CPU loop, large ones against cuBLAS. The `./run.sh --profile` part of
the same job (full model, profile table) was still running when this entry was written and is logged separately.

## 22. Hand-in run, part 2: full model with the submitted kernels and per-operation profile

**Need.** The second half of the hand-in output: the transcription benchmark and the profile table from the same job and node as entry 21.

```bash
# same job as entry 21 (handin.sbatch); output file handin-<JOBID>.out
./run.sh --profile            # all four tasks on our kernels: tile 32, 16 attention warps
```
```text
Loaded in 179.9 s   (cold file cache on this node; not inference time)
run 1: 2697.3 ms   run 2: 2708.7 ms   run 3: 2728.0 ms   (13 tokens each)
Time: 2711.3 ms per transcription (+/- 12.7 ms), 208.6 ms per token
Peak GPU memory: 8.56 GiB
Transcription: Concord returned to its place amidst the tents.   Accuracy: 100.0%   Status: PASS
GPU time per operation (profile pass, 2846.5 ms in kernels):
| operation | runs on | calls | total ms | share |
| linear (GEMM) | cuda | 2757 | 2449.5 | 86.1% |
| attention | cuda | 396 | 378.3 | 13.3% |
| rmsnorm | cuda | 741 | 6.7 | 0.2% |
| silu_mul | cuda | 364 | 3.5 | 0.1% |
| rope | cuda | 396 | 3.0 | 0.1% |
| layernorm | cuda | 65 | 2.4 | 0.1% |
| softmax | cuda | 13 | 1.5 | 0.1% |
| embedding | cuda | 13 | 1.3 | 0.0% |
| im2col | cuda | 2 | 0.2 | 0.0% |
```
**Explanation.**
- Submitted configuration: 2711.3 ms per transcription, 208.6 ms per token, peak memory 8.56 GiB, transcript exactly correct.
- Consistency check: the same configuration measured earlier on a different node gave 2728.1 ms (entry 16), a 0.6% difference, so our kernel timings are
  reproducible across nodes of the same GPU model (unlike the PyTorch path, entry 16). Compared with the handout configuration (3288 to 3294 ms,
  entries 15 and 16) the submitted settings are about 18% faster.
- Where the time goes: the GEMM (linear layers) is 86% of GPU time and attention 13%; the four memory-bound kernels (RoPE, both norms, softmax) plus embedding
  and im2col together are under 1%. This is Amdahl's law: all remaining tuning effort belongs to the GEMM (register tiling, vectorised loads; entry 21).
- Profile times are measured with per-operation synchronisation, so the total (2846.5 ms) is slightly above the wall-clock time per transcription (2711.3 ms);
  use the shares, not the absolute milliseconds.
**Still missing for the hand-in comparison:** a PyTorch-only run (`./run.sh --cuda none --profile`) in the same job and on the same node as our kernels. The
PyTorch baseline depends on the node (806, 1215 and 992.9 ms were seen on different nodes), so the ratio is only valid when both runs share a node.

## 23. PyTorch baseline and our kernels on one node, in one job (the hand-in comparison)

**Need.** Entry 16 showed the PyTorch-only time depends on the node, so the ratio between our kernels and PyTorch is only valid if both are
measured in the same job. One batch job ran three full-model benchmarks back to back on one node (RTX 2080 Ti, `nvidia-smi -L` printed):
PyTorch only, our kernels (submitted configuration: tile 32, 16 attention warps), PyTorch only again.

```bash
sbatch -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=16G --time=00:45:00 --output=compare-%j.out --wrap='
  ... conda activate mls-asr; cd hw1-asr; hostname; nvidia-smi -L;
  ./run.sh --cuda none --profile; ./run.sh --profile; ./run.sh --cuda none --profile'
grep -E "^<node>|^GPU 0|Time:|Peak GPU|Status" compare-<JOBID>.out
```
| run (same node, same job) | ms per transcription (+/-) | ms per token | peak GPU memory | status |
|---|---|---|---|---|
| PyTorch only (first) | 1096.2 (1.8) | 84.3 | 8.88 GiB | PASS |
| our kernels, tile 32 + 16 warps | 2698.8 (7.9) | 207.6 | 8.56 GiB | PASS |
| PyTorch only (second) | 1103.8 (4.9) | 84.9 | 8.88 GiB | PASS |

**Explanation.**
- The node is stable during the job: the two PyTorch runs differ by 0.7%. Ours versus PyTorch (mean 1100 ms) is 2.45x slower, so the
  hand-written kernels reach about 41% of the PyTorch (cuBLAS-backed) speed on this node.
- Our kernels use 0.32 GiB less peak GPU memory (8.56 vs 8.88 GiB), consistent with the tiled/online-softmax attention not materialising the full score matrix.
- Across nodes the PyTorch baseline varied (806, 1096 to 1104, 1215 ms; 992.9 ms in the course slides) while our kernels stayed near 2700 ms
  (2698.8, 2711.3, 2728.1 in three jobs). The ratio therefore depends on the node: about 2.25x (1215 ms node), 2.45x (this node) and, for the
  faster-baseline node, 3.4x if our submitted configuration (not measured there) behaves as elsewhere. Reason (hypothesis, untested): the
  PyTorch path launches thousands of small operations from Python and is sensitive to the host CPU, while our kernels are GPU-bound.
- Report this as: "on one node, same job: PyTorch 1.10 s, ours 2.70 s (2.45x), with the ratio ranging about 2.2x to 3.4x across nodes".
**Limits.** One 3.5 s clip, three runs per benchmark, one job. The profile tables from this job were not collected here; entry 22 holds the profile
of our kernels (same configuration) and entry 16 the PyTorch profile from another node.

## 24. Adaptive retrieval depth (implemented change for plan A; no ELM calls)

**Need.** Entry 20 suggested that the extra records hurt queries whose top record is already relevant, and help those where it is not.
Idea: choose k per query: small k when the retriever is confident about its top record, large k otherwise. This is an implemented technical
change (script `rag/extensions/adaptive_k.py` in the team overlay) with a protocol fixed before looking at answer quality:
(1) on the practice split, using only retrieval labels, choose the confidence feature (top-1 cosine score or the margin top-1 minus top-2) by AUC
for "is the rank-1 record relevant?" and the threshold by Youden's J; (2) freeze it and evaluate on dev (held out); (3) replay on already judged
answers: the adaptive answer for a query is the existing answer at the depth the policy picks, so no new ELM calls.

```bash
~/mls-rag-env/bin/python rag/extensions/adaptive_k.py           # tune on practice, check on dev, replay
~/mls-rag-env/bin/python docs/analysis/fixed_k_matched_context.py   # fixed k with the same mean context, for a fair comparison
```
Policy found: k = 1 if (score margin) >= 0.0122, else k = 20. AUC for predicting a relevant rank-1 record: 0.719 (practice), 0.691 (dev).

| split | policy | mean k | recall | hit | mean context words |
|---|---|---|---|---|---|
| practice (350) | fixed k=5 | 5 | 0.421 | 0.591 | 897 |
| practice | fixed k=10 | 10 | 0.480 | - | 1814 |
| practice | **adaptive** | 14.3 | 0.481 | 0.680 | 2624 |
| practice | fixed k=14 | 14 | 0.534 | - | 2538 |
| practice | fixed k=20 | 20 | 0.579 | 0.737 | 3640 |
| dev (125) | fixed k=5 | 5 | 0.379 | 0.584 | 830 |
| dev | fixed k=10 | 10 | 0.475 | - | 1748 |
| dev | **adaptive** | 15.1 | 0.503 | 0.712 | 2679 |
| dev | fixed k=14 / 15 | 14 / 15 | 0.539 / 0.553 | - | 2473 / 2644 |
| dev | fixed k=20 | 20 | 0.580 | 0.768 | 3598 |

Replay on the 28 queries judged at every depth (no new calls): judge score (95% CI), prompt tokens:
fixed k=1 0.250 (305 tokens); fixed k=5 0.268 (1249); fixed k=20 0.292 (4730); adaptive 0.277 (3198; +0.009 vs k=5, interval -0.122 to +0.122);
oracle (k=1 if rank 1 really relevant, else 20) 0.315 (2932 tokens).

**Explanation (a negative result).**
- The confidence signal is weak: AUC about 0.7 means the score margin only modestly separates queries whose top record is relevant from the rest.
  The policy picks k=1 for only 26 to 30% of queries and with precision 0.66 to 0.73, so roughly a third of the "confident" queries get k=1 wrongly.
- At the same average context, a plain fixed k beats it on recall: the adaptive policy (mean 14.3 to 15.1, about 2,650 words) reaches recall 0.481 (practice)
  and 0.503 (dev), but fixed k=14 to 15 with the same context reaches 0.534 to 0.553. Retrieval quality per word of context is therefore worse than
  not adapting. The policy mostly saves context compared with k=20 (-28%) at a cost of 10 recall points.
- On judged answers the adaptive policy is indistinguishable from every fixed depth (differences below 0.03, intervals about +/- 0.12); even the oracle,
  which knows the label, gains only +0.05 (not significant). So even a perfect confidence estimate would give a small benefit here: the ceiling is low.
- Why: the dilution effect (entry 20) is small and the zero-score floor of the judge dominates (entries 17b, 19); a gate on retrieval confidence cannot fix
  retrieval misses or judge strictness.
- Conclusion for the report: the idea was tested honestly with a pre-registered protocol and a held-out split; it does not beat fixed depth at matched
  context. This is a well-supported negative result with a mechanism (weak confidence signal, low oracle ceiling).
**Limits.** The judged replay uses 28 practice queries that were also in the tuning split (mild leakage that would favour the policy, yet it still shows no gain);
dev has no judged answers; one feature family (top-1 score, margin) and one depth pair (1, 20) tried; judge noise unmeasured.

## 25. Held-out dev replication (28 dev queries) and adaptive k on held-out answers

Logged in full as E19 in the team repo's `docs/report/experiment-log.md` (authoritative log). Headline: dev n = 27, judge score k = 1 / 5 / 20 = 0.284 / 0.321 / 0.253 (all paired intervals include zero); pooled n = 55: 0.267 / 0.294 / 0.273, differences at most 0.03; adaptive k 0.272 vs fixed k = 5 0.321 (-0.049, 3.3x the prompt tokens); oracle 0.306. Judge could not parse 1 of 28 dev answers (3 of 58 overall). About 169 ELM calls; running total about 360.

---

## Open items (next entries)
- Test the node hypothesis (`--cuda none` on both nodes).
- Cluster RAG venv + `.env` + `check_setup.py` (not reported yet; the baseline ran on the laptop).
- Decide: more queries / finer depths (k=2,3,10), measure judge noise (repeats), or move to plans C and B.
- Investigation question, ELM budget, who does what (undecided).
