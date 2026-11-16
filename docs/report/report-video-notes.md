# Notes for the report and the video (2026-11-16)

Key answers to the key questions, with the log entry behind every number (`docs/report/experiment-log.md`).
Requirements as updated by the course team on 2026-11-16: one question per ASR task (four in total), anywhere in the report but easy to find; "Try this" optional; long `make test` / profile output may go to the appendix; supplementary code = our files only; video ends with two fixed questions (about one minute each).

## 1. The four guide answers (one per task)

Put them together in one subsection (e.g. "Answers to the task questions") and point to it from the introduction.

**Task 1, Q1. Bytes moved and share of peak, audio-encoder case.**
The case rotates 1500 positions x 20 heads x 16 pairs (32 of 64 dims), for q and k: 960,000 pairs. Each pair reads 2 floats and writes 2 floats, 16 bytes, so 15.36 MB in total. Measured 0.031 ms, so 491.8 GB/s, **80% of the 2080 Ti's 616 GB/s** (K08; K02 gave 493.6 GB/s). RoPE is memory-bound and already near the bandwidth limit, which is why block size barely matters (K02).

**Task 2, Q3. How many times softmax reads each element of x.**
Ours reads x **twice**: pass 1 computes the maximum and the sum together with the online (max, sum) merge, and pass 2 reads x again to write exp(x - max) / sum. The textbook version reads it **three times** (max, then sum, then write). Measured on the 59,264-wide vocabulary row: one-pass 0.118 ms vs textbook 0.150 ms, **27% slower** (K02). On shorter rows the two are equal at the timer's 0.001 ms resolution, probably because the row stays in cache.

**Task 3, Q1. Floats loaded for M = N = K = 1024 (requests, no caches).**
Naive: each of the 1024 x 1024 outputs loads a row of x and a column of w, 2K floats: 2 x 1024^3 = **2,147,483,648**. Tiled with TILE = 32: each value loaded into shared memory is used by 32 threads, so 2 x 1024^3 / 32 = **67,108,864**, 32x fewer. Measured speed-up over naive on the large layers is about 9x, not 32x (K08), because the naive kernel's repeated loads partly hit the cache, and our kernel is then limited by shared-memory reads and one output per thread. cuBLAS is another 3-6x faster (16-31% of it reached, K08): it computes several outputs per thread from registers (register tiling), uses vector loads and double buffering (Task 3 Q2, as an optional second answer).

**Task 4, Q4. Key loads with ATTN_WARPS = 4, and the limit.**
Each block loads all keys of its head once, and a head has 1500 / 4 = 375 blocks, so each key is loaded **375 times per head per layer** (94 times with our submitted 16 warps). Measured on the encoder case: 1, 2, 4, 8, 16 warps = 103.6, 30.1, 11.5, 8.3, 7.6 ms (K04). The gain flattens (27% from 4 to 8, 9% from 8 to 16) once key loading stops being the bottleneck. Limits: at most 32 warps (1024 threads) per block, shared memory and registers per block (16 warps use 512 threads and about 20 KB), and fewer, larger blocks leave SMs idle at the end of the grid.

Extra answers are drafted for all 16 questions in `docs/TASK_ANSWERS.md` (laptop copy; review before using).

## 2. The two video questions (about one minute each)

**Question 1: one important design choice, and why.**
Recommended: *we tuned only the GEMM and attention, chosen from a profile, and handed in tile 32 + 16 attention warps.*
- Why: the profile showed matrix multiplies take 84-86% of kernel time and attention 13-16%; RoPE, norms and softmax together under 1% (K06, K08). By Amdahl's law only those two kernels were worth tuning.
- Evidence: tile 32 alone saves 13%, 16 warps 4%, together 17.2% end to end (3294 to 2728 ms) with the transcript still exact (K07).
- Trade-off we accepted: the 32-wide tile needs the +1 padding column (without it, all 32 lanes hit one shared-memory bank and fc1 is 4.5x slower, K03), and it is slower than naive on lm_head, a single-row multiply (63 vs 111 GFLOP/s, K08), where tile 8 is best. One constant cannot fit every shape; a shape-dependent tile is the next step.
- RAG alternative: we fixed the adaptive-k protocol before looking at answer quality (tune on practice labels, check on held-out dev), so the negative result cannot be an artefact of tuning (E17).

**Question 2: how we checked that it worked, with an example.**
- Four layers of checks: the course tests compare every kernel with a CPU or cuBLAS reference on edge cases (odd sizes, all-negative and +-100 softmax inputs, a single new query); compute-sanitizer memcheck and racecheck report 0 errors (K01); the full model on our kernels transcribes the test clip exactly (100%, K01, K08); repeated timings across jobs and nodes differ by only 0.3%, so differences above about 1% are real (K07).
- Example: the `--use_fast_math` variant looked like a free speed-up, but the tests failed on exactly one case, the audio-encoder RoPE, which uses the largest angles; the fast sin/cos approximation loses accuracy there (K02). We rejected it.
- Second example (experiments, not code): at 15 queries k = 20 looked clearly better than k = 1 (0.361 vs 0.178, E12); we extended to 28 queries with identical settings and the gain disappeared (E15), so we withdrew the claim and logged E12 as superseded.
- RAG example: before spending ELM calls we ran a 5-query smoke test (E10) and checked the judge output; 2 queries came back unparseable and were excluded and reported, not scored as zero (E15).

## 3. Key answers for the report

**System and question.** A spoken question goes through ASR (GLM-ASR, our CUDA kernels), a 384-dim encoder, exact dense retrieval over 24,000 records, and an LLM answer (EuroLLM-22B via ELM), judged by the same LLM. Question: *how much retrieved evidence should the generator get when the query comes from ASR?* (retrieval depth k; the slides' own example). It matters because context is the main generation cost.

**Baseline.** Course pipeline, k = 5, practice split (350 queries), dev (125) for confirmation. ASR baseline: the PyTorch path, 1096-1104 ms per transcription on the same node and job as ours (K09).

**Our work (vs supplied).** Supplied: the model, tests, RAG framework, data. Ours: the four kernels (RoPE, warp/block reductions with RMSNorm/LayerNorm/one-read softmax, tiled GEMM with fused bias + GELU, tiled online-softmax attention with GQA and causal masking); the ablation harness (`-D` knobs); the RAG extensions (cached transcripts, depth sweep, answer-quality runner with token usage, paired bootstrap summary, adaptive k).

**Evaluation.** ASR: correctness tests + sanitizers + exact transcript; time per transcription, 3 runs, same node; per-operation profile; GB/s and GFLOP/s against peak, naive and cuBLAS. RAG: Recall@k, hit@k, MRR/NDCG@10 (all queries); judge score 0-1 (correctness, grounding, completeness, temperature 0.1) on 28 practice queries at k = 1, 5, 20, paired, bootstrap 95% CIs (2000 resamples); prompt tokens, generation latency, context words. Transcripts cached, so RAG timings exclude ASR.

**Results.**
- ASR: all correct; 2699 ms vs PyTorch 1100 ms, **2.45x slower**, same node and job (K09); GEMM 86% of kernel time (K08); best config 17.2% faster than the handout defaults (K07).
- RAG retrieval: Recall@k rises 0.229 (k=1), 0.421 (5), 0.579 (20) on practice; dev agrees (E09, E11). hit@20 = 0.737: a quarter of queries have no relevant record in the top 20 (E14).
- RAG answers: judge score 0.250 / 0.268 / 0.292 at k = 1 / 5 / 20, all differences inside the noise (+-0.12), while prompt tokens grow 305 to 4730 and generation time 1.8 to 4.3 s (E15). **Quality is flat, cost is not.**
- Adaptive k: no gain over fixed k; even a perfect gate only +0.05 (n.s.) (E17).
- End to end at k = 5: about 5.4 s per spoken query with PyTorch ASR (generation 70%), 7.6 s with ours (half ASR, half generation) (E18).

**Why.** More records help queries whose rank-1 record is wrong (coverage) and hurt queries whose rank-1 record is right (dilution: 0.447 at k=1 vs 0.386 at k=20 for those), and the two cancel on average (E16). Queries with no relevant record in the top 20 score zero at every k (E13), which caps any depth policy. ASR side: our GEMM computes one output per thread from shared memory, cuBLAS uses register tiling and vector loads.

**Limitations.** n = 28 judged queries (wide CIs), one judge run, same model as generator and judge; judge cannot see whether ASR changed the question (no reference transcripts exist for the queries); RAG stages measured on different machines; ASR on one GPU type (2080 Ti); kernel tuning only over two knobs.

## 4. Anonymity checklist
No names, UUN, group ID, repo link or personal paths in the report, video or code zip; neutral file names (report.pdf, video.mp4, supplementary_code.zip); names and UUNs only on the final declaration page; AI use declared there.
