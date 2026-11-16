# Notes for the report and the video (key answers to the key questions)

Written 2026-11-16 from `docs/EXPERIMENT_LOG.md` (entry numbers E#). Not the report. Numbers are from our runs; hardware RTX 2080 Ti unless stated. The owner writes and speaks the final text.

## A. One-paragraph story (the "chain of reasoning" the rubric asks for)
- **System.** Spoken question -> ASR (GLM-ASR-Nano, 2.3B parameters, our hand-written CUDA kernels for RoPE, norms/softmax, GEMM, attention) -> query encoder (enc_a) -> exact top-k retrieval over 24,000 records ->
  prompt -> LLM answer (ELM) -> judged answer.
- **Question (RAG, plan A).** How far can retrieval depth k be cut before answer quality drops, and what does it save in prompt size and time?
- **Baseline.** k = 5 (course default), exact dense retrieval, same cached transcripts, same generator and judge for every depth.
- **Change and controls.** k in {1, 5, 20} on 28 judged practice queries (paired, bootstrap intervals), retrieval metrics on 350 practice and 125 dev queries; adaptive k with a protocol fixed before looking at answers.
- **Results.** Recall and hit rate rise strongly with k (Recall@5 0.42, @20 0.58, @100 0.77; hit@k 0.59 at k=5, 0.74 at k=20; 12% of queries never reach a relevant record), retrieval time is flat (0.45 ms), but judged answer quality is flat
  (0.250, 0.268, 0.292, intervals about +/- 0.14) while prompt size grows about 4x per step (305, 1249, 4730 tokens) and generation time 1.8, 3.8, 4.3 s. [E10, E12, E18, E19]
- **Mechanism.** Coverage vs dilution: queries whose relevant record is at rank 2 to 20 gain from depth (0.21 -> 0.34), queries whose top record is already relevant lose slightly (0.45 -> 0.39); the judge scores against the gold evidence only and gives zero to 14 to 16 of 28 answers. [E20, E17b]
- **Tested idea, negative result.** Adaptive k (small k when the top-two score margin is large): AUC about 0.7, so it is a weak gate; at equal context a fixed k reaches higher recall (0.53 to 0.55 vs 0.48 to 0.50); the oracle gate gains only +0.05 on judged answers. [E24]
- **ASR side.** Four kernels pass all tests and sanitizers; GEMM is 86% of GPU time, so tuning concentrates there; submitted settings (tile 32, 16 attention warps) are 18% faster than the handout settings; on one node ours is 2.45x slower than PyTorch (cuBLAS-backed),
  and the ratio varies 2.2x to 3.4x across nodes because the PyTorch path depends on the node. [E13 to E16, E21 to E23]
- **Limits.** n = 28 judged queries, one judge repeat, one judge model; the judge cannot check that ASR kept the question; no reference text exists for the queries; two queries excluded because the judge returned malformed JSON; timings come from several nodes.

## B. The four ASR guide questions to answer (one per task; my suggestion, with reasons)
| Task | Suggested question | Why this one | Where the numbers are |
|---|---|---|---|
| 1 RoPE | Q1: bytes moved and share of peak bandwidth | ties directly to a measurement (80% of 616 GB/s) and shows the kernel is memory-bound | `docs/TASK_ANSWERS.md`, E21 |
| 2 Reductions | Q2: `expf(100)` overflow and why subtracting the max is exact | short, precise, tests numerical understanding (and the "large inputs" test case) | `docs/TASK_ANSWERS.md` |
| 3 GEMM | Q1 or Q2: loads naive vs tiled (32x fewer) or why cuBLAS is faster | pairs with the measured naive/tiled/cuBLAS table (230 / 2,100 / 13,300 GFLOP/s) | E21 |
| 4 Attention | Q1 (score matrix 180 MB vs 20 KiB) or Q4 (375 vs 94 key loads) | Q4 matches the measured warp sweep (103.6 ... 7.58 ms) | E14 |
Place them together in one labelled subsection ("Answers to the four selected kernel questions") and say where in the introduction; put test/profile excerpts in the main body and full outputs in an appendix.

## C. Video: the two required answers (about one minute each) — draft talking points
**1. Main design choice and why.** Pick one:
- *RAG:* "I transcribed all 475 query recordings once and cached the transcripts, so every retrieval and answer experiment starts from identical text. That makes the depth comparison controlled (only k changes), removes ASR cost and noise from
  the comparison, and lets me report ASR time separately." [E04, E06]
- *ASR:* "I used the per-operation profile to decide where to spend effort: the GEMM is 86% of GPU time, so I tuned it (tile 32, found shape-dependent: 35% faster on large encoder GEMMs but slower on the one-row `lm_head`) and left the memory-bound kernels,
  which are under 1% of the time (Amdahl's law)." [E14, E22]
**2. How I checked my work (with an example).**
- *Kernels:* tests against a CPU/cuBLAS reference on the model's real shapes, `compute-sanitizer` memcheck and racecheck (0 errors, 0 hazards), then the full model must reproduce the exact transcript. Example: the all-negative softmax row, which a block max that fills
  missing warps with 0 instead of -infinity would get wrong. [E13, TASK_ANSWERS]
- *Experiments:* consistency and repeat checks. The depth sweep's MRR@10 and NDCG@10 equalled the baseline's (0.489, 0.420), repeats agreed within 0.2 to 0.4%, and a PyTorch baseline measured in a different job (806 vs 1215 ms) exposed a node effect, so I only report same-job ratios.
  Example: when the first 15 answers suggested a quality gain from deeper retrieval, extending to 28 queries removed it, so I withdrew the claim. [E10, E16, E19, E23]

## D. Likely questions in the oral Q&A and the one-line answers
- *Why is retrieval time flat in k?* Exact search computes all 24,000 similarities regardless of k; only the top-k selection changes.
- *Why does recall rise but answer quality not?* More relevant records are found (coverage) but extra text dilutes answers and the judge scores against specific gold evidence; the net is flat.
- *Why is your GEMM 6x slower than cuBLAS?* Register tiling, vector loads, double buffering, per-architecture tuning (E21).
- *Why does padding matter?* Without +1, the 32 lanes read the same shared-memory bank (32-way conflict): 4.7x slower attention, 2.2x slower GEMM.
- *Why does the ratio to PyTorch change with the node?* Hypothesis: PyTorch launches thousands of small operations and depends on host CPU speed; our kernels are GPU-bound and stable across nodes (2698 to 2728 ms).
- *What would you do with more time?* Register-tile the GEMM; more queries and judge repeats; a second judge model; a reranker.
