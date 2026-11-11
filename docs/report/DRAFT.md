# Report draft (skeleton, 6-page main body)

Working title: to be chosen (anonymous).
Rubric weights: technical work 30, experimental evidence 30, analysis 30, communication 10. A well-explained negative result scores well; improvement over baseline is not a criterion.

## 1. Context and question (about 0.5 page)
- System: audio query to ASR (our CUDA kernels) to local encoder to exact retrieval to context to ELM answer.
- Question (Plan A, to be confirmed): how far can retrieval depth k and an evidence budget cut prompt size and latency under ASR errors before answer quality drops? Backups: B (enc_a to enc_b map), C (ASR-error query adapter).
- Why it matters: prompt size drives generation cost; ASR errors change the query that retrieval sees.

## 2. Baseline and what is supplied vs ours (about 1 page)
- Supplied: model, Python layer, tests, benchmark, RAG pipeline (frozen, verified by checksum, E01/E02).
- Ours: four CUDA kernels (RoPE, RMSNorm/softmax reductions, tiled GEMM with fused GELU, FlashAttention-style attention) and the RAG investigation.
- ASR baseline on PyTorch: 1215.4 ms per transcription, 93.5 ms per token, 8.88 GiB, PASS (E03).

## 3. Technical work (about 1.5 pages)
- Kernel designs and why, with the per-task "try this" experiments: PENDING (no kernels written yet).
- RAG change and design choices: PENDING.

## 4. Evaluation design (about 1 page)
- Fixed inputs: cached PyTorch-ASR transcripts, 350 practice (tuning) and 125 dev (final) queries (E04).
- Hardware: RTX 2080 Ti, one node type for every comparison; timings state what they include (model load excluded).
- Quality: Recall@5/10/20/100, MRR@10, NDCG@10; judge correctness/grounding/completeness (0-4), combined /12, judge settings and repeats reported.
- Cost: per-stage latency, prompt size, ELM call count.

## 5. Results and analysis (about 1.5 pages)
- Baseline retrieval (practice, k up to 100, exact search): Recall@5/10/20/100 = 0.421/0.480/0.579/0.770, MRR@10 0.489, NDCG@10 0.420 (E07); dev (125 queries): 0.379/0.475/0.580/0.724, MRR@10 0.460, NDCG@10 0.398 (E08), same shape with a few points of split-to-split noise. Diminishing returns in k and a hard ceiling at Recall@100 (0.770) for any reranking or budgeting method.
- Retrieval-depth sweep (practice, E09): retrieval time is flat at about 0.45 ms for k=1..100 (exact search), while context grows from 175 words (k=1) to about 18,000 words (k=100). Recall@1/5/20/50/100 = 0.229/0.421/0.579/0.684/0.770 (CIs about ±4 points). The cost of depth is answer-model input, not retrieval.
- Smoke test (E10, 5 queries, k=5): the path works end to end; generation is 2.3-6.2 s, over 95% of end-to-end time. Not a result at n=5.
- Everything else PENDING.

## 6. Limitations (about 0.5 page)
PENDING. Known already: single baseline node so far (E03); judge is an automated LLM score; the public judge scores the ASR transcript, which cannot detect ASR errors.

## Evidence index
E01 pinned code/data, E02 guard, E03 ASR baseline, E04 transcripts (+verification), E05 RAG setup, E06 transcripts copied, E07 baseline retrieval (practice), E08 baseline retrieval (dev), E09 retrieval-depth sweep (practice), E10 five-query smoke test, E11 retrieval-depth sweep (dev), K01 kernel correctness, K02-K04 kernel ablations (Tasks 1-4), K05 end-to-end timing, K06 per-operation profile, K07 combined best configuration, E12 answer quality vs depth, E13 zero-score queries, E14 hit@k, E15 answer quality at n = 28, E16 coverage vs dilution (see experiment-log.md).

## Kernel results so far (ASR part; RTX 2080 Ti; all numbers within one job, see K01-K05)

- All four kernels pass the course tests, compute-sanitizer memcheck/racecheck (0 errors, 0 hazards) and the full-model transcription (K01).
- End to end, our RoPE and reductions are faster than PyTorch (0.87x), our tiled GEMM is the bottleneck (3.72x with GEMM, 4.08x with attention too) (K05).
- Best single tuning: GEMM tile 32 (-13% end to end); tile 8 is +52%. lm_head (M = 1) prefers tile 8, so no fixed tile is best for all shapes (K03, K05).
- Shared-memory padding is essential: no padding is 1.7-2.3x slower in GEMM and 4.7x slower in attention, from bank conflicts (K03, K04).
- Attention gains from more queries per block (fewer key loads) in the micro-benchmark (13.7x from 1 to 16 warps), but only 3% end to end at 8 warps (K04, K05).
- Reductions: best block size depends on shape; one-read softmax and shuffle reductions beat the textbook and atomic versions (K02).
- Profile (K06): with our kernels GEMM is 83.6% and attention 15.9% of kernel time; RoPE, norms and softmax together 0.5%. PyTorch's own RoPE (25%) and RMSNorm are slow chains of small ops, hence our kernels beat them.
- Combined best (K07): GEMM tile 32 + 16 attention warps is 17.2% faster than the default end to end (2728.1 vs 3294.0 ms, 100% accuracy); the effects are additive. Repeat spread of the default across jobs and nodes: 0.3%. The ratio to PyTorch depends on the node (4.1x on damnii08, about 2.7x for the default on opencast) because PyTorch's own time is host-bound.

- Answer quality vs depth (E12, 15 practice queries, 90 ELM calls): judge score 0.178 / 0.317 / 0.361 at k = 1 / 5 / 20; the step 1 to 5 is +0.139 (95% CI -0.006 to +0.311), 5 to 20 only +0.044, while context grows 4.1x and generation time 1.5x. Tentative (n = 15), to be extended to 30 queries.
- Coverage (E14, reported, re-run pending): hit@k = 0.41 / 0.59 / 0.74 / 0.88 at k = 1 / 5 / 20 / 100 on practice (dev 0.38 / 0.58 / 0.77 / 0.88); about 12% of queries have no relevant record in the top 100, a ceiling for any choice of k. Two of three zero-score queries inspected (E13) were retrieval misses.
- Answer quality vs depth, final (E15, n = 28 paired): judge score 0.250 / 0.268 / 0.292 at k = 1 / 5 / 20, differences not significant, while prompt tokens grow 305 -> 4730 and generation time 1.8 -> 4.3 s. The E12 trend did not replicate. Plan A result: under this judge, small k matches large k at a fraction of the cost.
- Mechanism (E16, small groups): rank-1 queries lose a little with more records (dilution, 0.447 -> 0.386), lower-ranked ones gain (coverage); misses stay at 0. Follow-up: reranking or adaptive k.
