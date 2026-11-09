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
- Everything else PENDING.

## 6. Limitations (about 0.5 page)
PENDING. Known already: single baseline node so far (E03); judge is an automated LLM score; the public judge scores the ASR transcript, which cannot detect ASR errors.

## Evidence index
E01 pinned code/data, E02 guard, E03 ASR baseline, E04 transcripts (+verification), E05 RAG setup, E06 transcripts copied, E07 baseline retrieval (practice), E08 baseline retrieval (dev) (see experiment-log.md).
