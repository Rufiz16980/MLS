# Plan summary

Full details: https://claude.ai/artifact/PDHfaidEkNQ2chxR6afhBi (INFR11269, 2026/27).

## What is assessed
- Group report (70%): 6-page main body in the supplied template + a video of at most 10 minutes (last 2 minutes answer two released questions). Marked out of 100: technical work 30, experimental evidence 30, analysis 30, communication 10. Beating the baseline is not a criterion.
- Individual review of another group's report (30%).
- ASR (hw1-asr CUDA kernels) and RAG (one systems investigation) are assessed together.

## Frozen (do not edit; guard them from outside)
hw1-asr: `glm_asr_cuda/ops.py`, `model.py`, `benchmark.py`, `run.sh`, `tests/`, `csrc/provided.cu`, `common.cuh`, `asr_kernels.h`, and every given function name/signature (incl. `launch_*`).
Editable: only the blanks in `csrc/task1_rope.cu`, `task2_reduce.cuh`, `task2_reduce.cu`, `task3_gemm.cu`, `task4_attention.cu`. No cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch inside a task.
RAG: keep supplied scripts and the judge unchanged; add new code in `extensions/` or beside the baseline.

## Research plans
- **A (lead, training-free):** how far can retrieval depth k and an evidence budget shrink prompts and latency, under ASR errors, before answer quality drops?
- **B:** representation migration: map enc_a queries into the enc_b corpus space (small learned map), compare to the baseline and re-encoding cost.
- **C:** do ASR errors hurt retrieval, and does a query adapter or ASR setting recover it?
We prototype cheaply and keep the most convincing one for the report.

## Metrics
Kernel tests all PASS; transcription PASS; PyTorch baseline on a 2080 Ti is 992.9 ms per transcription, 76.4 ms per token, 8.88 GiB. RAG: Recall@5/10/20/100, MRR@10, NDCG@10, judge correctness/grounding/completeness (0-4, combined /12), per-stage latency and context size.
