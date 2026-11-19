# Supplementary code

Only files we wrote or changed are included. Everything else (model code, tests, `ops.py`, `model.py`,
`benchmark.py`, `provided.cu`, the RAG framework in `rag/src` and `rag/scripts`, data and weights) is the
supplied course code, unchanged. Copy each folder below over the same path in the course repository.

## ASR (`hw1-asr/`)
| file | what |
| --- | --- |
| `csrc/task1_rope.cu` | Task 1: RoPE, q and k rotated in one launch |
| `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu` | Task 2: warp/block sum and max, online (max, sum) merge; RMSNorm, LayerNorm, one-read softmax |
| `csrc/task3_gemm.cu` | Task 3: shared-memory tiled GEMM, padded W tile, fused bias + GELU epilogue |
| `csrc/task4_attention.cu` | Task 4: tiled attention with online softmax, GQA, causal mask, one warp per query |
| `ablation/` | experiment harness: builds variants with `-D` flags, runs tests and the full model, writes summary tables (`ablation/README.md` lists the knobs) |

Submitted settings: `ASR_TILE 32`, `ASR_ATTN_WARPS 16` (handout values 16 and 4), `ASR_ROPE_THREADS 256`, `ASR_REDUCE_THREADS 256`.

```bash
cd hw1-asr
make test && ./run.sh --profile          # on a GPU node (CUDA 12.8, the course conda env)
sbatch ablation/handin.sbatch            # same, as one Slurm job, plus nvidia-smi -L
sbatch ablation/ablate.sbatch --stage sweep   # kernel ablations; stages: check sweep e2e best
```

## RAG (`rag/extensions/`), run from the repository root
| file | what |
| --- | --- |
| `transcribe_split.py` | run ASR once per query (GPU) and cache transcripts |
| `retrieval_sweep.py` | Recall@k, hit@k, MRR/NDCG@10, retrieval latency, context size for k = 1..100 (no API calls) |
| `generate_from_transcripts.py` | answers from cached transcripts at a given k; records prompt/completion tokens; resumable |
| `run_answer_quality.sh` | generate + judge for k in {1, 5, 20} on the first N queries |
| `answer_quality_summary.py` | paired comparison across k with bootstrap 95% CIs; `--split practice dev --bootstrap 5000 --csv` gives the pooled n = 55 numbers of the report and a per-query CSV |
| `adaptive_k.py` | adaptive depth (k = 1 when the top-2 score margin is large, else 20): tuned on practice, checked on dev, replayed on judged answers (no API calls) |
| `rank_groups.py` | post hoc analyses of the judged answers from the per-query CSV: score differences by the rank of the first relevant record, zero scores with a relevant record in the prompt, power, and replays of the gate and oracle with k = 20 or k = 5 as the high arm (no API calls) |
| `judge_retest_summary.py` | agreement between the original judge scores and a second judge pass on 56 stored answers (`outputs/rag/judge_retest.csv`; no API calls) |
| `demo_query.py` | two-stage demo of one spoken query: `asr` on the GPU with our kernels, then `answer` on a machine with the ELM key (retrieval, one generation call, per-stage times) |

```bash
python rag/extensions/transcribe_split.py --split practice        # GPU
python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl
EXCLUDE="q_d442a6395dbca951 q_4e29b0ce6fa96d1a" bash rag/extensions/run_answer_quality.sh 30 practice   # needs the ELM key in rag/.env
EXCLUDE="q_7f3179f293749f08" bash rag/extensions/run_answer_quality.sh 28 dev
python rag/extensions/rank_groups.py --csv outputs/rag/aq_pooled_practice_dev_summary.csv --adaptive outputs/rag/adaptive_k.json
python rag/extensions/adaptive_k.py
```
Dependencies: the course environments (`hw1-asr/setup_env.sh`, `rag/requirements.txt`); numpy only beyond them.

## Outputs and log
- `outputs/handin_make_test_and_profile.txt`: `make test` and `./run.sh --profile` from one job on one node (RTX 2080 Ti).
- `outputs/pytorch_vs_ours_same_job.txt`: PyTorch-only vs our kernels in one job.
- `outputs/rag/`: per-query judge scores at k = 1, 5, 20 for the 55 judged queries (`aq_pooled_practice_dev_summary.csv`, no answer or record text), the pooled summary, and the adaptive-k result (`adaptive_k.json`). With them every judged-answer and adaptive-policy number in the report can be recomputed without ELM calls (`answer_quality_summary.py`, `rank_groups.py`, `adaptive_k.py`). Retrieval metrics need the course assets and the cached transcripts (`retrieval_sweep.py`).
- `EXPERIMENT_LOG.md`: every experiment, command, number and interpretation (entries E01-E22, K01-K09).
