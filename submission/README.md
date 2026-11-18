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
| `answer_quality_summary.py` | paired comparison across k with bootstrap 95% CIs |
| `adaptive_k.py` | adaptive depth (k = 1 when the top-2 score margin is large, else 20): tuned on practice, checked on dev, replayed on judged answers (no API calls) |
| `demo_query.py` | two-stage demo of one spoken query: `asr` on the GPU with our kernels, then `answer` on a machine with the ELM key (retrieval, one generation call, per-stage times) |

```bash
python rag/extensions/transcribe_split.py --split practice        # GPU
python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl
bash rag/extensions/run_answer_quality.sh 30 practice                # needs the ELM key in rag/.env
python rag/extensions/adaptive_k.py
```
Dependencies: the course environments (`hw1-asr/setup_env.sh`, `rag/requirements.txt`); numpy only beyond them.

## Outputs and log
- `outputs/handin_make_test_and_profile.txt`: `make test` and `./run.sh --profile` from one job on one node (RTX 2080 Ti).
- `outputs/pytorch_vs_ours_same_job.txt`: PyTorch-only vs our kernels in one job.
- `EXPERIMENT_LOG.md`: every experiment, command, number and interpretation (entries E01-E19, K01-K09).
