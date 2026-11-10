# Kernel ablations

`run_ablations.py` builds the five task files with different `-D` flags (one
`build/ablation/<variant>/` folder each), runs the course tests on every variant and
writes `ablation/results/<UTC time>/summary.md` plus `results.csv` and raw logs.
The default build (`build/sm_XX`) and every course file stay untouched.

Knobs (all default to the values the handout ships with):

| Flag | File | Default | Guide item |
| --- | --- | --- | --- |
| `ASR_ROPE_THREADS` | task1_rope.cu | 256 | Task 1 Try 1 |
| `--use_fast_math` | (nvcc) | off | Task 1 Try 3 |
| `ASR_REDUCE_THREADS` | task2_reduce.cu | 256 | Task 2 Try 1 |
| `ASR_ABL_RMS_ATOMIC` | task2_reduce.cu | off | Task 2 Try 4 |
| `ASR_ABL_SOFTMAX_3PASS` | task2_reduce.cu | off | Task 2 Question 3 |
| `ASR_ABL_NO_FINAL_SYNC` | task2_reduce.cuh | off | Task 2 Try 3 (race, expected to break) |
| `ASR_TILE` | task3_gemm.cu | 16 | Task 3 Try 1 |
| `ASR_WS_PAD` | task3_gemm.cu | 1 | Task 3 Try 2 |
| `ASR_ABL_W_STRIDED` | task3_gemm.cu | off | Task 3 Try 3 |
| `ASR_ABL_NO_SECOND_SYNC` | task3_gemm.cu | off | Task 3 Try 4 (race, expected to break) |
| `ASR_ATTN_WARPS` | task4_attention.cu | 4 | Task 4 Try 1 |
| `ASR_K_PAD` | task4_attention.cu | 1 | Task 4 Try 2 |

Stages: `check` (make test, compute-sanitizer memcheck and racecheck on the small cases,
`./run.sh`), `sweep` (every variant, 3 repeats, median), `e2e` (full model with
`--cuda none`, cumulative `task1..task3`, `all`, and the TILE / ATTN_WARPS variants;
profiles for none, task1-3 and all), `best` (full-model runs of tile 32 + 16 attention warps and its parts, plus repeats), `race` (opt-in).

Submit with `sbatch ablation/ablate.sbatch [--stage ...]`, or run inside a GPU shell with
`python ablation/run_ablations.py --stage ...`.
