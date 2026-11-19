# Every task of the coursework, how we solved it, and which files hold the work

Where files live: **source of truth** = `~/MLS/overlay/...` (our repo, branch `rufiz`); `scripts/sync_overlay.sh` copies them over the course checkout (`edin-mls-26-autumn/...`, same relative paths), on the laptop and on the cluster. Course files not listed here were NOT changed (a checksum guard, `tools/guard/`, verifies 55 frozen files on every sync). Entry ids (K = kernels, E = RAG) point into `docs/report/experiment-log.md`.

## Part 0. Setup
| # | Task | How we solved it | Files / commands |
| --- | --- | --- | --- |
| 0.1 | Get cluster access and the ASR environment | Cloned the course repo on the head node, ran `hw1-asr/setup_env.sh` (conda env `mls-asr`, PyTorch for CUDA 12.8, 4.5 GB model); one GPU per session via Slurm | course file `hw1-asr/setup_env.sh` (unchanged) |
| 0.2 | Get an ELM key, install the RAG environment and course data | Own key in a git-ignored `rag/.env` (laptop only, mode 600); venv with CPU PyTorch + `rag/requirements.txt`; data from the LFS archive `rag-assets.zip` via `rag/scripts/install_assets.py`; `check_setup.py` all PASS | course scripts, unchanged; `scripts/setup_rag_local.sh` (ours, helper) |

## Part A. ASR: four CUDA kernels (`hw1-asr/csrc/`, five files edited)
| # | Task | What it is | How we solved it | File (lines) |
| --- | --- | --- | --- | --- |
| 1 | RoPE | Rotate query and key vectors by an angle that depends on token position (positions encoded without extra inputs) | One launch for q and k; one thread per (row, pair): blank 1.1 global index, 1.2 switch to k after q's pairs, 1.3 derive row, pair index and position, 1.4 `sincosf` then rotate `(v[i], v[i+half])` in place. Memory-bound; 256 threads/block | `task1_rope.cu` (92) |
| 2 | Reductions, RMSNorm, softmax | Sum/max across threads, then normalisation layers and softmax built on them | 2.1/2.2 warp sum/max with register shuffles (butterfly `__shfl_xor_sync`); 2.3 block max through shared memory with `-INF` for missing warps (0 would break all-negative rows); 2.4 online (max, sum) merge; 2.5 RMSNorm (shuffle reduction); 2.6 softmax pass 1 reads the row once with the online merge; 2.7 pass 2 writes `exp(x-max)/sum`. One block per row, 256 threads | `task2_reduce.cuh` (99), `task2_reduce.cu` (122) |
| 3 | Tiled GEMM | The linear layer `y = x W^T + b` (86% of GPU time), with optional GELU | 3.1 load x tile (0 outside); 3.2 load W tile coalesced into a padded shared array `[TILE][TILE+1]` (padding avoids bank conflicts); 3.3 accumulate over the tile; 3.4 epilogue: add bias, GELU, store. Submitted tile 32 (handout 16) | `task3_gemm.cu` (97) |
| 4 | Attention | Softmax attention with grouped-query heads and a causal mask, without storing the 1500x1500 score matrix | 4.1 GQA map query head to key head; 4.2 stage a 32-key tile of K and V in shared memory (K row padded by 1); 4.3 each lane scores one key; 4.4 online-softmax update (running max `m`, sum `l`, rescale `corr`) with warp reductions; 4.5 accumulate `p * V` using shuffles; 4.6 write `acc / l`. One warp per query; submitted 16 warps per block (handout 4) | `task4_attention.cu` (190) |
| 5 | Prove correctness | Every `make test` case PASS, transcription PASS | Course tests against CPU/cuBLAS on model shapes, `compute-sanitizer` memcheck + racecheck (0 errors, 0 hazards), full-model transcript exact (K01, K08) | no code changed; run via `hw1-asr/ablation/run_ablations.py --stage check` |
| 6 | Required comparisons | kernels vs PyTorch; GEMM vs naive vs cuBLAS; GB/s as share of peak; (optional "Try this") | `./run.sh --profile` vs `--cuda none --profile` in one job: 2698.8 ms vs 1096-1104 ms (2.45x slower; K09); `make test3` prints naive/ours/cuBLAS (K08); GB/s from tests 1 and 2; ablations by recompiling with `-D` flags (`ASR_TILE`, `ASR_ATTN_WARPS`, `ASR_WS_PAD`, ...): tile 32 -13%, 16 warps -4%, together -17.2% (K02-K07) | `hw1-asr/ablation/` (`run_ablations.py`, `ablate.sbatch`, `handin.sbatch`, `compare_runs.py`, `README.md`); the `#ifdef ASR_*` knobs sit inside the five task files |
| 7 | Hand-in outputs | five files + `make test` + `./run.sh --profile` + `nvidia-smi -L` from one node | One Slurm job (`handin.sbatch`) on the RTX 2080 Ti; outputs saved | `docs/outputs/handin_make_test_and_profile.txt`, `pytorch_vs_ours_same_job.txt`; in the zip as `outputs/` |
| 8 | Answer one guide question per task | Explanation questions at the end of each task guide | Chosen: T1 Q1 (bytes, 80% of peak), T2 Q3 (softmax reads x twice vs three times), T3 Q1 (32x fewer loads), T4 Q4 (375 vs 94 key loads); all 16 drafted | report Section 4.2 (`submission/report/report.tex`); drafts `docs/laptop/TASK_ANSWERS.md` |

## Part B. RAG investigation (`rag/extensions/`, all new files; `rag/src`, `rag/scripts` unchanged)
| # | Task | How we solved it | File |
| --- | --- | --- | --- |
| 9 | Understand and run the baseline pipeline (audio, ASR, encoder, exact retrieval, prompt, LLM answer, judge) | `check_setup.py`; baseline retrieval with the course `evaluate_rag.py` (practice 350, dev 125 queries); answers and judging through our runner using the course judge. `run_audio_query.py` was not run as is: the ASR step is replaced by our two-stage demo (row 15) | course scripts (unchanged) |
| 10 | Choose a precise question | "How much retrieved evidence (depth k) should the generator get, and what does extra depth cost?" (the slides' own example direction) | `docs/PLAN.md` |
| 11 | Hold inputs fixed | Transcribed all 475 query recordings once on the GPU and cached the text; every later experiment reuses it (and ASR time is reported separately) | `transcribe_split.py` |
| 12 | Measure retrieval quality and cost vs k | Recall@k, hit@k, MRR@10, NDCG@10, retrieval latency, context size for k=1..100 with bootstrap intervals; retrieval time flat (0.45 ms), context grows linearly | `retrieval_sweep.py`; `docs/laptop/analysis/hit_at_k.py` |
| 13 | Measure answer quality vs cost | Judge score on the same queries at k = 1, 5, 20 (practice 28, dev 27 held-out), prompt tokens, generation latency, paired bootstrap; result flat in k (pooled n=55: 0.267/0.294/0.273) | `generate_from_transcripts.py`, `run_answer_quality.sh`, `answer_quality_summary.py` |
| 14 | A change of our own | Adaptive k (k=1 if the top-two score margin is large, else 20), tuned on practice labels only, tested on held-out dev, replayed on judged answers; negative result (weak gate, low oracle ceiling) | `adaptive_k.py`; `docs/laptop/analysis/fixed_k_matched_context.py` |
| 15 | End-to-end evidence and demo | Cost per spoken question (ASR vs retrieval vs generation; derived, E18); two-stage demo: real ASR on the cluster GPU, then retrieval and one ELM answer on the laptop | `demo_query.py` |

## Part C. Deliverables
| # | Task | State | Where |
| --- | --- | --- | --- |
| 16 | Group report (6-page body, supplied template, AI declaration on the final page) | Draft complete; placeholders on the last page | `submission/report/report.tex`, `references.bib`, template style files (unchanged), `report.pdf` |
| 17 | Video (<= 10 min, demo, last 2 min = two questions) | Runbook and slides ready, not recorded | `docs/video/VIDEO_RUNBOOK.md` |
| 18 | Supplementary code (optional) | Zip builder ready (our files only, no history, personal-data check) | `scripts/make_supplementary_zip.sh`, `submission/README.md` |
| 19 | Individual review of another group's report (30%) | Not started (needs the assigned report) | Learn |
| 20 | Anonymity and hand-in | Neutral names `report.pdf`, `video.mp4`, `supplementary_code.zip`; names/UUN/Group ID only on the report's last page | see `docs/AUDIT.md` |
