# Status

Last updated: 2026-11-08 (night, end of day 1)

## Where we are

| # | Phase | State |
| --- | --- | --- |
| 0 | Access and environment | Laptop done: RAG venv (CPU torch, ~/mls-rag-env), course data installed, ELM key in rag/.env (laptop only), all setup checks PASS. Cluster: access verified (RTX 2080 Ti); course clone + `setup_env.sh` (~20 GB, run in background) started tonight. Check `~/setup_env.log` ends with "Setup finished". |
| 0b | Guard + CI | Built: `tools/guard/` (guard.py, frozen.json from course commit 6bbdfbc), `scripts/sync_overlay.sh`, pre-commit hook, `.github/workflows/ci.yml`. Not yet enabled as a required check. |
| 1 | ASR kernels Tasks 1-4 | Not started. Pristine task files are in `overlay/hw1-asr/csrc/`. |
| 2 | ASR experiments | Not started |
| 3 | RAG baseline | Not started. Needs cached transcripts (see next steps). |
| 4 | Investigation (Plan A, training-free; B and C backups) | Chosen: A. See PLAN.md |
| 5-7 | Report, video, packaging | Not started |

## Next steps (in order)

1. **Cluster, once setup finished:** `conda env list` shows mls-asr; model in `~/.cache/huggingface`. Then `cd ~/edin-mls-26-autumn && python3 ~/MLS/tools/guard/guard.py check --course .` (clone our repo on the cluster too: `git clone https://github.com/Rufiz16980/MLS.git ~/MLS`).
2. `python rag/scripts/install_assets.py --from-dir ~/mls-rag-assets` using the conda env python (numpy). Do NOT put the ELM key on the cluster.
3. GPU session, then `cd hw1-asr && make && ./run.sh --cuda none`: expect ~993 ms, PASS (reference baseline, 2080 Ti). Save the output and `nvidia-smi -L`.
4. Transcribe all 475 queries once with PyTorch ASR into `rag/.cache/{practice,dev}_transcripts.jsonl` (use `rag/scripts/evaluate_rag.py --transcript-cache ... --split practice|dev` with `ASR_CUDA=none` on the GPU), copy the caches to the laptop.
5. Laptop: `python rag/extensions/generate_from_transcripts.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/baseline_k5.jsonl --limit 5` (after `bash scripts/sync_overlay.sh <course dir>`), then judge dry-run, then the real judge on 5 queries.
6. Kernels: implement Task 1 to 4 in `overlay/hw1-asr/csrc/` (or edit in the cluster clone and run `scripts/pull_overlay.sh`), always test with `python3 tools/guard/guard.py run --course <course dir> -- make testN`.

## Open items

- [ ] Create `main` (repo default branch is currently `claude/project-thread-ix3sq3`)
- [ ] Agree who does which ASR task (suggested: Tasks 1-2 / Tasks 3-4) and the RAG study
- [ ] Look up deadlines, upload route and group size on Learn
- [ ] Every cluster/laptop result from the teammate is double-checked by the other side before it goes in the report

## Decisions log

- 2026-11-08: no model training is required. Plan A is training-free; training only if it clearly helps.
- 2026-11-08: ELM key lives on the laptop only; the repo is public, so no data, keys or weights are committed.
- 2026-11-08: frozen course files are never edited. Our files live in `overlay/`; the guard verifies the course checkout.
- Note: the cloud copy of `hw1-asr/test_audio.wav` differed from the git blob at 6bbdfbc (size 117970 vs 112204). Verify on the cluster clone; the guard will flag it if real.
