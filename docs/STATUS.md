# Status

Last updated: 2026-11-09

## Verified tonight (cluster, user `<UUN>` on the head node)

- Course repo cloned to `~/edin-mls-26-autumn` at `6bbdfbc` (= the commit our guard pins).
- `rag-assets.zip` (196,521,100 bytes) fetched, sha256 OK, `unzip -tq` no errors; extracted to `~/mls-rag-assets` (492 files, 231 MB apparent). `install_assets.py` NOT yet run on the cluster.
- Files are `chmod go-rwx`; no real `.env` and no key on the cluster. Cluster shell history has no key.
- `setup_env.sh` was started with nohup (PID 827793) and was at the Miniconda stage (home 1.3 GB). **NOT verified after logout**: the re-check ssh was run from the laptop straight to `mlp` and timed out (mlp is only reachable through the gateway: `ssh <UUN>@ssh.inf.ed.ac.uk`, then `ssh mlp.inf.ed.ac.uk`). First thing next session: `pgrep -u $USER -fa setup_env.sh; tail -n 5 ~/setup_env.log; du -sh ~`. If it died, re-run `cd ~/edin-mls-26-autumn/hw1-asr && (yes | nohup bash setup_env.sh > ~/setup_env.log 2>&1 &)` (safe to repeat).
- Also check `stat -c '%a' ~` is 700 or 750; `chmod 700 ~` if not.

## Where we are

| # | Phase | State |
| --- | --- | --- |
| 0 | Access and environment | Laptop done: RAG venv (CPU torch, ~/mls-rag-env), course data installed, ELM key in rag/.env (laptop only), all setup checks PASS. Cluster: access verified (RTX 2080 Ti); course clone + `setup_env.sh` (~20 GB, run in background) started tonight. Check `~/setup_env.log` ends with "Setup finished". |
| 0b | Guard + CI | Built: `tools/guard/` (guard.py, frozen.json from course commit 6bbdfbc), `scripts/sync_overlay.sh`, pre-commit hook, `.github/workflows/ci.yml`. Not yet enabled as a required check. |
| 1 | ASR kernels Tasks 1-4 | Implemented in `overlay/hw1-asr/csrc/`; pass all course tests and sanitizers on a 2080 Ti (K01). Ablations done (K02-K04); GEMM is the bottleneck (K05). Still to do: profiles, combined best config, cuBLAS comparison, report questions. |
| 2 | ASR experiments | Sweep and end-to-end runs done (K02-K05); per-operation profiles and a repeat run pending |
| 3 | RAG baseline | Retrieval baseline done (E07). Next: 5-query generation + judge with the ELM key, then the full practice split. |
| 4 | Investigation (Plan A, training-free; B and C backups) | Chosen: A. See PLAN.md |
| 5-7 | Report, video, packaging | Not started |

## Done since the first status

- 475 transcripts made once on the GPU with PyTorch ASR (E04), copied to the laptop (E06). ASR baseline 1215.4 ms (E03).
- Retrieval baseline on practice: Recall@5/10/20/100 0.421/0.480/0.579/0.770, MRR@10 0.489, NDCG@10 0.420 (E07). Dev baseline: 0.379/0.475/0.580/0.724, MRR@10 0.460, NDCG@10 0.398 (E08). Retrieval-depth sweep (E09): retrieval time flat at ~0.45 ms for all k; context grows ~180 words per record (17,957 words at k=100); Recall@100 = 0.770. Five-query smoke test (E10): path works, generation is over 95% of end-to-end time. Dev sweep (E11) confirms the shape: Recall@1/5/20/100 = 0.209/0.379/0.580/0.724, context 183-18,101 words, retrieval ~0.5 ms. Full detail in docs/report/experiment-log.md.

## Next steps (current, supersedes the numbered list below where they differ)

1. Done: 5-query smoke test (E10), practice and dev sweeps (E09, E11).
2. Laptop: answer-quality experiment (ELM budget needs the owner's approval): fixed subset of practice queries x k in {1, 5, 20}, generate + judge. Inspect the low-scoring smoke-test queries first.
3. GPU (needs a new session): Task 1 RoPE kernel, then Tasks 2-4. Repeat the ASR baseline on a second allocation.
4. Report: keep `docs/report/DRAFT.md` and the experiment log updated after every run.

## Next steps (original list)

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
