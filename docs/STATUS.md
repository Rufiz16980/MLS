# Status

Last updated: 2026-11-08

## Phases

| # | Phase | State |
| --- | --- | --- |
| 0 | Access and environment (cluster, conda env, RAG venv, ELM key, assets) | In progress: cluster access verified (RTX 2080 Ti); ELM key requested; env setup pending |
| 1 | ASR kernels: Task 1 RoPE, 2 reductions, 3 tiled GEMM, 4 attention | Not started |
| 2 | ASR experiments ("Try this" per task, comparisons vs PyTorch/cuBLAS) | Not started |
| 3 | RAG baseline (retrieval metrics, answers, judge on a subset) | Not started |
| 4 | Systems investigation (Plan A, with B and C as backups) | Plan chosen: A (see PLAN.md); not started |
| 5 | Report (6-page main body) | Not started |
| 6 | Video (max 10 min, last 2 min for the released questions) | Not started |
| 7 | Packaging, anonymity check, individual reviews | Not started |

## Immediate tasks

- [ ] Each member: get GPU access working (setup/gpu-access.md)
- [ ] Each member: request your own ELM API key (setup/elm-api-key.md); never share keys
- [ ] Build the frozen-file guard (checksum manifest + wrapper + pre-commit + CI)
- [ ] Decide repo layout for the course code (pristine copy plus our overlay files)
- [ ] Agree who does which ASR task (suggested split: Tasks 1-2 / Tasks 3-4)
- [ ] Look up deadlines, upload routes and group size on Learn

## Decisions log

- 2026-11-08: Training a model is not required by the coursework. Plan A is training-free; training is optional and only if it helps the investigation.
