# Status

Last updated: 2026-11-18 (supersedes all earlier status text; history is in `docs/report/experiment-log.md`)

## Where we are
| # | Phase | State |
| --- | --- | --- |
| 0 | Access, environment, data (laptop and cluster) | DONE. RAG venv (CPU torch) on the laptop, course data installed, ELM key only in the laptop `rag/.env`; ASR conda env and model on the cluster. |
| 1 | ASR kernels, tasks 1 to 4 | DONE. All tests pass, sanitizers clean, exact transcript. Submitted constants: GEMM tile 32, 16 attention warps. |
| 2 | ASR experiments | DONE. Ablations (K02-K04), end to end (K05), profile (K06), combined best (K07), hand-in run (K08), same-job PyTorch comparison: ours 2.45x slower (K09). |
| 3 | RAG baseline and depth study | DONE. Retrieval (E07-E11, E14), answer quality vs k on practice n=28 (E15) and dev n=27 (E19), pooled n=55: flat in k; mechanism (E16). |
| 4 | Our RAG change | DONE. Adaptive k: negative result, held-out dev check (E17, E19). |
| 5 | End-to-end cost | DONE (derived, E18). |
| 6 | Report | COMPLETE: `submission/report/report.tex` + PDF (format checker PASS, 6-page body, audited 2026-11-18, see `docs/AUDIT.md`). Needs your read-through, the final page (group ID, names, UUNs) and your AI-declaration wording. |
| 7 | Video | RUNBOOK AND SLIDES COMPLETE: `docs/video/VIDEO_RUNBOOK.md`, `docs/video/slides.tex` (9:40, demo option A via `rag/extensions/demo_query.py`). Needs rehearsal and recording. |
| 8 | Hand-in package | `supplementary_code.zip` built and checked (28 files, our files only, no personal data); rebuild with `scripts/make_supplementary_zip.sh` after any code change. |
| 9 | Individual review | NOT STARTED (needs the assigned report). |

Course announcements of 2026-11-16 are applied: one question per ASR task (four answers in report Section 4.2), "Try this" optional, long outputs may go to the appendix, video questions known.

## Next steps (see `docs/TODO.md` for the full list with commands)
1. Owner: read and understand the four kernel answers and the report; rehearse the oral questions.
2. Owner: rehearse the demo (option A, decided), record the video, finish the report's last page, run the format checker.
3. Build the code zip, rename files, upload on Learn. Write the individual review when the report arrives.
4. Compare with the teammate's version. Everything in this repo is pushed to `rufiz` (2026-11-18).

## Open decisions
- Which figure, if any, to add to the report.
- Create `main` on GitHub (the default branch is still `claude/project-thread-ix3sq3`; work happens on `rufiz`).

## Decisions log
- 2026-11-08: no training; plan A is training-free. ELM key on the laptop only; the repo is public, so no data, keys or weights are committed. Frozen course files are never edited (our files live in `overlay/`; the guard checks).
- 2026-11-10: plans B and C set aside; one well-supported investigation (A) plus a tested negative result is enough. Kernel tuning stops at tile 32 + 16 warps.
- 2026-11-11: work only on branch `rufiz` of the team repo.
- 2026-11-16: dev replication run approved and done (168 ELM calls, E19); no further ELM calls planned.
- Timings: always state the node; quote PyTorch-vs-ours ratios only from one job (node effect, K07-K09).
