# Project documentation

Shared notes for the team. Keep this folder up to date; it is the single place for plans and status.

| File | What it is |
| --- | --- |
| [STATUS.md](STATUS.md) | Current status, phases, immediate tasks, owners |
| [PLAN.md](PLAN.md) | Plan summary: requirements, frozen files, research plans A/B/C |
| [setup/gpu-access.md](setup/gpu-access.md) | How to reach the Informatics GPU cluster (each person does this) |
| [setup/data-install.md](setup/data-install.md) | Install course data and the RAG venv (one script) |
| [TODO.md](TODO.md) | Everything left to finish, with commands (report, video, hand-in, GitHub) |
| [../submission/report/report.tex](../submission/report/report.tex) | **The report** (course template; PDF next to it; format checker PASS, 6-page body) |
| [video/VIDEO_RUNBOOK.md](video/VIDEO_RUNBOOK.md) | **The video runbook**: step by step, what is on screen, the command, the expected output and the words to say (9:40) |
| [video/slides.tex](video/slides.tex) | The 13 video slides (build with the course slides saved as `course.pdf` next to it) |
| [report/report-video-notes.md](report/report-video-notes.md) | Key answers: the four task questions, the two video questions, report summary |
| [report/latex/report.tex](report/latex/report.tex) | Alternative report draft written in parallel on the laptop (cross-check only) |
| [laptop/](laptop/) | Laptop-side docs: task answers, notes, requirements checklist, analysis scripts |
| [report/README.md](report/README.md) | Report draft and the experiment log (every result with its command) |
| [setup/cluster-workflow.md](setup/cluster-workflow.md) | What persists on the cluster, per-session routine |
| [setup/elm-api-key.md](setup/elm-api-key.md) | How to request and handle your own ELM API key |
| [reference/elm-terms-and-key-agreement.md](reference/elm-terms-and-key-agreement.md) | Saved copy of the ELM terms and key-request agreement |

Full planning document (Claude Doc): https://claude.ai/artifact/PDHfaidEkNQ2chxR6afhBi

**Anonymity:** this `docs/` folder holds personal and planning material. It must NOT go into
`supplementary_code.zip` or the submitted report/video. Never put names, UUNs, emails or keys in the submission.

## Repo layout

| Path | Purpose |
| --- | --- |
| `overlay/` | OUR files only (task kernels, RAG extensions), copied into a course checkout by `scripts/sync_overlay.sh` |
| `tools/guard/` | Frozen-file guard (`guard.py`, `frozen.json`, pre-commit hook). Install the hook: `ln -sf ../../tools/guard/pre-commit .git/hooks/pre-commit` |
| `scripts/` | `setup_rag_local.sh`, `sync_overlay.sh`, `pull_overlay.sh`, `make_supplementary_zip.sh` |
| `submission/` | Hand-in material: `report/` (LaTeX + PDF), `README.md` for the code zip |
| `.github/workflows/ci.yml` | CI: guard on a pristine course clone, tamper test, secret/data hygiene |

Guard usage: `python3 tools/guard/guard.py run --course <course dir> -- make test1` (checks before and after the command).
