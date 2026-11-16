# To do: everything needed to finish (complete list, 2026-11-16)

Legend: [x] done, [ ] open. "You" = the owner. Commands run from the course checkout (`<course-root>`) unless stated.

## 0. What is finished (do not redo)
- [x] ASR: four kernels correct (tests, sanitizers, exact transcript), ablations, same-job comparison with PyTorch (2.45x slower), profile, GEMM naive/ours/cuBLAS, hand-in outputs (`docs/outputs/`).
- [x] RAG: baseline retrieval (practice 350, dev 125), depth sweep, answer quality vs k (practice n=28, dev n=27, pooled n=55), mechanism, adaptive k (negative, held-out checked), end-to-end cost.
- [x] Drafts: report (`docs/report/latex/report.tex`, compiles, main body ends on page 5), video script (`docs/video/VIDEO_SCRIPT.md`), notes (`docs/report/report-video-notes.md`), 16 guide answers (`docs/laptop/TASK_ANSWERS.md`).
- [x] ELM budget used: about 360 calls. No more calls are needed.

## 1. Understand everything (your job; do first, it protects the grade)
- [ ] Read `docs/laptop/TASK_ANSWERS.md` and explain each of the four chosen answers aloud without notes (T1 Q1, T2 Q3, T3 Q1, T4 Q4), then the other 12.
- [ ] Read the report draft once; for every number find its entry in `docs/report/experiment-log.md`.
- [ ] Rehearse the oral questions in `docs/report/report-video-notes.md` section D / `docs/laptop/NOTES_REPORT_VIDEO.md` (one-line answers).
- [ ] Be able to explain: online softmax (m, l, corr); why padding avoids bank conflicts; why tiling cuts loads by the tile width; why recall rises but judged quality is flat (coverage vs dilution); why adaptive k fails (weak gate, low oracle ceiling).

## 2. Report (6-page main body; template in `report/latex_template`)
- [ ] Open `docs/report/latex/report.tex` (see its README to build). Read it as a reviewer: is every claim supported by a table? Edit wording in your own voice.
- [ ] Fill the last page: Group ID, names, UUNs; keep the AI declaration (it says an AI assistant wrote the kernels and scripts and helped draft text; adjust if your use differs).
- [ ] Decide whether to add one figure (e.g. judge score vs k with intervals, or time breakdown). Optional; tables already carry the evidence. The main body has about one page of room.
- [ ] Run the format checker (needs PyMuPDF): see `docs/report/latex/README.md`. Fix any WARN/FAIL.
- [ ] Check anonymity: no names, UUN, repo link, personal paths in the body or appendix; PDF properties have no author; file name `report.pdf`.
- [ ] Reconcile with your teammate's version (compare numbers, pick the better sections; keep one source of truth).

## 3. Video (at most 10 minutes; script in `docs/video/VIDEO_SCRIPT.md`)
- [ ] Decide the demo setup: Option A (recommended): ASR demo on the cluster GPU, answer demo on the laptop from the cached transcript, cut stated in the video. Option B (download the ASR model to the laptop, about 4.5 GB, minutes per query) needs your explicit yes. Option C (ELM key on the cluster) is not recommended.
- [ ] Rehearse the demo commands once (cluster: `make test3`, `./run.sh --cuda none`, `./run.sh`; laptop: `generate_from_transcripts.py` with a new output name, `docs/video/demo_show.py`, `retrieval_sweep.py`).
- [ ] Record (OBS), cut loading waits and say so, last two minutes = Question 1 and Question 2, export `video.mp4`, check playback with sound, check no name/path/notification is visible.

## 4. Hand-in package
- [ ] Build the code zip (our files only, no Git history, no personal data; the script refuses to run if it finds personal data):
  ```bash
  cd ~/MLS && git switch rufiz && bash scripts/make_supplementary_zip.sh ~/submission
  ```
  Result: `~/submission/supplementary_code.zip` (five kernel files, ablation harness, RAG extensions, outputs, README). Open it once and read the file list.
- [ ] Rename files neutrally: `report.pdf`, `video.mp4`, `supplementary_code.zip`.
- [ ] Look up the deadline, upload route and group size on Learn; both members upload the same report PDF if the group has two members.
- [ ] Individual review (30% of the coursework): when the anonymised report you must review arrives, write Summary (150 words), Strengths (at least 3, 300 words), Weaknesses (at least 3, 300 words), Detailed Comments (300 words). Not started.

## 5. GitHub and housekeeping
- [ ] Push the laptop commits (needs your GitHub login; the assistant has no credentials): `cd ~/MLS && git switch rufiz && git push` (use a personal access token as the password, or your credential helper).
- [ ] Teammate comparison: send the teammate's repo URL; compare kernels, ablations and RAG results; merge the best of both.
- [ ] Keep the repository free of keys, data, weights and caches (`.gitignore`); `docs/` is not part of the submission zip.
- [ ] After the final submission: rotate the ELM key if you want to be safe; delete the cluster scratch files (`~/demo_asr`, `~/mls-rag-assets`, `~/.cache/huggingface` if not needed) and the laptop `rag/results/` if you do not need them.

## 6. Optional extras (not required; only if time remains)
- Racecheck variants (task 2 Try 3, task 3 Try 4): `sbatch ablation/ablate.sbatch --stage race` on the cluster.
- Shape-dependent GEMM tile (32 for large layers, 8 for `lm_head`); register tiling to close the gap to cuBLAS.
- More judged queries or a second judge model (ELM calls, needs approval); a reranker as an alternative to adaptive k.
- Test the node hypothesis for the PyTorch baseline (`./run.sh --cuda none` on two nodes in one session).
