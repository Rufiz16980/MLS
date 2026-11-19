# Audit of the hand-in (2026-11-18)

This audit checks every course requirement against the report, video script and code zip, then lists what was fixed and what is still open. Two passes were made the same day: one on the laptop and one in the cloud. Their fixes are merged here.

**Sources re-read.**
- `01_submission_requirements.pdf` and `02_group_report_rubric.pdf`.
- The Week 4 ASR and RAG slides.
- The course `README.md`, `hw1-asr/README.md`, `rag/README.md` and the data README.
- The four task guides.
- The 2026-11-16 announcements.

**Deliverables checked.**
- `submission/report/report.tex` and `report.pdf`.
- `docs/report/VIDEO_SCRIPT.md`.
- `scripts/make_supplementary_zip.sh`, `submission/README.md` and the built `supplementary_code.zip`.
- `overlay/` (the five kernel files and the RAG extensions).

**Verdict.** Every requirement that can be met without you is met. The open items are all yours, listed in the "Still open" section.

## 1. Written report (submission requirements)
| Requirement | State | Evidence |
| --- | --- | --- |
| Supplied template, style, geometry and anonymous review mode kept | met | The template's `.sty` and `.bst` files are unchanged. The course format checker gives PASS: 8 pages, 10 pt body, coursework header on 8/8 pages, review ruler present. |
| Main body at most 6 pages | met | References start on page 6, at about 40% of the page height. About half a page of text could still be added before the body overflows. |
| References, appendices and the final page are outside the limit | met | Appendix A holds the hand-in output, B the ablations and profiles, C the extra RAG detail, D the code. |
| System context, precise question and why it matters | met | Section 1 (Introduction) and Section 2. |
| Baseline, own technical work, reasons for design choices | met | Section 2.1 (kernels) and 2.2 (RAG); the main design choice ("tune only what the profile says matters") is in Section 2.1, with its evidence in Section 4.1 and Appendix B. |
| Evaluation conditions: baseline, split, sample size, settings, what changes | met | Section 3 (Methodology): practice 350 and dev 125 queries, n = 55 judged, k = 5 baseline, one change at a time. |
| What each timing includes (loading, ASR, retrieval, API) | met | Section 3: ASR excludes model loading; RAG timings exclude ASR (cached transcripts); generation latency is the API call. |
| Caching, repeated measurements, variability | met | Transcript cache; 3 runs per ASR timing; node effect on PyTorch timings; 95% bootstrap intervals (2000 resamples per split, 5000 pooled). |
| Quality/correctness measure and systems-cost measure | met | ASR: exact transcript and tests vs ms, ms per token, GB/s, GFLOP/s, peak memory. RAG: Recall@k, hit@k and judge score vs prompt tokens, context words, latency. |
| Tables labelled with metrics and units | met | Tables 1 to 4 and the appendix tables. |
| Judge settings and limits of automated scores | met | Same model as the generator, temperature 0, one call per answer; limits in Section 5 (same-model bias, credits only gold records, cannot see ASR errors). |
| Interpretation, mechanism, limitations, alternative explanations | met | Section 4.4 (coverage vs dilution, adaptive k and oracle bound) and Section 5. |
| Negative results explained accurately | met | Ours is 2.45x slower than PyTorch, with the reason; adaptive k does not beat fixed k, with the reason; a withdrawn 15-query claim. |
| Supplied vs own contributions distinguished | met | Section 2, "Supplied" and "Ours" paragraphs. |
| AI declaration and student information on one final page | met (2026-11-19) | Page 8: AI declaration and group information for both members. The filled page comes from a git-ignored `group_info.tex`; the repo copy keeps the placeholders. |
| Anonymous title, no names/UUN/repo link in body or appendix, no PDF author metadata | met | The PDF title and author fields are empty, and a grep of the source finds no personal data and no repo link. |

## 2. Rubric (group report, 100 marks)
| Criterion | How the report answers it |
| --- | --- |
| Technical work and system understanding (30) | Four kernels with design reasons (one-launch RoPE, one-read softmax, padded tiled GEMM with fused bias and GELU, online-softmax attention). The adaptive-k change was implemented and tested. Alternatives are discussed: cuBLAS register tiling, the three-pass softmax (27% slower), fast-math (rejected). |
| Experimental design and evidence (30) | Kernels are switched on one at a time; PyTorch is compared only within one job; the GPU type is pinned. The depth study varies only k. Adaptive k is tuned on practice labels and tested on held-out dev; paired bootstrap intervals; dev replication. |
| Analysis and conclusions (30) | Amdahl argument from the profile; why the GEMM reaches 16-31% of cuBLAS; recall rises but judged quality is flat, explained by rank-of-first-relevant; oracle ceiling; uncertainty from n = 55. |
| Communication (10) | One coherent account, four tables in the body, the four guide answers in one subsection (4.2) that the introduction points to, and traceable references. |

## 3. ASR hand-in (`hw1-asr/README.md`, slides, 2026-11-16 announcement)
| Requirement | State | Evidence |
| --- | --- | --- |
| Five task files, signatures kept, no cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch inside the tasks | met | `overlay/hw1-asr/csrc/`; checked by grep. |
| `make test`, `./run.sh --profile` and `nvidia-smi -L` from the same GPU node | met | Hand-in job K08; output in Appendix A and `docs/outputs/handin_make_test_and_profile.txt`. |
| Kernels vs PyTorch: ms per transcription, per token, peak memory, per-op | met | Table 1 and Section 4.1: 2.45x slower in the same job; 207.6 vs 84.3 ms per token; peak memory 8.56 GiB for ours vs 8.88 GiB for PyTorch; profile shares. |
| GEMM: naive vs ours vs cuBLAS, with the reason | met | Table 2: 233 / 2107 / 13,363 GFLOP/s. |
| Memory-bound kernels: achieved GB/s | met | Section 4.1: RoPE 80%, LayerNorm 62%, RMSNorm 27% of peak, vocabulary softmax 1%. |
| One guide question answered per task, easy to find | met | Section 4.2: T1 Q1, T2 Q3, T3 Q1, T4 Q4. |
| "Try this" experiments | optional now | Done anyway: Appendix B ablations. |
| Supplied framework unchanged | met | The guard checks 55 supplied files each time our files are synchronised into the course checkout: all intact. |

## 4. RAG investigation (slides: "fair experiment", "chain of reasoning")
| Requirement | State | Evidence |
| --- | --- | --- |
| Precise question, one change against a stated baseline | met | Retrieval depth k, baseline k = 5. |
| Controls: same transcripts, prompt, model, judge | met | Cached transcripts; supplied prompt, model and judge. |
| Quality and cost | met | Table 3. |
| Mechanism | met | Table 4 (left): queries grouped by the rank of their first relevant record. |
| Own change with a held-out test | met | Adaptive k, fixed protocol, tuned on practice and tested on dev: no gain. Oracle bound shown. |
| Robustness | met | Dev replication (n = 27) and pooled n = 55. |
| End-to-end cost | met | Section 4.5. |
| ELM rules: credits, key never shared | met | About 360 calls in total. The key exists only in the laptop's `rag/.env`; a scan of the whole repo history found no key. |

## 5. Video (submission requirements, 2026-11-16 announcement)
| Requirement | State | Evidence in `docs/report/VIDEO_SCRIPT.md` |
| --- | --- | --- |
| At most 10 minutes | met in script | 9:45 target with a 15 s margin; each slot's narration is word-counted at about 130 words per minute. |
| Presentation and demo up to 8 minutes | met in script | 0:00 to 7:45. |
| A recorded run of the system, its input and output, configuration named, cuts made clear | met in script | Both demo stages, with the configuration said aloud and the loading wait cut and captioned. |
| Final two minutes answer the two released questions, introduced as "Question 1" and "Question 2" | met in script | 7:45 to 9:45: main design choice, then how we checked, with the fast-math and 15-query examples. |
| Screen recording with narration; no webcam, names, paths, account names or notifications | met in script | The script sets `PS1='$ '`, crops the GPU UUID, keeps `rag/.env` closed and has a checklist in section 4. |
| MP4, neutral name, no author metadata | needs you | Upload checklist in section 4. |
| Recorded and checked | needs you | Not recorded yet. `demo_query.py` has not yet run on real data, so do a rehearsal first. |

## 6. Optional code and anonymity
| Requirement | State | Evidence |
| --- | --- | --- |
| README with files, dependencies and commands; our changes distinguished from supplied code | met | `submission/README.md`. |
| Source only: no weights, datasets, environments, caches, credentials or personal data | met | `supplementary_code.zip`: 28 files. Its kernel sources match the overlay byte for byte. The script's personal-data check passes and a manual scan found nothing. |
| No Git history, relative paths | met | The zip is built from a staging folder, and home paths in outputs are replaced by `<home>`. |
| Neutral file names | needs you | Use `report.pdf`, `video.mp4` and `supplementary_code.zip`. |

## 7. Fixes made in this audit
**Report**
- The design-choice paragraph now points to Section 4 and Appendix B.
- The three-pass softmax comparison now gives the number: 27% slower.
- The bootstrap resample counts are stated.
- The judge is described as run once per answer at temperature 0.
- The text says that transcripts were produced once on the PyTorch ASR path (mean 1555 ms) and cached.
- Added the PyTorch per-op shares, the per-token times, the peak memory and the GB/s shares.
- Added a sentence on the judge's sub-scores.
- Moved the judge-bias citation to where the bias is discussed.
- The Table 1 caption now names the handout settings.
- The node effect on PyTorch timings is now worded as a suspicion that was not tested.
- The guard sentence now says the guard runs at each synchronisation, not "before every run".
- "Dominates the error budget" was replaced by a supported statement.
- The RoFormer authors and pages are corrected.

**Video script**
- The timeline was rewritten to 9:45.
- Both question answers are trimmed to fit their one-minute slots.
- The retrieval time now reads "a few milliseconds", which matches what the single cold query prints.
- The first profile (84/16) is labelled as the handout-settings profile, so it does not clash with the final 86/13.
- The cluster demo now switches to the `rufiz` branch before syncing, because `demo_query.py` is only on that branch.

**Code zip**
- The personal-data check in `scripts/make_supplementary_zip.sh` no longer contains a real student ID; it uses a pattern instead.
- The README now lists log entries E01 to E19.
- The zip was rebuilt.

## 8. Still open (all need you)
1. **Final page of the report.** Fill in the Group ID, names and UUNs, and confirm that the AI declaration describes how you used AI. Then rebuild the PDF and run the format checker again. The body has about half a page of room.
2. **Rehearse the demo once.** `demo_query.py` has never run on real data, and the first real run will need the GPU node and 2 ELM calls.
3. **Record the video** and check it against the upload checklist (section 4 of the script).
4. **Upload.** Look up the deadline and upload route on Learn, and use neutral file names. If the group has two members, both upload the same PDF.
5. **Individual review** (30% of the coursework). It can only start once the report you are assigned arrives.
6. **Teammate's version.** Not seen yet. It needs comparing and combining into one report.
7. **Understanding for questions.** Read `docs/laptop/TASK_ANSWERS.md` and `docs/report/report-video-notes.md`.

## 9. Risks
- **Page limit.** About half a page of room is left. Any edit to the body must be followed by a rebuild and the format checker.
- **Student ID in old public commits.** It came from the old check pattern in the zip script. Nothing submitted contains it, and rewriting history is not recommended. Making the repository private would remove the exposure.
- **Single-run demo.** The demo depends on the cluster and the ELM service on the day. The script has a fallback that uses the cached transcript and the recorded hand-in output, and the narration then says so.
