# Requirements check (2026-11-11)

Sources re-read: `01_submission_requirements.pdf`, `02_group_report_rubric.pdf`, `week4coursework-combined.pdf` (ASR and RAG slides), course `README.md`, `rag/README.md`, `hw1-asr/README.md`, `rag/data/rag_corpus/README.md`. Report and video are excluded from this check.

## ASR (hw1-asr)
| Requirement (source) | State | Evidence / gap |
| --- | --- | --- |
| Four kernels implemented, tests pass, transcription passes (hw1 README) | done | K01 |
| Hand in the five task files (hw1 README, slides p.13) | ready; defaults are now tile 32 + 16 warps (K07), verified in K08 | overlay/hw1-asr/csrc |
| `make test` AND `./run.sh --profile` from the SAME GPU node, plus `nvidia-smi -L` (hw1 README, slides p.13) | done | K08 (one job, damnii07, submitted config) |
| At least one "Try this" per task: change, numbers, explanation (slides p.13) | done, several per task | K02-K04 |
| Answer at least one question from each of the four guides, in the main body (hw1 README) | drafted, not reviewed | all 16 drafted on the laptop in docs/TASK_ANSWERS.md (tagged measured / derived / prediction); needs user review and push |
| Task 3: compare tiled vs naive vs cuBLAS (test prints all three) | done | K08 table: ours about 9x naive, 16-31% of cuBLAS; lm_head slower than naive at tile 32 |
| Same GPU type for compared numbers, report it (slides p.12) | done | 2080 Ti pinned; node named per run |

## RAG systems investigation
| Requirement (source) | State | Evidence / gap |
| --- | --- | --- |
| Baseline pipeline run and understood | done | E07-E11, E10 |
| Precise question, one change vs baseline under controls (slides p.15-16) | done | Plan A = the slides' own example (retrieval depth; slides use baseline k = 5, depths 2, 5, 10, 20) |
| Quality AND systems-cost measures | done | Recall/hit@k, judge scores; context words, prompt tokens, generation latency, retrieval latency |
| Timing boundaries, caching, repeats, variability stated (submission req.) | done for ASR, mostly for RAG | ASR: repeats and node effects (K05, K07, K09); RAG: transcripts cached, bootstrap CIs, ELM calls single run |
| Judge settings reported, limits of automated scores (submission req.) | done | E12, E13, E15 |
| Explain why (mechanisms), limitations, alternatives (rubric) | done | E16 coverage vs dilution, E17 adaptive k + oracle bound, E18 cost breakdown |
| Technical substance of the RAG change (rubric: technical work 30) | done | E17 adaptive k: implemented, protocol fixed in advance, tuned on practice labels, held-out dev check, replay on judged answers, oracle bound; negative result with mechanism |
| Trustworthy evidence, robustness tests (rubric "Excellent") | GAP | answer quality only on 28 practice queries; no dev-split confirmation; k = 2 and 10 not run; no judge-repeat reliability check |

## Real ASR data question
There is no reference (true) text for the 475 audio queries: the slides list the supplied data as corpus, audio, manifest (query IDs, split, audio paths), qrels and embeddings; `rag_corpus/README.md` says "Audio provides the query input; qrels support evaluation"; the slides say public judging "cannot verify that ASR preserved the intended question". Only the single hw1 test clip has a reference (`test_audio.txt`). So word error rate cannot be measured on the RAG queries. Real-ASR experiments are still possible without references: perturb the audio (noise, speed) and re-transcribe on the GPU, or compare two ASR implementations' transcripts, and measure the change in retrieval. The slides also allow "ASR optimisations as extensions".

## Other
- Anonymity: no names/UUN/repo link in report, video or code zip; the GitHub repo is public and its history contains the UUN, so do not reference it in the report.
- Individual review (30%): happens after submission, on an assigned report.


## Added 2026-11-11
- End-to-end cost breakdown joining ASR, retrieval and generation: E18 (derived from logged numbers).

## Deliverables still open (2026-11-11 15:20)
- Report (6-page body, template, anonymous, AI-use declaration on the final page): not started; DRAFT.md is the evidence index.
- Video (<= 10 min; actual recorded run of the system with input and output; last 2 min answer the two released questions): not started.
- Optional supplementary code zip: README, our files vs supplied, no Git history or personal paths.
- Merge with the teammate's version: their work is not in this repo.