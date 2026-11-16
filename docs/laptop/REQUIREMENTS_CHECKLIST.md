# Requirements checklist (written 2026-11-11 from the coursework PDFs, slides and repo guides)

Sources read: 01 submission requirements, 02 group-report rubric, 03/04 individual review requirements and rubric,
05 video guide, the combined ASR + RAG slides, `hw1-asr/docs/task1..4`, `rag/README.md`, `report/README.md`.
Status: DONE / PARTIAL / TODO / NOT STARTED. Evidence = experiment-log entry (E#) in `docs/EXPERIMENT_LOG.md`.

## How the group report is marked (70% of coursework; 100 marks)
Technical work and system understanding 30 | Experimental design and evidence 30 | Analysis and conclusions 30 |
Communication 10. Beating the baseline is NOT a criterion; a well-analysed negative result can score highly.
ASR and RAG are assessed together as one account. Video (<= 10 min, last 2 min answer two questions released one week
before the deadline) has no separate mark; the individual review of another group's report is the other 30%.

## ASR part (hw1-asr)
| Requirement | Status | Evidence / gap |
|---|---|---|
| Fill the blanks of the four kernels; keep given names and signatures; no cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch inside a task | DONE (written by a Claude session; must be declared; owner must be able to explain each blank) | E13; guard OK (55 frozen files intact) |
| Correctness: every `make test` case PASS and transcription PASS | DONE | E13 (all PASS, 0 sanitizer errors/hazards) |
| Kernels vs PyTorch: `./run.sh --profile` vs `./run.sh --cuda none --profile` (ms/transcription, ms/token, peak memory, time per operation) | DONE (needs one clean same-node pair for the hand-in) | E15, E16 |
| GEMM: naive vs tiled vs cuBLAS GFLOP/s per case and why cuBLAS is faster (`make test3` prints all three) | DONE | E21 |
| Memory-bound kernels: GB/s as a share of peak (`make test1`, `make test2`) | DONE for default build | E14 |
| At least one "Try this" per task with numbers before/after and explanation | DONE (several per task); racecheck variants (task 2 Try 3, task 3 Try 4) not run | E14, E15 |
| Written answers to the questions at the end of each task guide (task1: 3, task2: 4, task3: 4, task4: 5 = 16) | DRAFTED (owner must review and be able to explain) | `docs/TASK_ANSWERS.md` |
| Hand in: the five task files, `make test` + `./run.sh --profile` output from the same GPU node, `nvidia-smi -L` | DONE (submitted config tile 32 + 16 warps) | E21, E22, E23 |

## RAG part
| Requirement | Status | Evidence / gap |
|---|---|---|
| Understand the audio-to-answer pipeline; run it end to end | PARTIAL: all stages run, but ASR was run separately and cached; no single recorded `run_audio_query.py` run with real ASR | E04-E11; TODO for the video demo |
| Retrieval metrics Recall@k, MRR@10, NDCG@10 | DONE (practice 350, dev 125) | E07, E08, E10, E12, E18 |
| Answer quality (judge: correctness, grounding, completeness 0-4, combined /12) with judge settings reported | DONE for 28 queries (3 depths) | E17, E19, E20 |
| A precise question; a technical change; a baseline; controls; quality AND cost metrics | DONE: question A with quality + cost; implemented change = adaptive k with a pre-set protocol (negative result) | E19, E20, E24 |
| Interpretation: mechanism, trade-offs, limitations, alternatives | PARTIAL-GOOD: coverage vs dilution (E20), judge floor (E17b), hit@k ceiling (E18); small n, single judge repeat | more queries or judge repeats would strengthen |
| Plans B (enc_a -> enc_b query map) and C (ASR errors vs retrieval) | NOT STARTED (data for B available; for C only proxies, see below) | decision pending |
| ELM rules (free local model, own key, limit calls, no personal data/raw audio, key never shared with others or AI) | followed | ~190 calls used |

## ASR data for plan C: what exists
The asset archive holds only audio for the 475 queries (no reference text), the manifest (no text), qrels, corpus, embeddings and the enc_a model.
The slides state that public judging "cannot verify that ASR preserved the intended question". Only one clip (`hw1-asr/test_audio.wav`) has a reference
transcript. So WER on the 475 queries is impossible; possible proxies: (1) agreement between ASR run with our kernels and PyTorch ASR over all 475
queries (numerical fidelity, real ASR), (2) simulated transcript noise, (3) manual reading of retrieval failures.

## Combined account and process rules
- ASR and RAG must read as one system: end-to-end cost breakdown (ASR with our kernels vs PyTorch, retrieval, generation) is a natural bridge. TODO.
- State timings' boundaries (model load, ASR, retrieval, API calls), hardware/node, repeats, variability (node effect E16; first-run load 100-150 s is cold cache).
- Anonymity: names, UUNs, emails, Group ID only on the final page; no account names/paths/notifications in the video; neutral file names; supplementary code without
  personal details and without Git history; the team repo is public and carries a personal GitHub name, so submit code as a clean zip.
- AI-use declaration on the final page; the owner must be able to explain every design choice (oral Q&A).
- Individual review (30%): NOT STARTED; needs the assigned anonymised report (Summary 150, Strengths 300, Weaknesses 300, Detailed comments 300 words).
- Video: NOT STARTED; needs a recorded real run (shorten waits and say so) and the two released questions in the final two minutes.

## Course announcements (received 2026-11-16) — these override earlier assumptions
1. ASR guides: only ONE question per task has to be answered (four in total); all "Try this" sections are now OPTIONAL suggestions. The four answers may be anywhere in the
   report (main body or appendix) but must all be present and easy to find (clear formatting or a sentence in the introduction). NOTE: the course README (hand-in section) still says "no fewer than four
   in the main body"; the announcement is the later text. Safest: a short, clearly labelled subsection in the main body with the four chosen answers; extras in the appendix.
2. Supplementary code: include whatever demonstrates the system; the supplied framework code need not be resubmitted if unchanged (hand in the five edited kernel files; ops.py, model.py, benchmark.py,
   provided.cu and the tests must stay unchanged).
3. `make test` and `./run.sh --profile` outputs: put the most relevant lines in the main body, the rest in an appendix.
4. Video final two minutes, about one minute each: (1) your main design choice and why; (2) how you checked your work (give an example). Any part of the system; need not cover both ASR and RAG.
Status after the announcements: ASR requirements FULFILLED in evidence (four answers can be chosen from the 16 drafts; Try-this work is a bonus). RAG, combined account and video demo: see below.
