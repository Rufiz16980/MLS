# AI-use policy: what the written materials say (checked 2026-11-18) and how our work compares

Sources searched: the five coursework PDFs (submission requirements, both rubrics, review requirements, video guide), the ASR/RAG slides, the course repo (README files, report template and its comments, format checker), the ELM terms and University guidance saved in `docs/reference/elm-terms-and-key-agreement.md`, and the course announcements of 2026-11-16. Paraphrased, not quoted.

## Where the policy is written
1. **Report template, `report/latex_template/report.tex` (comment blocks at the top and above the "AI declaration" section).** All submitted work is expected to be your own, including material you generated with the help of AI tools (the list names ELM, ChatGPT, Bing Chat, Claude, Gemini, "any other automated assistance"). For all such material you must make a prominent, explicit acknowledgement and explain clearly what you did, in the AI declaration section. This covers using AI to generate text, code, figures or anything similar, or to complete other tasks. You remain entirely responsible for what you submit.
2. **Submission requirements PDF (report section).** The AI-use declaration and the student information go on one final page, using the template; that page is removed before peer review.
3. **Course README and `report/README.md`.** Both say to include the template's AI declaration.
4. **RAG slides (ELM key request text).** The key-request statement promises to treat generated answers and judge scores as fallible, inspect examples, and acknowledge AI use in line with the course's academic-integrity requirements; the slides add that you must follow the ELM rules and must not share the key with others or AI.
5. **University-level guidance (inside the ELM terms; source elm.edina.ac.uk).** Misconduct includes: presenting AI output as your own work; submitting AI-generated text, images/audio/video, mathematical reasoning or **computer code** without acknowledgement; citing AI-found sources without reading and verifying them; and **using an AI agent or AI browser to do any work inside the virtual learning environment (Learn)**; sharing login details with an agent is a separate breach. It says AI content needs an in-text citation or footnote plus a reference-list entry, but also says to follow the course's own guidance on how to acknowledge, and that a course may allow or restrict more than the University baseline.

## What is NOT in the written materials
- No sentence that AI is encouraged, or that AI may write the assessed kernels or the report. The instructors' statement you reported (AI encouraged, understanding is what matters) was verbal, so it is not documented here.
- One rule worth a second look: the ASR README ("Rules") says the kernels must be your own CUDA code and may not call cuBLAS, cuDNN, CUTLASS, Thrust or PyTorch. It is about not outsourcing the work to libraries, but "own" is not defined; an AI-written kernel is acknowledged AI-generated code under the template policy.
- Suggest getting the verbal policy in writing (Learn announcement, forum, or an e-mail reply from the course organiser) and keeping it.

## How our work compares
| Requirement | Our status |
| --- | --- |
| AI declaration on the final page, explicit about what AI did | Present (report page 8): kernels, scripts, analyses, experiment log, report text; ELM only as generator and judge; you remain responsible |
| Declaration covers every AI-made artefact | GAP: it does not mention the **video script/narration and slides**, nor the code zip README. Add them. |
| Acknowledge AI-generated code | Covered by the declaration; optionally add one line to the zip README ("written with AI assistance, see the report declaration") |
| In-text citation or reference entry for AI content (University guidance) | Not done; the course template points to the declaration only. Optional: a reference entry for the assistant (name, company, year, URL) |
| AI-found sources read and verified | **YOU must do this**: the seven references in `references.bib` were added by the assistant; read or at least verify each exists with the right authors, venue and year before submitting |
| No AI agent inside Learn | Do not let any assistant or agentic browser operate Learn; this includes the individual-review text fields. Write and paste the review yourself. |
| ELM key not shared with AI | The assistant only loaded it into a shell variable for leak checks and never displayed it; if your reading is stricter, rotate the key in the ELM portal |
| Accountable for errors | Read the report and be able to explain every number (oral Q&A) |

## Suggested additions to the declaration (edit to match what is true)
- "The video narration script and slides were drafted with the same assistant and then reviewed and rehearsed by us."
- "The supplementary code and its README were written with the assistant's help."
