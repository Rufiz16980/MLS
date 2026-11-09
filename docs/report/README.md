# Report workspace

| File | Purpose |
| --- | --- |
| [experiment-log.md](experiment-log.md) | Every result we will rely on: what we needed, the exact command, the raw output, and what it means. Append only; each entry has an ID (E01, E02, ...) that the report cites. |
| [DRAFT.md](DRAFT.md) | The report skeleton (6-page main body) with the evidence we already have slotted into the rubric's four criteria. |

Rules for entries
- Same GPU type for every comparison (RTX 2080 Ti). Always record the node name, because timings differ between nodes.
- Redact personal data before committing: replace the cluster login with `<UUN>`, home paths with `~`, and never paste keys, emails or names.
- Mark anything we have not seen with our own eyes as PENDING; do not fill numbers from memory or from the slides.
- The report itself must be anonymous. This folder is a working draft; the submitted PDF is built from the LaTeX template (`report/latex_template`).
