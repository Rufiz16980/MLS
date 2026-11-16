# Report source (LaTeX, course template)

`report.tex` + `references.bib` are the complete draft of the group report on the supplied template (anonymous title, 6-page main body limit). Compiled locally: the main body ends on page 5, references and appendix
on page 5, the AI declaration and group information on the last page (6 pages in total). Placeholders to fill: `Group ID`, member names and UUNs (final page only), and the AI declaration wording.

## Build
The style files belong to the course template and are not copied here. From the course checkout:
```bash
cd <this folder>
for f in iclr2027_conference.sty iclr2027_conference.bst natbib.sty fancyhdr.sty math_commands.tex; do cp <course-root>/report/latex_template/$f .; done
latexmk -pdf -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
```
Or upload `report.tex`, `references.bib` and the template files to Overleaf (main document `report.tex`).

## Check the format (needs PyMuPDF; install it in the RAG venv or any venv)
```bash
python -m pip install -r <course-root>/report/format_checker/requirements.txt
python <course-root>/report/format_checker/check_report_format.py report.pdf
```
Not yet run on this draft: the checker's dependency (PyMuPDF) is not installed on the laptop.

## Where each claim comes from
Every number is in `docs/report/experiment-log.md` (K entries: kernels; E entries: RAG). Table to entry map: Table 1 K05, K07, K09; Table 2 K06, K08; Table 3 K02-K04; Table 4 K08; Table 5 E09, E14;
Table 6 E15, E19; Table 7 E16; adaptive k E17, E19; end-to-end cost E18.

## Before submitting
- Remove line numbers? No: the template's review mode adds them; keep the supplied style.
- Put the final names/UUNs only on the last page; check the PDF metadata has no author; file name `report.pdf`.
- Re-read every sentence you cannot explain; delete it or ask for an explanation (oral Q&A).
