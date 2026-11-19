# Video script (final draft, 10 minutes)

Screen recording with narration only: no webcam, no names, no UUN, no group ID. Speaking pace about 130 words per minute; every section's narration fits its slot (checked by word count). Slides are listed in section 2, terminal commands in section 3, the anonymity checklist in section 4. The final two minutes answer the course team's two questions and are introduced as "Question 1" and "Question 2".

## 1. Timeline and narration

Target length 9:45, leaving a 15-second margin under the 10-minute limit. The two questions take the final two minutes (7:45 to 9:45). Word counts are sized for about 130 words per minute; the demo slots have extra time for the terminal output.

### 0:00-0:40 Title and question (slide 1)
> This is a spoken-question answering system. A question is recorded as audio, transcribed by a speech model, matched against 24,000 finance records, and answered by a large language model from the records it retrieves. We worked on both ends: we wrote the four GPU kernels the speech model runs on, and we asked how much retrieved evidence the answer model should get. I will show the system running, then the two main results, and then answer the two questions.

### 0:40-1:40 System and what is ours (slide 2)
> Supplied by the course: the speech model and its tests, the retrieval framework, the data, and the answer and judge model. Ours: four CUDA kernels. RoPE rotates queries and keys in one launch. The reduction kernels compute RMSNorm and a softmax that reads each row only twice; LayerNorm was a supplied example that runs on our warp sum. The matrix multiply stages tiles in shared memory and fuses bias and GELU. Attention processes one query per warp with a running maximum and sum, so the 1500 by 1500 score matrix is never stored. On the retrieval side we wrote the experiment scripts: cached transcripts so every setting sees identical text, a depth sweep, a paired answer-quality study, and an adaptive depth. A checksum guard confirmed that no supplied file was changed.

### 1:40-3:30 Demo, part 1: speech recognition on our kernels (GPU cluster terminal)
Show: `nvidia-smi -L`, the end of `make test` (four ALL PASS lines), then `demo_query.py asr`. Cut the model-loading wait and caption it "waiting time shortened".
> This is the GPU node, an RTX 2080 Ti. First the course tests: each kernel is compared with a CPU or cuBLAS reference on edge cases and on the model's real shapes, and all four pass. Now one real query from the practice split. This runs the full speech model with all four of our kernels switched on. Loading the model takes a few minutes, so I have cut that. Here is the transcript, and the recognition time for this query. Below it, the transcript the PyTorch version produced earlier, for comparison. The configuration is the one we hand in: tile 32 for the matrix multiply and 16 queries per attention block.

### 3:30-4:50 Demo, part 2: retrieval and answer (laptop terminal)
Show: `demo_query.py answer --k 5`, then `--k adaptive`.
> The answer model's key stays on the laptop, so the transcript file is carried over and the second half runs here. The transcript is encoded, all 24,000 records are searched exactly in a few milliseconds, and here are the top five records with their similarity scores. The ones marked relevant are the course's ground-truth labels. The answer model writes this answer from those five records. The table shows where the time went: recognition on the GPU, a few tens of milliseconds for encoding and retrieval, and a few seconds of generation. Now the same query with our adaptive depth. It looks at the gap between the top two scores and picks k equals 1 or 20. That is the change I evaluate in a moment.

### 4:50-6:15 Result 1: the speech side (slide 3)
> On the same GPU node in the same job, PyTorch takes 1.1 seconds for the test recording and our kernels take 2.7: 2.45 times slower, while transcribing it exactly. The profile explains why: the matrix multiply is 86 percent of kernel time and attention 13; the other kernels together are under one percent. Our RoPE and normalisation kernels are actually faster than PyTorch's chains of small operations. The matrix multiply is about 9 times faster than a naive kernel but reaches only 16 to 31 percent of cuBLAS, which computes several outputs per thread from registers. Choosing tile 32 and 16 warps from the ablations made the whole model 17 percent faster than the handout settings. Padding mattered more than any constant: without the extra column, shared-memory bank conflicts make the multiply up to 4.5 times slower.

### 6:15-7:45 Result 2: the retrieval side (slide 4)
> Recall rises steeply with depth, from 0.23 at k equals 1 to 0.58 at 20. Judged answer quality does not. Over 55 judged queries from two splits, k equals 1, 5 and 20 score 0.27, 0.29 and 0.27, all within noise, while the prompt grows 15 times and generation time more than doubles. Splitting queries by where their first relevant record ranks shows why: queries whose evidence is ranked low gain from more records, queries whose top record is already right lose a little, and the two cancel. We turned that into our own change, adaptive depth, and tested it with a protocol fixed in advance: tuned only on practice labels, evaluated on the held-out dev split. It does not beat a fixed depth. The confidence signal sends about a third of its confident queries to the wrong record, and even a perfect gate would gain little, because a quarter of queries have no relevant record in the top 20 at all. So in this system, retrieval depth is a cost decision, not a quality decision.

### 7:45-8:45 Question 1: our main design choice (slide 5)
> Question 1: our main design choice. We decided to tune only the kernels that the profile said matter. The first profile, with the handout settings, showed the matrix multiply at 84 percent of kernel time and attention at 16; RoPE, the norms and softmax together were under one percent, so by Amdahl's law even making them free would save almost nothing. We therefore chose final settings only for those two kernels: tile 32 instead of 16, and 16 queries per attention block instead of 4. Together they made the model 17 percent faster, with the transcript still exact. The cost we accepted: the single-row lm_head multiply is slower with tile 32, where tile 8 is best, so a shape-dependent tile is the next step.

### 8:45-9:45 Question 2: how we checked our work (slide 5)
> Question 2: how we checked that it worked. We used four layers of checks. The course tests compare every kernel with a reference on edge cases. Compute-sanitizer found no memory errors or races. The full model must reproduce the reference transcript exactly. And we repeated timings across jobs and nodes: our times varied by under one percent, so we only trusted larger differences. One example: the fast-math compiler flag looked like a free speed-up, but the RoPE test failed on exactly one case, the audio encoder with the largest angles, so we rejected it. In the retrieval study, a gain for k equals 20 after 15 queries disappeared when we added more queries, so we withdrew that claim.

## 2. Slides (five, plain, large text)
1. Title (anonymous): "Where the time and the evidence go: CUDA kernels and retrieval depth in a spoken-question RAG system". One line: audio -> ASR (our kernels) -> encoder -> exact retrieval (24,000 records) -> LLM answer.
2. Pipeline diagram with "supplied" in grey and "ours" in colour: 4 kernels; transcript cache, depth sweep, paired judge study, adaptive k.
3. Speech results: Table 1 of the report (same-job PyTorch vs ours; kernels switched on one at a time), the profile shares (86.1 / 13.3 / 0.5%), and the GEMM row "naive 233, ours 2107, cuBLAS 13,363 GFLOP/s".
4. Retrieval results: Table 3 of the report (Recall, judge score practice/dev/pooled, prompt tokens) and the rank-of-first-relevant table (Table 4 left), plus the adaptive and oracle rows.
5. "Question 1: design choice" and "Question 2: how we checked" with three bullets each (the numbers above).

## 3. Demo commands

Rehearse once end to end before recording. Before each recording: `export PS1='$ '` in every terminal (hides the user and host name), `clear`, enlarge the font, close notifications.

**Pick the demo query (laptop, once, off camera).** From the course repo root, with the venv active; lists practice queries the judge scored well at k = 5 and where retrieval put a relevant record first:
```bash
python3 - <<'EOF'
import json
d = json.load(open('rag/results/adaptive_k.json'))
pq = {r['query_id']: r for r in d['splits']['practice']['per_query']}
for l in open('rag/results/aq_practice_k5_judge/answer_judge_per_query.jsonl'):
    r = json.loads(l); s = r.get('normalised_answer_score')
    if s is not None and s >= 0.75:
        print(r['query_id'], round(s, 2), 'first relevant rank', pq[r['query_id']]['first_rel_rank'], 'policy k', pq[r['query_id']]['k'])
EOF
```
Prefer a query with first relevant rank 1 (the adaptive demo then picks k = 1). Call its id `QID` below.

**Part 1, cluster (GPU).** Off camera: log in (`ssh <UUN>@ssh.inf.ed.ac.uk`, then `ssh mlp.inf.ed.ac.uk`), then
```bash
cd ~/MLS && git fetch -q origin && git switch rufiz && git pull -q && bash scripts/sync_overlay.sh ~/edin-mls-26-autumn
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH; source ~/miniconda3/etc/profile.d/conda.sh; conda activate mls-asr
export PS1='$ '; cd ~/edin-mls-26-autumn; clear
```
On camera:
```bash
nvidia-smi -L
cd hw1-asr && make -s test 2>&1 | grep -E "ALL PASS|FAIL"; cd ..
python rag/extensions/demo_query.py asr --query-id QID
```
`nvidia-smi -L` prints a UUID: crop it or run `nvidia-smi -L | sed 's/ (UUID.*//'`.
Optional extra evidence on camera, from `hw1-asr` (each `run.sh` loads the model; cut the wait): `make test3 2>&1 | tail -n 22` (naive / ours / cuBLAS for every GEMM case, all PASS), then `./run.sh --cuda none` (PyTorch, about 1.1 s) and `./run.sh` (ours, about 2.7 s, the same exact transcript). To keep loading short, run `./run.sh --cuda none > /dev/null 2>&1` once off camera first (warms the file cache). The ASR step loads the model first (up to 3 minutes on a fresh node): cut the wait and caption it. Output: transcript, ASR time, the cached PyTorch transcript, `-> rag/results/demo_transcript.json`.

**Copy the transcript to the laptop (off camera)**, from the laptop's course repo root:
```bash
scp -o ProxyJump=<UUN>@ssh.inf.ed.ac.uk <UUN>@mlp.inf.ed.ac.uk:edin-mls-26-autumn/rag/results/demo_transcript.json rag/results/
```

**Part 2, laptop.** Off camera: `cd` to the course repo root, `source ~/mls-rag-env/bin/activate`, `export PS1='$ '`, `clear`. On camera (one ELM call each):
```bash
python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k 5
python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k adaptive
```
Output: transcript, top-k records with scores and RELEVANT marks, the answer, a per-stage time table, prompt and completion tokens.

Fallback if the cluster is unavailable on the day: run part 2 with the cached transcript, `python rag/extensions/demo_query.py answer --transcript "<cached transcript>" --query-id QID --k 5`, and show the hand-in output (`outputs/handin_make_test_and_profile.txt`) for the GPU side; say clearly that the speech step is a recording of an earlier run.

## 4. Before uploading
- Duration at most 10:00 (target 9:45), with the two questions in the last two minutes and announced as "Question 1" and "Question 2".
- No names, UUN, group ID, host names, home paths, e-mail or notifications on screen; the API key never on screen (do not open `rag/.env`).
- Export MP4 (H.264 + AAC), file name `video.mp4`, clear the title/author metadata, watch the exported file once.
