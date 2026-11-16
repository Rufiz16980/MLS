# Video script (10 minutes maximum) — complete draft, ready to record

Rules from the course: one video per group, at most 10 minutes; up to 8 minutes of presentation and demonstration, then the final two minutes answer the two released questions
(about one minute each, clearly introduced as "Question 1" and "Question 2"). Show a real run (input, output, configuration); you may cut waiting time if you say so. Screen recording with narration,
no webcam, no names, no UUN, no personal paths or notifications. File name `video.mp4`. Numbers below come from `docs/report/experiment-log.md` (K/E entries); say "about" for rounded values.

## Timeline

| Time | Segment | On screen |
| --- | --- | --- |
| 0:00 - 0:30 | Introduction | title slide, one-line pipeline diagram |
| 0:30 - 2:00 | System, question, what we built | pipeline slide, kernels slide |
| 2:00 - 3:45 | DEMO 1: ASR on the GPU (real run) | cluster GPU shell, section "Demo 1" below |
| 3:45 - 5:00 | Kernel results | tables: end to end, profile, GEMM |
| 5:00 - 6:30 | DEMO 2: from transcript to answer (real run) | laptop shell, section "Demo 2" below |
| 6:30 - 8:00 | RAG results and the negative result | tables: depth, quality, adaptive k |
| 8:00 - 9:00 | Question 1: main design choice | one slide |
| 9:00 - 10:00 | Question 2: how we checked our work | one slide |

Keep a 15 second safety margin: finish at 9:45.

## Narration (read at a calm pace, about 140 words per minute)

**0:00 Introduction (about 70 words).** "This report studies a spoken-question answering system with two parts. A speech recogniser turns a question into text; a retrieval-augmented generator answers it from 24,000 records.
In the speech part we wrote four CUDA kernels and asked where the time goes. In the retrieval part we asked how much retrieved evidence the answer model should receive, and what extra evidence costs."

**0:30 System (about 200 words).** "The pipeline has five stages: speech recognition with a 2.3-billion-parameter model, a small query encoder, exact dense retrieval, prompt construction, and an LLM answer that a second LLM call scores.
We wrote the model's four hot kernels: RoPE, which rotates queries and keys in one launch; reductions, with RMSNorm and a one-read online softmax; a shared-memory tiled matrix multiply with a fused bias and GELU; and tiled attention with an online softmax, one warp per query.
Everything else is the supplied course code, unchanged, and a guard checks that. For the retrieval study we transcribed all 475 recordings once and cached the text, so every experiment sees identical input and ASR cost is separated out.
We vary retrieval depth k, measure recall, answer quality and cost, and test one idea of our own, an adaptive depth."

**2:00 DEMO 1 (about 200 words, say while the commands run).** "This is the GPU node, an RTX 2080 Ti; here is its name. First the kernel test for the matrix multiply: it checks our kernel against cuBLAS and also times a naive kernel. Every case passes; on the large layers ours reaches about 2.1 teraflops,
the naive kernel about 0.23, and cuBLAS about 13. Now the full model on a 3.5 second recording. First the PyTorch path: about 1.1 seconds. Now our kernels: about 2.7 seconds, and the transcript is exactly the same: 'Concord returned to its place amidst the tents'.
I have cut the model-loading time. So our kernels are correct but about two and a half times slower than PyTorch on the same node, because the matrix multiply is the bottleneck."

**3:45 Kernel results (about 170 words).** "The profile explains it. With our kernels the matrix multiplies take 86 percent of GPU time and attention 13; RoPE, norms and softmax together are under one percent, so only the GEMM and attention were worth tuning.
Two constants did it: a 32-wide GEMM tile and 16 attention warps, 17 percent faster than the handout settings. Shared-memory padding matters a lot: without one extra column, all 32 lanes read the same bank and attention is 4.7 times slower.
Our timings are stable across nodes, around 2.7 seconds, but PyTorch's is not, 0.8 to 1.2 seconds on three nodes, so we only quote ratios from the same job."

**5:00 DEMO 2 (about 180 words).** "Now the answering side. The transcript for a real query recording was produced earlier by the same ASR on the GPU; I read it from the cache to keep this recording short. The query is embedded, the top five records are retrieved, a prompt is built, and the answer model replies.
Here is the transcript, the first evidence records, the answer, and the timings: embedding and retrieval take milliseconds, generation takes several seconds. That is the whole cost story in one line: retrieval is almost free, the answer model is not."

**6:30 RAG results (about 260 words).** "Retrieval depth: recall rises from 0.23 at k equal one to 0.58 at k equal twenty, and retrieval time stays flat at under half a millisecond, because exact search scores every vector anyway.
But judged answer quality does not follow. On 55 queries from two splits the judge score is 0.27, 0.29 and 0.27 at k of 1, 5 and 20; no difference is significant, while the prompt grows about four times per step, from 300 to 4,700 tokens.
Why? More records help queries whose relevant record is not at rank one, and slightly hurt queries where it is, and the two effects cancel. And 12 percent of queries never reach a relevant record at any depth.
We tested an adaptive depth: use one record when the top scores are well separated, twenty otherwise. The rule was fixed on practice data and checked on held-out dev data. It does not help: on dev it scores below fixed k of five while using three times the prompt, and even a perfect oracle for the same rule is not better.
That is a clear negative result, and the mechanism explains it."

**8:00 Question 1 (about 130 words).** "My main design choice was to tune only the kernels that the profile said matter. The profile showed the matrix multiply at 86 percent of GPU time and attention at 13, with everything else under one percent, so by Amdahl's law only those two were worth effort.
Tile 32 alone saved 13 percent end to end and 16 attention warps another four, together 17 percent, with the transcript still exact. The trade-off I accepted is that one tile size cannot fit every shape: it is slower than naive on the single-row output layer, where a smaller tile wins. A shape-dependent tile is the next step."

**9:00 Question 2 (about 130 words).** "I checked the kernels with four layers: the course tests against a CPU or cuBLAS reference on edge cases such as all-negative softmax rows; the sanitizer's memory and race checks, which reported zero problems; the full model reproducing the exact transcript; and repeated timings that agree within a third of a percent.
An example: the fast-math build looked like a free speed-up, but exactly one test case failed, the audio-encoder RoPE with the largest angles, so I rejected it. For the experiments: at 15 queries deeper retrieval looked better; I extended to 55 queries with identical settings, the gain disappeared, and I withdrew the claim."

## Demo 1 commands (cluster GPU shell; real ASR and kernels)

Before recording: connect, get a GPU, and warm the file cache so model loading is short. Do the login steps off-camera (the ssh command and the default prompt show your UUN).
```bash
# off camera
ssh <uun>@ssh.inf.ed.ac.uk                 # then: ssh mlp.inf.ed.ac.uk
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=16G --time=00:45:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH && conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr
./run.sh --cuda none > /dev/null 2>&1      # warm-up so loading is fast during the recording
export PS1='$ ' && clear                   # hides user and host from the prompt
```
On camera (each step about 20 to 40 seconds; the model load is the slow part: say you cut it):
```bash
nvidia-smi -L | sed 's/ (UUID.*//'          # the GPU name only
make test3 2>&1 | tail -n 22                # naive / ours / cuBLAS columns, all PASS
./run.sh --cuda none                        # PyTorch baseline: ~1.1 s per transcription, PASS
./run.sh                                    # our kernels: ~2.7 s, same transcript, PASS
```
Optional (if time): `./run.sh --profile` shows the per-operation table (linear 86%, attention 13%).
Real ASR on a query recording (optional, from the repo root of the course checkout on the cluster, `mls-asr` active):
```bash
cd ~/edin-mls-26-autumn && ASR_CUDA=all python rag/extensions/transcribe_split.py --split practice --limit 1 --out-dir ~/demo_asr && cat ~/demo_asr/practice_transcripts.jsonl
```

## Demo 2 commands (laptop; one ELM call)
Off camera: `export PS1='$ '`, `cd <course-root>`, `source ~/mls-rag-env/bin/activate`, `clear`. The key is read from `rag/.env` and is never printed.
```bash
python rag/extensions/generate_from_transcripts.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl --output rag/results/demo_k5.jsonl --k 5 --limit 1
python ~/MLS/docs/video/demo_show.py rag/results/demo_k5.jsonl 1        # transcript, evidence, answer, timings
python rag/extensions/retrieval_sweep.py --split practice --transcripts rag/.cache/practice_transcripts.jsonl   # table, no API calls, about 1 minute (cut the wait)
```
If `demo_k5.jsonl` already exists, choose a new name (the script refuses to overwrite). Optional extras with no API calls: `python rag/extensions/adaptive_k.py` (adaptive-k tables, about 1 minute).

## Slides (keep them plain, 6 to 8 slides, readable fonts)
1. Title and pipeline diagram (audio, ASR, encoder, retrieval, prompt, answer, judge). 2. The four kernels in one line each. 3. Kernel results: same-job table (PyTorch 1.10 s vs ours 2.70 s), profile shares. 4. GEMM: naive 0.23, ours 2.1, cuBLAS 13.3 TFLOP/s.
5. Retrieval depth: recall and hit rate vs k, retrieval time flat. 6. Answer quality vs k with intervals, prompt tokens. 7. Mechanism table and adaptive-k result. 8. Questions 1 and 2 (one bullet list each).

## Recording checklist
- [ ] Screen recorder (OBS) set to one window or one display; microphone test; 1080p; terminal font size 18 or larger; light theme or high contrast.
- [ ] Notifications off; no browser tabs, account names, file paths with a user name, hostnames or Git remotes visible; prompt set to `$ `.
- [ ] Say "I cut the model loading" when you cut; keep terminal output readable (pause two seconds on each result).
- [ ] Total length at most 10:00; the last two minutes are Question 1 and Question 2, introduced by name; export `video.mp4` (HandBrake if the file is too large); watch it once with sound.
- [ ] No webcam, no spoken names, no credits slide, no author metadata (check the file's properties).
