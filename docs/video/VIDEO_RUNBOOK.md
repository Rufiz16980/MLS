# Video runbook: what to show, what to run, what to say

Keep this file open while you record. Every step has up to four parts:

- **SLIDE**: what is on the screen.
- **RUN**: the command to type (copy and paste it).
- **EXPECT**: what the screen will show.
- **SAY**: the words to read.

Read slowly and use short sentences. The target length is **9:40**; the limit is 10:00.
The course rules are: up to 8 minutes of presentation and demo, then **the final two minutes answer Question 1 and Question 2** (one minute each, each introduced by name). Use narration only, with no webcam. Nothing on screen may show a name, UUN, group ID, user name, host name, home path, e-mail, notification or the API key.

## 0. The plan

| Step | Time | Screen | What happens |
| --- | --- | --- | --- |
| 1 | 0:00-0:25 | slide 1 | What the project is |
| 2 | 0:25-1:00 | slide 2 (course slide) | The pipeline |
| 3 | 1:00-1:40 | slides 3 (course slide) and 4 | The four kernels |
| 4 | 1:40-3:05 | cluster terminal | DEMO 1: speech recognition on the GPU with our kernels |
| 5 | 3:05-4:20 | laptop terminal | DEMO 2: from the transcript to an answer |
| 6 | 4:20-5:10 | slides 5 and 6 | Speech results |
| 7 | 5:10-7:40 | slides 7 to 11 | Answering results |
| 8 | **7:40-8:40** | slide 12 | **Question 1** |
| 9 | **8:40-9:40** | slide 13 | **Question 2** |

## 1. The slides

The deck is `slides.pdf` (13 slides), built from `docs/video/slides.tex`. Slides 2 and 3 are pages 16 and 6 of the course slides (`week4coursework-combined.pdf`), with a small "Source: course slides" credit. The other slides are ours. To rebuild it, put `slides.tex` and a copy of the course slides named `course.pdf` in one folder, then run `pdflatex slides.tex` twice.

| # | Slide |
| --- | --- |
| 1 | Title |
| 2 | The common pipeline (course slide) |
| 3 | Inside one layer: where your kernels run (course slide) |
| 4 | What we built |
| 5 | Speech: correct, but slower than PyTorch |
| 6 | The matrix multiply: naive vs ours vs cuBLAS |
| 7 | Our question about answering |
| 8 | More records find more of the right ones |
| 9 | But better answers? No visible difference |
| 10 | Why? Two effects cancel |
| 11 | Our own idea: adaptive k. It does not help. |
| 12 | Question 1: our main design choice |
| 13 | Question 2: how we checked our work |

Open the PDF in presentation mode (F5 in most viewers), so the file name and path are not visible.

## 2. Before you press record (off camera)

**The demo question (already chosen):** `q_a02dccd42055bf0e`, *"How can I determine the minimum due payment and due date for my ANZ Visa credit card balance?"* The right record is ranked first. In our judged results this question scored 1.00 with k = 5 and 0.00 with k = 1.

**A. Cluster terminal (GPU node).** Use its own terminal window, not a tab, because tabs show the user name.
```bash
ssh <UUN>@ssh.inf.ed.ac.uk
ssh mlp.inf.ed.ac.uk
cd ~/MLS && git fetch -q origin && git switch rufiz && git pull -q && bash scripts/sync_overlay.sh ~/edin-mls-26-autumn
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=16G --time=00:45:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH; source ~/miniconda3/etc/profile.d/conda.sh; conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr && make -s all
./run.sh --cuda none > /dev/null 2>&1
cd ~/edin-mls-26-autumn
export PS1='$ '; clear
```
- The sync prints `guard OK: 55 frozen files intact, 5 task files conform`.
- `make -s all` builds our kernel library and the tests for this GPU.
- The `./run.sh` line is a warm-up. It loads the model once, so the file cache is warm later. It takes a few minutes and prints nothing.
- `export PS1='$ '` makes the prompt a plain `$`, so your user name and the host name are hidden.

**B. Laptop terminal.** Again its own window.
```bash
cd "$HOME/Desktop/SEM1/Machine Learning Systems (2026-2027)[SEM1]/Week4_Assignment/edin-mls-26-autumn"
source ~/mls-rag-env/bin/activate
export PS1='$ '; clear
```
Do not open `rag/.env`, because the API key must never be on screen.

**C. Screen.** Make both terminals full screen (F11 in most terminals), because the window title can show your user name. Set the font size to 18 or more. Turn notifications off and close e-mail and chat apps.

**D. Rehearse once without recording.** Run steps 4 and 5 once. Each run of step 5 makes 2 ELM calls (5a and 5b), so a rehearsal plus the real take is about 4 calls. Avoid many retakes of step 5.

**E. Recorder.** OBS Studio, one screen, with the microphone tested. While the model loads in step 4c, press **Pause Recording**, then **Resume**. Say that you shortened the wait. In the editor (Shotcut) you can also add a short caption, "Waiting time shortened".

## 3. The steps

### Step 1. 0:00-0:25. SLIDE 1 (title)
**SAY:** "This is a spoken-question answering system. A speech model turns a spoken question into text, we find matching records, and a language model writes the answer. We worked on two parts: the speech model's GPU kernels, and how many records the answer model should read."

### Step 2. 0:25-1:00. SLIDE 2 (course slide: the common pipeline)
**SAY:** "This slide is from the course. Audio goes into speech recognition and comes out as text. The text is turned into a vector. We search 24,000 finance records for the closest ones. They go into a prompt, and the language model writes the answer. The course supplied this whole system. Our work is the speech model's kernels and the experiments on retrieval."

### Step 3. 1:00-1:40. SLIDE 3, then SLIDE 4
**SLIDE 3 (course slide: inside one layer). SAY:** "Also from the course: one layer of the speech model. Our four kernels run in these boxes."

**SLIDE 4 (what we built). SAY:** "RoPE marks each token's position. The reductions compute sums and maximums for RMSNorm and softmax. The matrix multiply is the biggest cost. Attention lets each token look at the others. For answering, we cached the transcripts, measured retrieval depth, and tested one idea of our own: adaptive depth."

### Step 4. 1:40-3:05. DEMO 1 (cluster terminal)
**SAY first:** "Now the real system. This terminal is on a GPU node of the course cluster."

**4a. RUN**
```bash
nvidia-smi -L | sed 's/ (UUID.*//'
```
**EXPECT:** one line, `GPU 0: NVIDIA GeForce RTX 2080 Ti`.
**SAY:** "The GPU is a GeForce RTX 2080 Ti."

**4b. RUN**
```bash
cd hw1-asr && make -s test 2>&1 | grep -E "ALL PASS|FAIL"; cd ..
```
**EXPECT:** after a few seconds, four lines:
```
Task 1 (RoPE): ALL PASS
Task 2 (reductions): ALL PASS
Task 3 (tiled GEMM): ALL PASS
Task 4 (attention): ALL PASS
```
**SAY:** "First, the course tests. Each kernel is compared with a reference answer, including edge cases. All four kernels pass."

**4c. RUN**
```bash
python rag/extensions/demo_query.py asr --query-id q_a02dccd42055bf0e 2> /tmp/asr_log.txt
```
**SAY while it starts:** "Now one real spoken question from the practice data. This runs the whole speech model with all four of our kernels, using the settings we hand in: tile 32, and 16 queries per attention block."
**EXPECT, first:**
```
query q_a02dccd42055bf0e (practice), audio audio/practice/....wav
ASR kernels: ASR_CUDA=all  (all = our four CUDA kernels, none = PyTorch only)
```
Then nothing for up to three minutes while the model loads. **Pause the recording here.** Resume when `model loaded in ... s` appears.
**SAY after you resume:** "Loading the model takes a few minutes, so I shortened the wait."
**EXPECT, then:**
```
TRANSCRIPT: How can I determine the minimum due payment and due date for my ANZ Visa credit card balance?
ASR time: <a few thousand> ms
cached PyTorch transcript: How can I determine the minimum due payment and due date for my ANZ Visa credit card balance?
identical: True
-> rag/results/demo_transcript.json
```
**SAY:** "Here is the transcript. Recognition took about [read the ASR time] milliseconds. Below it is the transcript that the PyTorch version made earlier. They are identical. This is one live run; the measured numbers come next."
- If it shows `identical: False`, say instead: "They differ in a few words. The data has no reference text, so we did not measure a word error rate."

**4d. Off camera, laptop:** copy the transcript file to the laptop.
```bash
scp -o ProxyJump=<UUN>@ssh.inf.ed.ac.uk <UUN>@mlp.inf.ed.ac.uk:edin-mls-26-autumn/rag/results/demo_transcript.json "$HOME/Desktop/SEM1/Machine Learning Systems (2026-2027)[SEM1]/Week4_Assignment/edin-mls-26-autumn/rag/results/"
```
Then run `clear` in the laptop terminal.

### Step 5. 3:05-4:20. DEMO 2 (laptop terminal)
**SAY first:** "The key for the answer model stays on my laptop, so this half runs here, from the transcript file."

**5a. RUN**
```bash
python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k 5 2> /tmp/answer_log.txt
```
**EXPECT:**
- `query q_a02dccd42055bf0e (practice), relevant records in qrels: 1`, then the `TRANSCRIPT:` line.
- `Top 5 records (exact cosine over 24000 records, ... ms):` and five lines. The first one is `1. 0.937  RELEVANT r_85a3373471291694  You are in luck, I have an ANZ credit card as well...`
- After a few seconds, `ANSWER (...)` with a few paragraphs about the minimum payment and the due date.
- A table: ASR time (from the cluster), query embedding (about 60 to 90 ms), retrieval (about 10 ms), generation (about 4 to 7 seconds), and the total.
- `prompt tokens ...` (about a thousand), then `-> rag/results/demo_answer.json`.

**SAY:** "We search all 24,000 records. These are the top five. The first is marked RELEVANT: the course labels say it is the right record. The language model wrote this answer from these five. The table shows each stage. Encoding and search take milliseconds. Writing the answer takes several seconds, so the answer model is the biggest cost."

**5b. RUN**
```bash
python rag/extensions/demo_query.py answer --from rag/results/demo_transcript.json --k adaptive 2> /tmp/answer_log.txt
```
**EXPECT:** `adaptive k: margin = 0.0648 >= tau 0.0122 -> k = 1`, then `Top 1 records` with the same first record marked RELEVANT, a shorter answer, a faster generation time, and about 300 prompt tokens.
**SAY:** "Now our own idea, adaptive depth. The gap between the top two scores is large, so it uses only one record, and the prompt is much shorter. But in our judged results, this question scored 1.0 with five records and 0 with one. So here the rule chose badly. I come back to this later."
- If the one-record answer looks good live, say instead: "Here the short answer looks fine. But on average this rule is weak, as I will show."

### Step 6. 4:20-5:10. SLIDE 5, then SLIDE 6 (speech results)
**SLIDE 5. SAY:** "Now the speech results. On the same GPU, in the same job, PyTorch needs 1.1 seconds for the test recording and our kernels 2.7. Both transcripts are exact. So ours is 2.45 times slower. The profile shows why: 86 percent of the GPU time is the matrix multiply, 13 percent attention, and everything else under one percent."

**SLIDE 6. SAY:** "Our tiled matrix multiply is nine times faster than a naive one, because each value is loaded once into shared memory and reused. But cuBLAS is six times faster again, because it keeps several results per thread in registers. That is the gap to PyTorch."

### Step 7. 5:10-7:40. SLIDES 7 to 11 (answering results)
**SLIDE 7. SAY:** "Now the answering side. How many records should the answer model read, and what does more cost? Every question was transcribed once and cached, so only k, the number of records, changes."

**SLIDE 8. SAY:** "More records find more of the right ones. Recall goes from 0.23 with one record to 0.42 with five and 0.58 with twenty. The search itself takes under a millisecond, whatever k is. But 12 percent of questions find no right record even in the top hundred."

**SLIDE 9. SAY:** "But do the answers get better? A judge model scored the answers from zero to one, on 55 questions from two splits. The scores are 0.27, 0.29 and 0.27 for one, five and twenty records. The differences are tiny and inside the noise. Our sample would likely miss differences below about 0.1. But the prompt grows fifteen times, and the answer takes longer."

**SLIDE 10. SAY:** "Why? We grouped the questions by the rank of their first right record. If it is already first, extra records seem to distract: the score falls from 0.45 to 0.39, but not significantly. If it is lower, extra records bring it in: 0.21 rises to 0.34, and pooled over both splits this gain is significant. The two effects largely net out. The groups are small and chosen after the fact, so this is an explanation, not proof."

**SLIDE 11. SAY:** "We tried to use this. Our rule picks one record when the top two scores are far apart, and twenty otherwise. We set it on practice data and tested it on held-out dev data. It does not help: 0.27, against 0.32 for plain five records, with three times the tokens. The rule is weak: 27 to 34 percent of its confident picks have the wrong top record. Even a perfect rule would gain little: a quarter of questions have no right record in the top twenty."

### Step 8. 7:40-8:40. SLIDE 12. Start with the words "Question 1".
**SAY:** "Question 1: our main design choice. We tuned only the kernels that the profile said matter. With the handout settings, the matrix multiply took 84 percent of the GPU time and attention 16. Everything else was under one percent, so even making it free would save almost nothing. So we tuned only those two: tile 32 instead of 16, and 16 queries per attention block instead of 4. That made the model 17 percent faster, and the transcript stayed exact. The cost: one tile size does not fit every shape. The one-row output layer is slower with tile 32, so a tile size per shape is the next step."

### Step 9. 8:40-9:40. SLIDE 13. Start with the words "Question 2".
**SAY:** "Question 2: how we checked our work. For the kernels we used four checks. The course tests compare every kernel with a reference, including edge cases. The memory and race checker found no errors. The full model must give the exact transcript. And we repeated timings on different nodes: our times agreed within about one percent, so we only trusted bigger differences. Two examples. The fast-math compiler flag looked like free speed, but a RoPE test failed, so we rejected it. And in the answering study, after 15 questions, twenty records looked clearly better. With 13 more questions the gain disappeared, so we withdrew that claim."

## 4. If something goes wrong

- **A command prints nothing or stops early.** Look at the error off camera: `tail -n 20 /tmp/asr_log.txt` (step 4c) or `tail -n 20 /tmp/answer_log.txt` (step 5). The `2> ...` part only keeps loading messages and warnings off the screen.
- **The GPU step fails or the cluster is down on the day.** Skip 4c and 4d. On the laptop run:
  ```bash
  python rag/extensions/demo_query.py answer --transcript "How can I determine the minimum due payment and due date for my ANZ Visa credit card balance?" --query-id q_a02dccd42055bf0e --k 5 2> /tmp/answer_log.txt
  ```
  Then say: "This transcript was made earlier by the same speech model on the GPU." The ASR row of the table then says "not run".
- **`identical: False`.** Read it as it is, and do not hide it.
- **The answer model fails (ELM error).** Wait a minute and try once more. Do not retry in a loop.
- **You run over 10 minutes.** Shorten slide 10 first, then slide 7. Never shorten Question 1 or Question 2.

## 5. After recording

1. Watch the whole file once, with sound.
2. Check that it is at most 10:00 long, and that the last two minutes are Question 1 and Question 2, each introduced by name.
3. Check that nothing identifying is on screen at any time: no user name, host name, home path, e-mail, notification, or `.env` file.
4. Export it as `video.mp4` (H.264 video and AAC audio). Clear the title and author fields in the file's metadata. Upload it to Media Hopper Create (not directly to Learn), with a neutral title and display name, and embed it in the Learn submission (see `docs/TODO.md`, section 4). If the file is too big, re-encode it with HandBrake (preset Fast 1080p30).
