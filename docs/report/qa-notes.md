# Notes for the Q&A (kept up to date as we go)

Purpose: after the results are in, this is the study sheet for answering questions about the design. Numbers live in `experiment-log.md` (entry ids in brackets). No personal data here.

## Ground rules we are working under
- AI policy (given verbally by the instructors): generative AI is encouraged for study and for the assignment. What matters is that each person understands the design and purpose of everything submitted and can answer questions about it. The AI-use declaration in the report must say that Claude wrote the kernels and the scripts.
- The kernels and the extension scripts were written with Claude. Each person must be able to explain every blank below without notes.

## Decisions so far (what, why)
1. No model training (not required; Plan A is training-free).
2. Plan A is the lead RAG question: how far can retrieved evidence (k) be cut before answer quality drops, and what does it save in context size and time. The course README names this very direction as an example. B (enc_a to enc_b mapping) and C (ASR errors versus retrieval) come after A; one of them goes into the report if time allows.
3. ELM (the university's hosted LLM, reached with a personal key) writes the answers and judges them. The README says to apply for a key and that generation and evaluation make ELM requests. The key stays on the laptop only. API budget: start with 90 calls (15 queries x k = 1, 5, 20 x generate + judge), extend to 180 total only if the result is unclear.
4. Our files live in `overlay/`; a guard outside the course repo checks every run that the supplied files are unchanged.
5. ASR is run once on the GPU and cached; every RAG experiment reuses the same transcripts (controlled comparison).
6. Kernel ablations are build-time flags (`-D...`), so one command builds and times every variant.
7. Stop kernel tuning after the combined best configuration (tile 32 + 16 attention warps, 17.2% faster, K07): speed is not marked and the ablations already tell the story.

## The four kernels in three sentences each (be able to say these)
- **Task 1, RoPE.** Each token's query/key head vector is rotated pair by pair: element i and i + half turn by the angle position x inv_freq[i]. One thread per pair, q's pairs first and then k's, in one launch. It is memory-bound: 80% of the GPU's bandwidth on the encoder case (K02), and fast-math sin/cos fails the accuracy test on the largest angles.
- **Task 2, reductions.** A warp adds 32 values with `__shfl_xor_sync` (five butterfly steps, no memory); a block adds warp results through shared memory, and the last `__syncthreads` protects the shared array when the function is called again. Max uses -infinity as the empty value. Softmax keeps a running (max, sum) summary that can be merged, so it reads the row once instead of twice; RMSNorm is one sum of squares.
- **Task 3, tiled GEMM.** A block computes a TILE x TILE piece of y; it walks along K one tile at a time, copying a tile of x and a tile of w into shared memory so every loaded value is reused TILE times. Bias and GELU are applied before storing, saving one read and one write of y. The +1 padding of the w tile avoids shared-memory bank conflicts (1.7 to 2.3x faster than without, K03).
- **Task 4, attention.** One warp per query; keys and values go through shared memory 32 at a time, and each query keeps a running max, sum and output, rescaling by exp(old max - new max) whenever the max grows, so the score matrix is never written to memory. Grouped-query attention maps query head h to kv head h / (heads / kv_heads); causal masking limits the last visible key. All threads of a block run the same number of tiles so the barriers are safe.

## What the results say (one line each)
- All four kernels pass the course tests, memcheck and racecheck, and the full model stays at 100% accuracy (K01).
- GEMM is 83.6% of kernel time with our kernels, attention 15.9%, everything else 0.5% (K06): Amdahl's law, only the GEMM is worth tuning.
- Best single changes: GEMM tile 32 (-13%), 16 attention warps (-4%); together -17.2%, additive (K05, K07).
- Padding and coalescing matter more than most tuning: no padding is up to 2.3x (GEMM) and 4.7x (attention) slower (K03, K04).
- PyTorch's own time depends on the node's CPU; our kernel time does not. Quote ratios with the node (K05, K07).
