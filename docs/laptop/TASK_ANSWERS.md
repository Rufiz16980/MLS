# Draft answers to the task-guide questions (hw1-asr, tasks 1 to 4) — FOR THE OWNER TO REVIEW AND UNDERSTAND

Status: DRAFT written by the assistant from the submitted kernel code and the measurements in `docs/EXPERIMENT_LOG.md` (entries 13 to 23). The owner must
be able to explain every line (oral Q&A) and should re-derive each number. Items marked **[measured]** come from our runs; **[derived]** are arithmetic from shapes;
**[prediction]** are reasoning that we did not test. Hardware: RTX 2080 Ti, 68 SMs, about 616 GB/s peak bandwidth. Submitted settings: 256 threads (RoPE, reductions),
GEMM tile 32, 16 attention warps (handout values 256 / 16 / 4).

## Task 1: RoPE (`task1_rope.cu`)

**Q1. Audio-encoder case: bytes read and written, and the share of peak bandwidth.**
Case: batch 1, 1500 tokens, 20 q heads and 20 k heads, head_dim 64, rotary_dim 32 (test shape). Each pair `(x[i], x[i+16])` is read (2 x 4 B) and written (2 x 4 B) = 16 B.
Pairs = (q rows + k rows) x half = (1500 x 20 + 1500 x 20) x 16 = 960,000, so total = 960,000 x 16 B = **15.36 MB** [derived; the test uses the same formula,
2 x (q_rows + k_rows) x rotary_dim x 4 B]. In 0.031 ms this is **491.8 GB/s = 80% of 616 GB/s** [measured, entry 21]. (`inv_freq` is 16 floats, negligible and cached;
the 32 unrotated elements per row are never touched.)

**Q2. A precomputed cos/sin table: faster or slower? Count bytes.**
A `[1500, 16]` table of cos and sin is 2 x 1500 x 16 x 4 B = 192 KB. The kernel would read 2 more floats per pair (cos, sin): 8 B more on top of 16 B, i.e. 24 B of load/store
traffic per pair instead of 16 (+50% requested bytes). Rows of the same position repeat across 20 heads and q/k, so most table reads hit cache and DRAM traffic grows by only about
192 KB (1.2% of 15.36 MB). But the kernel is already at 80% of peak bandwidth and the `sincosf` work is hidden behind memory latency, so the table cannot make it noticeably faster
and the extra requests make it equal or slightly slower [derived; not measured]. It would pay off only if the transcendental functions were the bottleneck.

**Q3. Why rotate q and k in one launch (396 calls per transcription)?**
Two launches would double the launches (792 instead of 396). Small cases take about 0.004 ms, which is the launch-and-latency floor (entry 21: "tiny" cases at 0.004 ms), so launch
overhead is as large as the useful work for decoder-sized inputs. One launch also gives a larger grid (q pairs then k pairs), which fills more of the 68 SMs; the text decoder case
has only 64 tokens, too few threads from q alone. In our design a thread index below `q_pairs` selects q and the rest selects k (Blanks 1.1/1.2).

**Try this results (entry 14):** threads 32 to 1024 barely change the audio-encoder case (0.031 to 0.038 ms; 32 threads slowest: tiny blocks leave the SM's warp slots underused);
tiny cases cost the same whatever the size (launch overhead); `--use_fast_math` fails the accuracy test on the audio-encoder case [measured: FAIL]. Likely reason [hypothesis, check the guide's hint about the test's angles]: the angle `pos x inv_freq` is large there (positions up to 1,500), and the hardware `__sinf/__cosf` approximations lose accuracy for large arguments.

## Task 2: reductions, RMSNorm, softmax (`task2_reduce.cuh`, `task2_reduce.cu`)

**Q1. A row for which `block_reduce_max` is wrong if missing warps load 0 instead of -INFINITY.**
With 256 threads there are 8 warps, so in the final step lanes 8 to 31 of each warp have no warp maximum to load. If they load 0, the result is max(true max, 0).
Any row with all values negative, e.g. `x = [-5, -3, -7, ...]` (true max -3), returns **0**. This is the "softmax, all negative (-200..-150)" test case.

**Q2. `expf(100.0f)` in float32, and how subtracting the max keeps softmax correct.**
e^100 is about 2.7e43, above the float32 maximum of about 3.4e38, so `expf(100.0f)` = **+infinity**; the sum is infinity and infinity/infinity = NaN.
Subtracting the row maximum makes every exponent <= 0, so every term is in (0, 1] and the maximum element gives exactly 1: no overflow. The result is unchanged because
exp(x - m) / sum exp(x - m) = (e^x e^-m) / (e^-m sum e^x) = e^x / sum e^x: the common factor e^-m cancels.

**Q3. Reads of each element of `x` from global memory.**
Our softmax reads each element **twice**: once in pass 1 (a single online max-and-sum, `MaxSum`) and once in pass 2 (`exp(x - max) / sum`), plus one write of `y`.
The textbook version reads three times (max, sum, write). The 3-pass ablation measured 0.150 ms against 0.118 ms for the vocabulary row, 27% slower [measured, entry 14].

**Q4. What goes wrong in LayerNorm without the final `__syncthreads()` in `block_reduce_sum`?**
LayerNorm calls `block_reduce_sum` twice in a row (mean, then variance), and both calls use the same shared array `partial[]`. Sequence [prediction; the race stage was not run]:
(1) warp 0 and warp 1 store their partial sums; (2) both wait at the barrier; (3) warp 0 is fast: it reads all `partial[]`, finishes the first call, computes squared differences
and enters the second call, where lane 0 writes its new partial sum into `partial[0]`; (4) warp 1 is still slow in step 3 of the first call and now reads `partial[0]`, which already holds
the second call's value; (5) warp 1 gets a wrong mean, different warps disagree on the mean, and the output is wrong or non-deterministic. `compute-sanitizer --tool racecheck` would report a
write-after-read hazard on `partial[]`. The final barrier makes every thread finish reading before anyone may write the array again.

**Try this results (entry 14):** more threads help the long vocabulary row (59,264 floats: 1024 threads 0.041 ms vs 0.118 ms at 256) but not short rows (LayerNorm over 1,280 columns:
128 threads 0.042 ms, 1024 threads 0.076 ms; many threads idle); the vocabulary row uses a single block, so only 1 of 68 SMs works (1% of peak bandwidth) [derived + measured];
an `atomicAdd` RMSNorm is 3.7x slower than the shuffle reduction (0.022 vs 0.006 ms; all threads serialise on one address) [measured].

## Task 3: tiled GEMM (`task3_gemm.cu`)

**Q1. M = N = K = 1024: floats loaded from global memory, naive vs tiled.**
Naive: every output reads a row of x and a row of w, 2K floats, so 1024 x 1024 x 2048 = **2,147,483,648 floats** (about 2.1 billion, 8.6 GB). Tiled with TILE = 32 (our setting): each block
(32 x 32 outputs) loads two 32 x 32 tiles per K step, 2,048 floats, over K/32 = 32 steps = 65,536 floats per block, times (1024/32)^2 = 1,024 blocks = **67,108,864 floats**
(2^26, about 67 million), 32 times fewer, i.e. a factor equal to the tile width [derived]. (With the handout TILE = 16 the factor is 16: 134 million.)

**Q2. One thing cuBLAS does differently.**
**Register tiling**: each thread computes a small block of outputs (for example 4 x 4 or 8 x 8) held in registers, so each value read from shared memory feeds many multiply-adds. Our kernel does
2 shared-memory loads for every one FMA, which limits it to about 2.1 TFLOP/s; cuBLAS reaches about 13.3 TFLOP/s, near the fp32 peak [measured, entry 21]. (Others: 128-bit vector loads, double buffering, tuning per GPU.)

**Q3. Traffic saved by fusing GELU into the `fc1` epilogue.**
`fc1` output: 1500 x 5120 floats = 7.68 million floats = 30.72 MB. A separate GELU kernel would read y (30.72 MB) and write it back (30.72 MB): **61.44 MB** saved, about 0.1 ms at 616 GB/s [derived],
about 1% of the 9.3 ms `fc1` call. The encoder has 32 layers, so about 3.2 ms per transcription, roughly 0.1% of 2,700 ms.

**Q4. Why must every thread reach both `__syncthreads()`, even outside y?**
A barrier completes only when all threads of the block arrive; a thread that returned early would make the others wait forever (hang or undefined behaviour). Also, threads outside y still have a job:
each thread loads one element of the x tile and one of the w tile (0 outside the matrices), and other threads read those shared-memory elements; a thread that skips the load leaves part of the tile missing.

**Try this results (entry 14):** tile 32 is 35% faster than 16 on the large encoder GEMMs (14.2 to 9.3 ms for fc1) but 78% slower on `lm_head` (M = 1: one useful row out of 32) where tile 8 wins (1.54 vs 2.72 ms); removing
the +1 padding of `ws` makes the large cases about 2.2x slower (bank conflicts); loading W the lecture's way (strided) costs about 41% [measured]. `lm_head` reads the 485 MB weight matrix, so GB/s is the useful
metric: 485 MB / 3.845 ms = 126 GB/s, about 20% of peak [derived from entry 21].

## Task 4: attention (`task4_attention.cu`)

**Q1. Memory of the full score matrix of one encoder layer, and shared memory per block here.**
20 x 1500 x 1500 x 4 B = **180,000,000 B (about 180 MB)** [derived]. One block of our kernel uses `k_tile` 32 x (64+1) floats + `v_tile` 32 x 64 + `q_all` 16 x 64 (16 warps) = 2,080 + 2,048 + 1,024 =
5,152 floats = **20,608 B (about 20 KiB)**; with the handout 4 warps: 17,536 B [derived]. So attention never materialises the matrix; the streaming tile is about 8,700 times smaller.

**Q2. Why must all threads of a block run the same number of tile iterations?**
Each iteration contains `__syncthreads()` (all threads cooperatively load the key/value tile). If a warp stopped at its own `last_key`, it would skip barriers that the other warps still wait at: deadlock or undefined behaviour,
and the tile would no longer be loaded by everyone. So the loop runs to the block's last needed key (`block_last_key`), and a warp's keys past its `last_key` get score -infinity and probability 0.

**Q3. What is `corr`? When can `m_new` be -INFINITY?**
`corr = exp(m - m_new)` rescales the running sum `l` and the accumulator `acc` when a new maximum arrives. If the tile does not raise the maximum then `m_new = m` and **`corr = exp(0) = 1`**, nothing is rescaled.
`m_new = -INFINITY` only while a query has seen no visible key at all, e.g. a spare warp in the last block (`last_key = -1`); then exp(-inf - -inf) would be NaN, so the code sets `corr = 1` and `p = 0`.

**Q4. Key loads per encoder head with `ATTN_WARPS = 4`, and the limit on more warps.**
A block handles 4 consecutive queries and streams all 1,500 keys, so each key of one head is loaded once per block: ceil(1500 / 4) = **375 times** per layer (and 94 with our 16 warps; 750, 188 for 2 and 8; 1,500 for 1 warp) [derived].
That follows the measurements: audio encoder case 103.6 / 30.1 / 11.5 / 8.34 / 7.58 ms for 1 / 2 / 4 / 8 / 16 warps [measured, entry 14]. More warps are limited by the 1,024-thread block maximum (32 warps),
by shared memory and registers per block (occupancy), and by having enough blocks to keep all 68 SMs busy (1,500 queries / 32 = 47 blocks per head, times 20 heads, is still plenty, but each block must also stay small
enough that several fit per SM); returns diminish once key reloads are no longer the bottleneck (16 warps was fastest of those tried).

**Q5. Why does `acc` have `MAX_DIMS_PER_LANE` entries though only `dims` are used? Where can the compiler keep it?**
The array length and the loops over it are compile-time constants (`MAX_DIMS_PER_LANE = 4`, `#pragma unroll`, with `if (t < dims)` inside), so every index is known at compile time and the compiler can keep `acc[0..3]`
in registers. With a run-time length (`dims` is only known at launch) it could not unroll fully, and an array indexed dynamically is placed in **local memory** (per-thread memory backed by the slow off-chip memory system,
cached in L1), which would make every update a memory access.

**Try this results (entries 14 to 16):** warps 8 and 16 are 27 to 34% faster on the encoder and 3 times faster in the small decoder case, but only 3% faster end to end (attention is 13% of the time; Amdahl's law);
removing the +1 padding of `k_tile` makes attention 4.7x slower (bank conflicts: with head_dim 64 and pitch 64, all 32 lanes reading element d of their own row hit the same bank, a 32-way conflict; with pitch 65 they hit 32 different banks) [measured + derived].
Attention against PyTorch's: use `./run.sh --cuda task1,task2,task3 --profile` (profile with PyTorch attention) versus `--cuda all` — numbers in entry 16 (attention 267.8 ms PyTorch vs 544.9 ms ours with the handout settings, profile from another node, so use the ratio only; our submitted settings give 378.3 ms, entry 22).

## What the owner should be able to explain without notes
1. Why RoPE is memory-bound and what 80% of peak means (Task 1 Q1).
2. Why subtracting the max works (Task 2 Q2) and the reduction in two levels (warp shuffle, then shared memory).
3. Why tiling cuts loads by the tile width and why register tiling would go further (Task 3 Q1 and Q2).
4. Online softmax in attention: `m`, `l`, `corr`, and why the score matrix is never stored (Task 4 Q1, Q3).
5. The ablation story: GEMM is 86% of time (Amdahl), padding matters (bank conflicts), best tile depends on the shape.
