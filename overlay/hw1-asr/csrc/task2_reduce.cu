// Task 2, part B: row-wise kernels built on the helpers in task2_reduce.cuh.
// Read docs/task2-reduce.md first.
//
// All three kernels work on a [rows, cols] matrix and use one block per row.
// The threads of a block walk along the row with a block-stride loop:
//
//     for (int c = threadIdx.x; c < cols; c += blockDim.x) ...
//
// so thread 0 handles columns 0, 256, 512, ..., thread 1 handles 1, 257, ...
// Neighbouring threads read neighbouring addresses, and any cols works.
#include "asr_kernels.h"
#include "task2_reduce.cuh"

// Tuning constant. Override at build time for ablations: -DASR_REDUCE_THREADS=512
#ifndef ASR_REDUCE_THREADS
#define ASR_REDUCE_THREADS 256
#endif
constexpr int REDUCE_THREADS = ASR_REDUCE_THREADS;  // threads per block: a multiple of 32, at most 1024

// Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
                                 float* y, int cols, float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;  // this block's row
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float sum = 0.0f;  // 1. mean
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += xr[c];
  const float mean = block_reduce_sum(sum) / cols;

  float sq = 0.0f;   // 2. variance
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float d = xr[c] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_reduce_sum(sq) / cols + eps);  // 1 / sqrt(var + eps)

  for (int c = threadIdx.x; c < cols; c += blockDim.x)  // 3. normalise and write
    yr[c] = (xr[c] - mean) * rstd * weight[c] + bias[c];
}

// RMSNorm (text decoder): y = x / sqrt(mean(x^2) + eps) * weight
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, int cols,
                               float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  // ======================== BLANK 2.5 =========================
  // RMSNorm of this row: y = x / sqrt(mean(x^2) + eps) * weight.
  float sq = 0.0f;  // this thread's share of sum(x^2)
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sq += xr[c] * xr[c];
  #ifdef ASR_ABL_RMS_ATOMIC
  // Ablation (Try this 4): every thread atomically adds into one shared float
  // instead of the shuffle-based block reduction.
  __shared__ float total;
  if (threadIdx.x == 0) total = 0.0f;
  __syncthreads();
  atomicAdd(&total, sq);
  __syncthreads();
  const float sum_sq = total;
  #else
  const float sum_sq = block_reduce_sum(sq);
  #endif
  const float rstd = rsqrtf(sum_sq / cols + eps);  // 1 / sqrt(mean(x^2) + eps)
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * rstd * weight[c];
  // ============================================================
}

// Softmax over each row: y = exp(x - max) / sum(exp(x - max)).
// The textbook version reads the row three times (max, sum, write). With the
// online-softmax MaxSum from part A, the first two passes become one.
__global__ void softmax_kernel(const float* x, float* y, int cols) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float row_max = 0.0f;  // max of the whole row
  float row_sum = 1.0f;  // sum of exp(x - row_max) over the whole row
  // ======================== BLANK 2.6 =========================
  // Pass 1: set row_max and row_sum (the sum of exp(x - row_max) over the row),
  // reading x only once, with MaxSum.
  #ifdef ASR_ABL_SOFTMAX_3PASS
  // Ablation: the textbook version reads the row once for the max and once more for the sum.
  float mx = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) mx = fmaxf(mx, xr[c]);
  row_max = block_reduce_max(mx);
  float sm = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sm += expf(xr[c] - row_max);
  row_sum = block_reduce_sum(sm);
  #else
  // One read: each thread keeps the MaxSum summary of its own columns ...
  MaxSum t = {-INFINITY, 0.0f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) t = maxsum_merge(t, MaxSum{xr[c], 1.0f});
  // ... then the block agrees on the max and rescales every thread's sum to it.
  // A thread with no columns has {-inf, 0}: its term is 0 * exp(-inf) = 0.
  row_max = block_reduce_max(t.m);
  row_sum = block_reduce_sum(t.l * expf(t.m - row_max));
  #endif
  // ============================================================

  // ======================== BLANK 2.7 =========================
  // Pass 2: y = exp(x - row_max) / row_sum.
  const float inv_sum = 1.0f / row_sum;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = expf(xr[c] - row_max) * inv_sum;
  // ============================================================
}

ASR_API int launch_rmsnorm(const float* x, const float* weight, float* y, int rows,
                           int cols, float eps, cudaStream_t stream) {
  rmsnorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_layernorm(const float* x, const float* weight, const float* bias,
                             float* y, int rows, int cols, float eps, cudaStream_t stream) {
  layernorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, bias, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_softmax(const float* x, float* y, int rows, int cols,
                           cudaStream_t stream) {
  softmax_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, y, cols);
  RETURN_LAUNCH_ERROR();
}
