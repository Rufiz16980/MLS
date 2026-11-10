// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include "asr_kernels.h"
#include "common.cuh"

// Tuning constants. Override at build time for ablations: -DASR_TILE=32, -DASR_WS_PAD=0
#ifndef ASR_TILE
#define ASR_TILE 16
#endif
#ifndef ASR_WS_PAD
#define ASR_WS_PAD 1
#endif
constexpr int TILE = ASR_TILE;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + ASR_WS_PAD];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ======================== BLANK 3.1 =========================
    // Load this thread's element of the x tile into xs. Use 0 outside x.
    {
      const int kx = k0 + tx;  // column of x this thread loads
      xs[ty][tx] = (row < M && kx < K) ? x[static_cast<size_t>(row) * K + kx] : 0.0f;
    }
    // ============================================================

    // ======================== BLANK 3.2 =========================
    // Load this thread's element of the w tile, so that ws[n][k] holds
    // w[blockIdx.x * TILE + n][k0 + k]. Use 0 outside w. Threads with consecutive
    // tx must read consecutive addresses of w.
    {
    #ifdef ASR_ABL_W_STRIDED
      // Ablation (Try this 3), the lecture's way of loading B: thread (ty, tx) loads
      // w row (n = tx), element k = ty, so neighbouring threads read addresses K floats apart.
      const int wn = blockIdx.x * TILE + tx;
      const int wk = k0 + ty;
      ws[tx][ty] = (wn < N && wk < K) ? w[static_cast<size_t>(wn) * K + wk] : 0.0f;
    #else
      // Coalesced: thread (ty, tx) loads w row n = ty, element k = tx, so a warp's
      // consecutive tx read consecutive addresses of one row of w.
      const int wn = blockIdx.x * TILE + ty;
      const int wk = k0 + tx;
      ws[ty][tx] = (wn < N && wk < K) ? w[static_cast<size_t>(wn) * K + wk] : 0.0f;
    #endif
    }
    __syncthreads();  // both tiles are complete before anyone reads them
    // ============================================================

    // ======================== BLANK 3.3 =========================
    // Add this K step's contribution to acc.
    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += xs[ty][kk] * ws[tx][kk];  // x[row][k] * w[col][k]
    #ifndef ASR_ABL_NO_SECOND_SYNC  // ablation only (Try this 4): removes the barrier, expect a race
    __syncthreads();  // everyone is done with this tile before the next loop iteration overwrites it
    #endif
    // ============================================================
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: for outputs inside y, add bias[col] (bias may be nullptr), apply
  // gelu() when act == ACT_GELU, and store.
  if (row < M && col < N) {
    float r = acc + (bias != nullptr ? bias[col] : 0.0f);
    if (act == ACT_GELU) r = gelu(r);
    y[static_cast<size_t>(row) * N + col] = r;
  }
  // ============================================================
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
  RETURN_LAUNCH_ERROR();
}
