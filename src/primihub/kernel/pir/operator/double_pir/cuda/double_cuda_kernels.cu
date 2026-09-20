/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA implementation of the DoublePIR LWE matrix multiply (task 2.2).
 * See double_cuda_kernels.h.
 */
#include "double_cuda_kernels.h"

#include <cuda_runtime.h>

namespace primihub::pir::doublepir::cuda {

namespace {

constexpr int kTile = 16;

// C = A * B mod 2^32. Tiled shared-memory matmul; uint32 accumulation wraps.
__global__ void MatMulMod2Pow32Kernel(std::uint32_t* c, const std::uint32_t* a,
                                       const std::uint32_t* b, int rows,
                                       int inner, int cols) {
  __shared__ std::uint32_t as[kTile][kTile];
  __shared__ std::uint32_t bs[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  std::uint32_t acc = 0;
  const int tiles = (inner + kTile - 1) / kTile;
  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * kTile + threadIdx.x;
    const int b_row = t * kTile + threadIdx.y;
    as[threadIdx.y][threadIdx.x] =
        (row < rows && a_col < inner) ? a[row * inner + a_col] : 0u;
    bs[threadIdx.y][threadIdx.x] =
        (b_row < inner && col < cols) ? b[b_row * cols + col] : 0u;
    __syncthreads();
    for (int k = 0; k < kTile; ++k)
      acc += as[threadIdx.y][k] * bs[k][threadIdx.x];  // wraps mod 2^32
    __syncthreads();
  }
  if (row < rows && col < cols) c[row * cols + col] = acc;
}

// Specialized matrix-vector path (cols == 1): the DoublePIR Answer hot path.
// Warp-per-row + coalesced uint4 loads + grid-stride + warp shuffle-reduce —
// far higher memory throughput than the generic tiled matmul for a single
// column (which wastes a 16x16 tile on one output element per row). Mirrors the
// bench kernel in bench/cuda_vs_avx2_bench.cu. b is the length-`inner` vector.
__global__ void MatVecMod2Pow32Kernel(std::uint32_t* c, const std::uint32_t* a,
                                      const std::uint32_t* b, std::size_t rows,
                                      std::size_t inner) {
  const unsigned lane = threadIdx.x & 31u;
  const std::size_t warps_per_block = blockDim.x >> 5;
  const std::size_t warp_id =
      static_cast<std::size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
  const std::size_t total_warps =
      static_cast<std::size_t>(gridDim.x) * warps_per_block;
  const bool vec_ok = (inner % 4u == 0u);

  for (std::size_t row = warp_id; row < rows; row += total_warps) {
    const std::uint32_t* arow = a + row * inner;
    std::uint32_t acc = 0;
    if (vec_ok) {
      const uint4* a4 = reinterpret_cast<const uint4*>(arow);
      const uint4* b4 = reinterpret_cast<const uint4*>(b);
      const std::size_t n4 = inner >> 2;
      for (std::size_t j = lane; j < n4; j += 32) {
        uint4 av = a4[j], bv = b4[j];
        acc += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
      }
    } else {
      for (std::size_t k = lane; k < inner; k += 32) acc += arow[k] * b[k];
    }
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) c[row] = acc;
  }
}

}  // namespace

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void LweMatMulMod2Pow32(std::uint32_t* c, const std::uint32_t* a,
                        const std::uint32_t* b, std::size_t rows,
                        std::size_t inner, std::size_t cols) {
  const std::size_t a_n = rows * inner, b_n = inner * cols, c_n = rows * cols;
  std::uint32_t *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, a_n * sizeof(std::uint32_t));
  cudaMalloc(&d_b, b_n * sizeof(std::uint32_t));
  cudaMalloc(&d_c, c_n * sizeof(std::uint32_t));
  cudaMemcpy(d_a, a, a_n * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_n * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

  if (cols == 1) {
    // Answer hot path: dispatch the tuned warp-per-row matvec.
    const int t = 256, wpb = t / 32;
    long want = long((rows + wpb - 1) / wpb);
    const int b = int(want > 65535 ? 65535 : want);
    MatVecMod2Pow32Kernel<<<b, t>>>(d_c, d_a, d_b, rows, inner);
  } else {
    const dim3 threads(kTile, kTile);
    const dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    MatMulMod2Pow32Kernel<<<blocks, threads>>>(d_c, d_a, d_b,
                                               static_cast<int>(rows),
                                               static_cast<int>(inner),
                                               static_cast<int>(cols));
  }
  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, c_n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}


// Packed squished-DB matrix-vector (basis=10, squishing=3). Warp-per-row,
// coalesced row loads, warp shuffle-reduce. Mirrors matMulVecPacked semantics:
// out[i] = sum_j sum_{s<3} ((a[i*cols+j]>>10s)&1023) * b[3j+s], mod 2^32.
__global__ void PackedMatVecKernel(std::uint32_t* out, const std::uint32_t* a,
                                   const std::uint32_t* b, std::size_t rows,
                                   std::size_t cols) {
  const unsigned lane = threadIdx.x & 31u;
  const std::size_t wpb = blockDim.x >> 5;
  const std::size_t warp_id =
      static_cast<std::size_t>(blockIdx.x) * wpb + (threadIdx.x >> 5);
  const std::size_t total = static_cast<std::size_t>(gridDim.x) * wpb;
  constexpr std::uint32_t kMask = 1023u;  // (1<<BASIS)-1
  for (std::size_t row = warp_id; row < rows; row += total) {
    const std::uint32_t* arow = a + row * cols;
    std::uint32_t acc = 0;
    for (std::size_t j = lane; j < cols; j += 32) {
      const std::uint32_t db = arow[j];
      acc += (db & kMask) * b[3 * j] +
             ((db >> 10) & kMask) * b[3 * j + 1] +
             ((db >> 20) & kMask) * b[3 * j + 2];  // wraps mod 2^32
    }
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
  }
}

void PackedMatVecMod2Pow32(std::uint32_t* out, const std::uint32_t* a,
                           const std::uint32_t* b, std::size_t rows,
                           std::size_t cols) {
  const std::size_t a_n = rows * cols, b_n = 3 * cols;
  std::uint32_t *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, a_n * sizeof(std::uint32_t));
  cudaMalloc(&d_b, b_n * sizeof(std::uint32_t));
  cudaMalloc(&d_out, rows * sizeof(std::uint32_t));
  cudaMemcpy(d_a, a, a_n * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_n * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
  const int t = 256, wpb = t / 32;
  long want = long((rows + wpb - 1) / wpb);
  const int blocks = int(want > 65535 ? 65535 : (want < 1 ? 1 : want));
  PackedMatVecKernel<<<blocks, t>>>(d_out, d_a, d_b, rows, cols);
  cudaDeviceSynchronize();
  cudaMemcpy(out, d_out, rows * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

}  // namespace primihub::pir::doublepir::cuda
