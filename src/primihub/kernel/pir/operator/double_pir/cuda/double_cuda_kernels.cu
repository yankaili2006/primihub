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

  const dim3 threads(kTile, kTile);
  const dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  MatMulMod2Pow32Kernel<<<blocks, threads>>>(d_c, d_a, d_b,
                                             static_cast<int>(rows),
                                             static_cast<int>(inner),
                                             static_cast<int>(cols));
  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, c_n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}  // namespace primihub::pir::doublepir::cuda
