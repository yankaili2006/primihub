/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA implementation of the SpiralPIR GSW external product + Galois
 * automorphism (task 2.1). See spiral_cuda_kernels.h.
 */
#include "spiral_cuda_kernels.h"

#include <cuda_runtime.h>

namespace primihub::pir::spiral::cuda {

namespace {

// out[r][c][i] = sum_k gsw[r][k][c][i] * decomp[k][c][i]  mod moduli[c].
// One thread per (r, c, i). gsw/decomp coeffs are < their modulus (< ~2^28), so
// each product is < ~2^56 and the running reduction keeps acc < q < 2^28.
__global__ void GswExtKernel(std::uint64_t* out, const std::uint64_t* gsw,
                             const std::uint64_t* decomp,
                             const std::uint64_t* moduli, std::size_t rows_k,
                             std::size_t crt_count, std::size_t poly_len) {
  const std::size_t total = 2 * crt_count * poly_len;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const std::size_t i = idx % poly_len;
  const std::size_t c = (idx / poly_len) % crt_count;
  const std::size_t r = idx / (poly_len * crt_count);
  const std::uint64_t q = moduli[c];

  std::uint64_t acc = 0;
  for (std::size_t k = 0; k < rows_k; ++k) {
    const std::uint64_t g = gsw[((r * rows_k + k) * crt_count + c) * poly_len + i];
    const std::uint64_t d = decomp[(k * crt_count + c) * poly_len + i];
    acc = (acc + (g % q) * (d % q)) % q;
  }
  out[(r * crt_count + c) * poly_len + i] = acc;
}

// out[c][i] = in[c][table[i]].  One thread per (c, i).
__global__ void GaloisKernel(std::uint64_t* out, const std::uint64_t* in,
                             const std::size_t* table, std::size_t crt_count,
                             std::size_t poly_len) {
  const std::size_t total = crt_count * poly_len;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const std::size_t i = idx % poly_len;
  const std::size_t c = idx / poly_len;
  out[c * poly_len + i] = in[c * poly_len + table[i]];
}

constexpr int kThreads = 256;

}  // namespace

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void GswExternalProductNtt(std::uint64_t* out, const std::uint64_t* gsw,
                           const std::uint64_t* decomp,
                           const std::uint64_t* moduli, std::size_t rows_k,
                           std::size_t crt_count, std::size_t poly_len) {
  const std::size_t out_n = 2 * crt_count * poly_len;
  const std::size_t gsw_n = 2 * rows_k * crt_count * poly_len;
  const std::size_t dec_n = rows_k * crt_count * poly_len;

  std::uint64_t *d_out, *d_gsw, *d_dec, *d_mod;
  cudaMalloc(&d_out, out_n * sizeof(std::uint64_t));
  cudaMalloc(&d_gsw, gsw_n * sizeof(std::uint64_t));
  cudaMalloc(&d_dec, dec_n * sizeof(std::uint64_t));
  cudaMalloc(&d_mod, crt_count * sizeof(std::uint64_t));
  cudaMemcpy(d_gsw, gsw, gsw_n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dec, decomp, dec_n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mod, moduli, crt_count * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  const int blocks = static_cast<int>((out_n + kThreads - 1) / kThreads);
  GswExtKernel<<<blocks, kThreads>>>(d_out, d_gsw, d_dec, d_mod, rows_k,
                                     crt_count, poly_len);
  cudaDeviceSynchronize();

  cudaMemcpy(out, d_out, out_n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  cudaFree(d_gsw);
  cudaFree(d_dec);
  cudaFree(d_mod);
}

void ApplyGaloisNtt(std::uint64_t* out, const std::uint64_t* in,
                    const std::size_t* table, std::size_t crt_count,
                    std::size_t poly_len) {
  const std::size_t n = crt_count * poly_len;
  std::uint64_t *d_out, *d_in;
  std::size_t* d_tab;
  cudaMalloc(&d_out, n * sizeof(std::uint64_t));
  cudaMalloc(&d_in, n * sizeof(std::uint64_t));
  cudaMalloc(&d_tab, poly_len * sizeof(std::size_t));
  cudaMemcpy(d_in, in, n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tab, table, poly_len * sizeof(std::size_t), cudaMemcpyHostToDevice);

  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  GaloisKernel<<<blocks, kThreads>>>(d_out, d_in, d_tab, crt_count, poly_len);
  cudaDeviceSynchronize();

  cudaMemcpy(out, d_out, n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  cudaFree(d_in);
  cudaFree(d_tab);
}

}  // namespace primihub::pir::spiral::cuda
