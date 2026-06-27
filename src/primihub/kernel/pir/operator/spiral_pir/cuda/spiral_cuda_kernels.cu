/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA implementation of the SpiralPIR GSW external product + Galois
 * automorphism (task 2.1). See spiral_cuda_kernels.h.
 */
#include "spiral_cuda_kernels.h"

#include <cuda_runtime.h>

#include <vector>

#include "ntt_device.cuh"

namespace primihub::pir::spiral::cuda {

namespace {

// out[r][c][i] = sum_k gsw[r][k][c][i] * decomp[k][c][i]  mod mods[c].
// One thread per (r, c, i). Uses the Barrett mul/add-mod reused from SIGMA
// (ntt_device.cuh) instead of the hand-rolled `% q`.
__global__ void GswExtKernel(std::uint64_t* out, const std::uint64_t* gsw,
                             const std::uint64_t* decomp, const ModParams* mods,
                             std::size_t rows_k, std::size_t crt_count,
                             std::size_t poly_len) {
  const std::size_t total = 2 * crt_count * poly_len;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const std::size_t i = idx % poly_len;
  const std::size_t c = (idx / poly_len) % crt_count;
  const std::size_t r = idx / (poly_len * crt_count);
  const ModParams m = mods[c];

  std::uint64_t acc = 0;
  for (std::size_t k = 0; k < rows_k; ++k) {
    const std::uint64_t g = gsw[((r * rows_k + k) * crt_count + c) * poly_len + i];
    const std::uint64_t d = decomp[(k * crt_count + c) * poly_len + i];
    acc = d_addmod(acc, d_mulmod(g % m.value, d % m.value, m), m);
  }
  out[(r * crt_count + c) * poly_len + i] = acc;
}

// ---- negacyclic NTT kernels (one block per polynomial) ----

// Coefficient-wise multiply g[i] *= scale[i] mod m  (psi pre/post weighting).
__global__ void WeightKernel(std::uint64_t* data, const std::uint64_t* scale,
                             ModParams m, std::size_t N) {
  std::uint64_t* g = data + static_cast<std::size_t>(blockIdx.x) * N;
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x)
    g[i] = d_mulmod(g[i] % m.value, scale[i], m);
}

// Cyclic radix-2 DIT NTT in shared memory. inv=false: forward with w=omega^k.
// inv=true: inverse with w=omega^{-k} and a final * n_inv. Input is permuted by
// `bitrev` on load so the output is in natural order.
__global__ void NttKernel(std::uint64_t* data, const std::uint64_t* w,
                          const std::uint32_t* bitrev, ModParams m,
                          std::size_t N, std::uint64_t n_inv, bool inv) {
  extern __shared__ std::uint64_t sh[];
  std::uint64_t* g = data + static_cast<std::size_t>(blockIdx.x) * N;
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x) sh[i] = g[bitrev[i]];
  __syncthreads();
  for (std::size_t len = 2; len <= N; len <<= 1) {
    const std::size_t half = len >> 1;
    const std::size_t step = N / len;
    for (std::size_t bf = threadIdx.x; bf < N / 2; bf += blockDim.x) {
      const std::size_t blk = bf / half;
      const std::size_t j = bf % half;
      const std::size_t base = blk * len;
      const std::uint64_t tw = w[step * j];
      const std::uint64_t u = sh[base + j];
      const std::uint64_t v = d_mulmod(sh[base + j + half], tw, m);
      sh[base + j] = d_addmod(u, v, m);
      sh[base + j + half] = d_submod(u, v, m);
    }
    __syncthreads();
  }
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x)
    g[i] = inv ? d_mulmod(sh[i], n_inv, m) : sh[i];
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

  // Barrett params per residue (reused from SIGMA via ntt_device.cuh).
  std::vector<ModParams> mods(crt_count);
  for (std::size_t c = 0; c < crt_count; ++c) mods[c] = ntt_host::make_mod(moduli[c]);

  std::uint64_t *d_out, *d_gsw, *d_dec;
  ModParams* d_mod;
  cudaMalloc(&d_out, out_n * sizeof(std::uint64_t));
  cudaMalloc(&d_gsw, gsw_n * sizeof(std::uint64_t));
  cudaMalloc(&d_dec, dec_n * sizeof(std::uint64_t));
  cudaMalloc(&d_mod, crt_count * sizeof(ModParams));
  cudaMemcpy(d_gsw, gsw, gsw_n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dec, decomp, dec_n * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mod, mods.data(), crt_count * sizeof(ModParams), cudaMemcpyHostToDevice);

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

namespace {

// Run a forward (inv=false) or inverse (inv=true) negacyclic NTT in place over
// data[crt_count][poly_len], each residue mod moduli[c]. Host pointer in/out.
void RunNttCrt(std::uint64_t* data, const std::uint64_t* moduli,
               std::size_t crt_count, std::size_t poly_len, bool inv) {
  const int blockThreads = poly_len >= 1024 ? 512 : 256;
  const std::size_t shmem = poly_len * sizeof(std::uint64_t);
  for (std::size_t c = 0; c < crt_count; ++c) {
    auto t = ntt_host::build_tables(poly_len, moduli[c]);
    std::uint64_t* slice = data + c * poly_len;

    std::uint64_t *d_data, *d_w, *d_psi;
    std::uint32_t* d_br;
    cudaMalloc(&d_data, poly_len * sizeof(std::uint64_t));
    cudaMalloc(&d_w, (poly_len / 2) * sizeof(std::uint64_t));
    cudaMalloc(&d_psi, poly_len * sizeof(std::uint64_t));
    cudaMalloc(&d_br, poly_len * sizeof(std::uint32_t));
    cudaMemcpy(d_data, slice, poly_len * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, (inv ? t.winv.data() : t.w.data()),
               (poly_len / 2) * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi, (inv ? t.psi_inv.data() : t.psi.data()),
               poly_len * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_br, t.bitrev.data(), poly_len * sizeof(std::uint32_t),
               cudaMemcpyHostToDevice);

    if (!inv) {
      // forward: pre-scale by psi^j, then cyclic NTT.
      WeightKernel<<<1, blockThreads>>>(d_data, d_psi, t.mod, poly_len);
      NttKernel<<<1, blockThreads, shmem>>>(d_data, d_w, d_br, t.mod, poly_len,
                                            t.n_inv, false);
    } else {
      // inverse: cyclic inverse NTT (* n_inv), then post-scale by psi^{-j}.
      NttKernel<<<1, blockThreads, shmem>>>(d_data, d_w, d_br, t.mod, poly_len,
                                            t.n_inv, true);
      WeightKernel<<<1, blockThreads>>>(d_data, d_psi, t.mod, poly_len);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(slice, d_data, poly_len * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_w);
    cudaFree(d_psi);
    cudaFree(d_br);
  }
}

}  // namespace

void ForwardNttCrt(std::uint64_t* data, const std::uint64_t* moduli,
                   std::size_t crt_count, std::size_t poly_len) {
  RunNttCrt(data, moduli, crt_count, poly_len, /*inv=*/false);
}

void InverseNttCrt(std::uint64_t* data, const std::uint64_t* moduli,
                   std::size_t crt_count, std::size_t poly_len) {
  RunNttCrt(data, moduli, crt_count, poly_len, /*inv=*/true);
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
