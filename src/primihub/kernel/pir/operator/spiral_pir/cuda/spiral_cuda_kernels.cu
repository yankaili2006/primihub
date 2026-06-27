/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA implementation of the SpiralPIR GSW external product + Galois
 * automorphism (task 2.1) and the negacyclic NTT (perf follow-up). See
 * spiral_cuda_kernels.h.
 *
 * NTT perf notes: the transform is a shared-memory DIT NTT, one block per
 * (polynomial, residue) instance, bit-reversed on load / natural on store. The
 * default path uses a radix-4 butterfly that FUSES two consecutive radix-2 DIT
 * stages with the 4 values kept in registers between them -- algebraically
 * identical to two radix-2 stages (same `bitrev` permutation, same `w[]` twiddle
 * table, indices stay < N/2), so it halves the shared-memory round trips and
 * __syncthreads() barriers (N=2048: 11 -> 6). The radix-2 kernel is retained for
 * apples-to-apples benchmarking. Both share the batched device-resident driver
 * (tables built once per modulus, all instances in one grid -- no per-residue
 * cudaMalloc/memcpy churn).
 */
#include "spiral_cuda_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
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

// ---- batched negacyclic NTT kernels (one block per (poly,residue) instance) ----
// data layout: [num_instances][N], instance index = poly*crt + c. Per-residue
// tables are arrays indexed by c = blockIdx.x % crt: w_all[crt][N/2],
// psi_all[crt][N], mods[crt], ninv[crt]. bitrev[N] is residue-independent.

// Coefficient-wise multiply g[i] *= scale[i] mod m  (psi pre/post weighting).
__global__ void WeightKernel(std::uint64_t* data, const std::uint64_t* psi_all,
                             const ModParams* mods, std::size_t N,
                             std::size_t crt) {
  const std::size_t inst = blockIdx.x;
  const std::size_t c = inst % crt;
  std::uint64_t* g = data + inst * N;
  const std::uint64_t* scale = psi_all + c * N;
  const ModParams m = mods[c];
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x)
    g[i] = d_mulmod(g[i] % m.value, scale[i], m);
}

// Radix-2 DIT NTT (reference). log2(N) stages, one __syncthreads() each.
__global__ void NttKernelR2(std::uint64_t* data, const std::uint64_t* w_all,
                            const std::uint32_t* bitrev, const ModParams* mods,
                            const std::uint64_t* ninv_all, std::size_t N,
                            std::size_t crt, bool inv) {
  extern __shared__ std::uint64_t sh[];
  const std::size_t inst = blockIdx.x;
  const std::size_t c = inst % crt;
  std::uint64_t* g = data + inst * N;
  const std::uint64_t* w = w_all + c * (N / 2);
  const ModParams m = mods[c];

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
  const std::uint64_t ninv = ninv_all[c];
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x)
    g[i] = inv ? d_mulmod(sh[i], ninv, m) : sh[i];
}

// Radix-4 DIT NTT: fuses pairs of radix-2 stages in registers. Each radix-4
// butterfly owns a disjoint quadruple {p0,p1,p2,p3} and does its two sub-stage
// butterflies (length L then length 2L) locally, so only ONE __syncthreads()
// per fused stage. For N=2^lg: floor(lg/2) fused stages + (lg odd ? 1 radix-2).
__global__ void NttKernelR4(std::uint64_t* data, const std::uint64_t* w_all,
                            const std::uint32_t* bitrev, const ModParams* mods,
                            const std::uint64_t* ninv_all, std::size_t N,
                            std::size_t crt, bool inv) {
  extern __shared__ std::uint64_t sh[];
  const std::size_t inst = blockIdx.x;
  const std::size_t c = inst % crt;
  std::uint64_t* g = data + inst * N;
  const std::uint64_t* w = w_all + c * (N / 2);
  const ModParams m = mods[c];

  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x) sh[i] = g[bitrev[i]];
  __syncthreads();

  int lg = 0;
  while ((static_cast<std::size_t>(1) << lg) < N) ++lg;

  std::size_t len = 2;  // smaller of the two fused stages
  int done = 0;
  while (lg - done >= 2) {
    const std::size_t L = len;              // sub-stage A length
    const std::size_t stepL = N / L;        // twiddle stride for length L
    const std::size_t step2 = N / (2 * L);  // twiddle stride for length 2L
    const std::size_t halfL = L >> 1;       // butterflies per L-block
    for (std::size_t bf = threadIdx.x; bf < N / 4; bf += blockDim.x) {
      const std::size_t blk = bf / halfL;   // which 2L-block
      const std::size_t j = bf % halfL;     // j in [0, L/2)
      const std::size_t base = blk * (2 * L);
      const std::size_t p0 = base + j;
      const std::size_t p1 = p0 + halfL;
      const std::size_t p2 = p0 + L;
      const std::size_t p3 = p2 + halfL;
      // sub-stage A (length L): two radix-2 butterflies sharing twiddle tA.
      const std::uint64_t tA = w[stepL * j];
      const std::uint64_t a0 = sh[p0];
      const std::uint64_t a1 = d_mulmod(sh[p1], tA, m);
      const std::uint64_t x0 = d_addmod(a0, a1, m);
      const std::uint64_t x1 = d_submod(a0, a1, m);
      const std::uint64_t a2 = sh[p2];
      const std::uint64_t a3 = d_mulmod(sh[p3], tA, m);
      const std::uint64_t x2 = d_addmod(a2, a3, m);
      const std::uint64_t x3 = d_submod(a2, a3, m);
      // sub-stage B (length 2L): pairs (x0,x2) and (x1,x3).
      const std::uint64_t y2 = d_mulmod(x2, w[step2 * j], m);
      const std::uint64_t y3 = d_mulmod(x3, w[step2 * (j + halfL)], m);
      sh[p0] = d_addmod(x0, y2, m);
      sh[p2] = d_submod(x0, y2, m);
      sh[p1] = d_addmod(x1, y3, m);
      sh[p3] = d_submod(x1, y3, m);
    }
    __syncthreads();
    len <<= 2;
    done += 2;
  }
  if (done < lg) {  // one leftover radix-2 stage (lg odd), len == N
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

  const std::uint64_t ninv = ninv_all[c];
  for (std::size_t i = threadIdx.x; i < N; i += blockDim.x)
    g[i] = inv ? d_mulmod(sh[i], ninv, m) : sh[i];
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

// Device-resident per-residue NTT tables, built once and reused across launches.
struct DeviceNttTables {
  std::uint64_t *w = nullptr, *winv = nullptr, *psi = nullptr, *psi_inv = nullptr;
  std::uint64_t* ninv = nullptr;
  std::uint32_t* bitrev = nullptr;
  ModParams* mods = nullptr;
  std::size_t crt = 0, N = 0;
};

DeviceNttTables UploadTables(const std::uint64_t* moduli, std::size_t crt,
                            std::size_t N) {
  DeviceNttTables d;
  d.crt = crt;
  d.N = N;
  std::vector<std::uint64_t> w(crt * (N / 2)), winv(crt * (N / 2));
  std::vector<std::uint64_t> psi(crt * N), psi_inv(crt * N);
  std::vector<std::uint64_t> ninv(crt);
  std::vector<ModParams> mods(crt);
  std::vector<std::uint32_t> bitrev;
  for (std::size_t c = 0; c < crt; ++c) {
    auto t = ntt_host::build_tables(N, moduli[c]);
    std::copy(t.w.begin(), t.w.end(), w.begin() + c * (N / 2));
    std::copy(t.winv.begin(), t.winv.end(), winv.begin() + c * (N / 2));
    std::copy(t.psi.begin(), t.psi.end(), psi.begin() + c * N);
    std::copy(t.psi_inv.begin(), t.psi_inv.end(), psi_inv.begin() + c * N);
    ninv[c] = t.n_inv;
    mods[c] = t.mod;
    if (c == 0) bitrev = t.bitrev;  // residue-independent
  }
  cudaMalloc(&d.w, w.size() * sizeof(std::uint64_t));
  cudaMalloc(&d.winv, winv.size() * sizeof(std::uint64_t));
  cudaMalloc(&d.psi, psi.size() * sizeof(std::uint64_t));
  cudaMalloc(&d.psi_inv, psi_inv.size() * sizeof(std::uint64_t));
  cudaMalloc(&d.ninv, ninv.size() * sizeof(std::uint64_t));
  cudaMalloc(&d.bitrev, bitrev.size() * sizeof(std::uint32_t));
  cudaMalloc(&d.mods, mods.size() * sizeof(ModParams));
  cudaMemcpy(d.w, w.data(), w.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.winv, winv.data(), winv.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.psi, psi.data(), psi.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.psi_inv, psi_inv.data(), psi_inv.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.ninv, ninv.data(), ninv.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.bitrev, bitrev.data(), bitrev.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d.mods, mods.data(), mods.size() * sizeof(ModParams), cudaMemcpyHostToDevice);
  return d;
}

void FreeTables(DeviceNttTables& d) {
  cudaFree(d.w); cudaFree(d.winv); cudaFree(d.psi); cudaFree(d.psi_inv);
  cudaFree(d.ninv); cudaFree(d.bitrev); cudaFree(d.mods);
}

// Launch a forward/inverse NTT over `num_inst` device-resident instances
// (instance = poly*crt + c). Tables device-resident. No host sync inside.
void LaunchNtt(std::uint64_t* d_data, const DeviceNttTables& t,
               std::size_t num_inst, bool inv, bool r4) {
  const int threads = t.N >= 1024 ? 512 : 256;
  const std::size_t shmem = t.N * sizeof(std::uint64_t);
  const std::uint64_t* w = inv ? t.winv : t.w;
  const std::uint64_t* psi = inv ? t.psi_inv : t.psi;
  const unsigned grid = static_cast<unsigned>(num_inst);
  if (!inv) {
    WeightKernel<<<grid, threads>>>(d_data, psi, t.mods, t.N, t.crt);
    if (r4)
      NttKernelR4<<<grid, threads, shmem>>>(d_data, w, t.bitrev, t.mods, t.ninv, t.N, t.crt, false);
    else
      NttKernelR2<<<grid, threads, shmem>>>(d_data, w, t.bitrev, t.mods, t.ninv, t.N, t.crt, false);
  } else {
    if (r4)
      NttKernelR4<<<grid, threads, shmem>>>(d_data, w, t.bitrev, t.mods, t.ninv, t.N, t.crt, true);
    else
      NttKernelR2<<<grid, threads, shmem>>>(d_data, w, t.bitrev, t.mods, t.ninv, t.N, t.crt, true);
    WeightKernel<<<grid, threads>>>(d_data, psi, t.mods, t.N, t.crt);
  }
}

// Forward/inverse NTT over a host buffer data[num_polys][crt][N] (in place).
// radix4=true is the production path; the radix-2 reference is bench-only.
void RunNttBatched(std::uint64_t* data, const std::uint64_t* moduli,
                   std::size_t num_polys, std::size_t crt, std::size_t N,
                   bool inv, bool radix4 = true) {
  const std::size_t total = num_polys * crt * N;
  DeviceNttTables t = UploadTables(moduli, crt, N);
  std::uint64_t* d_data;
  cudaMalloc(&d_data, total * sizeof(std::uint64_t));
  cudaMemcpy(d_data, data, total * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  LaunchNtt(d_data, t, num_polys * crt, inv, radix4);
  cudaDeviceSynchronize();
  cudaMemcpy(data, d_data, total * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  FreeTables(t);
}

}  // namespace

void ForwardNttCrt(std::uint64_t* data, const std::uint64_t* moduli,
                   std::size_t crt_count, std::size_t poly_len) {
  RunNttBatched(data, moduli, /*num_polys=*/1, crt_count, poly_len, /*inv=*/false);
}

void InverseNttCrt(std::uint64_t* data, const std::uint64_t* moduli,
                   std::size_t crt_count, std::size_t poly_len) {
  RunNttBatched(data, moduli, /*num_polys=*/1, crt_count, poly_len, /*inv=*/true);
}

#ifdef SPIRAL_NTT_BENCH
// Bench-only: end-to-end ms (incl. H2D/D2H + allocation) for ONE forward+inverse
// over `num_polys` polynomials. mode 0 = original path (per-residue cudaMalloc +
// 4 memcpy + free, host table rebuild, single-block <<<1,...>>> per residue,
// processed one polynomial at a time); mode 1 = batched device-resident path
// (tables uploaded once, all instances in one grid). Shows the host-path win.
double BenchEndToEndMs(std::size_t num_polys, const std::uint64_t* moduli,
                       std::size_t crt, std::size_t N, int iters, int mode) {
  const std::size_t total = num_polys * crt * N;
  std::vector<std::uint64_t> host(total);
  std::uint64_t s = 0xBEEF1234ull;
  for (std::size_t inst = 0; inst < num_polys * crt; ++inst) {
    const std::size_t c = inst % crt;
    for (std::size_t i = 0; i < N; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      host[inst * N + i] = s % moduli[c];
    }
  }
  const int threads = N >= 1024 ? 512 : 256;
  const std::size_t shmem = N * sizeof(std::uint64_t);

  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    std::vector<std::uint64_t> buf = host;
    if (mode == 0) {
      // Original path: per polynomial, per residue, single-block, rebuild tables.
      for (std::size_t p = 0; p < num_polys; ++p) {
        for (bool inv : {false, true}) {
          for (std::size_t c = 0; c < crt; ++c) {
            auto tb = ntt_host::build_tables(N, moduli[c]);
            std::uint64_t* slice = buf.data() + (p * crt + c) * N;
            std::uint64_t *d_data, *d_w, *d_psi;
            std::uint32_t* d_br;
            ModParams* d_m;
            std::uint64_t* d_ninv;
            cudaMalloc(&d_data, N * sizeof(std::uint64_t));
            cudaMalloc(&d_w, (N / 2) * sizeof(std::uint64_t));
            cudaMalloc(&d_psi, N * sizeof(std::uint64_t));
            cudaMalloc(&d_br, N * sizeof(std::uint32_t));
            cudaMalloc(&d_m, sizeof(ModParams));
            cudaMalloc(&d_ninv, sizeof(std::uint64_t));
            cudaMemcpy(d_data, slice, N * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_w, (inv ? tb.winv.data() : tb.w.data()), (N / 2) * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_psi, (inv ? tb.psi_inv.data() : tb.psi.data()), N * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_br, tb.bitrev.data(), N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_m, &tb.mod, sizeof(ModParams), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ninv, &tb.n_inv, sizeof(std::uint64_t), cudaMemcpyHostToDevice);
            if (!inv) {
              WeightKernel<<<1, threads>>>(d_data, d_psi, d_m, N, 1);
              NttKernelR2<<<1, threads, shmem>>>(d_data, d_w, d_br, d_m, d_ninv, N, 1, false);
            } else {
              NttKernelR2<<<1, threads, shmem>>>(d_data, d_w, d_br, d_m, d_ninv, N, 1, true);
              WeightKernel<<<1, threads>>>(d_data, d_psi, d_m, N, 1);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(slice, d_data, N * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_data); cudaFree(d_w); cudaFree(d_psi); cudaFree(d_br);
            cudaFree(d_m); cudaFree(d_ninv);
          }
        }
      }
    } else {
      RunNttBatched(buf.data(), moduli, num_polys, crt, N, false, /*radix4=*/true);
      RunNttBatched(buf.data(), moduli, num_polys, crt, N, true, /*radix4=*/true);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Bench-only: kernel-only ms for `iters` warm forward+inverse NTTs over
// `num_polys` polynomials (each crt residues x N), device-resident, tables and
// data preloaded; excludes H2D/D2H. radix4 selects the butterfly.
double BenchNttKernelMs(std::size_t num_polys, const std::uint64_t* moduli,
                        std::size_t crt, std::size_t N, int iters, bool radix4) {
  const std::size_t num_inst = num_polys * crt;
  const std::size_t total = num_inst * N;
  DeviceNttTables t = UploadTables(moduli, crt, N);

  std::vector<std::uint64_t> host(total);
  std::uint64_t s = 0xC0FFEEull;
  for (std::size_t inst = 0; inst < num_inst; ++inst) {
    const std::size_t c = inst % crt;
    for (std::size_t i = 0; i < N; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      host[inst * N + i] = s % moduli[c];
    }
  }
  std::uint64_t* d_data;
  cudaMalloc(&d_data, total * sizeof(std::uint64_t));
  cudaMemcpy(d_data, host.data(), total * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  LaunchNtt(d_data, t, num_inst, false, radix4);  // warmup
  LaunchNtt(d_data, t, num_inst, true, radix4);
  cudaDeviceSynchronize();

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int it = 0; it < iters; ++it) {
    LaunchNtt(d_data, t, num_inst, false, radix4);
    LaunchNtt(d_data, t, num_inst, true, radix4);
  }
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaFree(d_data);
  FreeTables(t);
  return static_cast<double>(ms);
}
#endif  // SPIRAL_NTT_BENCH

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
