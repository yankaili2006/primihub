/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Three-backend benchmark of the PIR Answer hot path (openspec change
 * primihub-pir-cuda-tiptoe, task 2.4): the LWE matrix-vector product
 * answer = A * q mod 2^32, where A is the (squished) database (~1e8 uint32
 * entries) and q is the query vector. Compares scalar CPU, AVX2 CPU, and CUDA
 * GPU, checks all three agree, and reports per-Answer latency + throughput.
 *
 * For CUDA the database is uploaded once (PIR preprocessing) and the timed
 * per-Answer cost is query upload + kernel + result download (the realistic
 * online metric), alongside a cold number that includes the DB upload.
 *
 * Build/run: see bench/cuda_vs_avx2.sh (needs nvcc + a GPU; AVX2 host path uses
 * a target attribute so no global -mavx2 is required).
 */
#include <cuda_runtime.h>
#include <immintrin.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point a) {
  return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

// ---- scalar CPU ----
static void MatVecScalar(const std::uint32_t* a, const std::uint32_t* q,
                         std::uint32_t* out, std::size_t rows, std::size_t inner) {
  for (std::size_t i = 0; i < rows; ++i) {
    std::uint32_t acc = 0;
    const std::uint32_t* row = a + i * inner;
    for (std::size_t k = 0; k < inner; ++k) acc += row[k] * q[k];
    out[i] = acc;
  }
}

// ---- AVX2 CPU (8-wide mullo/add, wraps mod 2^32) ----
__attribute__((target("avx2"))) static void MatVecAvx2(
    const std::uint32_t* a, const std::uint32_t* q, std::uint32_t* out,
    std::size_t rows, std::size_t inner) {
  for (std::size_t i = 0; i < rows; ++i) {
    const std::uint32_t* row = a + i * inner;
    __m256i acc = _mm256_setzero_si256();
    std::size_t k = 0;
    for (; k + 8 <= inner; k += 8) {
      __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + k));
      __m256i vq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + k));
      acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(va, vq));
    }
    std::uint32_t lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::uint32_t s = 0;
    for (int l = 0; l < 8; ++l) s += lanes[l];
    for (; k < inner; ++k) s += row[k] * q[k];
    out[i] = s;
  }
}

// ---- CUDA GPU (warp-per-row, coalesced uint4 loads, wraps mod 2^32) ----
// This matvec is memory-bound: the 4 GB DB (at 1e9) is streamed once per Answer,
// so the kernel is structured to read it at close to peak bandwidth.
//   * one WARP cooperates on one output row -> the 32 lanes read consecutive
//     elements, so each load is fully coalesced (the old thread-per-row layout
//     had adjacent threads reading addresses `inner` apart -> uncoalesced, which
//     is why its throughput only rose with scale as occupancy hid the latency).
//   * uint4 (16-byte) vectorized loads of both the row and the query when inner
//     is 4-aligned (the row base a+row*inner is then 16-byte aligned too); a
//     scalar path covers ragged inner.
//   * grid-stride over rows so the launch is occupancy-driven, not rows-driven.
// A final warp shuffle-reduce produces the row's dot product.
__global__ void MatVecKernel(const std::uint32_t* a, const std::uint32_t* q,
                             std::uint32_t* out, std::size_t rows,
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
      const uint4* q4 = reinterpret_cast<const uint4*>(q);
      const std::size_t n4 = inner >> 2;  // # of uint4 chunks
      for (std::size_t j = lane; j < n4; j += 32) {
        uint4 av = a4[j], qv = q4[j];
        acc += av.x * qv.x + av.y * qv.y + av.z * qv.z + av.w * qv.w;
      }
    } else {
      for (std::size_t k = lane; k < inner; k += 32) acc += arow[k] * q[k];
    }
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
  }
}

int main(int argc, char** argv) {
  // Dimensions are argv-overridable so the same bench runs at 1e8 (default) or
  // billion scale:  ./bench [rows] [inner]   e.g. 125000 8000 = 1e9 uint32.
  const std::size_t rows = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 12500;
  const std::size_t inner = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 8000;
  // argv[3] = K: number of distinct queries to time against a resident DB (the
  // realistic Answer-server pattern: DB uploaded once, many queries served).
  const int kQueries = (argc > 3) ? std::atoi(argv[3]) : 32;
  const std::size_t n = rows * inner;  // ~1e8 DB entries by default
  std::printf("DB = %zu x %zu = %.2e uint32 (%.0f MB); answer = %zu\n", rows,
              inner, double(n), double(n) * 4 / 1e6, rows);

  std::vector<std::uint32_t> a(n), q(inner), o_scalar(rows), o_avx2(rows), o_cuda(rows);
  std::uint64_t s = 0x9E3779B9u;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return std::uint32_t(s); };
  for (auto& v : a) v = rnd();
  for (auto& v : q) v = rnd();

  const int iters = 5;
  auto bench = [&](const char* name, auto fn) {
    fn();  // warmup
    auto t = clk::now();
    for (int it = 0; it < iters; ++it) fn();
    return ms_since(t) / iters;
  };

  double t_scalar = bench("scalar", [&]() {
    MatVecScalar(a.data(), q.data(), o_scalar.data(), rows, inner);
  });
  double t_avx2 = bench("avx2", [&]() {
    MatVecAvx2(a.data(), q.data(), o_avx2.data(), rows, inner);
  });

  // CUDA: upload DB once (preprocessing), then time the online per-answer cost.
  std::uint32_t *d_a, *d_q, *d_o;
  cudaMalloc(&d_a, n * 4);
  cudaMalloc(&d_q, inner * 4);
  cudaMalloc(&d_o, rows * 4);
  auto t_upload0 = clk::now();
  cudaMemcpy(d_a, a.data(), n * 4, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double t_db_upload = ms_since(t_upload0);

  // Warp-per-row launch: one warp per output row, grid-stride covers rows >
  // total warps. threads/32 = warps per block.
  const int threads = 256;
  const int warps_per_block = threads / 32;
  long want_blocks = long((rows + warps_per_block - 1) / warps_per_block);
  const int blocks = int(want_blocks > 65535 ? 65535 : want_blocks);

  auto answer_once = [&]() {
    cudaMemcpy(d_q, q.data(), inner * 4, cudaMemcpyHostToDevice);
    MatVecKernel<<<blocks, threads>>>(d_a, d_q, d_o, rows, inner);
    cudaMemcpy(o_cuda.data(), d_o, rows * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  };
  double t_cuda_warm = bench("cuda", answer_once);  // DB resident, query up + down
  double t_cuda_cold = t_cuda_warm + t_db_upload;   // + one-time DB upload

  // Correctness must be checked from the WARM result (original q) BEFORE the
  // resident loop below mutates q.
  std::size_t mis_avx2 = 0, mis_cuda = 0;
  for (std::size_t i = 0; i < rows; ++i) {
    if (o_avx2[i] != o_scalar[i]) ++mis_avx2;
    if (o_cuda[i] != o_scalar[i]) ++mis_cuda;
  }

  // Kernel-only (no PCIe): isolates compute from the per-query transfers, so the
  // gap to warm attributes the remaining cost to PCIe vs the kernel.
  auto t_k0 = clk::now();
  for (int it = 0; it < iters; ++it)
    MatVecKernel<<<blocks, threads>>>(d_a, d_q, d_o, rows, inner);
  cudaDeviceSynchronize();
  double t_kernel_only = ms_since(t_k0) / iters;

  // Resident, K distinct queries: DB stays on device, each query is uploaded,
  // answered, and downloaded. per-query = total / K. The production metric.
  auto t_r0 = clk::now();
  for (int k = 0; k < kQueries; ++k) {
    q[0] += 1u;  // make each query distinct (cheap)
    cudaMemcpy(d_q, q.data(), inner * 4, cudaMemcpyHostToDevice);
    MatVecKernel<<<blocks, threads>>>(d_a, d_q, d_o, rows, inner);
    cudaMemcpy(o_cuda.data(), d_o, rows * 4, cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  double t_resident = ms_since(t_r0) / double(kQueries);

  cudaFree(d_a); cudaFree(d_q); cudaFree(d_o);

  const double macs = double(n);
  auto gmacs = [&](double ms) { return macs / (ms / 1e3) / 1e9; };
  // GB/s of DB streamed: the kernel reads the n-uint32 DB once per answer.
  const double db_gb = double(n) * 4.0 / 1e9;
  auto gbps = [&](double ms) { return db_gb / (ms / 1e3); };
  std::printf("\n  backend             per-answer ms     GMAC/s     vs scalar    correct\n");
  std::printf("  scalar              %10.2f   %9.2f   %8s     %s\n", t_scalar, gmacs(t_scalar), "1.0x", "ref");
  std::printf("  avx2                %10.2f   %9.2f   %7.1fx     %s\n", t_avx2, gmacs(t_avx2), t_scalar / t_avx2, mis_avx2 ? "FAIL" : "ok");
  std::printf("  cuda (warm)         %10.2f   %9.2f   %7.1fx     %s\n", t_cuda_warm, gmacs(t_cuda_warm), t_scalar / t_cuda_warm, mis_cuda ? "FAIL" : "ok");
  std::printf("  cuda (resident,K=%d) %10.2f   %9.2f   %7.1fx     %s\n", kQueries, t_resident, gmacs(t_resident), t_scalar / t_resident, mis_cuda ? "FAIL" : "ok");
  std::printf("  cuda (kernel-only)  %10.2f   %9.2f   %7.1fx\n", t_kernel_only, gmacs(t_kernel_only), t_scalar / t_kernel_only);
  std::printf("  cuda (cold+DB)      %10.2f   %9.2f   %7.1fx\n", t_cuda_cold, gmacs(t_cuda_cold), t_scalar / t_cuda_cold);
  std::printf("  (DB upload one-time: %.2f ms; kernel-only bandwidth: %.0f GB/s)\n", t_db_upload, gbps(t_kernel_only));

  return (mis_avx2 == 0 && mis_cuda == 0) ? 0 : 1;
}
