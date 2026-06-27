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

// ---- CUDA GPU (one thread per output row) ----
__global__ void MatVecKernel(const std::uint32_t* a, const std::uint32_t* q,
                             std::uint32_t* out, std::size_t rows,
                             std::size_t inner) {
  std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  std::uint32_t acc = 0;
  const std::uint32_t* row = a + i * inner;
  for (std::size_t k = 0; k < inner; ++k) acc += row[k] * q[k];
  out[i] = acc;
}

int main(int argc, char** argv) {
  // Dimensions are argv-overridable so the same bench runs at 1e8 (default) or
  // billion scale:  ./bench [rows] [inner]   e.g. 125000 8000 = 1e9 uint32.
  const std::size_t rows = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 12500;
  const std::size_t inner = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 8000;
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

  const int threads = 256, blocks = int((rows + threads - 1) / threads);
  auto answer_once = [&]() {
    cudaMemcpy(d_q, q.data(), inner * 4, cudaMemcpyHostToDevice);
    MatVecKernel<<<blocks, threads>>>(d_a, d_q, d_o, rows, inner);
    cudaMemcpy(o_cuda.data(), d_o, rows * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  };
  double t_cuda_warm = bench("cuda", answer_once);  // DB resident
  double t_cuda_cold = t_cuda_warm + t_db_upload;   // + one-time DB upload

  cudaFree(d_a); cudaFree(d_q); cudaFree(d_o);

  // Correctness: AVX2 + CUDA must match scalar exactly.
  std::size_t mis_avx2 = 0, mis_cuda = 0;
  for (std::size_t i = 0; i < rows; ++i) {
    if (o_avx2[i] != o_scalar[i]) ++mis_avx2;
    if (o_cuda[i] != o_scalar[i]) ++mis_cuda;
  }

  const double macs = double(n);
  auto gmacs = [&](double ms) { return macs / (ms / 1e3) / 1e9; };
  std::printf("\n  backend         per-answer ms     GMAC/s     vs scalar    correct\n");
  std::printf("  scalar          %10.2f   %9.2f   %8s     %s\n", t_scalar, gmacs(t_scalar), "1.0x", "ref");
  std::printf("  avx2            %10.2f   %9.2f   %7.1fx     %s\n", t_avx2, gmacs(t_avx2), t_scalar / t_avx2, mis_avx2 ? "FAIL" : "ok");
  std::printf("  cuda (warm)     %10.2f   %9.2f   %7.1fx     %s\n", t_cuda_warm, gmacs(t_cuda_warm), t_scalar / t_cuda_warm, mis_cuda ? "FAIL" : "ok");
  std::printf("  cuda (cold+DB)  %10.2f   %9.2f   %7.1fx\n", t_cuda_cold, gmacs(t_cuda_cold), t_scalar / t_cuda_cold);
  std::printf("  (DB upload one-time: %.2f ms)\n", t_db_upload);

  return (mis_avx2 == 0 && mis_cuda == 0) ? 0 : 1;
}
