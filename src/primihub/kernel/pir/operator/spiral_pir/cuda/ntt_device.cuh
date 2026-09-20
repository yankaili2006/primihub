/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Self-contained negacyclic NTT for the SpiralPIR CUDA kernels (perf follow-up
 * to task 2.1). The device modular-arithmetic primitives (Barrett reduction) are
 * reused VERBATIM from pir-acc/SIGMA's CUDA-SEAL kernels
 * (src/primihub/kernel/pir-acc/SIGMA/src/kernelutils.cuh:59-126:
 * d_multiply_uint64[_hw64], d_add_uint64, d_barrett_reduce_128), refactored to
 * take a plain POD `ModParams` instead of SIGMA's `Modulus` class so this header
 * compiles standalone with no SIGMA / DeviceArray / KernelProvider / CMake
 * dependency. The NTT itself is a textbook weighted (psi-scaled) negacyclic
 * transform: pre-scale by psi^j, run a cyclic radix-2 DIT NTT with the N-th root,
 * (pointwise multiply in caller), inverse NTT * N^{-1}, post-scale by psi^{-j}.
 *
 * Coefficients are uint64 < modulus (< ~2^62 supported by Barrett). Header-only;
 * device code guarded by __CUDACC__, host table-builders are plain C++.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_NTT_DEVICE_CUH_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_NTT_DEVICE_CUH_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::spiral::cuda {

// POD modulus params. cr0/cr1 = low/high 64 bits of floor(2^128 / value), i.e.
// SEAL/SIGMA Modulus.const_ratio()[0..1] (see kernelutils.cuh d_barrett_reduce_128).
struct ModParams {
  std::uint64_t value;
  std::uint64_t cr0;
  std::uint64_t cr1;
};

#ifdef __CUDACC__
// ---- device Barrett primitives (from SIGMA kernelutils.cuh:59-126) ----
__device__ __forceinline__ void d_mul64(std::uint64_t a, std::uint64_t b,
                                        std::uint64_t* r) {
  unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
  r[0] = static_cast<std::uint64_t>(p);
  r[1] = static_cast<std::uint64_t>(p >> 64);
}
__device__ __forceinline__ std::uint64_t d_mulhi64(std::uint64_t a,
                                                   std::uint64_t b) {
  return static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(a) * b) >> 64);
}
__device__ __forceinline__ unsigned char d_add64(std::uint64_t a,
                                                 std::uint64_t b,
                                                 std::uint64_t* r) {
  *r = a + b;
  return static_cast<unsigned char>(*r < a);
}
// Base-2^64 Barrett reduction of a 128-bit input mod m (kernelutils.cuh:98-126).
__device__ __forceinline__ std::uint64_t d_barrett128(const std::uint64_t* in,
                                                      ModParams m) {
  std::uint64_t tmp1, tmp2[2], tmp3, carry;
  carry = d_mulhi64(in[0], m.cr0);
  d_mul64(in[0], m.cr1, tmp2);
  tmp3 = tmp2[1] + d_add64(tmp2[0], carry, &tmp1);
  d_mul64(in[1], m.cr0, tmp2);
  carry = tmp2[1] + d_add64(tmp1, tmp2[0], &tmp1);
  tmp1 = in[1] * m.cr1 + tmp3 + carry;
  const std::uint64_t mv = m.value;
  tmp3 = in[0] - tmp1 * mv;
  return (tmp3 >= mv) ? (tmp3 - mv) : tmp3;
}
__device__ __forceinline__ std::uint64_t d_mulmod(std::uint64_t x,
                                                  std::uint64_t y, ModParams m) {
  std::uint64_t pr[2];
  d_mul64(x, y, pr);
  return d_barrett128(pr, m);
}
__device__ __forceinline__ std::uint64_t d_addmod(std::uint64_t a,
                                                  std::uint64_t b, ModParams m) {
  a += b;
  return a >= m.value ? a - m.value : a;
}
__device__ __forceinline__ std::uint64_t d_submod(std::uint64_t a,
                                                  std::uint64_t b, ModParams m) {
  return a >= b ? a - b : a + m.value - b;
}
#endif  // __CUDACC__

// ---- host-side table construction (plain C++) ----
namespace ntt_host {

inline std::uint64_t powmod(std::uint64_t b, std::uint64_t e, std::uint64_t m) {
  unsigned __int128 r = 1, x = b % m;
  while (e) {
    if (e & 1) r = (r * x) % m;
    x = (x * x) % m;
    e >>= 1;
  }
  return static_cast<std::uint64_t>(r);
}

// floor(2^128 / p) for odd prime p == floor((2^128 - 1) / p) since p never
// divides 2^128. Split into low/high words.
inline ModParams make_mod(std::uint64_t p) {
  unsigned __int128 ratio = (~static_cast<unsigned __int128>(0)) / p;
  return ModParams{p, static_cast<std::uint64_t>(ratio),
                   static_cast<std::uint64_t>(ratio >> 64)};
}

inline std::uint64_t primitive_root(std::uint64_t p) {
  const std::uint64_t phi = p - 1;
  std::uint64_t n = phi;
  std::vector<std::uint64_t> f;
  for (std::uint64_t d = 2; d * d <= n; ++d)
    if (n % d == 0) {
      f.push_back(d);
      while (n % d == 0) n /= d;
    }
  if (n > 1) f.push_back(n);
  for (std::uint64_t g = 2;; ++g) {
    bool ok = true;
    for (std::uint64_t q : f)
      if (powmod(g, phi / q, p) == 1) {
        ok = false;
        break;
      }
    if (ok) return g;
  }
}

// Precomputed tables for one residue's negacyclic NTT of length N.
struct NttTables {
  std::uint64_t N = 0, p = 0, n_inv = 0;
  ModParams mod{0, 0, 0};
  std::vector<std::uint64_t> w;        // size N/2: omega^k
  std::vector<std::uint64_t> winv;     // size N/2: omega^{-k}
  std::vector<std::uint64_t> psi;      // size N: psi^j
  std::vector<std::uint64_t> psi_inv;  // size N: psi^{-j}
  std::vector<std::uint32_t> bitrev;   // size N: bit-reversal permutation
};

inline NttTables build_tables(std::uint64_t N, std::uint64_t p) {
  NttTables t;
  t.N = N;
  t.p = p;
  t.mod = make_mod(p);
  const std::uint64_t g = primitive_root(p);
  const std::uint64_t omega = powmod(g, (p - 1) / N, p);          // N-th root
  const std::uint64_t psi = powmod(g, (p - 1) / (2 * N), p);      // 2N-th root
  const std::uint64_t omega_inv = powmod(omega, p - 2, p);
  const std::uint64_t psi_inv = powmod(psi, p - 2, p);
  t.n_inv = powmod(N % p, p - 2, p);
  t.w.resize(N / 2);
  t.winv.resize(N / 2);
  std::uint64_t cw = 1, ciw = 1;
  for (std::uint64_t k = 0; k < N / 2; ++k) {
    t.w[k] = cw;
    t.winv[k] = ciw;
    cw = static_cast<std::uint64_t>((static_cast<unsigned __int128>(cw) * omega) % p);
    ciw = static_cast<std::uint64_t>((static_cast<unsigned __int128>(ciw) * omega_inv) % p);
  }
  t.psi.resize(N);
  t.psi_inv.resize(N);
  std::uint64_t cp = 1, cip = 1;
  for (std::uint64_t j = 0; j < N; ++j) {
    t.psi[j] = cp;
    t.psi_inv[j] = cip;
    cp = static_cast<std::uint64_t>((static_cast<unsigned __int128>(cp) * psi) % p);
    cip = static_cast<std::uint64_t>((static_cast<unsigned __int128>(cip) * psi_inv) % p);
  }
  t.bitrev.resize(N);
  int lg = 0;
  while ((static_cast<std::uint64_t>(1) << lg) < N) ++lg;
  for (std::uint64_t i = 0; i < N; ++i) {
    std::uint32_t r = 0;
    for (int b = 0; b < lg; ++b)
      if (i & (static_cast<std::uint64_t>(1) << b)) r |= 1u << (lg - 1 - b);
    t.bitrev[i] = r;
  }
  return t;
}

}  // namespace ntt_host

}  // namespace primihub::pir::spiral::cuda

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_NTT_DEVICE_CUH_
