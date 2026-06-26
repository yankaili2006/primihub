/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_kernel — C++ port of upstream ypir `src/kernel.rs`
 * (fast_batched_dot_product) + the params.rs crt_compose_2 it needs (task
 * 7.3 chunk 10e).
 *
 * FastBatchedDotProduct runtime-dispatches: on an AVX512F host it runs the
 * intrinsics path (mirroring upstream fast_batched_dot_product_avx512); on any
 * other x86_64 host (e.g. the Broadwell build host) it runs the scalar core.
 * Both produce identical results -- the scalar core is the correctness
 * reference the tests pin, and the AVX512 path is exercised + checked against
 * the same oracle on an AVX512 box.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {

// params.rs Params::crt_compose_2: CRT-reconstruct the value r in [0, Q) with
// r == x (mod moduli[0]) and r == y (mod moduli[1]), via
// barrett_reduction_u128(x*mod1_inv_mod0 + y*mod0_inv_mod1).
std::uint64_t CrtCompose2(const Params& p, std::uint64_t x, std::uint64_t y);

namespace kernel_detail {

// Scalar reference: per (batch k, column j) sum the condensed query's low/high
// 32-bit limbs against the transposed db (u128), reduce mod q0/q1, crt_compose,
// accumulate into c via barrett_u64.
template <typename T>
void FastBatchedDotProductScalar(const Params& p, std::size_t k_batch,
                                 std::uint64_t* c, const std::uint64_t* a,
                                 std::size_t a_elems, const T* b_t,
                                 std::size_t b_rows, std::size_t b_cols) {
  const std::uint64_t q0 = p.moduli[0];
  const std::uint64_t q1 = p.moduli[1];
  for (std::size_t k = 0; k < k_batch; ++k) {
    const std::uint64_t* ak = a + k * a_elems;
    for (std::size_t j = 0; j < b_cols; ++j) {
      const T* bj = b_t + j * b_rows;
      __uint128_t sum_lo = 0, sum_hi = 0;
      for (std::size_t kk = 0; kk < a_elems; ++kk) {
        const std::uint64_t av = ak[kk];
        const std::uint64_t b_val = static_cast<std::uint64_t>(bj[kk]);
        sum_lo += static_cast<__uint128_t>(av & 0xFFFFFFFFull) * b_val;
        sum_hi += static_cast<__uint128_t>(av >> 32) * b_val;
      }
      const std::uint64_t lo = static_cast<std::uint64_t>(sum_lo % q0);
      const std::uint64_t hi = static_cast<std::uint64_t>(sum_hi % q1);
      const std::uint64_t res = CrtCompose2(p, lo, hi);
      std::uint64_t& cell = c[k * b_cols + j];
      cell = BarrettRawU64(cell + res, p.barrett_cr_1_modulus, p.modulus);
    }
  }
}

#if defined(__x86_64__)
// Zero-extend 8 consecutive db elements into 8 u64 lanes (mirrors ToM512).
__attribute__((target("avx512f"))) inline __m512i LoadB8(const std::uint8_t* q) {
  return _mm512_cvtepu8_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q)));
}
__attribute__((target("avx512f"))) inline __m512i LoadB8(const std::uint16_t* q) {
  return _mm512_cvtepu16_epi64(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(q)));
}
__attribute__((target("avx512f"))) inline __m512i LoadB8(const std::uint32_t* q) {
  return _mm512_cvtepu32_epi64(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)));
}

// AVX512 path (mirrors kernel.rs fast_batched_dot_product_avx512): chunked,
// _mm512_mul_epu32 on the low/high 32-bit limbs, per-chunk horizontal sum +
// Barrett fold + crt_compose, accumulated into c. Same result as the scalar
// core. Requires a_elems % 8 == 0.
template <typename T>
__attribute__((target("avx512f"))) void FastBatchedDotProductAvx512(
    const Params& p, std::size_t k_batch, std::uint64_t* c,
    const std::uint64_t* a, std::size_t a_elems, const T* b_t,
    std::size_t b_rows, std::size_t b_cols) {
  const std::size_t simd_width = 8;
  std::size_t kpow = 1;
  while (kpow < k_batch) kpow <<= 1;
  const std::size_t chunk_size =
      std::min<std::size_t>(8192 / kpow, a_elems / simd_width);
  const std::size_t num_chunks =
      chunk_size == 0 ? 0 : (a_elems / simd_width) / chunk_size;

  for (std::size_t k = 0; k < k_batch; ++k) {
    const std::uint64_t* ak = a + k * a_elems;
    for (std::size_t j = 0; j < b_cols; ++j) {
      const T* bj = b_t + j * b_rows;
      for (std::size_t k_outer = 0; k_outer < num_chunks; ++k_outer) {
        __m512i sum_lo = _mm512_setzero_si512();
        __m512i sum_hi = _mm512_setzero_si512();
        for (std::size_t k_inner = 0; k_inner < chunk_size; ++k_inner) {
          const std::size_t base =
              simd_width * (k_outer * chunk_size + k_inner);
          const __m512i a_vec =
              _mm512_loadu_si512(reinterpret_cast<const void*>(ak + base));
          const __m512i a_hi = _mm512_srli_epi64(a_vec, 32);
          const __m512i b_vec = LoadB8(bj + base);
          sum_lo = _mm512_add_epi64(sum_lo, _mm512_mul_epu32(a_vec, b_vec));
          sum_hi = _mm512_add_epi64(sum_hi, _mm512_mul_epu32(a_hi, b_vec));
        }
        const std::uint64_t res_lo = _mm512_reduce_add_epi64(sum_lo);
        const std::uint64_t res_hi = _mm512_reduce_add_epi64(sum_hi);
        const std::uint64_t lo =
            BarrettRawU64(res_lo, p.barrett_cr_1[0], p.moduli[0]);
        const std::uint64_t hi =
            BarrettRawU64(res_hi, p.barrett_cr_1[1], p.moduli[1]);
        const std::uint64_t res = CrtCompose2(p, lo, hi);
        std::uint64_t& cell = c[k * b_cols + j];
        cell = BarrettRawU64(cell + res, p.barrett_cr_1_modulus, p.modulus);
      }
    }
  }
}
#endif  // __x86_64__

}  // namespace kernel_detail

// Public entry: runtime-dispatch to AVX512 when available, else scalar. See
// kernel_detail::FastBatchedDotProductScalar for the semantics. a is the
// CONDENSED query (k_batch * a_elems u64), b_t the transposed db (b[kk][j] =
// b_t[j*b_rows + kk]), a_elems == b_rows, c (k_batch * b_cols) accumulated into.
template <typename T>
void FastBatchedDotProduct(const Params& p, std::size_t k_batch,
                           std::uint64_t* c, const std::uint64_t* a,
                           std::size_t a_elems, const T* b_t,
                           std::size_t b_rows, std::size_t b_cols) {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx512f")) {
    kernel_detail::FastBatchedDotProductAvx512<T>(p, k_batch, c, a, a_elems, b_t,
                                                  b_rows, b_cols);
    return;
  }
#endif
  kernel_detail::FastBatchedDotProductScalar<T>(p, k_batch, c, a, a_elems, b_t,
                                                b_rows, b_cols);
}

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_
