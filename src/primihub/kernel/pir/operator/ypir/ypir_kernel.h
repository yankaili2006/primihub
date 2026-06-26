/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_kernel — C++ port of upstream ypir `src/kernel.rs`
 * (fast_batched_dot_product) + the params.rs crt_compose_2 it needs (task
 * 7.3 chunk 10e).
 *
 * This file lands the SCALAR reference of fast_batched_dot_product, which is
 * portable everywhere and is what the correctness tests pin. The AVX512
 * intrinsics path (the upstream fast_batched_dot_product_avx512 body) is a
 * follow-on that runtime-dispatches to AVX512 on a capable host and falls back
 * to this scalar core elsewhere -- so the kernel builds + runs on the Broadwell
 * build host as well as on an AVX512 box.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {

// params.rs Params::crt_compose_2: CRT-reconstruct the value r in [0, Q) with
// r == x (mod moduli[0]) and r == y (mod moduli[1]), via
// barrett_reduction_u128(x*mod1_inv_mod0 + y*mod0_inv_mod1).
std::uint64_t CrtCompose2(const Params& p, std::uint64_t x, std::uint64_t y);

// Scalar reference of kernel.rs fast_batched_dot_product_avx512. Computes, for
// each batch k in [0,K) and column j in [0,b_cols):
//   total_lo = sum_kk a_lo[k][kk] * b[kk][j]   (a_lo = low 32 bits of a[k][kk])
//   total_hi = sum_kk a_hi[k][kk] * b[kk][j]   (a_hi = high 32 bits)
//   c[k][j] = barrett_u64( c[k][j] + crt_compose_2(total_lo mod q0,
//                                                  total_hi mod q1) )
// where a is the CONDENSED query (K * a_elems u64, each packing the two CRT
// NTT limbs in lo/hi 32 bits), b_t is the transposed db (b[kk][j] =
// b_t[j*b_rows + kk]), a_elems == b_rows, and c (K * b_cols) is ACCUMULATED
// into (the caller zeroes it). T is the db element type (uint8/16/32_t).
template <typename T>
void FastBatchedDotProduct(const Params& p, std::size_t k_batch,
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
        const std::uint64_t a_lo = av & 0xFFFFFFFFull;
        const std::uint64_t a_hi = av >> 32;
        const std::uint64_t b_val = static_cast<std::uint64_t>(bj[kk]);
        sum_lo += static_cast<__uint128_t>(a_lo) * b_val;
        sum_hi += static_cast<__uint128_t>(a_hi) * b_val;
      }
      const std::uint64_t lo = static_cast<std::uint64_t>(sum_lo % q0);
      const std::uint64_t hi = static_cast<std::uint64_t>(sum_hi % q1);
      const std::uint64_t res = CrtCompose2(p, lo, hi);
      std::uint64_t& cell = c[k * b_cols + j];
      cell = BarrettRawU64(cell + res, p.barrett_cr_1_modulus, p.modulus);
    }
  }
}

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_KERNEL_H_
