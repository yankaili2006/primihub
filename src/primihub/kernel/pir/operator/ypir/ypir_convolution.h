/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_convolution — port of upstream menonsamir/ypir@a73e550a
 * src/convolution.rs. Chunk 6 of the YPIR port.
 *
 * This first landing ports the Spiral-free reference helpers — the
 * naive O(n^2) negacyclic convolution and the naive (wrapping u32)
 * matrix multiply — which are the oracles the optimised NTT path is
 * tested against. (negacyclic_matrix_u32 / negacyclic_perm_u32 are
 * already ported in ypir_negacyclic.)
 *
 * Deferred to a follow-up: the `Convolution` struct (ntt / raw /
 * pointwise_mul / convolve). Upstream builds it on spiral_rs's
 * 2-modulus CRT NTT (`Convolution::params_for` -> crt_count=2). In the
 * C++ port that maps to two `hexl::NTT(n, modulus_i)` instances + CRT
 * reconstruction (the @hexl facade is now wired and link-proven by
 * ypir_spiral_smoke_test); it is correctness-sensitive and will be
 * verified against NaiveNegacyclicConvolve here.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Naive O(n^2) negacyclic convolution over Z_{2^32}[X]/(X^n + 1):
// res[i] = sum_j a[j] * (i < j ? -b[(n+i-j)%n] : b[(n+i-j)%n]), with
// u32-wrapping arithmetic. Mirrors upstream naive_negacyclic_convolve.
// Requires a.size() == b.size().
std::vector<std::uint32_t> NaiveNegacyclicConvolve(
    const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b);

// Naive wrapping-u32 matrix multiply (the u32 specialisation of upstream
// naive_multiply_matrices). `a` is a_rows x a_cols row-major. When
// `is_b_transposed`, `b` is b_cols x b_rows row-major (i.e. column j is a
// contiguous row), indexed b[j*b_rows + k]; otherwise `b` is
// b_rows x b_cols row-major, indexed b[k*b_cols + j]. Requires
// a_cols == b_rows. Returns a_rows x b_cols.
std::vector<std::uint32_t> NaiveMultiplyMatrices(
    const std::vector<std::uint32_t>& a, std::size_t a_rows, std::size_t a_cols,
    const std::vector<std::uint32_t>& b, std::size_t b_rows, std::size_t b_cols,
    bool is_b_transposed);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_H_
