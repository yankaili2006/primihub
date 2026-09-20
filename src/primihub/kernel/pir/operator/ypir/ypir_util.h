/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_util — the Spiral-free subset of upstream menonsamir/ypir@
 * a73e550a src/util.rs. Chunk 13a of the YPIR port (see
 * docs/pir/ypir-port-plan.md). Three pure-arithmetic helpers
 * port today; the Spiral-bound helpers (get_negacylic /
 * reduce_copy / add_into_no_reduce / scalar_multiply_avx /
 * test_params / multiply_matrices_raw_not_transposed / etc.) are
 * deferred to chunk 13b because they take PolyMatrixRaw,
 * PolyMatrixNTT, or Params.
 *
 * Distinction from chunk 6a's negacyclic helpers in
 * ypir_negacyclic.{h,cc}:
 *
 *   * ypir_negacyclic.h functions (`NegacyclicMatrixU32`,
 *     `NaiveNegacyclicConvolveU32`, `NegacyclicPermU32`) are the
 *     u32-wrapping versions from convolution.rs, with shift fixed
 *     to 0.
 *   * ypir_util.h functions here (`NegacyclicPermU64Mod`,
 *     `NegacyclicMatrixU64Mod`) are the u64-modular versions from
 *     util.rs, with the shift parameterised. They reduce to the
 *     u32 versions when modulus = 2^32 and shift = 0.
 *
 * Both pairs co-exist because upstream Rust has the same split —
 * lwe.rs uses the u32 wrapping ones, packing.rs uses the u64
 * modular ones, and forcing a single port would require either a
 * generic-over-T template (drags trait machinery) or losing the
 * upstream byte-for-byte fidelity that chunk 6a maintains.
 *
 * Functions:
 *   * NegacyclicPermU64Mod(a, shift, modulus) — column `shift` of
 *     the negacyclic matrix when a is the basis vector. Mirrors
 *     upstream util.rs `negacyclic_perm(a, shift, modulus)`.
 *   * NegacyclicMatrixU64Mod(a, modulus) — column-major n×n
 *     matrix whose i-th column is `NegacyclicPermU64Mod(a, i,
 *     modulus)`. Mirrors upstream util.rs `negacyclic_matrix`.
 *   * ConcatHorizontalU64(v_a, a_rows, a_cols) — horizontally
 *     concatenate `v_a.size()` row-major matrices, each of shape
 *     a_rows × a_cols, into a single row-major matrix of shape
 *     a_rows × (a_cols * v_a.size()). Mirrors upstream
 *     `concat_horizontal`.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_UTIL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Negacyclic permutation: returns the `shift`-th column of the
// n×n negacyclic matrix built from `a`, modulo `modulus`.
//   out[i]     = a[shift - i]                      for 0 <= i <= shift
//   out[i]     = -a[n - (i - shift)] (mod modulus) for i > shift
//
// Returns an empty vector when `a` is empty. When `shift >= a.size()`
// the result is still defined (upstream Rust does not bounds-check
// the first loop), but `a[shift - i]` will access out-of-range; we
// return empty in that degenerate case to keep the C++ wrapper safe.
std::vector<std::uint64_t> NegacyclicPermU64Mod(
    const std::vector<std::uint64_t>& a, std::size_t shift,
    std::uint64_t modulus);

// Negacyclic matrix: column-major n×n matrix whose i-th column is
// `NegacyclicPermU64Mod(a, i, modulus)`. Output layout is
// `out[j * n + i]` = entry (i, j), matching upstream byte-for-byte.
// Returns an empty vector when `a` is empty.
std::vector<std::uint64_t> NegacyclicMatrixU64Mod(
    const std::vector<std::uint64_t>& a, std::uint64_t modulus);

// Horizontally concatenate `v_a.size()` row-major matrices of shape
// (a_rows × a_cols) into a single row-major matrix of shape
// (a_rows × (a_cols * v_a.size())). Output cell (i, k*a_cols + j)
// equals v_a[k][i*a_cols + j]. Returns an empty vector if any
// input matrix has wrong size or `v_a` is empty.
std::vector<std::uint64_t> ConcatHorizontalU64(
    const std::vector<std::vector<std::uint64_t>>& v_a,
    std::size_t a_rows, std::size_t a_cols);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_UTIL_H_
