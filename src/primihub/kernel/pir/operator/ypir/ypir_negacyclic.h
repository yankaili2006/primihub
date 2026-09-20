/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_negacyclic — three standalone negacyclic helpers from upstream
 * menonsamir/ypir@a73e550a src/convolution.rs. Chunk 6a of the YPIR
 * port (see docs/pir/ypir-port-plan.md): the Spiral-free subset of
 * convolution.rs. The Convolution struct, NTT / IRT methods, and the
 * trait-templated `naive_multiply_matrices<T: ToU64>` are deferred
 * to chunk 6b because they depend on Spiral's PolyMatrixRaw / Params
 * / NTT primitives (task 3 / Phase 3 partial).
 *
 * What's here (all wrap-around u32 arithmetic, zero crypto-library
 * dependency):
 *   * NegacyclicMatrixU32(b) — build the n×n matrix M such that
 *     M·a = naive_negacyclic_convolve(a, b) when a is a column.
 *     Output is upstream's _transposed_ layout: M[j*n + i] is the
 *     (i, j) cell (column-major), matching the Rust comment
 *     `nb: transposed`. Used by lwe.rs LWEClient::encrypt_many
 *     (chunk 3b unblocker).
 *   * NaiveNegacyclicConvolveU32(a, b) — direct O(n²) reference
 *     convolution. Same input/output as multiplying via the
 *     negacyclic matrix; kept as a reference for the matrix-vs-
 *     convolution equivalence test.
 *   * NegacyclicPermU32(a) — single-shift negacyclic permutation:
 *     res[0] = a[0]; res[i] = -a[(n-i) mod n] for i > 0. Used by
 *     packing.rs (chunk 9, blocked).
 *
 * Wrapping semantics: all arithmetic is `wrapping_add` /
 * `wrapping_mul` / `wrapping_neg` on u32, which in C++ unsigned
 * arithmetic is already defined modular behavior — no special
 * helpers needed.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_NEGACYCLIC_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_NEGACYCLIC_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Returns the column-major negacyclic matrix M for input vector b
// of length n. Reading order is `res[j * n + i]` for cell (i, j).
// Mirrors upstream `negacyclic_matrix_u32(b)` byte-for-byte.
//
// Returns an empty vector if `b` is empty (n == 0); does not allocate.
std::vector<std::uint32_t> NegacyclicMatrixU32(const std::vector<std::uint32_t>& b);

// Naive O(n^2) negacyclic convolution. `a` and `b` must have the
// same length; mismatched-length inputs return an empty vector to
// preserve "ignore mistakes" wrapper semantics (upstream Rust would
// `assert!()`-panic, which is harsher than we want at our boundary).
// Mirrors upstream `naive_negacyclic_convolve(a, b)`.
std::vector<std::uint32_t> NaiveNegacyclicConvolveU32(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b);

// Single-shift negacyclic permutation:
//   res[0] = a[0]
//   res[i] = (-a[(n - i) mod n])  for 1 <= i < n
// Mirrors upstream `negacyclic_perm_u32(a)`.
//
// Returns an empty vector if `a` is empty.
std::vector<std::uint32_t> NegacyclicPermU32(
    const std::vector<std::uint32_t>& a);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_NEGACYCLIC_H_
