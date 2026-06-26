/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_client — C++ port of upstream ypir `src/client.rs` (chunk 12). This
 * sub-chunk lands the self-contained query/hint helper generate_matrix_ring,
 * which expands a ChaCha20 public-randomness stream into a block-negacyclic
 * u32 matrix (each n x n block is the negacyclic matrix of n freshly sampled
 * coefficients). It is used by server.rs generate_hint_0 and the LWE query
 * path.
 *
 * NOTE: the RLWE query path (YClient::generate_query_impl) depends on the
 * spiral RLWE Client (encrypt_matrix_reg / encrypt_matrix_scaled_reg), which
 * is not yet ported; it lands in a later sub-chunk once that encryption (or a
 * RegevEncrypt-based equivalent) is wired with an end-to-end decode test.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CLIENT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"

namespace primihub::pir::ypir {

// Port of client.rs generate_matrix_ring. Fills a rows x cols u32 matrix
// (row-major) from public randomness: for each (i, j) block (i over rows/n,
// j over cols/n), sample n u32 coefficients from rng_pub (in order), build
// their n x n negacyclic matrix (NegacyclicMatrixU32), and place it at block
// (i, j): out[(i*n + k)*cols + (j*n + l)] = mat[k*n + l].
//
// Requires rows % n == 0 and cols % n == 0.
std::vector<std::uint32_t> GenerateMatrixRing(ChaChaRng& rng_pub, std::size_t n,
                                              std::size_t rows, std::size_t cols);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CLIENT_H_
