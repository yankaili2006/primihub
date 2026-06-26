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
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

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

// Port of client.rs YClient — the RLWE query generation + response decoding
// layer (chunk 12b-3). Wraps a spiral Client (borrowed). GSW / LWE-branch
// (generate_query SEED_0) paths are later sub-chunks.
class YClient {
 public:
  YClient(const NttContext& ctx, const Client& client)
      : ctx_(&ctx), client_(&client) {}

  // Port of YClient::generate_query_impl. Produces (1<<dim_log2) raw 2x1
  // ciphertexts: a one-hot selector for `index` (scale_k = modulus/pt_modulus
  // at coefficient index%poly_len of block index/poly_len), optionally scaled
  // by poly_len^{-1} (packing), each encrypted via the Client's
  // encrypt_matrix_scaled_reg with scale = poly_len^{-1} (so the query noise
  // is e*poly_len^{-1}, cancelled by the *poly_len in downstream packing).
  // rng_pub is derived from get_seed(public_seed_idx); `noise_rng` supplies
  // the per-sample Gaussian noise (caller-seeded for determinism; upstream
  // uses from_entropy).
  std::vector<PolyMatrixRaw> GenerateQueryImpl(std::uint8_t public_seed_idx,
                                               std::size_t dim_log2, bool packing,
                                               std::size_t index,
                                               ChaChaRng& noise_rng) const;

  // Port of YClient::decode_response. `response` is (poly_len+1) x db_cols
  // (row-major): the LWE answer. For each column, phase =
  // sum_i response[i][col]*sk_reg[i] + response[poly_len][col] (mod modulus),
  // then rescaled modulus->pt_modulus. Returns db_cols values.
  std::vector<std::uint64_t> DecodeResponse(
      const std::vector<std::uint64_t>& response, std::size_t db_cols) const;

 private:
  const NttContext* ctx_;
  const Client* client_;
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CLIENT_H_
