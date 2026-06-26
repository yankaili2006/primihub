/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_server — C++ port of upstream ypir `src/server.rs` (chunk 10 of
 * the 14-chunk YPIR port plan, docs/pir/ypir-port-plan.md).
 *
 * NOTE ON SCOPE (sub-chunk 10a): the bulk of server.rs (the YServer<T>
 * struct, multiply_*_with_db, hint generation, and the offline/online
 * computation pipelines) sits at the TOP of the YPIR dependency graph —
 * it calls into scheme.rs (SEED_*, get_seed, YPIRParams), client.rs
 * (Client / YClient::generate_query_impl), packing.rs' pack_many_lwes /
 * precompute_pack (NOT yet ported — see docs/pir/server-port-plan.md),
 * and the AVX512 fast_batched_dot_product kernel (kernel.rs, needs an
 * AVX512 host). Those land in later sub-chunks once their dependencies
 * exist. This file ports only the self-contained leaf functions that
 * depend solely on already-ported C++ primitives and have clean,
 * self-verifying oracles.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Holds the per-level Y monomial constants used by the ring-packing
// step. Both vectors have `params.poly_len_log2` entries; entry k (for
// k in 0..poly_len_log2) corresponds to upstream num_cts_log2 = k+1,
// i.e. the monomial X^(poly_len / 2^(k+1)).
struct YConstants {
  std::vector<PolyMatrixNTT> y;      // NTT( X^(poly_len/2^(k+1)) )
  std::vector<PolyMatrixNTT> neg_y;  // NTT( -X^(poly_len/2^(k+1)) )
};

// Port of server.rs `generate_y_constants`. For num_cts_log2 in
// 1..=poly_len_log2: Y = X^(poly_len/num_cts) (coeff 1), neg_Y has
// that coeff set to modulus-1; both are returned in NTT form.
YConstants GenerateYConstants(const NttContext& ctx);

// Port of server.rs `split_alloc`. Re-chunks a per-column bitstream
// from `inp_mod_bits` granularity to `pt_bits` granularity.
//
// Input  `buf` is rows*cols u64's (row-major). Output is out_rows*cols
// u16's (row-major). For each column, the `rows` inputs are written
// contiguously (inp_mod_bits each) into a scratch byte buffer — except
// the LAST input, which is written at bit offset `special_bit_offs` —
// then re-read in `pt_bits`-wide chunks into the output column.
//
// Requires out_rows >= rows and inp_mod_bits >= pt_bits.
std::vector<std::uint16_t> SplitAlloc(const std::vector<std::uint64_t>& buf,
                                      std::size_t special_bit_offs,
                                      std::size_t rows, std::size_t cols,
                                      std::size_t out_rows,
                                      std::size_t inp_mod_bits,
                                      std::size_t pt_bits);

// Port of server.rs `DbRowsPadded` trait / YServer::db_rows_padded.
// The padded-rows formula is commented out upstream, so both pad_rows
// settings currently return the same value: 1 << (db_dim_1 +
// poly_len_log2). `pad_rows` is kept for faithful call-site mirroring.
std::size_t DbRowsPadded(const Params& params, bool pad_rows);

// Port of server.rs YServer::db_cols. is_simplepir ? instances*poly_len
// : 1 << (db_dim_2 + poly_len_log2).
std::size_t DbCols(const Params& params, bool is_simplepir);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_
