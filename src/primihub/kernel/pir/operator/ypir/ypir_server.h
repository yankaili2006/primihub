/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_server — C++ port of upstream ypir `src/server.rs` (chunk 10 of
 * the 14-chunk YPIR port plan, docs/pir/ypir-port-plan.md).
 *
 * Sub-chunk 10a ported the self-contained leaf functions (GenerateYConstants
 * / SplitAlloc / DbRowsPadded / DbCols). Sub-chunk 10b adds the YServer<T>
 * data container: the transposed DB buffer + new() + element accessors. This
 * is the data layer only; the crypto pipelines (multiply_*_with_db, hint
 * generation, offline/online computation) and the AVX512 dot-product land in
 * later sub-chunks (see docs/pir/server-port-plan.md).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"
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

// Port of server.rs `YServer<'a, T>` (sub-chunk 10b: container + new() +
// accessors only). T is the per-element plaintext type (uint8_t / uint16_t
// / uint32_t). The DB is stored TRANSPOSED (column-major over padded rows):
// element (row, col) lives at col*db_rows_padded + row.
//
// new() consumes db_rows*db_cols elements from `db` in row-major order
// (row outer, col inner); `inp_transposed` switches the storage index to
// row-major (i*db_cols + j). The AlignedMemory64 of upstream is mirrored by
// a std::vector<uint64_t> (8-byte aligned, enough for T) reinterpreted as T.
// For the non-simplepir case new() also derives smaller_params (the
// "DoublePIR round" parameters) exactly as upstream.
template <typename T>
class YServer {
 public:
  YServer(const Params& params, const std::vector<T>& db, bool is_simplepir,
          bool inp_transposed, bool pad_rows)
      : params_(&params),
        smaller_params_(params),
        is_simplepir_(is_simplepir),
        pad_rows_(pad_rows) {
    const std::size_t db_rows = static_cast<std::size_t>(1)
                                << (params.db_dim_1 + params.poly_len_log2);
    db_rows_padded_ = DbRowsPadded(params, pad_rows);
    db_cols_ = DbCols(params, is_simplepir);

    const std::size_t sz_bytes = db_rows_padded_ * db_cols_ * sizeof(T);
    db_buf_.assign((sz_bytes + 7) / 8, 0);
    T* ptr = reinterpret_cast<T*>(db_buf_.data());

    std::size_t cnt = 0;
    for (std::size_t i = 0; i < db_rows; ++i) {
      for (std::size_t j = 0; j < db_cols_; ++j) {
        const std::size_t idx =
            inp_transposed ? (i * db_cols_ + j) : (j * db_rows_padded_ + i);
        ptr[idx] = db[cnt++];
      }
    }

    // Parameters for the second ("DoublePIR") round.
    if (!is_simplepir) {
      const LweParams lwe = LweParams::Default();
      const std::size_t pt_bits = static_cast<std::size_t>(
          std::floor(std::log2(static_cast<double>(params.pt_modulus))));
      const double blowup_factor =
          static_cast<double>(lwe.q2_bits) / static_cast<double>(pt_bits);
      smaller_params_.db_dim_1 = params.db_dim_2;
      smaller_params_.db_dim_2 = static_cast<std::size_t>(
          std::ceil(std::log2(blowup_factor *
                              static_cast<double>(lwe.n + 1) /
                              static_cast<double>(params.poly_len))));
    }
  }

  std::size_t DbRowsPaddedSelf() const { return db_rows_padded_; }
  std::size_t DbColsSelf() const { return db_cols_; }
  const Params& smaller_params() const { return smaller_params_; }

  // Reinterpret the aligned u64 buffer as the element type T.
  const T* Db() const { return reinterpret_cast<const T*>(db_buf_.data()); }

  // Element (row, col) — stored transposed.
  T GetElem(std::size_t row, std::size_t col) const {
    return Db()[col * db_rows_padded_ + row];
  }

  // The db_cols() elements of a logical row.
  std::vector<T> GetRow(std::size_t row) const {
    std::vector<T> res;
    res.reserve(db_cols_);
    for (std::size_t col = 0; col < db_cols_; ++col)
      res.push_back(GetElem(row, col));
    return res;
  }

 private:
  const Params* params_;
  Params smaller_params_;
  std::vector<std::uint64_t> db_buf_;  // stored transposed; reinterpreted as T
  std::size_t db_rows_padded_ = 0;
  std::size_t db_cols_ = 0;
  bool is_simplepir_ = false;
  bool pad_rows_ = false;
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SERVER_H_
