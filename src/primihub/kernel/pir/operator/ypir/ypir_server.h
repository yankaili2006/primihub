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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution_ntt.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_transpose.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

namespace primihub::pir::ypir {

// scheme.rs public-seed indices (scheme.rs is chunk 11; these two trivial
// constants are needed earlier by multiply_with_db_ring's negacyclic-perm
// gate and are defined here until scheme lands).
inline constexpr std::uint8_t kSeed0 = 0;
inline constexpr std::uint8_t kSeed1 = 1;

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

  // Port of server.rs YServer::multiply_with_db_ring (sub-chunk 10c). For each
  // column in [col_start, col_end), forms the ring (RLWE) product
  //   sum_row preprocessed_query[row] * db_poly(col, row)
  // where db_poly(col,row) is the poly_len-coefficient polynomial at
  // db[col*db_rows + row*poly_len ..], accumulates in the NTT domain,
  // inverse-NTTs, applies negacyclic_perm for the SEED_0 (first-mul,
  // non-simplepir) case, and transposes the col-major result to
  // poly_len x num_cols. `ctx` supplies the HEXL NTT (upstream carries NTT
  // tables on Params). preprocessed_query has 1<<db_dim_1 entries (1x1 NTT).
  //
  // NOTE: upstream accumulates lazily (add_into_no_reduce except the last
  // row); here AddNtt reduces every step, yielding the identical sum mod q.
  std::vector<std::uint64_t> MultiplyWithDbRing(
      const NttContext& ctx,
      const std::vector<PolyMatrixNTT>& preprocessed_query,
      std::size_t col_start, std::size_t col_end,
      std::uint8_t seed_idx) const {
    const Params& params = *params_;
    const std::size_t db_rows_poly = static_cast<std::size_t>(1)
                                     << params.db_dim_1;
    const std::size_t db_rows = static_cast<std::size_t>(1)
                                << (params.db_dim_1 + params.poly_len_log2);
    assert(preprocessed_query.size() == db_rows_poly);

    const T* db = Db();
    std::vector<std::uint64_t> result;
    result.reserve((col_end - col_start) * params.poly_len);

    for (std::size_t col = col_start; col < col_end; ++col) {
      PolyMatrixNTT sum = ctx.ZeroNtt(1, 1);
      for (std::size_t row = 0; row < db_rows_poly; ++row) {
        PolyMatrixRaw db_elem_poly = ctx.ZeroRaw(1, 1);
        for (std::size_t z = 0; z < params.poly_len; ++z)
          db_elem_poly.data[z] = static_cast<std::uint64_t>(
              db[col * db_rows + row * params.poly_len + z]);
        const PolyMatrixNTT db_elem_ntt = ctx.ToNtt(db_elem_poly);
        const PolyMatrixNTT prod =
            MultiplyNtt(params, preprocessed_query[row], db_elem_ntt);
        sum = AddNtt(params, sum, prod);
      }

      const PolyMatrixRaw sum_raw = ctx.FromNtt(sum);
      const std::uint64_t* s = sum_raw.Poly(0, 0, params.poly_len);
      if (seed_idx == kSeed0 && !is_simplepir_) {
        std::vector<std::uint64_t> col_poly(s, s + params.poly_len);
        const std::vector<std::uint64_t> t =
            NegacyclicPermU64Mod(col_poly, 0, params.modulus);
        result.insert(result.end(), t.begin(), t.end());
      } else {
        result.insert(result.end(), s, s + params.poly_len);
      }
    }

    return TransposeGeneric<std::uint64_t>(result, col_end - col_start,
                                           params.poly_len);
  }

  // Port of server.rs YServer::generate_hint_0_ring. Computes the offline
  // hint H0 = A * DB (n x db_cols) where A (n x db_rows) is the block-
  // negacyclic pseudorandom matrix from get_seed(SEED_0): for each n-row
  // block, sample n u32, negacyclic_perm_u32 them, and ring-convolve with the
  // db column slice (CRT NTT). Products are accumulated lazily in the NTT
  // domain (up to max_adds before a Barrett fold + Raw reconstruction) to
  // avoid u64 overflow, mirroring upstream. Returns hint_0 (n*db_cols,
  // row-major), each entry reduced mod 2^32. Uses the LWE n (1024); requires
  // db_rows a multiple of n. db element values must be < pt_modulus (the noise
  // budget / no-Q-overflow analysis assumes this).
  std::vector<std::uint64_t> GenerateHint0Ring() const {
    const Params& p = *params_;
    const std::size_t db_rows = static_cast<std::size_t>(1)
                                << (p.db_dim_1 + p.poly_len_log2);
    const std::size_t db_cols = db_cols_;
    const LweParams lwe = LweParams::Default();
    const std::size_t n = lwe.n;
    Convolution conv(n);
    std::vector<std::uint64_t> hint_0(n * db_cols, 0);
    const std::size_t convd_len = conv.CrtCount() * conv.PolyLen();
    const std::size_t num_outer = db_rows / n;

    ChaChaRng rng_pub = ChaChaRng::FromSeed(GetSeed(kSeed0));
    std::vector<std::vector<std::uint32_t>> v_nega_perm_a;
    v_nega_perm_a.reserve(num_outer);
    for (std::size_t k = 0; k < num_outer; ++k) {
      std::vector<std::uint32_t> a(n);
      for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng_pub.NextU32();
      v_nega_perm_a.push_back(conv.Ntt(NegacyclicPermU32(a)));
    }

    const std::uint64_t log2_conv_output =
        Log2(lwe.modulus) + Log2(static_cast<std::uint64_t>(lwe.n)) +
        Log2(lwe.pt_modulus);
    const std::uint64_t log2_modulus = Log2(conv.ProductModulus());
    assert(log2_modulus > log2_conv_output + 1);
    const std::uint64_t log2_max_adds = log2_modulus - log2_conv_output - 1;
    const std::size_t max_adds = static_cast<std::size_t>(1) << log2_max_adds;

    const T* db = Db();
    for (std::size_t col = 0; col < db_cols; ++col) {
      std::vector<std::uint64_t> tmp_col(convd_len, 0);
      for (std::size_t outer_row = 0; outer_row < num_outer; ++outer_row) {
        const std::size_t start_idx = col * db_rows_padded_ + outer_row * n;
        std::vector<std::uint32_t> pt_col_u32(n);
        for (std::size_t z = 0; z < n; ++z)
          pt_col_u32[z] = static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(db[start_idx + z]));
        const std::vector<std::uint32_t> pt_ntt = conv.Ntt(pt_col_u32);
        const std::vector<std::uint32_t> convolved =
            conv.PointwiseMul(v_nega_perm_a[outer_row], pt_ntt);
        for (std::size_t r = 0; r < convd_len; ++r) tmp_col[r] += convolved[r];

        if (outer_row % max_adds == max_adds - 1 ||
            outer_row == num_outer - 1) {
          std::vector<std::uint32_t> col_poly_u32(convd_len, 0);
          for (std::size_t i = 0; i < conv.CrtCount(); ++i)
            for (std::size_t j = 0; j < conv.PolyLen(); ++j)
              col_poly_u32[i * conv.PolyLen() + j] =
                  static_cast<std::uint32_t>(conv.BarrettCoeff(
                      tmp_col[i * conv.PolyLen() + j], i));
          const std::vector<std::uint32_t> col_poly_raw = conv.Raw(col_poly_u32);
          for (std::size_t i = 0; i < n; ++i) {
            hint_0[i * db_cols + col] += col_poly_raw[i];
            hint_0[i * db_cols + col] %= (static_cast<std::uint64_t>(1) << 32);
          }
          std::fill(tmp_col.begin(), tmp_col.end(), 0);
        }
      }
    }
    return hint_0;
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
