/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_api.h"

#include <sstream>
#include <utility>

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_format.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_lwe_consts.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

namespace primihub::pir::frodo {

// ---------- Query ----------

Query::Query(std::vector<std::uint32_t> data)
    : data_(std::move(data)) {}

// ---------- Response ----------

Response::Response(std::vector<std::uint32_t> data)
    : data_(std::move(data)) {}

std::vector<std::uint32_t> Response::ParseOutputAsRow(
    const QueryParams& qp) const {
  // Upstream:
  //   let rounding_factor = get_rounding_factor(qp.plaintext_bits);
  //   let rounding_floor  = get_rounding_floor(qp.plaintext_bits);
  //   let plaintext_size  = get_plaintext_size(qp.plaintext_bits);
  //   (0..Database::get_matrix_width(qp.elem_size, qp.plaintext_bits))
  //     .map(|i| {
  //       let unscaled = self.0[i].wrapping_sub(qp.rhs[i]);
  //       let scaled_res = unscaled / rounding_factor;
  //       let scaled_rem = unscaled % rounding_factor;
  //       let mut rounded = scaled_res;
  //       if scaled_rem > rounding_floor { rounded += 1; }
  //       rounded % plaintext_size
  //     })
  //     .collect()
  const std::size_t pt_bits = qp.GetPlaintextBits();
  const std::uint32_t rounding_factor = GetRoundingFactor(pt_bits);
  const std::uint32_t rounding_floor = GetRoundingFloor(pt_bits);
  const std::uint32_t plaintext_size = GetPlaintextSize(pt_bits);
  const std::size_t row_width =
      Database::GetMatrixWidth(qp.GetElemSize(), pt_bits);

  std::vector<std::uint32_t> row;
  row.reserve(row_width);
  for (std::size_t i = 0; i < row_width; ++i) {
    // C++ unsigned subtraction wraps mod 2^32 by spec.
    const std::uint32_t unscaled = data_[i] - qp.GetRhs()[i];
    // Guard against the chunk-1 saturation case (plaintext_bits
    // >= 32 → rounding_factor == 0). Upstream would panic on /0.
    if (rounding_factor == 0u) {
      row.push_back(0u);
      continue;
    }
    const std::uint32_t scaled_res = unscaled / rounding_factor;
    const std::uint32_t scaled_rem = unscaled % rounding_factor;
    std::uint32_t rounded = scaled_res;
    if (scaled_rem > rounding_floor) {
      rounded += 1;
    }
    // plaintext_size == 0 in the saturation case; guard to avoid
    // a UB modulo-by-zero.
    if (plaintext_size == 0u) {
      row.push_back(rounded);
    } else {
      row.push_back(rounded % plaintext_size);
    }
  }
  return row;
}

std::vector<std::uint8_t> Response::ParseOutputAsBytes(
    const QueryParams& qp) const {
  const auto row = ParseOutputAsRow(qp);
  return BytesFromU32Slice(row, qp.GetPlaintextBits(),
                           qp.GetElemSize());
}

std::string Response::ParseOutputAsBase64(
    const QueryParams& qp) const {
  const auto row = ParseOutputAsRow(qp);
  return Base64FromU32Slice(row, qp.GetPlaintextBits(),
                            qp.GetElemSize());
}

// ---------- QueryParams ----------

QueryParams::QueryParams()
    : elem_size_(0), plaintext_bits_(0), used_(false) {}

retcode QueryParams::New(const CommonParams& cp, const BaseParams& bp,
                         QueryParams* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "QueryParams::New: out must be non-null";
    return retcode::FAIL;
  }
  const auto s = RandomTernaryVector(bp.GetDim());
  std::vector<std::uint32_t> lhs;
  const auto rc_lhs = cp.MultLeft(s, &lhs, err);
  if (rc_lhs != retcode::SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "QueryParams::New: CommonParams::MultLeft failed: "
          << *err;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  std::vector<std::uint32_t> rhs;
  const auto rc_rhs = bp.MultRight(s, &rhs, err);
  if (rc_rhs != retcode::SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "QueryParams::New: BaseParams::MultRight failed: "
          << *err;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  out->lhs_ = std::move(lhs);
  out->rhs_ = std::move(rhs);
  out->elem_size_ = bp.GetElemSize();
  out->plaintext_bits_ = bp.GetPlaintextBits();
  out->used_ = false;
  return retcode::SUCCESS;
}

retcode QueryParams::GenerateQuery(std::size_t row_index, Query* out,
                                   std::string* err) {
  if (out == nullptr) {
    if (err) *err = "QueryParams::GenerateQuery: out must be non-null";
    return retcode::FAIL;
  }
  if (used_) {
    if (err) {
      *err = "QueryParams::GenerateQuery: ErrorQueryParamsReused — "
             "this QueryParams instance has already produced a "
             "Query; the LWE secret would leak on reuse.";
    }
    return retcode::FAIL;
  }
  if (row_index >= lhs_.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "QueryParams::GenerateQuery: row_index=" << row_index
          << " out of range [0, " << lhs_.size() << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  used_ = true;
  const std::uint32_t query_indicator =
      GetRoundingFactor(plaintext_bits_);
  // Mirror upstream's overflowing_add check. C++ has no direct
  // overflowing_add; detect via post-condition.
  const std::uint32_t before = lhs_[row_index];
  const std::uint32_t sum = before + query_indicator;
  // Overflow iff sum < before (unsigned wrap detection).
  if (sum < before) {
    if (err) {
      std::ostringstream oss;
      oss << "QueryParams::GenerateQuery: ErrorOverflownAdd — "
          << "lhs[row_index=" << row_index << "]=" << before
          << " + rounding_factor=" << query_indicator
          << " would overflow u32";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  std::vector<std::uint32_t> q = lhs_;  // clone (upstream clones too)
  q[row_index] = sum;
  *out = Query(std::move(q));
  return retcode::SUCCESS;
}

// ---------- Shard ----------

Shard::Shard() = default;

retcode Shard::FromBase64StringsWithSeed(
    const std::vector<std::string>& base64_strs, std::size_t dim,
    std::size_t m, std::size_t elem_size, std::size_t plaintext_bits,
    const SeedBytes& seed, Shard* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "Shard::FromBase64StringsWithSeed: out must be non-null";
    return retcode::FAIL;
  }
  Database db;
  const auto rc_db =
      Database::New(base64_strs, m, elem_size, plaintext_bits, &db, err);
  if (rc_db != retcode::SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "Shard::FromBase64StringsWithSeed: Database::New failed: "
          << *err;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  BaseParams bp;
  const auto rc_bp = BaseParams::NewWithSeed(db, dim, seed, &bp, err);
  if (rc_bp != retcode::SUCCESS) {
    if (err) {
      std::ostringstream oss;
      oss << "Shard::FromBase64StringsWithSeed: BaseParams::NewWithSeed "
          << "failed: " << *err;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Move DB into the shard. DB's column-form is what BaseParams
  // expects too (the rhs was built against the DB we just gave to
  // bp, so storing the same db keeps Respond's vec_mult valid).
  out->db_ = std::move(db);
  out->base_params_ = std::move(bp);
  return retcode::SUCCESS;
}

retcode Shard::FromBase64Strings(
    const std::vector<std::string>& base64_strs, std::size_t dim,
    std::size_t m, std::size_t elem_size, std::size_t plaintext_bits,
    Shard* out, std::string* err) {
  return FromBase64StringsWithSeed(base64_strs, dim, m, elem_size,
                                   plaintext_bits, GenerateSeed(),
                                   out, err);
}

retcode Shard::Respond(const Query& query, Response* out,
                       std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Shard::Respond: out must be non-null";
    return retcode::FAIL;
  }
  // Upstream:
  //   Response((0..self.db.get_matrix_width_self())
  //     .map(|i| self.db.vec_mult(q, i))
  //     .collect())
  const std::size_t w = db_.GetMatrixWidthSelf();
  std::vector<std::uint32_t> data;
  data.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    // db.VecMult precondition: query length == db column length.
    // We trust the caller (matches upstream's lack of validation).
    data.push_back(db_.VecMult(query.AsSlice(), i));
  }
  *out = Response(std::move(data));
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::frodo
